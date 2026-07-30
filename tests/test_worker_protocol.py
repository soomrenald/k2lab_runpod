from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from k2core.inference import GenerationRequest

from k2_region_lab.semantic_conditioning import SemanticAttentionError
from k2_region_lab.agent.jobs import (
    JobError,
    SubprocessWorkerExecutor,
)
from k2_region_lab.worker import entrypoint as worker_entrypoint
from k2_region_lab.worker.entrypoint import (
    model_directories,
    registered_native_model,
    selected_inference_backend,
)
from k2_region_lab.worker.protocol import CommandKind, classify_worker_error


def descriptor(shape: list[int], dtype: str = "BF16") -> dict:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, 2]}


def write_header(path: Path, tensors: dict) -> None:
    encoded = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def write_compatible_artifacts(root: Path) -> tuple[Path, Path, Path]:
    diffusion = root / "diffusion_models"
    text = root / "text_encoders"
    vae = root / "vae"
    diffusion.mkdir()
    text.mkdir()
    vae.mkdir()

    transformer = {
        "first.weight": descriptor([6144, 64]),
        "blocks.0.attn.wq.weight": descriptor([6144, 6144], "F8_E4M3"),
        "blocks.0.attn.wk.weight": descriptor([1536, 6144], "F8_E4M3"),
        "blocks.0.attn.wv.weight": descriptor([1536, 6144], "F8_E4M3"),
        "blocks.27.attn.wq.weight": descriptor([6144, 6144], "F8_E4M3"),
        "txtfusion.projector.weight": descriptor([1, 12]),
        "txtfusion.layerwise_blocks.0.prenorm.scale": descriptor([2560]),
        "last.linear.weight": descriptor([64, 6144]),
        "blocks.0.attn.wq.weight_scale": descriptor([], "F32"),
    }
    for index in range(1, 27):
        transformer[f"blocks.{index}.placeholder"] = descriptor([1])
    write_header(diffusion / "krea2_turbo_fp8_scaled.safetensors", transformer)

    text_tensors = {
        "model.embed_tokens.weight": descriptor([151936, 2560]),
        "model.layers.0.self_attn.q_proj.weight": descriptor([4096, 2560], "F8_E4M3"),
        "model.layers.35.self_attn.q_proj.weight": descriptor([4096, 2560], "F8_E4M3"),
        "model.norm.weight": descriptor([2560]),
    }
    for index in range(1, 35):
        text_tensors[f"model.layers.{index}.placeholder"] = descriptor([1])
    write_header(text / "qwen3vl_4b_fp8_scaled.safetensors", text_tensors)

    write_header(
        vae / "qwen_image_vae.safetensors",
        {
            "encoder.conv1.weight": descriptor([96, 3, 3, 3, 3]),
            "decoder.conv1.weight": descriptor([384, 16, 3, 3, 3]),
            "conv1.weight": descriptor([32, 32, 1, 1, 1]),
            "conv2.weight": descriptor([16, 16, 1, 1, 1]),
        },
    )
    return diffusion, text, vae


class WorkerProtocolTests(unittest.TestCase):
    def test_image_edit_has_a_dedicated_worker_command(self) -> None:
        self.assertEqual(CommandKind.EDIT_IMAGE.value, "edit_image")

    def test_system_ram_guard_is_not_misreported_as_a_lora_failure(self) -> None:
        code, message = classify_worker_error(
            MemoryError("available system RAM is below the 12.0 GiB guard"),
            CommandKind.GENERATE_BASELINE,
        )

        self.assertEqual(code, "worker_ram_low")
        self.assertIn("system RAM", message)

    def test_semantic_errors_keep_their_stable_browser_code(self) -> None:
        code, message = classify_worker_error(
            SemanticAttentionError("private attention detail"),
            CommandKind.GENERATE_BASELINE,
        )

        self.assertEqual(code, "semantic_attention_failed")
        self.assertNotIn("private attention detail", message)

    def test_lora_directory_is_distinct_from_lora_job_specifications(self) -> None:
        payload = {
            "diffusion_models": "/workspace/models/diffusion_models",
            "text_encoders": "/workspace/models/text_encoders",
            "vae": "/workspace/models/vae",
            "lora_directory": "/workspace/models/loras",
            "loras": [{"id": "character", "path": "/workspace/models/loras/person.safetensors"}],
        }

        directories = model_directories(payload)

        self.assertEqual(directories.loras, Path("/workspace/models/loras"))

    def test_native_backend_selection_and_registered_model_use_supplied_hashes(
        self,
    ) -> None:
        payload = {
            "inference_backend": "native",
            "diffusion_model_file": "/workspace/transformer.safetensors",
            "diffusion_model_sha256": "1" * 64,
            "text_encoder_file": "/workspace/text.safetensors",
            "text_encoder_sha256": "2" * 64,
            "vae_file": "/workspace/vae.safetensors",
            "vae_sha256": "3" * 64,
            "tokenizer_path": "/workspace/tokenizer",
            "tokenizer_sha256": "4" * 64,
        }

        model = registered_native_model(payload)

        self.assertEqual(selected_inference_backend(payload), "native")
        self.assertEqual(model.architecture, "krea2")
        self.assertEqual(model.transformer.sha256, "1" * 64)
        self.assertEqual(model.text_encoder.sha256, "2" * 64)
        self.assertEqual(model.vae.sha256, "3" * 64)
        self.assertIsNotNone(model.tokenizer)
        assert model.tokenizer is not None
        self.assertEqual(model.tokenizer.sha256, "4" * 64)

    def test_worker_rejects_unknown_backend_before_gpu_work(self) -> None:
        with self.assertRaisesRegex(Exception, "Unsupported inference backend"):
            selected_inference_backend({"inference_backend": "mystery"})

    def test_native_worker_loads_generates_emits_output_and_unloads(self) -> None:
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "gate11"
            / "native_clean_generation.json"
        )
        generation_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        payload = {
            **generation_fixture,
            "inference_backend": "native",
            "comfyui_root": "/opt/ComfyUI",
            "diffusion_models": "/workspace/models/diffusion_models",
            "text_encoders": "/workspace/models/text_encoders",
            "vae": "/workspace/models/vae",
            "diffusion_model_file": "/workspace/transformer.safetensors",
            "diffusion_model_sha256": "1" * 64,
            "text_encoder_file": "/workspace/text.safetensors",
            "text_encoder_sha256": "2" * 64,
            "vae_file": "/workspace/vae.safetensors",
            "vae_sha256": "3" * 64,
            "tokenizer_path": "/workspace/tokenizer",
            "tokenizer_sha256": "4" * 64,
        }
        commands = (
            {"command_id": "load", "kind": "load_model", "payload": payload},
            {
                "command_id": "generate",
                "kind": "generate_baseline",
                "payload": payload,
            },
        )
        backend = Mock()
        backend.pipeline = None

        def load(_config):
            backend.pipeline = SimpleNamespace(loaded=True)
            return SimpleNamespace(metadata={"model_name": "krea2-runpod"})

        backend.load.side_effect = load
        backend.generate.return_value = SimpleNamespace(
            to_payload=lambda: {
                "image_path": "/workspace/outputs/native-contract.png"
            }
        )
        stdin = io.StringIO(
            "".join(json.dumps(command) + "\n" for command in commands)
        )
        stdout = io.StringIO()
        with (
            patch.object(worker_entrypoint, "configure_debug_logging"),
            patch.object(worker_entrypoint, "NativeK2Backend", return_value=backend),
            patch.object(
                worker_entrypoint,
                "discover_native_model_artifacts",
                return_value=SimpleNamespace(complete=True),
            ),
            patch.object(worker_entrypoint.sys, "stdin", stdin),
            patch.object(worker_entrypoint.sys, "stdout", stdout),
        ):
            exit_code = worker_entrypoint.main()

        self.assertEqual(exit_code, 0)
        backend.load.assert_called_once()
        backend.generate.assert_called_once()
        request = backend.generate.call_args.args[0]
        self.assertIsInstance(request, GenerationRequest)
        self.assertEqual(request.correlation_id, "generate")
        self.assertEqual(request.prompt, generation_fixture["prompt"])
        self.assertEqual(request.seed, generation_fixture["seed"])
        self.assertEqual(request.sampler, generation_fixture["sampler"])
        self.assertEqual(request.scheduler, generation_fixture["scheduler"])
        self.assertEqual(request.regions[0].region_id, "teapot")
        self.assertEqual(request.prompt_emphases[0].phrase, "ceramic")
        backend.unload.assert_called_once()
        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        output = next(
            event
            for event in events
            if event["command_id"] == "generate"
            and event["state"] == "ready"
        )
        self.assertEqual(output["payload"]["backend"], "native")
        self.assertEqual(
            output["payload"]["image_path"],
            "/workspace/outputs/native-contract.png",
        )

    def test_external_worker_probes_validates_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffusion, text, vae = write_compatible_artifacts(root)
            manifests = root / "manifests"
            payload = {
                "comfyui_root": str(root),
                "diffusion_models": str(diffusion),
                "text_encoders": str(text),
                "vae": str(vae),
                "manifest_directory": str(manifests),
            }
            commands = [
                {"command_id": "probe", "kind": "probe", "payload": payload},
                {
                    "command_id": "diagnose",
                    "kind": "diagnose_accelerator",
                    "payload": payload,
                },
                {"command_id": "validate", "kind": "validate_models", "payload": payload},
                {"command_id": "stop", "kind": "shutdown", "payload": payload},
            ]
            environment = os.environ.copy()
            project_root = Path(__file__).resolve().parents[1]
            environment["PYTHONPATH"] = str(project_root / "src")
            environment["K2LAB_DATA_DIR"] = str(root / "worker-data")
            process = subprocess.run(
                [sys.executable, "-m", "k2_region_lab.worker.entrypoint"],
                input="".join(json.dumps(command) + "\n" for command in commands),
                text=True,
                capture_output=True,
                env=environment,
                cwd=project_root,
                timeout=15,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            events = [json.loads(line) for line in process.stdout.splitlines()]
            validation = next(
                event
                for event in events
                if event["command_id"] == "validate" and event["state"] == "ready"
            )
            self.assertTrue(validation["payload"]["complete"])
            self.assertTrue(all(item["compatible"] for item in validation["payload"]["manifests"]))
            self.assertEqual(len(tuple(manifests.glob("*_tensor_manifest.json"))), 3)
            diagnostic = next(
                event
                for event in events
                if event["command_id"] == "diagnose"
                and event["message"] == "Accelerator diagnostics complete"
            )
            self.assertIn("python_executable", diagnostic["payload"])
            self.assertIn("device_paths", diagnostic["payload"])
            self.assertTrue(diagnostic["payload"]["recommendations"])


class SubprocessWorkerTimeoutTests(unittest.IsolatedAsyncioTestCase):
    class Stdin:
        def __init__(self) -> None:
            self.closing = False

        def write(self, _content: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closing = True

        def is_closing(self) -> bool:
            return self.closing

        async def wait_closed(self) -> None:
            return None

    class Stderr:
        async def read(self, _size: int) -> bytes:
            return b""

    class Stdout:
        def __init__(
            self, events: list[dict] | None = None, *, eof: bool = False
        ) -> None:
            self.events = list(events or [])
            self.eof = eof
            self.blocker = asyncio.Event()

        async def readline(self) -> bytes:
            if self.events:
                return (
                    json.dumps(self.events.pop(0), separators=(",", ":")) + "\n"
                ).encode()
            if self.eof:
                return b""
            await self.blocker.wait()
            return b""

    class Process:
        def __init__(self, stdout) -> None:
            self.stdin = SubprocessWorkerTimeoutTests.Stdin()
            self.stdout = stdout
            self.stderr = SubprocessWorkerTimeoutTests.Stderr()
            self.returncode = None

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    @staticmethod
    async def _on_event(_event: dict) -> None:
        return None

    @staticmethod
    def _commands(**payload) -> list[dict]:
        return [
            {
                "command_id": "generation",
                "kind": "generate_baseline",
                "payload": payload,
            }
        ]

    async def test_worker_logs_are_persisted_without_forwarding_secrets(
        self,
    ) -> None:
        executor = SubprocessWorkerExecutor(Path(sys.executable), Path.cwd())
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "K2LAB_AGENT_SESSION_TOKEN": "must-not-leak",
            },
            clear=True,
        ):
            environment = executor._worker_environment()

        self.assertEqual(environment["K2LAB_DATA_DIR"], str(Path.cwd()))
        self.assertNotIn("K2LAB_AGENT_SESSION_TOKEN", environment)

    async def test_worker_startup_timeout_is_distinct(self) -> None:
        process = self.Process(self.Stdout())
        executor = SubprocessWorkerExecutor(Path(sys.executable), Path.cwd())
        with patch(
            "k2_region_lab.agent.jobs.asyncio.create_subprocess_exec",
            return_value=process,
        ):
            with self.assertRaises(JobError) as caught:
                await executor.run(
                    self._commands(worker_startup_timeout_seconds=0.01),
                    self._on_event,
                )

        self.assertEqual(caught.exception.code, "worker_startup_timeout")
        self.assertEqual(process.returncode, -15)

    async def test_generation_timeout_starts_at_first_running_event(self) -> None:
        process = self.Process(
            self.Stdout(
                [
                    {
                        "command_id": "generation",
                        "state": "running",
                        "message": "Generation started",
                        "payload": {},
                    }
                ]
            )
        )
        executor = SubprocessWorkerExecutor(Path(sys.executable), Path.cwd())
        with patch(
            "k2_region_lab.agent.jobs.asyncio.create_subprocess_exec",
            return_value=process,
        ):
            with self.assertRaises(JobError) as caught:
                await executor.run(
                    self._commands(
                        worker_startup_timeout_seconds=1,
                        generation_timeout_seconds=0.01,
                    ),
                    self._on_event,
                )

        self.assertEqual(caught.exception.code, "generation_timeout")
        self.assertEqual(process.returncode, -15)

    async def test_worker_disconnect_requires_a_terminal_event(self) -> None:
        process = self.Process(self.Stdout(eof=True))
        executor = SubprocessWorkerExecutor(Path(sys.executable), Path.cwd())
        with patch(
            "k2_region_lab.agent.jobs.asyncio.create_subprocess_exec",
            return_value=process,
        ):
            with self.assertRaises(JobError) as caught:
                await executor.run(
                    self._commands(worker_startup_timeout_seconds=1),
                    self._on_event,
                )

        self.assertEqual(caught.exception.code, "worker_disconnected")

if __name__ == "__main__":
    unittest.main()
