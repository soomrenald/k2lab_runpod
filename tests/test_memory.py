from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from k2_region_lab.memory import (
    MEMORY_POLICIES,
    SystemMemorySnapshot,
    configure_comfy_vram_args,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
    oom_recovery_reserve_vram_gb,
    resolve_vram_mode,
    system_memory_snapshot,
)
from k2_region_lab.worker.runtime import CriticalGpuMemoryPressure, ComfyBaselineRuntime


class MemoryPolicyTests(unittest.TestCase):
    def test_cgroup_v2_memory_limit_takes_precedence_over_host_ram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.current").write_text("42500000000\n", encoding="utf-8")
            (root / "memory.max").write_text("50000000000\n", encoding="utf-8")
            (root / "memory.stat").write_text(
                "anon 32000000000\nfile 9000000000\n",
                encoding="utf-8",
            )

            snapshot = system_memory_snapshot(root)

        self.assertEqual(
            snapshot,
            SystemMemorySnapshot(
                source="cgroup_v2",
                total_bytes=50_000_000_000,
                used_bytes=42_500_000_000,
                available_bytes=7_500_000_000,
                anonymous_bytes=32_000_000_000,
                file_cache_bytes=9_000_000_000,
            ),
        )

    def test_cgroup_v1_memory_limit_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            memory.mkdir()
            (memory / "memory.usage_in_bytes").write_text("30\n", encoding="utf-8")
            (memory / "memory.limit_in_bytes").write_text("100\n", encoding="utf-8")
            (memory / "memory.stat").write_text(
                "total_rss 20\ntotal_cache 8\n",
                encoding="utf-8",
            )

            snapshot = system_memory_snapshot(root)

        self.assertEqual(snapshot.source, "cgroup_v1")
        self.assertEqual(snapshot.available_bytes, 70)
        self.assertEqual(snapshot.anonymous_bytes, 20)
        self.assertEqual(snapshot.file_cache_bytes, 8)

    def test_safe_16gb_policy_keeps_four_gib_free(self) -> None:
        policy = memory_policy("safe_16gb")
        self.assertEqual(policy.reserve_vram_gb, 4.0)
        self.assertEqual(policy.minimum_system_ram_gb, 14.0)
        self.assertTrue(policy.oom_recovery)

    def test_policy_keys_are_unique(self) -> None:
        keys = [policy.key for policy in MEMORY_POLICIES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_saved_value_cannot_weaken_policy_floor(self) -> None:
        self.assertEqual(effective_reserve_vram_gb("safe_16gb", 2.0), 4.0)
        self.assertEqual(effective_reserve_vram_gb("emergency", 4.0), 5.5)
        self.assertEqual(effective_reserve_vram_gb("balanced", 3.5), 3.5)
        self.assertEqual(effective_minimum_system_ram_gb("safe_16gb", 12.0), 14.0)
        self.assertEqual(effective_minimum_system_ram_gb("emergency", 14.0), 16.0)
        self.assertEqual(effective_minimum_system_ram_gb("balanced", 13.0), 13.0)

    def test_custom_policy_allows_tuning_for_unlisted_gpu_sizes(self) -> None:
        self.assertEqual(effective_reserve_vram_gb("custom", 0.75), 0.75)
        self.assertEqual(effective_minimum_system_ram_gb("custom", 6.0), 6.0)

    def test_oom_recovery_reserve_scales_with_gpu_capacity(self) -> None:
        self.assertEqual(oom_recovery_reserve_vram_gb(1.0, 8.0), 1.5)
        self.assertEqual(oom_recovery_reserve_vram_gb(4.0, 16.0), 5.0)
        self.assertEqual(oom_recovery_reserve_vram_gb(3.0, 24.0), 4.5)
        self.assertEqual(oom_recovery_reserve_vram_gb(5.0, 8.0), 5.0)

    def test_auto_vram_mode_uses_high_vram_only_on_large_devices(self) -> None:
        self.assertEqual(resolve_vram_mode("auto", 48.0), "high_vram")
        self.assertEqual(resolve_vram_mode("auto", 39.99), "dynamic")
        self.assertEqual(resolve_vram_mode("low_vram", 48.0), "low_vram")
        with self.assertRaisesRegex(ValueError, "unknown VRAM mode"):
            resolve_vram_mode("unsafe", 48.0)

    def test_comfy_vram_modes_are_mutually_exclusive(self) -> None:
        args = SimpleNamespace(
            gpu_only=True,
            novram=True,
            highvram=False,
            lowvram=True,
            enable_dynamic_vram=False,
            disable_dynamic_vram=True,
        )
        configure_comfy_vram_args(args, "high_vram")
        self.assertTrue(args.highvram)
        self.assertFalse(args.lowvram)
        self.assertTrue(args.disable_dynamic_vram)
        configure_comfy_vram_args(args, "dynamic")
        self.assertFalse(args.highvram)
        self.assertFalse(args.lowvram)
        self.assertTrue(args.enable_dynamic_vram)
        self.assertFalse(args.disable_dynamic_vram)

    def test_critical_pressure_uses_the_single_oom_recovery_path(self) -> None:
        self.assertTrue(
            ComfyBaselineRuntime._is_oom(CriticalGpuMemoryPressure("1.4 GiB free"))
        )

    def test_generation_retries_only_once_with_same_request_after_oom(self) -> None:
        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.model = object()
        runtime.clip = object()
        runtime.vae = object()
        runtime.oom_recovery = True
        expected = {"image_path": "/tmp/result.png", "oom_recovered": True}
        runtime._generate_once = Mock(
            side_effect=[RuntimeError("HIP out of memory"), expected]
        )
        runtime._recover_from_oom = Mock()
        runtime.memory_snapshot = Mock(return_value={"stage": "OOM detected"})

        with tempfile.TemporaryDirectory() as directory:
            result = runtime.generate(
                prompt="a teapot",
                width=256,
                height=256,
                steps=1,
                sampler="dpmpp_2m",
                scheduler="karras",
                seed=42,
                output_directory=Path(directory),
            )

        self.assertEqual(result, expected)
        self.assertEqual(runtime._generate_once.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in runtime._generate_once.call_args_list],
            [42, 42],
        )
        self.assertEqual(
            [call.kwargs["sampler"] for call in runtime._generate_once.call_args_list],
            ["dpmpp_2m", "dpmpp_2m"],
        )
        self.assertEqual(
            [call.kwargs["scheduler"] for call in runtime._generate_once.call_args_list],
            ["karras", "karras"],
        )
        runtime._recover_from_oom.assert_called_once()

    def test_high_vram_resident_cache_keeps_transformer_only_above_reserve(self) -> None:
        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.model = object()
        runtime.keep_model_loaded = True
        runtime.vram_mode = "high_vram"
        runtime.reserve_vram_gb = 2.0
        runtime.memory_snapshot = Mock(
            side_effect=[
                {
                    "gpu_free_bytes": 8 * 1024**3,
                    "ram_available_bytes": 20 * 1024**3,
                    "minimum_ram_bytes": 14 * 1024**3,
                },
                {
                    "gpu_free_bytes": 3 * 1024**3,
                    "ram_available_bytes": 20 * 1024**3,
                    "minimum_ram_bytes": 14 * 1024**3,
                },
            ]
        )
        management = SimpleNamespace(
            load_models_gpu=Mock(),
            unload_all_models=Mock(),
            soft_empty_cache=Mock(),
        )
        comfy = SimpleNamespace(model_management=management)

        with patch.dict(
            "sys.modules",
            {"comfy": comfy, "comfy.model_management": management},
        ):
            result = runtime.retain_baseline_model_if_safe()

        self.assertTrue(result["resident"])
        management.load_models_gpu.assert_called_once_with(
            [runtime.model],
            minimum_memory_required=0,
            force_full_load=True,
        )
        management.unload_all_models.assert_not_called()
        self.assertEqual(result["retained_components"], ["baseline_transformer"])
        self.assertFalse(result["regional_loras_retained"])

    def test_resident_cache_releases_baseline_when_pod_ram_reserve_is_not_met(
        self,
    ) -> None:
        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.model = object()
        runtime.keep_model_loaded = True
        runtime.vram_mode = "high_vram"
        runtime.reserve_vram_gb = 2.0
        runtime.memory_snapshot = Mock(
            side_effect=[
                {
                    "gpu_free_bytes": 8 * 1024**3,
                    "ram_available_bytes": 16 * 1024**3,
                    "minimum_ram_bytes": 14 * 1024**3,
                },
                {
                    "gpu_free_bytes": 3 * 1024**3,
                    "ram_available_bytes": 4 * 1024**3,
                    "minimum_ram_bytes": 14 * 1024**3,
                },
                {
                    "gpu_free_bytes": 40 * 1024**3,
                    "ram_available_bytes": 30 * 1024**3,
                    "minimum_ram_bytes": 14 * 1024**3,
                },
            ]
        )
        management = SimpleNamespace(
            load_models_gpu=Mock(),
            unload_all_models=Mock(),
            soft_empty_cache=Mock(),
        )
        comfy = SimpleNamespace(model_management=management)

        with patch.dict(
            "sys.modules",
            {"comfy": comfy, "comfy.model_management": management},
        ):
            result = runtime.retain_baseline_model_if_safe()

        self.assertFalse(result["resident"])
        self.assertEqual(result["reason"], "pod_ram_reserve_not_met")
        management.load_models_gpu.assert_called_once_with(
            [runtime.model],
            minimum_memory_required=0,
            force_full_load=True,
        )
        management.unload_all_models.assert_called_once()

    def test_vae_decode_stays_inside_torch_inference_mode(self) -> None:
        active = False

        class InferenceMode:
            def __enter__(self):
                nonlocal active
                active = True

            def __exit__(self, *_):
                nonlocal active
                active = False

        def decode(samples):
            self.assertTrue(active)
            return samples

        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.vae = SimpleNamespace(decode=decode)
        fake_torch = SimpleNamespace(inference_mode=InferenceMode)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = runtime._decode_vae("latent")

        self.assertEqual(result, "latent")
        self.assertFalse(active)

    def test_vae_encode_disables_gradient_tracking(self) -> None:
        active = False

        class NoGrad:
            def __enter__(self):
                nonlocal active
                active = True

            def __exit__(self, *_):
                nonlocal active
                active = False

        def encode(pixels):
            self.assertTrue(active)
            return pixels

        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.vae = SimpleNamespace(encode=encode)
        fake_torch = SimpleNamespace(no_grad=NoGrad)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = runtime._encode_vae("pixels")

        self.assertEqual(result, "pixels")
        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()
