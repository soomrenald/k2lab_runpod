from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gate12_native_soak_probe import (
    Worker,
    atomic_json,
    base_payload,
    command,
    nvidia_memory_used_mib,
    pixel_sha256,
    sha256_file,
)


T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare identical ComfyUI and native release requests on one GPU and "
            "evaluate the Gate 12 peak-VRAM threshold."
        )
    )
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--comfy-python", type=Path, required=True)
    parser.add_argument("--k2core-package", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--transformer-sha256", required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--text-encoder-sha256", required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--vae-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("--native-peak-ratio-limit", type=float, default=1.15)
    parser.add_argument("--hourly-cost-usd", type=float, default=0.44)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "fixtures"
            / "gate11"
            / "native_clean_generation.json"
        ),
    )
    return parser.parse_args()


class GpuMemoryMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.samples.append(nvidia_memory_used_mib())
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> GpuMemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15)
        self.samples.append(nvidia_memory_used_mib())


def measure(
    operation: Callable[[], T],
    *,
    interval_seconds: float,
) -> tuple[T, dict[str, Any]]:
    started_at = time.monotonic()
    with GpuMemoryMonitor(interval_seconds) as monitor:
        result = operation()
    return result, {
        "seconds": time.monotonic() - started_at,
        "samples": len(monitor.samples),
        "minimum_used_mib": min(monitor.samples),
        "maximum_used_mib": max(monitor.samples),
        "median_used_mib": statistics.median(monitor.samples),
    }


def wait_for_cleanup(*, initial_mib: int, timeout_seconds: float = 60.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    observed = nvidia_memory_used_mib()
    while observed > initial_mib + 8 and time.monotonic() < deadline:
        time.sleep(0.25)
        observed = nvidia_memory_used_mib()
    return observed


def release_comparison(
    *,
    native_peak_mib: int,
    comfy_peak_mib: int,
    ratio_limit: float,
) -> dict[str, Any]:
    if native_peak_mib < 0 or comfy_peak_mib <= 0:
        raise ValueError("peak GPU memory values must be positive")
    if ratio_limit <= 0:
        raise ValueError("peak ratio limit must be positive")
    ratio = native_peak_mib / comfy_peak_mib
    return {
        "native_peak_used_mib": native_peak_mib,
        "comfyui_peak_used_mib": comfy_peak_mib,
        "native_to_comfyui_ratio": ratio,
        "limit": ratio_limit,
        "passed": ratio <= ratio_limit,
    }


def validate_inputs(args: argparse.Namespace) -> None:
    for executable in (args.native_python, args.comfy_python):
        if not executable.expanduser().resolve().is_file():
            raise FileNotFoundError(executable)
    if not (args.k2core_package / "__init__.py").resolve().is_file():
        raise FileNotFoundError(args.k2core_package)
    for path, expected in (
        (args.transformer, args.transformer_sha256),
        (args.text_encoder, args.text_encoder_sha256),
        (args.vae, args.vae_sha256),
    ):
        if sha256_file(path.expanduser().resolve()) != expected.casefold():
            raise ValueError(f"model hash mismatch: {path}")
    if args.width <= 0 or args.height <= 0 or args.steps <= 0:
        raise ValueError("width, height, and steps must be positive")
    if args.timeout_seconds <= 0 or args.sample_interval_seconds <= 0:
        raise ValueError("timeouts and sample interval must be positive")
    if args.hourly_cost_usd < 0:
        raise ValueError("hourly cost cannot be negative")


def run_backend(
    backend: str,
    *,
    executable: Path,
    args: argparse.Namespace,
    payload: dict[str, Any],
    initial_gpu_mib: int,
) -> dict[str, Any]:
    backend_output = args.output_directory / backend
    backend_output.mkdir(parents=True, exist_ok=True)
    configured_payload = {
        **payload,
        "inference_backend": backend,
        "output_directory": str(backend_output),
        "filename_prefix": f"gate12_{backend}_vram",
    }
    worker = Worker(
        timeout=args.timeout_seconds,
        log_path=args.state.with_name(f"{args.state.stem}.{backend}.worker.log"),
        executable=executable,
        backend=backend,
        k2core_package=args.k2core_package,
    )
    cleanup: dict[str, Any] = {}
    try:
        worker.execute(
            command(f"gate12-{backend}-probe", "probe", configured_payload),
            terminal_state="unloaded",
        )
        _load_events, load = measure(
            lambda: worker.execute(
                command(f"gate12-{backend}-load", "load_model", configured_payload),
                terminal_state="ready",
            ),
            interval_seconds=args.sample_interval_seconds,
        )
        events, generation = measure(
            lambda: worker.execute(
                command(
                    f"gate12-{backend}-generation",
                    "generate_baseline",
                    configured_payload,
                ),
                terminal_state="complete",
            ),
            interval_seconds=args.sample_interval_seconds,
        )
        ready = next(event for event, _received_at in events if event["state"] == "ready")
        result = ready["payload"]
        image_path = Path(result["image_path"]).resolve(strict=True)
        return {
            "backend": backend,
            "worker_python": str(executable),
            "cold_start_seconds": worker.cold_start_seconds,
            "load": load,
            "generation": generation,
            "lifecycle_peak_used_mib": max(
                int(load["maximum_used_mib"]),
                int(generation["maximum_used_mib"]),
            ),
            "image_sha256": sha256_file(image_path),
            "pixel_sha256": pixel_sha256(image_path),
            "output_bytes": image_path.stat().st_size,
            "result_memory": result.get("memory", {}),
            "cleanup": cleanup,
        }
    finally:
        close_seconds = worker.close()
        cleanup.update(
            {
                "seconds": close_seconds,
                "terminal_gpu_used_mib": wait_for_cleanup(initial_mib=initial_gpu_mib),
            }
        )


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    args.fixture = args.fixture.expanduser().resolve()
    args.transformer = args.transformer.expanduser().resolve()
    args.text_encoder = args.text_encoder.expanduser().resolve()
    args.vae = args.vae.expanduser().resolve()
    args.tokenizer = args.tokenizer.expanduser().resolve()
    args.native_python = args.native_python.expanduser().resolve()
    args.comfy_python = args.comfy_python.expanduser().resolve()
    args.k2core_package = args.k2core_package.expanduser().resolve()
    args.output_directory = args.output_directory.expanduser().resolve()
    args.state = args.state.expanduser().resolve()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    initial_gpu_mib = nvidia_memory_used_mib()
    started_at = time.monotonic()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    payload = base_payload(args, fixture)
    results = {
        backend: run_backend(
            backend,
            executable=executable,
            args=args,
            payload=payload,
            initial_gpu_mib=initial_gpu_mib,
        )
        for backend, executable in (
            ("comfyui", args.comfy_python),
            ("native", args.native_python),
        )
    }
    comparison = release_comparison(
        native_peak_mib=int(results["native"]["lifecycle_peak_used_mib"]),
        comfy_peak_mib=int(results["comfyui"]["lifecycle_peak_used_mib"]),
        ratio_limit=args.native_peak_ratio_limit,
    )
    terminal_gpu_mib = wait_for_cleanup(initial_mib=initial_gpu_mib)
    session_seconds = time.monotonic() - started_at
    cleanup_passed = all(
        int(result["cleanup"]["terminal_gpu_used_mib"]) <= initial_gpu_mib + 8
        for result in results.values()
    ) and terminal_gpu_mib <= initial_gpu_mib + 8
    state = {
        "schema_version": "k2lab-gate12-backend-vram/1",
        "status": "passed" if comparison["passed"] and cleanup_passed else "failed",
        "fixture_sha256": sha256_file(args.fixture),
        "workload": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "sampler": fixture["sampler"],
            "scheduler": fixture["scheduler"],
            "seed": fixture["seed"],
        },
        "model_hashes": {
            "transformer": args.transformer_sha256,
            "text_encoder": args.text_encoder_sha256,
            "vae": args.vae_sha256,
            "tokenizer": args.tokenizer_sha256,
        },
        "initial_gpu_used_mib": initial_gpu_mib,
        "backends": results,
        "comparison": comparison,
        "cleanup_passed": cleanup_passed,
        "terminal_gpu_used_mib": terminal_gpu_mib,
        "session_seconds": session_seconds,
        "estimated_session_cost_usd": session_seconds / 3600 * args.hourly_cost_usd,
    }
    atomic_json(args.state, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
