from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable, bounded resident-worker native K2 soak and write "
            "prompt-safe release evidence."
        )
    )
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
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        pixels = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(pixels.width.to_bytes(4, "big"))
        digest.update(pixels.height.to_bytes(4, "big"))
        digest.update(pixels.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command(command_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"command_id": command_id, "kind": kind, "payload": payload}


def nvidia_memory_used_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    values = [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(values) != 1:
        raise RuntimeError("the soak probe requires exactly one visible GPU")
    return values[0]


def worker_rss_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("worker RSS is unavailable")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return (
        sum(
            (index - x_mean) * (value - y_mean)
            for index, value in enumerate(values)
        )
        / denominator
    )


def growth_summary(
    values: list[int],
    *,
    tolerance: int,
    warmup: int = 5,
    window: int = 10,
) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "growth": 0,
            "slope_per_job": 0.0,
            "tolerance": tolerance,
            "passed": False,
        }
    stable = values[min(warmup, len(values) - 1) :]
    first = stable[: min(window, len(stable))]
    last = stable[-min(window, len(stable)) :]
    first_median = int(statistics.median(first))
    last_median = int(statistics.median(last))
    growth = last_median - first_median
    return {
        "samples": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "first_window_median": first_median,
        "last_window_median": last_median,
        "growth": growth,
        "slope_per_job": linear_slope([float(value) for value in stable]),
        "tolerance": tolerance,
        "passed": growth <= tolerance,
    }


def performance_summary(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(item["duration_seconds"]) for item in iterations]
    timing_names = (
        "text_encoding_seconds",
        "transformer_seconds",
        "vae_decode_seconds",
        "image_output_seconds",
    )
    result: dict[str, Any] = {
        "jobs": len(iterations),
        "total_seconds": sum(durations),
        "mean_generation_seconds": statistics.fmean(durations) if durations else 0.0,
        "p50_generation_seconds": percentile(durations, 0.50),
        "p95_generation_seconds": percentile(durations, 0.95),
        "minimum_generation_seconds": min(durations, default=0.0),
        "maximum_generation_seconds": max(durations, default=0.0),
    }
    for name in timing_names:
        values = [float(item["timings"][name]) for item in iterations]
        result[f"mean_{name}"] = statistics.fmean(values) if values else 0.0
    return result


class Worker:
    def __init__(
        self,
        *,
        timeout: float,
        log_path: Path,
        executable: Path | None = None,
        backend: str = "native",
        k2core_package: Path | None = None,
    ) -> None:
        self.timeout = timeout
        self.backend = backend
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.setdefault("K2LAB_DATA_DIR", str(log_path.parent))
        environment["K2LAB_BACKEND"] = backend
        if k2core_package is not None:
            environment["K2LAB_K2CORE_PACKAGE"] = str(k2core_package)
        self.started_at = time.monotonic()
        self.process = subprocess.Popen(
            [
                str(executable or Path(sys.executable)),
                "-m",
                "k2_region_lab.worker.entrypoint",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
            env=environment,
        )
        if self.process.stdout is None or self.process.stdin is None:
            raise RuntimeError("worker pipes were not created")
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        initial, _received_at = self._read_event()
        if initial.get("state") != "unloaded":
            raise RuntimeError(f"unexpected worker startup event: {initial}")
        self.cold_start_seconds = time.monotonic() - self.started_at

    def _read_event(self) -> tuple[dict[str, Any], float]:
        ready = self._selector.select(self.timeout)
        if not ready:
            raise TimeoutError("worker event deadline exceeded")
        assert self.process.stdout is not None
        encoded = self.process.stdout.readline()
        if not encoded:
            raise RuntimeError(
                f"worker exited before a terminal event (code {self.process.poll()})"
            )
        event = json.loads(encoded)
        if not isinstance(event, dict):
            raise RuntimeError("worker emitted a non-object event")
        return event, time.monotonic()

    def execute(
        self,
        item: dict[str, Any],
        *,
        terminal_state: str,
    ) -> list[tuple[dict[str, Any], float]]:
        assert self.process.stdin is not None
        command_id = str(item["command_id"])
        self.process.stdin.write(json.dumps(item, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        events: list[tuple[dict[str, Any], float]] = []
        while True:
            event, received_at = self._read_event()
            if str(event.get("command_id")) != command_id:
                continue
            events.append((event, received_at))
            if event.get("state") == "error":
                raise RuntimeError(
                    f"worker command {command_id} failed: {event.get('payload')}"
                )
            if event.get("state") == terminal_state:
                return events

    def close(self) -> float:
        started_at = time.monotonic()
        try:
            if self.process.poll() is None:
                self.execute(
                    command(
                        "gate12-soak-shutdown",
                        "shutdown",
                        {"inference_backend": self.backend},
                    ),
                    terminal_state="complete",
                )
                self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self._selector.close()
            self._log.close()
        return time.monotonic() - started_at


def generation_timings(
    events: list[tuple[dict[str, Any], float]],
    *,
    started_at: float,
) -> dict[str, Any]:
    phase_events: dict[str, list[float]] = {}
    diffusion_steps: list[float] = []
    ready_at = started_at
    for event, received_at in events:
        payload = event.get("payload")
        if isinstance(payload, dict):
            phase = payload.get("phase")
            if isinstance(phase, str):
                phase_events.setdefault(phase, []).append(received_at)
                if phase == "diffusion":
                    diffusion_steps.append(received_at)
        if event.get("state") == "ready":
            ready_at = received_at
    text = phase_events.get("text_encoding", [])
    vae = phase_events.get("vae_decode", [])
    text_end = text[-1] if text else started_at
    transformer_end = diffusion_steps[-1] if diffusion_steps else text_end
    vae_start = vae[0] if vae else transformer_end
    vae_end = vae[-1] if vae else vae_start
    step_starts = [text_end, *diffusion_steps[:-1]]
    step_seconds = [
        completed - step_start
        for step_start, completed in zip(
            step_starts,
            diffusion_steps,
            strict=True,
        )
    ]
    return {
        "text_encoding_seconds": max(0.0, text_end - text[0]) if text else 0.0,
        "transformer_seconds": max(0.0, transformer_end - text_end),
        "per_step_transformer_seconds": step_seconds,
        "vae_decode_seconds": max(0.0, vae_end - vae_start),
        "image_output_seconds": max(0.0, ready_at - vae_end),
    }


def base_payload(args: argparse.Namespace, fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        **fixture,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "inference_backend": "native",
        "comfyui_root": "/opt/ComfyUI",
        "diffusion_models": str(args.transformer.parent),
        "text_encoders": str(args.text_encoder.parent),
        "vae": str(args.vae.parent),
        "diffusion_model_file": str(args.transformer),
        "diffusion_model_sha256": args.transformer_sha256,
        "text_encoder_file": str(args.text_encoder),
        "text_encoder_sha256": args.text_encoder_sha256,
        "vae_file": str(args.vae),
        "vae_sha256": args.vae_sha256,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": args.tokenizer_sha256,
        "output_directory": str(args.output_directory),
        "memory_policy": "custom",
        "reserve_vram_gb": 4.0,
        "minimum_system_ram_gb": 12.0,
        "cpu_vae": False,
        "oom_recovery": True,
        "keep_model_loaded": True,
    }


def load_state(args: argparse.Namespace) -> dict[str, Any]:
    workload = {
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "sampler": "euler",
        "scheduler": "simple",
    }
    if args.state.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if state.get("target_count") != args.count:
            raise ValueError("existing state target count does not match --count")
        if state.get("fixture_sha256") != sha256_file(args.fixture):
            raise ValueError("existing state fixture does not match --fixture")
        if state.get("workload") != workload:
            raise ValueError("existing state workload does not match requested workload")
        return state
    return {
        "schema_version": "k2lab-native-soak/1",
        "status": "running",
        "target_count": args.count,
        "fixture_sha256": sha256_file(args.fixture),
        "workload": workload,
        "model_hashes": {
            "transformer": args.transformer_sha256,
            "text_encoder": args.text_encoder_sha256,
            "vae": args.vae_sha256,
            "tokenizer": args.tokenizer_sha256,
        },
        "iterations": [],
        "sessions": [],
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.count <= 1000:
        raise ValueError("count must be between 1 and 1000")
    if args.width <= 0 or args.height <= 0 or args.steps <= 0:
        raise ValueError("width, height, and steps must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if args.hourly_cost_usd < 0:
        raise ValueError("hourly cost cannot be negative")
    args.output_directory = args.output_directory.expanduser().resolve()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.state = args.state.expanduser().resolve()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    state = load_state(args)
    state["status"] = "running"
    atomic_json(args.state, state)
    initial_gpu_mib = nvidia_memory_used_mib()
    session_started_at = time.monotonic()
    worker = Worker(
        timeout=args.timeout_seconds,
        log_path=args.state.with_suffix(".worker.log"),
    )
    cleanup_seconds = 0.0
    try:
        payload = base_payload(args, fixture)
        probe_started_at = time.monotonic()
        worker.execute(
            command("gate12-soak-probe", "probe", payload),
            terminal_state="unloaded",
        )
        probe_seconds = time.monotonic() - probe_started_at
        load_started_at = time.monotonic()
        worker.execute(
            command("gate12-soak-load", "load_model", payload),
            terminal_state="ready",
        )
        load_seconds = time.monotonic() - load_started_at
        start_index = len(state["iterations"])
        state["sessions"].append(
            {
                "start_iteration": start_index + 1,
                "cold_start_seconds": worker.cold_start_seconds,
                "probe_seconds": probe_seconds,
                "model_load_seconds": load_seconds,
            }
        )
        atomic_json(args.state, state)
        for index in range(start_index, args.count):
            iteration = index + 1
            run_payload = {
                **payload,
                "filename_prefix": f"gate12_soak_{iteration:03d}",
            }
            command_id = f"gate12-soak-generation-{iteration:03d}"
            started_at = time.monotonic()
            events = worker.execute(
                command(command_id, "generate_baseline", run_payload),
                terminal_state="complete",
            )
            duration = time.monotonic() - started_at
            ready_event = next(
                event
                for event, _received_at in events
                if event.get("state") == "ready"
            )
            result = ready_event["payload"]
            image_path = Path(str(result["image_path"])).resolve(strict=True)
            memory = result.get("memory", {})
            record = {
                "iteration": iteration,
                "command_id": command_id,
                "duration_seconds": duration,
                "timings": generation_timings(events, started_at=started_at),
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "pixel_sha256": pixel_sha256(image_path),
                "output_bytes": image_path.stat().st_size,
                "gpu_used_mib": nvidia_memory_used_mib(),
                "worker_rss_bytes": worker_rss_bytes(worker.process.pid),
                "gpu_allocated_bytes": int(memory.get("gpu_allocated_bytes", 0)),
                "gpu_reserved_bytes": int(memory.get("gpu_reserved_bytes", 0)),
                "gpu_peak_allocated_bytes": int(
                    memory.get("gpu_peak_allocated_bytes", 0)
                ),
                "gpu_peak_reserved_bytes": int(
                    memory.get("gpu_peak_reserved_bytes", 0)
                ),
                "ram_available_bytes": int(memory.get("ram_available_bytes", 0)),
                "ram_total_bytes": int(memory.get("ram_total_bytes", 0)),
            }
            state["iterations"].append(record)
            atomic_json(args.state, state)
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "target": args.count,
                        "duration_seconds": round(duration, 3),
                        "gpu_used_mib": record["gpu_used_mib"],
                        "worker_rss_mib": round(record["worker_rss_bytes"] / MIB, 1),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    finally:
        cleanup_seconds = worker.close()
    terminal_gpu_mib = nvidia_memory_used_mib()
    iterations = state["iterations"]
    pixel_hashes = {item["pixel_sha256"] for item in iterations}
    gpu_growth = growth_summary(
        [int(item["gpu_used_mib"]) for item in iterations],
        tolerance=64,
    )
    rss_growth = growth_summary(
        [int(item["worker_rss_bytes"]) for item in iterations],
        tolerance=256 * MIB,
    )
    session_seconds = time.monotonic() - session_started_at
    state.update(
        {
            "status": (
                "passed"
                if len(iterations) == args.count
                and len(pixel_hashes) == 1
                and gpu_growth["passed"]
                and rss_growth["passed"]
                and terminal_gpu_mib <= initial_gpu_mib + 8
                else "failed"
            ),
            "completed_count": len(iterations),
            "pixel_exact": len(pixel_hashes) == 1,
            "pixel_sha256": next(iter(pixel_hashes), None),
            "performance": performance_summary(iterations),
            "memory_growth": {
                "nvidia_used_mib": gpu_growth,
                "worker_rss_bytes": rss_growth,
            },
            "cleanup": {
                "seconds": cleanup_seconds,
                "initial_gpu_used_mib": initial_gpu_mib,
                "terminal_gpu_used_mib": terminal_gpu_mib,
            },
            "last_session_seconds": session_seconds,
            "estimated_last_session_cost_usd": (
                session_seconds / 3600 * args.hourly_cost_usd
            ),
        }
    )
    atomic_json(args.state, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
