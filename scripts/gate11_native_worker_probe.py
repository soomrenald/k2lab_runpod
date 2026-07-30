from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one bounded Gate 11 fixture through the RunPod worker entrypoint."
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
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def command(command_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "kind": kind,
        "payload": payload,
    }


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        **fixture,
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
        "output_directory": str(output_directory),
        "memory_policy": "custom",
        "reserve_vram_gb": 4.0,
        "minimum_system_ram_gb": 12.0,
        "cpu_vae": False,
        "oom_recovery": True,
        "keep_model_loaded": False,
    }
    commands = (
        command("gate11-runpod-probe", "probe", payload),
        command("gate11-runpod-load", "load_model", payload),
        command("gate11-runpod-generation", "generate_baseline", payload),
    )
    environment = os.environ.copy()
    environment.setdefault("K2LAB_DATA_DIR", "/tmp/k2lab-gate11")
    started_at = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-m", "k2_region_lab.worker.entrypoint"],
        input="".join(
            json.dumps(item, separators=(",", ":")) + "\n"
            for item in commands
        ),
        text=True,
        capture_output=True,
        env=environment,
        timeout=args.timeout_seconds,
        check=False,
    )
    events = [
        json.loads(line)
        for line in process.stdout.splitlines()
        if line.strip()
    ]
    ready = next(
        (
            event
            for event in events
            if event.get("command_id") == "gate11-runpod-generation"
            and event.get("state") == "ready"
        ),
        None,
    )
    if process.returncode != 0 or ready is None:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "returncode": process.returncode,
                    "duration_seconds": time.monotonic() - started_at,
                    "events": events,
                    "stderr_tail": process.stderr[-4000:],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    result = ready.get("payload", {})
    image_path = Path(str(result["image_path"])).resolve(strict=True)
    summary = {
        "status": "passed",
        "fixture_sha256": sha256_file(args.fixture),
        "duration_seconds": time.monotonic() - started_at,
        "backend": result.get("backend"),
        "correlation_id": result.get("correlation_id"),
        "seed": result.get("seed"),
        "width": result.get("width"),
        "height": result.get("height"),
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "event_count": len(events),
        "terminal_states": [
            event.get("state")
            for event in events
            if event.get("command_id") == "gate11-runpod-generation"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
