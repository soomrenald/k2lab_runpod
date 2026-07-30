from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from k2core.depth import (
    DepthControlSettings,
    DepthNormalizationMode,
    DepthNormalizationSettings,
    inspect_depth_checkpoint,
)
from k2_region_lab.model import discover_model_artifacts
from k2_region_lab.config import ModelDirectories
from k2_region_lab.depth.runtime import prepare_depth_control
from k2_region_lab.worker.runtime import ComfyBaselineRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public Krea 2 depth adapter outside the web request path."
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-image", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path, default=Path("~/ComfyUI"))
    parser.add_argument("--mode", choices=("raw", "turbo"), default="turbo")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--vram-mode",
        choices=("auto", "high_vram", "dynamic", "low_vram"),
        default="auto",
    )
    parser.add_argument("--reserve-vram-gb", type=float, default=4.0)
    parser.add_argument(
        "--normalization",
        choices=tuple(mode.value for mode in DepthNormalizationMode),
        default=DepthNormalizationMode.MINMAX.value,
    )
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate artifacts and preprocessing without loading the base model.",
    )
    return parser


def _artifact_set(args: argparse.Namespace):
    directories = ModelDirectories(
        diffusion_models=args.base_model.parent,
        text_encoders=args.text_encoder.parent,
        vae=args.vae.parent,
        diffusion_model_file=args.base_model,
        text_encoder_file=args.text_encoder,
        vae_file=args.vae,
    )
    return discover_model_artifacts(directories)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / "validation.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, report_path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = inspect_depth_checkpoint(args.depth_checkpoint)
    if not checkpoint.compatible:
        raise SystemExit("; ".join(checkpoint.errors))
    normalization = DepthNormalizationSettings(
        mode=DepthNormalizationMode(args.normalization),
        invert=args.invert,
        gamma=args.gamma,
    )
    settings = DepthControlSettings(
        enabled=True,
        checkpoint=args.depth_checkpoint,
        depth_image=args.depth_image,
        global_strength=args.strength,
        normalization=normalization,
    )
    steps = args.steps if args.steps is not None else (28 if args.mode == "raw" else 8)
    cfg = args.cfg if args.cfg is not None else (3.5 if args.mode == "raw" else 0.0)
    report: dict[str, Any] = {
        "version": 1,
        "status": "inspected" if args.inspect_only else "running",
        "mode": args.mode,
        "steps": steps,
        "cfg": cfg,
        "vram_mode": args.vram_mode,
        "reserve_vram_gb": args.reserve_vram_gb,
        "checkpoint": checkpoint.document(),
        "depth_control": settings.to_payload(),
    }
    preprocessing = prepare_depth_control(
        settings,
        regions=(),
        width=args.width,
        height=args.height,
        steps=steps,
    )
    report["preprocessing"] = preprocessing.document()
    if args.inspect_only:
        _write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return 0

    runtime = ComfyBaselineRuntime(args.comfyui_root.expanduser().resolve())
    started = time.monotonic()
    report["model_load"] = runtime.load(
        _artifact_set(args),
        memory_policy_key="safe_16gb",
        vram_mode=args.vram_mode,
        reserve_vram_gb=args.reserve_vram_gb,
    )
    report["model_load_seconds"] = time.monotonic() - started
    generation_started = time.monotonic()
    result = runtime.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=steps,
        cfg=cfg,
        seed=args.seed,
        output_directory=args.output,
        filename_prefix=f"depth-validation-{args.mode}",
        regional_prompting=False,
        depth_control=settings,
    )
    report["generation_seconds"] = time.monotonic() - generation_started
    report["generation"] = result
    report["status"] = "complete"
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
