from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .checkpoint_metadata import metadata, write_metadata
except ImportError:
    from checkpoint_metadata import metadata, write_metadata


PINNED_TRAINER_COMMIT = "909682ae0bdd9eb87c8258894c0003224db00d0b"


def validate_checkout(path: Path) -> None:
    required = (
        "k2_lora.py",
        "mmdit.py",
        "trainer/train_control_lora.py",
    )
    missing = [relative for relative in required if not (path / relative).is_file()]
    if missing:
        raise ValueError("upstream trainer checkout is missing: " + ", ".join(missing))
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_TRAINER_COMMIT:
        raise ValueError(
            f"upstream trainer must be pinned to {PINNED_TRAINER_COMMIT}; found {commit}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--caption-dropout", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    validate_checkout(args.upstream_checkout)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    manifest_hash = str(manifest["manifest_sha256"])
    command = [
        sys.executable,
        str(args.upstream_checkout / "trainer/train_control_lora.py"),
        "--data-dir",
        str(args.data_dir),
        "--ckpt-dir",
        str(args.checkpoint_dir),
        "--raw-ckpt",
        str(args.raw_checkpoint),
        "--control-type",
        "k2_volumetric_pose_v1",
        "--rank",
        str(args.rank),
        "--lr",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.gradient_accumulation),
        "--warmup",
        str(args.warmup),
        "--caption-dropout",
        str(args.caption_dropout),
        "--max-steps",
        str(args.max_steps),
        "--save-every",
        str(args.checkpoint_interval),
        "--val-every",
        str(args.validation_interval),
    ]
    if args.no_wandb:
        command.append("--no-wandb")
    if args.resume:
        # The pinned trainer restores weights and step only. This is explicitly
        # approximate: optimizer, scheduler, RNG, data position, and gradient
        # accumulation state are not restored.
        command.extend(("--resume", str(args.resume)))
        print("WARNING: upstream resume is approximate, not an exact continuation.")
    subprocess.run(command, check=True)
    additions = metadata(
        rank=args.rank,
        dataset_manifest_sha256=manifest_hash,
        training_commit=args.training_commit,
    )
    for checkpoint in args.checkpoint_dir.rglob("step_*.safetensors"):
        verified = checkpoint.with_name(checkpoint.stem + ".k2lab.safetensors")
        if not verified.exists():
            write_metadata(checkpoint, verified, additions)


if __name__ == "__main__":
    main()
