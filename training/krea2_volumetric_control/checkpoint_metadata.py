from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from safetensors import safe_open

from k2_region_lab.volumetric_control import (
    K2_VOLUMETRIC_CONTROL_FORMAT,
    K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
)


TRAINER_REPOSITORY = "https://github.com/Tanmaypatil123/Krea-2-controlnet"
TRAINER_COMMIT = "909682ae0bdd9eb87c8258894c0003224db00d0b"


def metadata(
    *,
    rank: int,
    dataset_manifest_sha256: str,
    training_commit: str,
    created_at: str | None = None,
) -> dict[str, str]:
    return {
        "k2lab_adapter_kind": "krea2_control_lora",
        "k2lab_adapter_version": "1",
        "k2lab_control_format": K2_VOLUMETRIC_CONTROL_FORMAT,
        "k2lab_control_format_sha256": K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
        "k2lab_renderer_version": "1",
        "k2lab_base_model": "krea/Krea-2-Raw",
        "k2lab_inference_targets": "krea/Krea-2-Raw,krea/Krea-2-Turbo",
        "k2lab_rank": str(rank),
        "k2lab_expanded_input_projection": "true",
        "k2lab_expected_transformer_blocks": "28",
        "k2lab_control_channel_mode": "rgb",
        "k2lab_control_normalize": "none",
        "k2lab_control_invert": "false",
        "k2lab_dataset_manifest_sha256": dataset_manifest_sha256,
        "k2lab_trainer_repository": TRAINER_REPOSITORY,
        "k2lab_trainer_commit": TRAINER_COMMIT,
        "k2lab_training_commit": training_commit,
        "k2lab_created_at": created_at or datetime.now(UTC).isoformat(),
    }


def write_metadata(source: Path, destination: Path, additions: Mapping[str, str]) -> None:
    from safetensors.torch import load_file, save_file

    with safe_open(source, framework="pt") as handle:
        current = dict(handle.metadata() or {})
    current.update({str(key): str(value) for key, value in additions.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(load_file(source), destination, metadata=current)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--training-commit", required=True)
    args = parser.parse_args()
    write_metadata(
        args.source,
        args.destination,
        metadata(
            rank=args.rank,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            training_commit=args.training_commit,
        ),
    )


if __name__ == "__main__":
    main()
