from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from k2_region_lab.volumetric_control import (
    K2_VOLUMETRIC_CONTROL_FORMAT,
    K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
    render_krea_volumetric_control,
)

try:
    from .coco_pose import coco_person, filter_reason, person_region
    from .dataset import (
        CanvasTransform,
        canonical_json_sha256,
        choose_bucket,
        normalized_caption,
        sha256_file,
    )
except ImportError:
    from coco_pose import coco_person, filter_reason, person_region
    from dataset import (
        CanvasTransform,
        canonical_json_sha256,
        choose_bucket,
        normalized_caption,
        sha256_file,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_box_for_subject(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    margin: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    image_width, image_height = image_size
    return (
        max(0.0, x0 - width * margin),
        max(0.0, y0 - height * margin),
        min(float(image_width), x1 + width * margin),
        min(float(image_height), y1 + height * margin),
    )


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(args: argparse.Namespace) -> dict[str, Any]:
    keypoints = _load(args.keypoints)
    captions_document = _load(args.captions)
    image_records = {int(item["id"]): item for item in keypoints["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in keypoints["annotations"]:
        if int(annotation.get("category_id", 1)) == 1:
            annotations[int(annotation["image_id"])].append(annotation)
    captions: dict[int, list[str]] = defaultdict(list)
    for caption in captions_document["annotations"]:
        captions[int(caption["image_id"])].append(str(caption["caption"]))

    split_root = args.output / args.split
    image_output = split_root / "images"
    control_output = split_root / "controls"
    image_output.mkdir(parents=True, exist_ok=True)
    control_output.mkdir(parents=True, exist_ok=True)
    metadata_path = split_root / "metadata.jsonl"
    rng = random.Random(args.seed)
    excluded = Counter()
    bucket_counts = Counter()
    distribution = Counter()
    metadata: list[dict[str, Any]] = []

    for image_id in sorted(image_records):
        image_record = image_records[image_id]
        source_path = args.images / image_record["file_name"]
        if not source_path.is_file():
            excluded["missing_image"] += 1
            continue
        source = Image.open(source_path).convert("RGB")
        people = []
        for annotation in annotations[image_id]:
            try:
                people.append(coco_person(annotation))
            except (KeyError, TypeError, ValueError):
                excluded["invalid_annotation"] += 1
        if not people:
            excluded["no_person"] += 1
            continue

        subject_focused = (
            args.split == "train"
            and rng.random() >= args.full_image_fraction
        )
        selected = people[:1] if subject_focused else people
        source_box = (
            _source_box_for_subject(
                selected[0].bbox,
                source.size,
                margin=rng.uniform(args.crop_margin_min, args.crop_margin_max),
            )
            if subject_focused
            else (0.0, 0.0, float(source.width), float(source.height))
        )
        bucket = choose_bucket(
            round(source_box[2] - source_box[0]),
            round(source_box[3] - source_box[1]),
        )
        flip = args.split == "train" and args.horizontal_flip and rng.random() < 0.5
        transform = CanvasTransform.cover(source_box, bucket, horizontal_flip=flip)
        regions = []
        fallbacks = []
        annotation_ids = []
        for person in selected:
            reason = filter_reason(
                person,
                transform,
                minimum_joints=args.minimum_joints,
                minimum_height=args.minimum_person_height,
            )
            if reason is not None:
                excluded[reason] += 1
                continue
            try:
                region, fallback = person_region(
                    person,
                    transform,
                    region_id=f"coco-{person.annotation_id}",
                    priority=len(selected) - len(regions),
                )
            except (ValueError, KeyError):
                excluded["invalid_transformed_pose"] += 1
                continue
            regions.append(region)
            fallbacks.append(fallback)
            annotation_ids.append(person.annotation_id)
        if not regions:
            excluded["no_qualifying_person"] += 1
            continue

        target = transform.image(source)
        control = render_krea_volumetric_control(
            regions=regions,
            width=bucket[0],
            height=bucket[1],
        )
        if control.coverage < args.minimum_control_coverage:
            excluded["control_coverage_low"] += 1
            continue
        if control.coverage > args.maximum_control_coverage:
            excluded["control_coverage_high"] += 1
            continue
        sample_id = f"{image_id:012d}-{len(metadata):02d}"
        image_name = f"{sample_id}.jpg"
        control_name = f"{sample_id}.png"
        target.save(image_output / image_name, quality=95, subsampling=0)
        (control_output / control_name).write_bytes(control.png_bytes)
        line = {
            "file_name": image_name,
            "control_file_name": control_name,
            "text": normalized_caption(captions[image_id]),
            "source_image_id": image_id,
            "person_annotation_ids": annotation_ids,
            "bucket": list(bucket),
            "control_format": K2_VOLUMETRIC_CONTROL_FORMAT,
            "control_sha256": control.sha256,
            "target_sha256": sha256_file(image_output / image_name),
            "transform": transform.document(),
            "head_fallback": fallbacks[0] if len(set(fallbacks)) == 1 else fallbacks,
            "sample_kind": "subject_crop" if subject_focused else "full_image",
        }
        metadata.append(line)
        bucket_counts[f"{bucket[0]}x{bucket[1]}"] += 1
        distribution["multi_person" if len(regions) > 1 else "single_person"] += 1
        distribution[line["sample_kind"]] += 1

    metadata_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in metadata),
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "split": args.split,
        "source_annotations": {
            "keypoints_sha256": sha256_file(args.keypoints),
            "captions_sha256": sha256_file(args.captions),
        },
        "k2lab_commit": _git_commit(),
        "renderer_format": K2_VOLUMETRIC_CONTROL_FORMAT,
        "renderer_format_sha256": K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
        "upstream_trainer_commit": "909682ae0bdd9eb87c8258894c0003224db00d0b",
        "filters": {
            "iscrowd": False,
            "minimum_joints": args.minimum_joints,
            "minimum_person_height": args.minimum_person_height,
            "control_coverage": [
                args.minimum_control_coverage,
                args.maximum_control_coverage,
            ],
            "full_image_fraction": args.full_image_fraction,
            "crop_margin": [args.crop_margin_min, args.crop_margin_max],
            "horizontal_flip": args.horizontal_flip and args.split == "train",
        },
        "counts": {"included": len(metadata), "excluded": sum(excluded.values())},
        "bucket_distribution": dict(sorted(bucket_counts.items())),
        "sample_distribution": dict(sorted(distribution.items())),
        "caption_statistics": {
            "nonempty": sum(bool(item["text"]) for item in metadata),
            "maximum_characters": max((len(item["text"]) for item in metadata), default=0),
        },
        "excluded_sample_reasons": dict(sorted(excluded.items())),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (split_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--images", type=Path, required=True)
    value.add_argument("--keypoints", type=Path, required=True)
    value.add_argument("--captions", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--split", choices=("train", "validation"), required=True)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--full-image-fraction", type=float, default=0.70)
    value.add_argument("--crop-margin-min", type=float, default=0.15)
    value.add_argument("--crop-margin-max", type=float, default=0.40)
    value.add_argument("--minimum-joints", type=int, default=8)
    value.add_argument("--minimum-person-height", type=float, default=96.0)
    value.add_argument("--minimum-control-coverage", type=float, default=0.01)
    value.add_argument("--maximum-control-coverage", type=float, default=0.75)
    value.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=True)
    return value


if __name__ == "__main__":
    build(parser().parse_args())
