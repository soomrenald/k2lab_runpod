from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from k2_region_lab.volumetric_control import K2_VOLUMETRIC_CONTROL_FORMAT
from training.krea2_volumetric_control.build_coco_pairs import build
from training.krea2_volumetric_control.coco_pose import (
    COCO_JOINT_NAMES,
    coco_person,
    person_region,
)
from training.krea2_volumetric_control.dataset import (
    CanvasTransform,
    canonical_json_sha256,
    choose_bucket,
    normalized_caption,
)
from training.krea2_volumetric_control.prepare_latent_shards import prepare


def _annotation() -> dict[str, object]:
    points = {
        "nose": (50, 20),
        "left_eye": (46, 18),
        "right_eye": (54, 18),
        "left_ear": (42, 20),
        "right_ear": (58, 20),
        "left_shoulder": (40, 40),
        "right_shoulder": (60, 40),
        "left_elbow": (32, 65),
        "right_elbow": (68, 65),
        "left_wrist": (28, 90),
        "right_wrist": (72, 90),
        "left_hip": (44, 90),
        "right_hip": (56, 90),
        "left_knee": (42, 130),
        "right_knee": (58, 130),
        "left_ankle": (40, 175),
        "right_ankle": (60, 175),
    }
    keypoints = [
        value
        for name in COCO_JOINT_NAMES
        for value in (*points[name], 2)
    ]
    return {"id": 7, "bbox": [20, 10, 60, 180], "keypoints": keypoints, "iscrowd": 0}


def test_coco_neck_head_and_horizontal_semantic_swap() -> None:
    person = coco_person(_annotation())
    regular, fallback = person_region(
        person,
        CanvasTransform.cover((0, 0, 100, 200), (512, 1024)),
        region_id="person",
    )
    flipped, _ = person_region(
        person,
        CanvasTransform.cover((0, 0, 100, 200), (512, 1024), horizontal_flip=True),
        region_id="person",
    )
    assert np.isclose(regular.pose.joint("neck").x, 0.5)
    assert fallback == "face_points"
    assert np.isclose(
        flipped.pose.joint("left_shoulder").x,
        1.0 - regular.pose.joint("right_shoulder").x,
    )


def test_bucket_and_caption_selection_are_deterministic() -> None:
    assert choose_bucket(1000, 1000) == (1024, 1024)
    assert normalized_caption([" short ", "the   much longer caption"]) == (
        "the much longer caption"
    )


class _FakeEncoder:
    def encode(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        return np.zeros((16, height // 8, width // 8), dtype=np.float16)


def test_shard_preparer_validates_and_resumes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "controls").mkdir()
    target = dataset / "images" / "sample.jpg"
    control = dataset / "controls" / "sample.png"
    Image.new("RGB", (64, 64), "white").save(target)
    Image.new("RGB", (64, 64), "black").save(control)
    control_hash = hashlib.sha256(control.read_bytes()).hexdigest()
    (dataset / "metadata.jsonl").write_text(
        json.dumps(
            {
                "file_name": target.name,
                "control_file_name": control.name,
                "text": "a person",
                "bucket": [64, 64],
                "control_sha256": control_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "shards"
    assert prepare(dataset, output, samples_per_shard=1, encoder=_FakeEncoder()) == 1
    assert prepare(dataset, output, samples_per_shard=1, encoder=_FakeEncoder()) == 1
    assert (output / "shard00000" / "_DONE").is_file()
    with np.load(output / "shard00000" / "sample.npz") as data:
        assert data["latent"].shape == data["control"].shape == (16, 8, 8)


def _shifted_annotation(
    annotation_id: int,
    *,
    shift_x: float = 0,
    hide_face: bool = False,
    iscrowd: int = 0,
) -> dict[str, object]:
    annotation = _annotation()
    annotation["id"] = annotation_id
    annotation["iscrowd"] = iscrowd
    annotation["bbox"] = [
        float(annotation["bbox"][0]) + shift_x,
        *annotation["bbox"][1:],
    ]
    points = list(annotation["keypoints"])
    for index, _name in enumerate(COCO_JOINT_NAMES):
        points[index * 3] = float(points[index * 3]) + shift_x
        if hide_face and index < 5:
            points[index * 3 + 2] = 0
    annotation["keypoints"] = points
    return annotation


def _write_coco_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "coco-images"
    images.mkdir()
    source = Image.new("RGB", (240, 200), "white")
    source.save(images / "0001.jpg", quality=100, subsampling=0)
    keypoints = tmp_path / "keypoints.json"
    captions = tmp_path / "captions.json"
    keypoints.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 240, "height": 200}],
                "annotations": [
                    {"image_id": 1, "category_id": 1, **_shifted_annotation(7)},
                    {
                        "image_id": 1,
                        "category_id": 1,
                        **_shifted_annotation(8, shift_x=120, hide_face=True),
                    },
                    {
                        "image_id": 1,
                        "category_id": 1,
                        **_shifted_annotation(9, shift_x=60, iscrowd=1),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    captions.write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": 1, "caption": "two people standing"},
                    {"image_id": 1, "caption": "  two   people standing together  "},
                ]
            }
        ),
        encoding="utf-8",
    )
    return images, keypoints, captions


def _build_args(
    images: Path,
    keypoints: Path,
    captions: Path,
    output: Path,
    *,
    split: str,
    full_image_fraction: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        images=images,
        keypoints=keypoints,
        captions=captions,
        output=output,
        split=split,
        seed=17,
        full_image_fraction=full_image_fraction,
        crop_margin_min=0.25,
        crop_margin_max=0.25,
        minimum_joints=8,
        minimum_person_height=96.0,
        minimum_control_coverage=0.001,
        maximum_control_coverage=0.95,
        horizontal_flip=False,
    )


def test_coco_builder_manifest_filters_alignment_and_subject_crop(tmp_path: Path) -> None:
    images, keypoints, captions = _write_coco_fixture(tmp_path)
    output = tmp_path / "dataset"
    manifest = build(
        _build_args(
            images,
            keypoints,
            captions,
            output,
            split="validation",
            full_image_fraction=1.0,
        )
    )
    lines = [
        json.loads(line)
        for line in (output / "validation" / "metadata.jsonl").read_text().splitlines()
    ]
    assert manifest["counts"] == {"included": 1, "excluded": 1}
    assert manifest["excluded_sample_reasons"] == {"iscrowd": 1}
    assert manifest["sample_distribution"]["multi_person"] == 1
    assert lines[0]["person_annotation_ids"] == [7, 8]
    assert lines[0]["head_fallback"] == ["face_points", "shoulder_fallback"]
    assert lines[0]["control_format"] == K2_VOLUMETRIC_CONTROL_FORMAT
    manifest_hash = manifest.pop("manifest_sha256")
    assert manifest_hash == canonical_json_sha256(manifest)

    target_path = output / "validation" / "images" / lines[0]["file_name"]
    control_path = output / "validation" / "controls" / lines[0]["control_file_name"]
    with Image.open(target_path) as target, Image.open(control_path) as control:
        assert target.size == control.size == tuple(lines[0]["bucket"])
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == lines[0]["target_sha256"]
    assert hashlib.sha256(control_path.read_bytes()).hexdigest() == lines[0]["control_sha256"]

    crop_manifest = build(
        _build_args(
            images,
            keypoints,
            captions,
            output,
            split="train",
            full_image_fraction=0.0,
        )
    )
    crop_line = json.loads(
        (output / "train" / "metadata.jsonl").read_text().splitlines()[0]
    )
    assert crop_manifest["sample_distribution"]["subject_crop"] == 1
    assert crop_line["sample_kind"] == "subject_crop"
    assert crop_line["person_annotation_ids"] == [7]
    assert crop_line["transform"]["source_box"] != [0.0, 0.0, 240.0, 200.0]
