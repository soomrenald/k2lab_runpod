from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from k2_region_lab.pose import (
    VOLUMETRIC_POSE_FORMAT,
    VOLUMETRIC_POSE_JOINT_NAMES,
    PoseHeadEllipse,
    VolumetricPoseJoint,
    VolumetricSubjectPose,
    default_volumetric_subject_pose,
    mirror_volumetric_subject_pose,
    squatting_volumetric_subject_pose,
    volumetric_subject_pose_document,
    volumetric_subject_pose_from_document,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.volumetric_pose import render_volumetric_masks


def _subject(
    region_id: str,
    box: PixelBox,
    *,
    priority: int,
    pose: VolumetricSubjectPose | None = None,
) -> RegionDefinition:
    return RegionDefinition(
        region_id=region_id,
        name=region_id,
        box=box,
        prompt="a person",
        priority=priority,
        spatial_role="subject",
        region_type="subject",
        pose=pose or default_volumetric_subject_pose(),  # type: ignore[arg-type]
    )


def test_volumetric_pose_has_exact_body_joints_and_head() -> None:
    pose = default_volumetric_subject_pose()

    assert pose.format == VOLUMETRIC_POSE_FORMAT
    assert tuple(joint.name for joint in pose.joints) == VOLUMETRIC_POSE_JOINT_NAMES
    assert len(pose.joints) == 13
    assert not {"nose", "left_eye", "right_eye", "left_ear", "right_ear"} & {
        joint.name for joint in pose.joints
    }
    assert pose.head.rx > 0
    assert pose.head.ry > 0


def test_volumetric_pose_round_trip_and_mirror_involution() -> None:
    pose = replace(
        squatting_volumetric_subject_pose(),
        head=PoseHeadEllipse(0.44, 0.25, 0.08, 0.11),
    )
    restored = volumetric_subject_pose_from_document(
        volumetric_subject_pose_document(pose)
    )
    mirrored_twice = mirror_volumetric_subject_pose(
        mirror_volumetric_subject_pose(restored)
    )

    for expected, actual in zip(restored.joints, mirrored_twice.joints, strict=True):
        assert actual.name == expected.name
        assert actual.x == pytest.approx(expected.x)
        assert actual.y == pytest.approx(expected.y)
    assert mirrored_twice.head.cx == pytest.approx(restored.head.cx)
    assert mirrored_twice.head.cy == pytest.approx(restored.head.cy)
    assert mirrored_twice.head.rx == pytest.approx(restored.head.rx)
    assert mirrored_twice.head.ry == pytest.approx(restored.head.ry)


def test_legacy_openpose_document_derives_head_and_discards_face_nodes() -> None:
    legacy = {
        "enabled": True,
        "joints": [
            {"name": joint.name, "x": joint.x, "y": joint.y, "enabled": True}
            for joint in default_volumetric_subject_pose().joints
        ]
        + [
            {"name": "nose", "x": 0.47, "y": 0.10, "enabled": True},
            {"name": "left_eye", "x": 0.51, "y": 0.08, "enabled": True},
            {"name": "right_eye", "x": 0.43, "y": 0.08, "enabled": True},
        ],
    }

    migrated = volumetric_subject_pose_from_document(legacy)
    document = volumetric_subject_pose_document(migrated)

    assert migrated.head.cx == pytest.approx(0.47)
    assert isinstance(document["joints"], dict)
    assert "nose" not in document["joints"]


def test_invalid_head_and_nonfinite_joint_are_rejected() -> None:
    with pytest.raises(ValueError, match="radii"):
        PoseHeadEllipse(0.5, 0.1, 0.0, 0.1)
    with pytest.raises(ValueError, match="finite"):
        VolumetricPoseJoint("neck", float("nan"), 0.2)


def test_masks_are_binary_soft_containing_and_exclusive() -> None:
    regions = (
        _subject("front", PixelBox(50, 20, 210, 250), priority=10),
        _subject("back", PixelBox(130, 20, 290, 250), priority=5),
    )
    bundle = render_volumetric_masks(regions=regions, width=320, height=256)

    assert bundle.summary.subject_count == 2
    assert bundle.summary.priority_region_ids == ("front", "back")
    assert set(np.unique(bundle.union_core)).issubset({0.0, 1.0})
    assert np.all(bundle.union_support >= bundle.union_core)
    assert np.all((bundle.union_support >= 0.0) & (bundle.union_support <= 1.0))
    assert np.count_nonzero(bundle.union_support) > np.count_nonzero(bundle.union_core)
    assert bundle.summary.overlap_pixels > 0
    front, back = bundle.subjects
    overlap = front.support > 0.0
    assert np.count_nonzero(back.ownership[overlap]) == 0


def test_out_of_box_limb_renders_until_canvas_boundary() -> None:
    pose = default_volumetric_subject_pose()
    joints = tuple(
        VolumetricPoseJoint(joint.name, -0.5, joint.y)
        if joint.name == "left_wrist"
        else joint
        for joint in pose.joints
    )
    pose = replace(pose, joints=joints)
    region = _subject("person", PixelBox(100, 10, 220, 250), priority=1, pose=pose)

    bundle = render_volumetric_masks(regions=(region,), width=256, height=256)

    assert np.count_nonzero(bundle.union_core[:, :100]) > 0


def test_empty_subject_set_produces_zero_bundle() -> None:
    bundle = render_volumetric_masks(regions=(), width=64, height=96)

    assert bundle.summary.subject_count == 0
    assert bundle.union_support.shape == (96, 64)
    assert np.count_nonzero(bundle.union_support) == 0
