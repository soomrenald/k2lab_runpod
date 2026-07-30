from k2_region_lab.pose import PoseJoint, SubjectPose, default_subject_pose
from k2_region_lab.pose_control import render_openpose_map
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.pose_gating import (
    PoseGateHookIncompatibleError,
    PoseGateRuntimeError,
    PoseGateScheduleError,
    PoseMaskBuildError,
)
from k2_region_lab.worker.protocol import CommandKind, classify_worker_error


def subject(region_id: str, box: PixelBox, *, enabled: bool = True) -> RegionDefinition:
    return RegionDefinition(
        region_id=region_id,
        name=region_id,
        box=box,
        spatial_role="subject",
        region_type="subject",
        pose=default_subject_pose(),
        enabled=enabled,
    )


def test_pose_map_composes_subjects_and_ignores_regular_regions() -> None:
    regions = (
        subject("left", PixelBox(0, 0, 200, 400)),
        RegionDefinition(
            region_id="background",
            name="background",
            box=PixelBox(0, 0, 600, 400),
            spatial_role="background",
        ),
        subject("right", PixelBox(400, 0, 600, 400)),
    )

    image, summary = render_openpose_map(600, 400, regions)

    assert image.size == (600, 400)
    assert summary.subject_count == 2
    assert summary.joint_count == 36
    assert summary.connection_count == 34
    bounds = image.getbbox()
    assert bounds is not None
    assert bounds[0] < 100
    assert bounds[2] > 500


def test_pose_map_does_not_clip_limbs_to_subject_box() -> None:
    pose = default_subject_pose()
    joints = tuple(
        PoseJoint(name=joint.name, x=1.5 if joint.name == "left_wrist" else joint.x, y=joint.y)
        for joint in pose.joints
    )
    region = RegionDefinition(
        region_id="interaction",
        name="interaction",
        box=PixelBox(100, 50, 300, 450),
        spatial_role="subject",
        region_type="subject",
        pose=SubjectPose(joints=joints),
    )

    image, summary = render_openpose_map(500, 500, (region,))

    assert summary.subject_count == 1
    assert image.crop((300, 0, 500, 500)).getbbox() is not None


def test_disabled_subject_does_not_contribute_pose() -> None:
    image, summary = render_openpose_map(
        256,
        256,
        (subject("disabled", PixelBox(0, 0, 256, 256), enabled=False),),
    )

    assert summary.subject_count == 0
    assert image.getbbox() is None


def test_pose_gate_errors_are_actionable() -> None:
    code, message = classify_worker_error(
        PoseGateRuntimeError("volumetric pose gate callback diverged"),
        CommandKind.GENERATE_BASELINE,
    )

    assert code == "pose_gate_runtime_failed"
    assert "Volumetric pose gating failed" in message


def test_pose_gate_hook_conflict_has_distinct_diagnostic() -> None:
    code, message = classify_worker_error(
        PoseGateHookIncompatibleError(
            "a pre-existing denoise-mask callback is incompatible with volumetric pose gating"
        ),
        CommandKind.GENERATE_BASELINE,
    )

    assert code == "pose_gate_hook_incompatible"
    assert "conflicts" in message


def test_unrelated_sampling_error_does_not_blame_lora() -> None:
    code, message = classify_worker_error(
        RuntimeError("sampler produced a non-finite tensor"),
        CommandKind.GENERATE_BASELINE,
    )

    assert code == "generation_failed"
    assert "LoRA" not in message


def test_pose_mask_and_schedule_errors_have_distinct_codes() -> None:
    mask_code, _ = classify_worker_error(
        PoseMaskBuildError("empty mask"),
        CommandKind.GENERATE_BASELINE,
    )
    schedule_code, _ = classify_worker_error(
        PoseGateScheduleError("invalid phases"),
        CommandKind.GENERATE_BASELINE,
    )

    assert mask_code == "pose_mask_build_failed"
    assert schedule_code == "pose_gate_schedule_invalid"
