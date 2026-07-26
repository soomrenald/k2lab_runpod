from k2_region_lab.pose import PoseJoint, SubjectPose, default_subject_pose
from k2_region_lab.pose_control import render_openpose_map
from k2_region_lab.regions import PixelBox, RegionDefinition
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


def test_pose_control_errors_are_actionable() -> None:
    code, message = classify_worker_error(
        RuntimeError("selected file is not a supported ControlNet"),
        CommandKind.GENERATE_BASELINE,
    )

    assert code == "pose_control_failed"
    assert "Qwen Image ControlNet Union" in message
