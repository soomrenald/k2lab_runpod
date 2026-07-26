from __future__ import annotations

from dataclasses import dataclass
from typing import Any


POSE_JOINT_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)


@dataclass(frozen=True, slots=True)
class PoseJoint:
    """One OpenPose-compatible joint normalized against its subject box."""

    name: str
    x: float
    y: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.name not in POSE_JOINT_NAMES:
            raise ValueError(f"unsupported pose joint: {self.name!r}")
        # Limbs may extend outside the owning subject box to make contact with
        # another subject. The generous bound still catches corrupt projects.
        if not -2.0 <= self.x <= 3.0 or not -2.0 <= self.y <= 3.0:
            raise ValueError("pose joint coordinates must be between -2 and 3")


@dataclass(frozen=True, slots=True)
class SubjectPose:
    """An articulated body pose owned by a subject region."""

    joints: tuple[PoseJoint, ...]
    enabled: bool = True

    def __post_init__(self) -> None:
        names = [joint.name for joint in self.joints]
        if len(names) != len(set(names)):
            raise ValueError("pose joint names must be unique")
        missing = set(POSE_JOINT_NAMES) - set(names)
        if missing:
            raise ValueError("subject pose is missing joints: " + ", ".join(sorted(missing)))

    def joint(self, name: str) -> PoseJoint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)


def default_subject_pose() -> SubjectPose:
    """Return a neutral standing pose in normalized subject-box coordinates."""

    points = {
        "nose": (0.50, 0.09),
        "neck": (0.50, 0.20),
        "right_shoulder": (0.36, 0.23),
        "right_elbow": (0.29, 0.40),
        "right_wrist": (0.28, 0.57),
        "left_shoulder": (0.64, 0.23),
        "left_elbow": (0.71, 0.40),
        "left_wrist": (0.72, 0.57),
        "right_hip": (0.43, 0.51),
        "right_knee": (0.41, 0.72),
        "right_ankle": (0.39, 0.94),
        "left_hip": (0.57, 0.51),
        "left_knee": (0.59, 0.72),
        "left_ankle": (0.61, 0.94),
        "right_eye": (0.46, 0.075),
        "left_eye": (0.54, 0.075),
        "right_ear": (0.42, 0.09),
        "left_ear": (0.58, 0.09),
    }
    return SubjectPose(
        joints=tuple(
            PoseJoint(name=name, x=points[name][0], y=points[name][1]) for name in POSE_JOINT_NAMES
        )
    )


def subject_pose_document(pose: SubjectPose) -> dict[str, Any]:
    return {
        "enabled": pose.enabled,
        "joints": [
            {
                "name": joint.name,
                "x": joint.x,
                "y": joint.y,
                "enabled": joint.enabled,
            }
            for joint in pose.joints
        ],
    }


def subject_pose_from_document(value: object) -> SubjectPose:
    if not isinstance(value, dict):
        return default_subject_pose()
    raw_joints = value.get("joints")
    if not isinstance(raw_joints, list):
        return default_subject_pose()
    by_name: dict[str, PoseJoint] = {}
    for item in raw_joints:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if name not in POSE_JOINT_NAMES:
            continue
        by_name[name] = PoseJoint(
            name=name,
            x=float(item.get("x", 0.5)),
            y=float(item.get("y", 0.5)),
            enabled=bool(item.get("enabled", True)),
        )
    defaults = default_subject_pose()
    joints = tuple(by_name.get(name, defaults.joint(name)) for name in POSE_JOINT_NAMES)
    return SubjectPose(joints=joints, enabled=bool(value.get("enabled", True)))
