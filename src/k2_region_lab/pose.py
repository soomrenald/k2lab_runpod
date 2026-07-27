from __future__ import annotations

import math
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


VOLUMETRIC_POSE_FORMAT = "k2-volumetric-pose-v1"
VOLUMETRIC_POSE_JOINT_NAMES = (
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True, slots=True)
class VolumetricPoseJoint:
    name: str
    x: float
    y: float

    def __post_init__(self) -> None:
        if self.name not in VOLUMETRIC_POSE_JOINT_NAMES:
            raise ValueError(f"unsupported volumetric pose joint: {self.name!r}")
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("volumetric pose joint coordinates must be finite")
        if not -2.0 <= self.x <= 3.0 or not -2.0 <= self.y <= 3.0:
            raise ValueError("volumetric pose joint coordinates must be between -2 and 3")


@dataclass(frozen=True, slots=True)
class PoseHeadEllipse:
    cx: float
    cy: float
    rx: float
    ry: float

    def __post_init__(self) -> None:
        values = (self.cx, self.cy, self.rx, self.ry)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose head ellipse values must be finite")
        if not -2.0 <= self.cx <= 3.0 or not -2.0 <= self.cy <= 3.0:
            raise ValueError("pose head center must be between -2 and 3")
        if not 0.005 <= self.rx <= 1.5 or not 0.005 <= self.ry <= 1.5:
            raise ValueError("pose head radii must be between 0.005 and 1.5")


@dataclass(frozen=True, slots=True)
class VolumetricSubjectPose:
    joints: tuple[VolumetricPoseJoint, ...]
    head: PoseHeadEllipse
    enabled: bool = True
    format: str = VOLUMETRIC_POSE_FORMAT

    def __post_init__(self) -> None:
        if self.format != VOLUMETRIC_POSE_FORMAT:
            raise ValueError(f"unsupported volumetric pose format: {self.format!r}")
        names = [joint.name for joint in self.joints]
        if len(names) != len(set(names)):
            raise ValueError("volumetric pose joint names must be unique")
        missing = set(VOLUMETRIC_POSE_JOINT_NAMES) - set(names)
        extra = set(names) - set(VOLUMETRIC_POSE_JOINT_NAMES)
        if missing or extra:
            raise ValueError(
                "volumetric pose must contain exactly these joints: "
                + ", ".join(VOLUMETRIC_POSE_JOINT_NAMES)
            )

    def joint(self, name: str) -> VolumetricPoseJoint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)


_STANDING_VOLUMETRIC_POINTS = {
    "neck": (0.50, 0.20),
    "left_shoulder": (0.39, 0.24),
    "right_shoulder": (0.61, 0.24),
    "left_elbow": (0.34, 0.42),
    "right_elbow": (0.66, 0.42),
    "left_wrist": (0.31, 0.60),
    "right_wrist": (0.69, 0.60),
    "left_hip": (0.44, 0.52),
    "right_hip": (0.56, 0.52),
    "left_knee": (0.43, 0.72),
    "right_knee": (0.57, 0.72),
    "left_ankle": (0.42, 0.94),
    "right_ankle": (0.58, 0.94),
}

_SQUATTING_VOLUMETRIC_POINTS = {
    "neck": (0.50, 0.36),
    "left_shoulder": (0.37, 0.38),
    "right_shoulder": (0.63, 0.38),
    "left_elbow": (0.31, 0.51),
    "right_elbow": (0.69, 0.51),
    "left_wrist": (0.24, 0.61),
    "right_wrist": (0.76, 0.61),
    "left_hip": (0.44, 0.60),
    "right_hip": (0.56, 0.60),
    "left_knee": (0.29, 0.71),
    "right_knee": (0.71, 0.71),
    "left_ankle": (0.21, 0.92),
    "right_ankle": (0.79, 0.92),
}


def _volumetric_pose_from_points(
    points: dict[str, tuple[float, float]],
    *,
    head: PoseHeadEllipse,
) -> VolumetricSubjectPose:
    return VolumetricSubjectPose(
        joints=tuple(
            VolumetricPoseJoint(name, *points[name])
            for name in VOLUMETRIC_POSE_JOINT_NAMES
        ),
        head=head,
    )


def default_volumetric_subject_pose() -> VolumetricSubjectPose:
    return _volumetric_pose_from_points(
        _STANDING_VOLUMETRIC_POINTS,
        head=PoseHeadEllipse(cx=0.50, cy=0.105, rx=0.075, ry=0.105),
    )


def squatting_volumetric_subject_pose() -> VolumetricSubjectPose:
    return _volumetric_pose_from_points(
        _SQUATTING_VOLUMETRIC_POINTS,
        head=PoseHeadEllipse(cx=0.50, cy=0.265, rx=0.075, ry=0.105),
    )


def mirror_volumetric_subject_pose(pose: VolumetricSubjectPose) -> VolumetricSubjectPose:
    by_name = {joint.name: joint for joint in pose.joints}
    mirrored: list[VolumetricPoseJoint] = []
    for name in VOLUMETRIC_POSE_JOINT_NAMES:
        if name.startswith("left_"):
            source_name = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            source_name = "left_" + name.removeprefix("right_")
        else:
            source_name = name
        source = by_name[source_name]
        mirrored.append(VolumetricPoseJoint(name=name, x=1.0 - source.x, y=source.y))
    return VolumetricSubjectPose(
        joints=tuple(mirrored),
        head=PoseHeadEllipse(
            cx=1.0 - pose.head.cx,
            cy=pose.head.cy,
            rx=pose.head.rx,
            ry=pose.head.ry,
        ),
        enabled=pose.enabled,
    )


def volumetric_subject_pose_document(pose: VolumetricSubjectPose) -> dict[str, Any]:
    return {
        "enabled": pose.enabled,
        "format": pose.format,
        "joints": {
            joint.name: {"x": joint.x, "y": joint.y}
            for joint in pose.joints
        },
        "head": {
            "cx": pose.head.cx,
            "cy": pose.head.cy,
            "rx": pose.head.rx,
            "ry": pose.head.ry,
        },
    }


def _legacy_head_ellipse(
    by_name: dict[str, tuple[float, float]],
    neck: VolumetricPoseJoint,
) -> PoseHeadEllipse:
    face_points = [
        by_name[name]
        for name in ("right_eye", "left_eye", "right_ear", "left_ear")
        if name in by_name
    ]
    if "nose" in by_name:
        cx, cy = by_name["nose"]
    elif face_points:
        cx = sum(point[0] for point in face_points) / len(face_points)
        cy = sum(point[1] for point in face_points) / len(face_points)
    else:
        cx, cy = neck.x, neck.y - 0.095
    horizontal = [abs(point[0] - cx) for point in face_points]
    rx = max(horizontal, default=0.075)
    ry = max(0.055, min(0.20, abs(neck.y - cy)))
    return PoseHeadEllipse(
        cx=max(-2.0, min(3.0, cx)),
        cy=max(-2.0, min(3.0, cy)),
        rx=max(0.04, min(0.20, rx)),
        ry=ry,
    )


def volumetric_subject_pose_from_document(value: object) -> VolumetricSubjectPose:
    defaults = default_volumetric_subject_pose()
    if not isinstance(value, dict):
        return defaults
    raw_joints = value.get("joints")
    by_name: dict[str, tuple[float, float]] = {}
    if isinstance(raw_joints, dict):
        for name, item in raw_joints.items():
            if name in VOLUMETRIC_POSE_JOINT_NAMES and isinstance(item, dict):
                by_name[name] = (float(item.get("x", 0.5)), float(item.get("y", 0.5)))
    elif isinstance(raw_joints, list):
        for item in raw_joints:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if name in {*VOLUMETRIC_POSE_JOINT_NAMES, "nose", "left_eye", "right_eye",
                        "left_ear", "right_ear"}:
                by_name[name] = (float(item.get("x", 0.5)), float(item.get("y", 0.5)))

    joints = tuple(
        VolumetricPoseJoint(
            name,
            *(by_name.get(name) or (defaults.joint(name).x, defaults.joint(name).y)),
        )
        for name in VOLUMETRIC_POSE_JOINT_NAMES
    )
    head_value = value.get("head")
    if value.get("format") == VOLUMETRIC_POSE_FORMAT and isinstance(head_value, dict):
        head = PoseHeadEllipse(
            cx=float(head_value.get("cx", defaults.head.cx)),
            cy=float(head_value.get("cy", defaults.head.cy)),
            rx=float(head_value.get("rx", defaults.head.rx)),
            ry=float(head_value.get("ry", defaults.head.ry)),
        )
    else:
        head = _legacy_head_ellipse(by_name, next(joint for joint in joints if joint.name == "neck"))
    return VolumetricSubjectPose(
        joints=joints,
        head=head,
        enabled=bool(value.get("enabled", True)),
    )
