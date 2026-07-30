from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from k2_region_lab.pose import (
    PoseHeadEllipse,
    VOLUMETRIC_POSE_JOINT_NAMES,
    VolumetricPoseJoint,
    VolumetricSubjectPose,
)
from k2_region_lab.regions import PixelBox, RegionDefinition

try:
    from .dataset import CanvasTransform
except ImportError:
    from dataset import CanvasTransform


COCO_JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
LEFT_RIGHT_PAIRS = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)


@dataclass(frozen=True, slots=True)
class CocoPerson:
    annotation_id: int
    bbox: tuple[float, float, float, float]
    points: Mapping[str, tuple[float, float, int]]
    iscrowd: bool

    @property
    def usable_joint_count(self) -> int:
        return sum(visibility > 0 for _x, _y, visibility in self.points.values())


def coco_person(annotation: Mapping[str, Any]) -> CocoPerson:
    keypoints = list(annotation.get("keypoints", []))
    if len(keypoints) != len(COCO_JOINT_NAMES) * 3:
        raise ValueError("COCO person keypoints must contain 17 x/y/visibility triples")
    points = {
        name: (
            float(keypoints[index * 3]),
            float(keypoints[index * 3 + 1]),
            int(keypoints[index * 3 + 2]),
        )
        for index, name in enumerate(COCO_JOINT_NAMES)
    }
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return CocoPerson(
        annotation_id=int(annotation["id"]),
        bbox=(x, y, x + width, y + height),
        points=points,
        iscrowd=bool(annotation.get("iscrowd", False)),
    )


def filter_reason(
    person: CocoPerson,
    transform: CanvasTransform,
    *,
    minimum_joints: int = 8,
    minimum_height: float = 96.0,
) -> str | None:
    if person.iscrowd:
        return "iscrowd"
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        if person.points[name][2] <= 0:
            return f"missing_{name}"
    if person.usable_joint_count < minimum_joints:
        return "insufficient_body_joints"
    if not all(math.isfinite(value) for point in person.points.values() for value in point[:2]):
        return "nonfinite_coordinates"
    _x0, y0 = transform.point(person.bbox[0], person.bbox[1])
    _x1, y1 = transform.point(person.bbox[2], person.bbox[3])
    if abs(y1 - y0) < minimum_height:
        return "person_too_small"
    return None


def _swap_anatomical_points(
    points: dict[str, tuple[float, float, int]],
) -> dict[str, tuple[float, float, int]]:
    swapped = dict(points)
    for left, right in LEFT_RIGHT_PAIRS:
        swapped[left], swapped[right] = points[right], points[left]
    return swapped


def _head(
    points: Mapping[str, tuple[float, float, int]],
    neck: tuple[float, float],
    shoulder_width: float,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    visible = [
        points[name][:2]
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if points[name][2] > 0
    ]
    if len(visible) >= 2:
        center = (
            sum(point[0] for point in visible) / len(visible),
            sum(point[1] for point in visible) / len(visible),
        )
        span = max(point[0] for point in visible) - min(point[0] for point in visible)
        radius_x = max(span * 0.65, shoulder_width * 0.12)
        radius_y = max(abs(neck[1] - center[1]) * 0.85, radius_x * 1.2)
        return center, (radius_x, radius_y), "face_points"
    radius_x = max(shoulder_width * 0.25, 2.0)
    radius_y = radius_x * 1.35
    return (neck[0], neck[1] - radius_y * 1.05), (radius_x, radius_y), "shoulder_fallback"


def person_region(
    person: CocoPerson,
    transform: CanvasTransform,
    *,
    region_id: str,
    priority: int = 0,
) -> tuple[RegionDefinition, str]:
    transformed = {
        name: (*transform.point(x, y), visibility)
        for name, (x, y, visibility) in person.points.items()
    }
    if transform.horizontal_flip:
        transformed = _swap_anatomical_points(transformed)
    x0, y0 = transform.point(person.bbox[0], person.bbox[1])
    x1, y1 = transform.point(person.bbox[2], person.bbox[3])
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    box = PixelBox(
        max(0.0, left),
        max(0.0, top),
        min(float(transform.target_width), right),
        min(float(transform.target_height), bottom),
    )
    left_shoulder = transformed["left_shoulder"]
    right_shoulder = transformed["right_shoulder"]
    neck = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    shoulder_width = math.dist(left_shoulder[:2], right_shoulder[:2])
    head_center, head_radii, fallback = _head(transformed, neck, shoulder_width)
    absolute = {"neck": neck}
    absolute.update(
        {
            name: transformed[name][:2]
            for name in VOLUMETRIC_POSE_JOINT_NAMES
            if name != "neck"
        }
    )
    pose = VolumetricSubjectPose(
        joints=tuple(
            VolumetricPoseJoint(
                name=name,
                x=(absolute[name][0] - box.x0) / box.width,
                y=(absolute[name][1] - box.y0) / box.height,
            )
            for name in VOLUMETRIC_POSE_JOINT_NAMES
        ),
        head=PoseHeadEllipse(
            cx=(head_center[0] - box.x0) / box.width,
            cy=(head_center[1] - box.y0) / box.height,
            rx=max(0.005, min(1.5, head_radii[0] / box.width)),
            ry=max(0.005, min(1.5, head_radii[1] / box.height)),
        ),
    )
    return (
        RegionDefinition(
            region_id=region_id,
            name=region_id,
            box=box,
            prompt="person",
            enabled=True,
            priority=priority,
            spatial_role="subject",
            region_type="subject",
            pose=pose,
        ),
        fallback,
    )
