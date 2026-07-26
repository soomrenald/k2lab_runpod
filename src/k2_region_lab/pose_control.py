from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from k2_region_lab.regions import RegionDefinition


OPENPOSE_CONNECTIONS = (
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("neck", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("neck", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("nose", "left_eye"),
    ("left_eye", "left_ear"),
)

# OpenPose-compatible colors keep each limb distinguishable to pose preprocessors
# and ControlNets trained against the standard rendered skeleton.
OPENPOSE_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


@dataclass(frozen=True, slots=True)
class PoseControlSummary:
    subject_count: int
    joint_count: int
    connection_count: int
    width: int
    height: int

    def document(self) -> dict[str, int]:
        return {
            "subject_count": self.subject_count,
            "joint_count": self.joint_count,
            "connection_count": self.connection_count,
            "width": self.width,
            "height": self.height,
        }


def render_openpose_map(
    width: int,
    height: int,
    regions: tuple[RegionDefinition, ...],
) -> tuple[Image.Image, PoseControlSummary]:
    """Compose all enabled subject mannequins into one unclipped canvas map."""

    if width <= 0 or height <= 0:
        raise ValueError("pose control dimensions must be positive")
    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    line_width = max(4, round(min(width, height) / 180)) * scale
    radius = max(3, round(min(width, height) / 240)) * scale
    subject_count = 0
    joint_count = 0
    connection_count = 0

    for region in regions:
        if (
            not region.enabled
            or region.region_type != "subject"
            or region.pose is None
            or not region.pose.enabled
        ):
            continue
        subject_count += 1
        points = {
            joint.name: (
                (region.box.x0 + joint.x * region.box.width) * scale,
                (region.box.y0 + joint.y * region.box.height) * scale,
            )
            for joint in region.pose.joints
            if joint.enabled
        }
        joint_count += len(points)
        for index, (first_name, second_name) in enumerate(OPENPOSE_CONNECTIONS):
            first = points.get(first_name)
            second = points.get(second_name)
            if first is None or second is None:
                continue
            draw.line(
                (first, second),
                fill=OPENPOSE_COLORS[index % len(OPENPOSE_COLORS)],
                width=line_width,
            )
            connection_count += 1
        for index, joint in enumerate(region.pose.joints):
            point = points.get(joint.name)
            if point is None:
                continue
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=OPENPOSE_COLORS[index % len(OPENPOSE_COLORS)],
            )

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image, PoseControlSummary(
        subject_count=subject_count,
        joint_count=joint_count,
        connection_count=connection_count,
        width=width,
        height=height,
    )
