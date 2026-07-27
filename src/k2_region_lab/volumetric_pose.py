from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from k2_region_lab.pose import VOLUMETRIC_POSE_FORMAT, VolumetricSubjectPose
from k2_region_lab.pose_gating import PoseMaskBuildError
from k2_region_lab.regions import RegionDefinition


@dataclass(frozen=True, slots=True)
class VolumetricPoseStyle:
    format: str = VOLUMETRIC_POSE_FORMAT
    supersampling: int = 4
    torso_expansion: float = 1.08
    unit_fraction: float = 0.0125
    minimum_unit_pixels: float = 1.5
    upper_arm_widths: tuple[float, float] = (4.5, 4.0)
    forearm_widths: tuple[float, float] = (4.0, 3.0)
    thigh_widths: tuple[float, float] = (6.5, 5.5)
    calf_widths: tuple[float, float] = (5.5, 3.8)
    neck_widths: tuple[float, float] = (3.5, 3.5)
    dilation_units: float = 2.5
    minimum_dilation_pixels: float = 8.0
    feather_units: float = 1.0
    minimum_feather_pixels: float = 4.0


VOLUMETRIC_POSE_STYLE = VolumetricPoseStyle()


@dataclass(frozen=True, slots=True)
class SubjectPoseMaskSummary:
    region_id: str
    core_pixels: int
    support_pixels: int
    core_coverage: float
    support_coverage: float
    dilation_radius: float
    feather_radius: float

    def document(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "core_pixels": self.core_pixels,
            "support_pixels": self.support_pixels,
            "core_coverage": self.core_coverage,
            "support_coverage": self.support_coverage,
            "dilation_radius": self.dilation_radius,
            "feather_radius": self.feather_radius,
        }


@dataclass(frozen=True, slots=True)
class PoseMaskSummary:
    subject_count: int
    subjects: tuple[SubjectPoseMaskSummary, ...]
    union_core_coverage: float
    union_support_coverage: float
    overlap_pixels: int
    priority_region_ids: tuple[str, ...]
    style_version: str
    warnings: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        return {
            "subject_count": self.subject_count,
            "subjects": [subject.document() for subject in self.subjects],
            "union_core_coverage": self.union_core_coverage,
            "union_support_coverage": self.union_support_coverage,
            "overlap_pixels": self.overlap_pixels,
            "priority_region_ids": list(self.priority_region_ids),
            "style_version": self.style_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SubjectPoseMasks:
    region_id: str
    core: np.ndarray
    support: np.ndarray
    ownership: np.ndarray
    summary: SubjectPoseMaskSummary


@dataclass(frozen=True, slots=True)
class VolumetricMaskBundle:
    width: int
    height: int
    subjects: tuple[SubjectPoseMasks, ...]
    union_core: np.ndarray
    union_support: np.ndarray
    summary: PoseMaskSummary


def _point(region: RegionDefinition, x: float, y: float, scale: int) -> tuple[float, float]:
    return (
        (region.box.x0 + x * region.box.width) * scale,
        (region.box.y0 + y * region.box.height) * scale,
    )


def _disk(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    *,
    fill: int = 255,
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _tapered_capsule(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    start_radius: float,
    end_radius: float,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        _disk(draw, start, max(start_radius, end_radius))
        return
    px, py = -dy / length, dx / length
    polygon = (
        (start[0] + px * start_radius, start[1] + py * start_radius),
        (end[0] + px * end_radius, end[1] + py * end_radius),
        (end[0] - px * end_radius, end[1] - py * end_radius),
        (start[0] - px * start_radius, start[1] - py * start_radius),
    )
    draw.polygon(polygon, fill=255)
    _disk(draw, start, start_radius)
    _disk(draw, end, end_radius)


def _ellipse_neck_boundary(
    center: tuple[float, float],
    radii: tuple[float, float],
    neck: tuple[float, float],
) -> tuple[float, float]:
    dx, dy = neck[0] - center[0], neck[1] - center[1]
    denominator = math.sqrt((dx / radii[0]) ** 2 + (dy / radii[1]) ** 2)
    if denominator < 1e-9:
        return center[0], center[1] + radii[1]
    return center[0] + dx / denominator, center[1] + dy / denominator


def _expanded_polygon(
    points: tuple[tuple[float, float], ...],
    factor: float,
) -> tuple[tuple[float, float], ...]:
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return tuple((cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points)


def render_volumetric_core(
    region: RegionDefinition,
    pose: VolumetricSubjectPose,
    *,
    width: int,
    height: int,
    style: VolumetricPoseStyle = VOLUMETRIC_POSE_STYLE,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise PoseMaskBuildError("pose mask canvas dimensions must be positive")
    scale = style.supersampling
    image = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(image)
    joints = {
        name: _point(region, pose.joint(name).x, pose.joint(name).y, scale)
        for name in (joint.name for joint in pose.joints)
    }
    unit = max(
        style.minimum_unit_pixels,
        style.unit_fraction * min(region.box.width, region.box.height),
    ) * scale

    head_center = _point(region, pose.head.cx, pose.head.cy, scale)
    head_radii = (
        pose.head.rx * region.box.width * scale,
        pose.head.ry * region.box.height * scale,
    )
    draw.ellipse(
        (
            head_center[0] - head_radii[0],
            head_center[1] - head_radii[1],
            head_center[0] + head_radii[0],
            head_center[1] + head_radii[1],
        ),
        fill=255,
    )

    torso = (
        joints["left_shoulder"],
        joints["right_shoulder"],
        joints["right_hip"],
        joints["left_hip"],
    )
    expanded_torso = _expanded_polygon(torso, style.torso_expansion)
    draw.polygon(expanded_torso, fill=255)
    torso_corner_radius = 2.25 * unit
    for point in expanded_torso:
        _disk(draw, point, torso_corner_radius)

    neck_boundary = _ellipse_neck_boundary(
        head_center, head_radii, joints["neck"]
    )
    _tapered_capsule(
        draw,
        joints["neck"],
        neck_boundary,
        style.neck_widths[0] * unit / 2,
        style.neck_widths[1] * unit / 2,
    )

    segments = (
        ("left_shoulder", "left_elbow", style.upper_arm_widths),
        ("right_shoulder", "right_elbow", style.upper_arm_widths),
        ("left_elbow", "left_wrist", style.forearm_widths),
        ("right_elbow", "right_wrist", style.forearm_widths),
        ("left_hip", "left_knee", style.thigh_widths),
        ("right_hip", "right_knee", style.thigh_widths),
        ("left_knee", "left_ankle", style.calf_widths),
        ("right_knee", "right_ankle", style.calf_widths),
    )
    for start, end, widths in segments:
        _tapered_capsule(
            draw,
            joints[start],
            joints[end],
            widths[0] * unit / 2,
            widths[1] * unit / 2,
        )

    downsampled = image.resize((width, height), Image.Resampling.LANCZOS)
    return (np.asarray(downsampled, dtype=np.uint8) >= 128).astype(np.float32)


def _odd_filter_size(radius: float) -> int:
    return max(3, 2 * int(math.ceil(radius)) + 1)


def _support_mask(
    core: np.ndarray,
    *,
    dilation_radius: float,
    feather_radius: float,
) -> np.ndarray:
    image = Image.fromarray((core * 255.0).astype(np.uint8), mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(_odd_filter_size(dilation_radius)))
    feathered = dilated.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    support = np.asarray(feathered, dtype=np.float32) / 255.0
    return np.maximum(core, np.clip(support, 0.0, 1.0)).astype(np.float32)


def _active_subjects(
    regions: Iterable[RegionDefinition],
) -> tuple[tuple[RegionDefinition, VolumetricSubjectPose], ...]:
    active = []
    for region in regions:
        pose = region.pose
        if (
            region.enabled
            and region.region_type == "subject"
            and isinstance(pose, VolumetricSubjectPose)
            and pose.enabled
        ):
            active.append((region, pose))
    active.sort(key=lambda item: -item[0].priority)
    return tuple(active)


def render_volumetric_masks(
    *,
    regions: Iterable[RegionDefinition],
    width: int,
    height: int,
    style: VolumetricPoseStyle = VOLUMETRIC_POSE_STYLE,
) -> VolumetricMaskBundle:
    active = _active_subjects(regions)
    total_pixels = width * height
    if total_pixels <= 0:
        raise PoseMaskBuildError("pose mask canvas dimensions must be positive")
    zeros = np.zeros((height, width), dtype=np.float32)
    cores: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    summaries: list[SubjectPoseMaskSummary] = []
    for region, pose in active:
        core = render_volumetric_core(
            region, pose, width=width, height=height, style=style
        )
        unit = max(
            style.minimum_unit_pixels,
            style.unit_fraction * min(region.box.width, region.box.height),
        )
        dilation = max(style.minimum_dilation_pixels, style.dilation_units * unit)
        feather = max(style.minimum_feather_pixels, style.feather_units * unit)
        support = _support_mask(
            core, dilation_radius=dilation, feather_radius=feather
        )
        core_pixels = int(np.count_nonzero(core))
        support_pixels = int(np.count_nonzero(support))
        cores.append(core)
        supports.append(support)
        summaries.append(
            SubjectPoseMaskSummary(
                region_id=region.region_id,
                core_pixels=core_pixels,
                support_pixels=support_pixels,
                core_coverage=core_pixels / total_pixels,
                support_coverage=support_pixels / total_pixels,
                dilation_radius=dilation,
                feather_radius=feather,
            )
        )

    union_core = np.maximum.reduce(cores) if cores else zeros.copy()
    union_support = np.maximum.reduce(supports) if supports else zeros.copy()
    overlap_count = (
        int(np.count_nonzero(np.sum([support > 0.0 for support in supports], axis=0) > 1))
        if supports
        else 0
    )
    claimed = np.zeros((height, width), dtype=bool)
    ownerships: list[np.ndarray] = []
    for support in supports:
        available = ~claimed
        ownerships.append(np.where(available, support, 0.0).astype(np.float32))
        claimed |= support > 0.0

    warnings: list[str] = []
    coverage = float(np.count_nonzero(union_support)) / total_pixels
    if active and coverage < 0.01:
        warnings.append("Mannequin support covers less than 1% of the canvas.")
    if coverage > 0.85:
        warnings.append("Mannequin support covers more than 85% of the canvas.")
    subject_masks = tuple(
        SubjectPoseMasks(
            region_id=region.region_id,
            core=core,
            support=support,
            ownership=ownership,
            summary=summary,
        )
        for (region, _pose), core, support, ownership, summary in zip(
            active, cores, supports, ownerships, summaries, strict=True
        )
    )
    summary = PoseMaskSummary(
        subject_count=len(active),
        subjects=tuple(summaries),
        union_core_coverage=float(np.count_nonzero(union_core)) / total_pixels,
        union_support_coverage=coverage,
        overlap_pixels=overlap_count,
        priority_region_ids=tuple(region.region_id for region, _pose in active),
        style_version=style.format,
        warnings=tuple(warnings),
    )
    return VolumetricMaskBundle(
        width=width,
        height=height,
        subjects=subject_masks,
        union_core=union_core,
        union_support=union_support,
        summary=summary,
    )
