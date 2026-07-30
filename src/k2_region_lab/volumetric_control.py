from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from k2_region_lab.pose import VolumetricSubjectPose
from k2_region_lab.pose_gating import PoseMaskBuildError
from k2_region_lab.regions import RegionDefinition
from k2_region_lab.volumetric_pose import (
    VOLUMETRIC_POSE_STYLE,
    VolumetricPoseStyle,
    _disk,
    _ellipse_neck_boundary,
    _expanded_polygon,
    _point,
    _tapered_capsule,
)


K2_VOLUMETRIC_CONTROL_FORMAT = "k2-volumetric-pose-control-v1"
K2_VOLUMETRIC_CONTROL_RENDERER_VERSION = 1

_PALETTE_ITEMS = (
    ("background", (0, 0, 0)),
    ("head", (255, 255, 255)),
    ("neck", (255, 215, 0)),
    ("torso", (255, 128, 0)),
    ("left_upper_arm", (255, 32, 64)),
    ("left_forearm", (255, 32, 192)),
    ("right_upper_arm", (32, 128, 255)),
    ("right_forearm", (32, 255, 255)),
    ("left_thigh", (128, 32, 255)),
    ("left_calf", (224, 32, 255)),
    ("right_thigh", (32, 255, 64)),
    ("right_calf", (160, 255, 32)),
)
K2_VOLUMETRIC_CONTROL_PALETTE: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    dict(_PALETTE_ITEMS)
)
K2_VOLUMETRIC_CONTROL_RENDER_ORDER = (
    "torso",
    "left_thigh",
    "left_calf",
    "right_thigh",
    "right_calf",
    "left_upper_arm",
    "left_forearm",
    "right_upper_arm",
    "right_forearm",
    "neck",
    "head",
)


def _format_document(style: VolumetricPoseStyle = VOLUMETRIC_POSE_STYLE) -> dict[str, Any]:
    geometry = asdict(style)
    geometry.pop("dilation_units")
    geometry.pop("minimum_dilation_pixels")
    geometry.pop("feather_units")
    geometry.pop("minimum_feather_pixels")
    return {
        "format": K2_VOLUMETRIC_CONTROL_FORMAT,
        "renderer_version": K2_VOLUMETRIC_CONTROL_RENDERER_VERSION,
        "channel_mode": "rgb",
        "normalize": "none",
        "invert": False,
        "background": "opaque_black",
        "anatomical_sides": True,
        "render_order": list(K2_VOLUMETRIC_CONTROL_RENDER_ORDER),
        "palette": {name: list(color) for name, color in _PALETTE_ITEMS},
        "geometry": geometry,
        "downsample": "lanczos",
        "canvas": "full_generation_canvas",
    }


K2_VOLUMETRIC_CONTROL_FORMAT_DOCUMENT = MappingProxyType(_format_document())
K2_VOLUMETRIC_CONTROL_FORMAT_SHA256 = hashlib.sha256(
    json.dumps(
        dict(K2_VOLUMETRIC_CONTROL_FORMAT_DOCUMENT),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class KreaVolumetricControlFormat:
    format_id: str = K2_VOLUMETRIC_CONTROL_FORMAT
    format_sha256: str = K2_VOLUMETRIC_CONTROL_FORMAT_SHA256
    renderer_version: int = K2_VOLUMETRIC_CONTROL_RENDERER_VERSION
    channel_mode: str = "rgb"
    normalize: str = "none"
    invert: bool = False

    def document(self) -> dict[str, Any]:
        return dict(K2_VOLUMETRIC_CONTROL_FORMAT_DOCUMENT)


@dataclass(frozen=True, slots=True)
class KreaVolumetricControlImage:
    width: int
    height: int
    region_ids: tuple[str, ...]
    rgb: np.ndarray
    png_bytes: bytes
    sha256: str
    coverage: float
    format_id: str = K2_VOLUMETRIC_CONTROL_FORMAT
    format_sha256: str = K2_VOLUMETRIC_CONTROL_FORMAT_SHA256
    renderer_version: int = K2_VOLUMETRIC_CONTROL_RENDERER_VERSION

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.shape != (self.height, self.width, 3):
            raise ValueError("control raster must be RGB uint8 at the declared dimensions")
        self.rgb.flags.writeable = False

    def document(self) -> dict[str, Any]:
        return {
            "format": self.format_id,
            "format_sha256": self.format_sha256,
            "renderer_version": self.renderer_version,
            "width": self.width,
            "height": self.height,
            "region_ids": list(self.region_ids),
            "sha256": self.sha256,
            "coverage": self.coverage,
        }


@dataclass(frozen=True, slots=True)
class KreaVolumetricControlBundle:
    full: KreaVolumetricControlImage
    subjects: Mapping[str, KreaVolumetricControlImage]
    format: KreaVolumetricControlFormat = KreaVolumetricControlFormat()


def _active_subjects(
    regions: Iterable[RegionDefinition],
) -> tuple[tuple[RegionDefinition, VolumetricSubjectPose], ...]:
    active = [
        (region, region.pose)
        for region in regions
        if (
            region.enabled
            and region.region_type == "subject"
            and isinstance(region.pose, VolumetricSubjectPose)
            and region.pose.enabled
        )
    ]
    return tuple(
        (region, pose)
        for region, pose in sorted(
            active,
            key=lambda item: (item[0].priority, item[0].region_id),
        )
    )


def _draw_subject(
    draw: ImageDraw.ImageDraw,
    region: RegionDefinition,
    pose: VolumetricSubjectPose,
    *,
    style: VolumetricPoseStyle,
) -> None:
    scale = style.supersampling
    joints = {
        joint.name: _point(region, joint.x, joint.y, scale)
        for joint in pose.joints
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
    torso = _expanded_polygon(
        (
            joints["left_shoulder"],
            joints["right_shoulder"],
            joints["right_hip"],
            joints["left_hip"],
        ),
        style.torso_expansion,
    )
    draw.polygon(torso, fill=K2_VOLUMETRIC_CONTROL_PALETTE["torso"])
    for point in torso:
        _disk(
            draw,
            point,
            2.25 * unit,
            fill=K2_VOLUMETRIC_CONTROL_PALETTE["torso"],
        )

    segments = (
        ("left_thigh", "left_hip", "left_knee", style.thigh_widths),
        ("left_calf", "left_knee", "left_ankle", style.calf_widths),
        ("right_thigh", "right_hip", "right_knee", style.thigh_widths),
        ("right_calf", "right_knee", "right_ankle", style.calf_widths),
        ("left_upper_arm", "left_shoulder", "left_elbow", style.upper_arm_widths),
        ("left_forearm", "left_elbow", "left_wrist", style.forearm_widths),
        ("right_upper_arm", "right_shoulder", "right_elbow", style.upper_arm_widths),
        ("right_forearm", "right_elbow", "right_wrist", style.forearm_widths),
    )
    for part, start, end, widths in segments:
        _tapered_capsule(
            draw,
            joints[start],
            joints[end],
            widths[0] * unit / 2,
            widths[1] * unit / 2,
            fill=K2_VOLUMETRIC_CONTROL_PALETTE[part],
        )

    neck_boundary = _ellipse_neck_boundary(
        head_center,
        head_radii,
        joints["neck"],
    )
    _tapered_capsule(
        draw,
        joints["neck"],
        neck_boundary,
        style.neck_widths[0] * unit / 2,
        style.neck_widths[1] * unit / 2,
        fill=K2_VOLUMETRIC_CONTROL_PALETTE["neck"],
    )
    draw.ellipse(
        (
            head_center[0] - head_radii[0],
            head_center[1] - head_radii[1],
            head_center[0] + head_radii[0],
            head_center[1] + head_radii[1],
        ),
        fill=K2_VOLUMETRIC_CONTROL_PALETTE["head"],
    )


def render_krea_volumetric_control(
    *,
    regions: Iterable[RegionDefinition],
    width: int,
    height: int,
    subject_region_id: str | None = None,
    style: VolumetricPoseStyle = VOLUMETRIC_POSE_STYLE,
) -> KreaVolumetricControlImage:
    if width <= 0 or height <= 0:
        raise PoseMaskBuildError("control image canvas dimensions must be positive")
    active = _active_subjects(regions)
    if subject_region_id is not None:
        active = tuple(item for item in active if item[0].region_id == subject_region_id)
        if not active:
            raise PoseMaskBuildError(
                f"subject {subject_region_id!r} has no enabled volumetric mannequin"
            )
    scale = style.supersampling
    canvas = Image.new(
        "RGB",
        (width * scale, height * scale),
        K2_VOLUMETRIC_CONTROL_PALETTE["background"],
    )
    draw = ImageDraw.Draw(canvas)
    for region, pose in active:
        _draw_subject(draw, region, pose, style=style)
    image = canvas.resize((width, height), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8).copy()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    png_bytes = buffer.getvalue()
    covered = np.any(rgb != 0, axis=-1)
    return KreaVolumetricControlImage(
        width=width,
        height=height,
        region_ids=tuple(region.region_id for region, _pose in active),
        rgb=rgb,
        png_bytes=png_bytes,
        sha256=hashlib.sha256(png_bytes).hexdigest(),
        coverage=float(np.count_nonzero(covered)) / float(width * height),
    )


def render_krea_volumetric_control_bundle(
    *,
    regions: Iterable[RegionDefinition],
    width: int,
    height: int,
    include_subjects: bool = True,
) -> KreaVolumetricControlBundle:
    region_tuple = tuple(regions)
    full = render_krea_volumetric_control(
        regions=region_tuple,
        width=width,
        height=height,
    )
    subject_images = (
        {
            region_id: render_krea_volumetric_control(
                regions=region_tuple,
                width=width,
                height=height,
                subject_region_id=region_id,
            )
            for region_id in full.region_ids
        }
        if include_subjects
        else {}
    )
    return KreaVolumetricControlBundle(
        full=full,
        subjects=MappingProxyType(subject_images),
    )
