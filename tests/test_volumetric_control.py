from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from k2_region_lab.pose import default_volumetric_subject_pose
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.volumetric_control import (
    K2_VOLUMETRIC_CONTROL_FORMAT,
    K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
    K2_VOLUMETRIC_CONTROL_PALETTE,
    render_krea_volumetric_control_bundle,
)
from k2_region_lab.volumetric_pose import render_volumetric_core


def _subject(region_id: str, x0: int, *, priority: int = 0) -> RegionDefinition:
    return RegionDefinition(
        region_id=region_id,
        name=region_id,
        box=PixelBox(x0, 24, x0 + 160, 248),
        prompt="a person",
        enabled=True,
        priority=priority,
        spatial_role="subject",
        region_type="subject",
        pose=default_volumetric_subject_pose(),
    )


def test_control_format_palette_and_frontend_artifact_are_exact() -> None:
    assert K2_VOLUMETRIC_CONTROL_FORMAT == "k2-volumetric-pose-control-v1"
    assert dict(K2_VOLUMETRIC_CONTROL_PALETTE) == {
        "background": (0, 0, 0),
        "head": (255, 255, 255),
        "neck": (255, 215, 0),
        "torso": (255, 128, 0),
        "left_upper_arm": (255, 32, 64),
        "left_forearm": (255, 32, 192),
        "right_upper_arm": (32, 128, 255),
        "right_forearm": (32, 255, 255),
        "left_thigh": (128, 32, 255),
        "left_calf": (224, 32, 255),
        "right_thigh": (32, 255, 64),
        "right_calf": (160, 255, 32),
    }
    assert re.fullmatch(r"[0-9a-f]{64}", K2_VOLUMETRIC_CONTROL_FORMAT_SHA256)
    frontend = Path("web/client/src/volumetricControlFormat.ts").read_text()
    assert K2_VOLUMETRIC_CONTROL_FORMAT in frontend
    for color in K2_VOLUMETRIC_CONTROL_PALETTE.values():
        assert f"[{', '.join(str(channel) for channel in color)}]" in frontend


def test_full_and_subject_controls_are_deterministic_full_canvas_pngs() -> None:
    subjects = (_subject("a", 8, priority=2), _subject("b", 164, priority=1))
    first = render_krea_volumetric_control_bundle(
        regions=subjects,
        width=336,
        height=272,
    )
    second = render_krea_volumetric_control_bundle(
        regions=reversed(subjects),
        width=336,
        height=272,
    )
    assert first.full.sha256 == second.full.sha256
    assert first.full.rgb.shape == (272, 336, 3)
    assert set(first.subjects) == {"a", "b"}
    assert all(image.rgb.shape == first.full.rgb.shape for image in first.subjects.values())
    assert first.subjects["a"].sha256 != first.subjects["b"].sha256
    assert Image.open(__import__("io").BytesIO(first.full.png_bytes)).mode == "RGB"
    assert np.all(first.subjects["a"].rgb[:, 220:] == 0)
    assert np.all(first.subjects["b"].rgb[:, :130] == 0)


def test_control_geometry_uses_exact_volumetric_core_without_support_dilation() -> None:
    subject = _subject("a", 48)
    bundle = render_krea_volumetric_control_bundle(
        regions=(subject,),
        width=256,
        height=272,
    )
    control_coverage = np.any(bundle.full.rgb != 0, axis=-1)
    core = render_volumetric_core(
        subject,
        subject.pose,
        width=256,
        height=272,
    )
    # Lanczos retains a narrow antialias fringe. Every binary core pixel must
    # be represented, and there must be no wide support-mask expansion.
    assert np.all(control_coverage[core > 0])
    assert np.count_nonzero(control_coverage) < np.count_nonzero(core) * 1.35
    assert bundle.full.coverage < 0.75
