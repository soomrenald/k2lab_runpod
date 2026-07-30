import argparse
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from k2core.depth import DepthRegionMode
from k2_region_lab.depth.validate import _depth_region, _pixel_sha256


def test_depth_region_argument_builds_matching_settings_and_geometry() -> None:
    settings, geometry = _depth_region("subject:relax:0.25:10,20,110,220")

    assert settings.region_id == geometry.region_id == "subject"
    assert settings.mode == DepthRegionMode.RELAX
    assert settings.strength == 0.25
    assert geometry.box.x0 == 10
    assert geometry.box.y1 == 220


def test_depth_region_argument_rejects_bad_shape() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _depth_region("subject:relax:0.25:10,20,110")


def test_pixel_digest_ignores_png_metadata(tmp_path: Path) -> None:
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(pixels).save(first)
    Image.fromarray(pixels).save(second)

    assert _pixel_sha256(first) == _pixel_sha256(second)
