from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from k2core.depth import (
    DepthNormalizationMode,
    DepthNormalizationSettings,
    load_depth_image,
    normalize_depth,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an explicit Krea depth preprocessing calibration grid."
    )
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, nargs="+", default=(0.7, 1.0, 1.4))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = load_depth_image(args.depth)
    tiles: list[tuple[str, Image.Image, dict[str, object]]] = []
    for mode in (DepthNormalizationMode.MINMAX, DepthNormalizationMode.PERCENTILE):
        for invert in (False, True):
            for gamma in args.gamma:
                normalized = normalize_depth(
                    source,
                    DepthNormalizationSettings(
                        mode=mode,
                        invert=invert,
                        gamma=gamma,
                    ),
                )
                pixels = np.rint(normalized.values * 255).astype(np.uint8)
                label = f"{mode.value} | invert={invert} | gamma={gamma:g}"
                tiles.append(
                    (
                        label,
                        Image.fromarray(pixels).convert("RGB"),
                        normalized.report.document(),
                    )
                )
    tile_width = min(512, source.info.width)
    tile_height = max(64, round(source.info.height * tile_width / source.info.width))
    label_height = 32
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    reports = []
    for index, (label, image, report) in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        grid.paste(image.resize((tile_width, tile_height), Image.Resampling.BILINEAR), (x, y))
        draw.text((x + 8, y + tile_height + 8), label, fill="black")
        reports.append({"label": label, **report})
    args.output.mkdir(parents=True, exist_ok=True)
    grid.save(args.output / "preprocessing-grid.png")
    (args.output / "calibration.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source": source.info.document(),
                "candidates": reports,
                "recommendation": (
                    "Start with minmax, invert=false, gamma=1.0. The public checkpoint "
                    "expects inverse depth with near objects white; confirm against a "
                    "generated response grid before changing inversion."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
