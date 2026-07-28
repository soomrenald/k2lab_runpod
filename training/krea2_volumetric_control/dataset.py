from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_BUCKETS = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
)


@dataclass(frozen=True, slots=True)
class CanvasTransform:
    source_box: tuple[float, float, float, float]
    target_width: int
    target_height: int
    scale: float
    offset_x: float
    offset_y: float
    horizontal_flip: bool = False

    @classmethod
    def cover(
        cls,
        source_box: tuple[float, float, float, float],
        target: tuple[int, int],
        *,
        horizontal_flip: bool = False,
    ) -> "CanvasTransform":
        left, top, right, bottom = source_box
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise ValueError("source transform box must have positive area")
        target_width, target_height = target
        scale = max(target_width / width, target_height / height)
        resized_width, resized_height = width * scale, height * scale
        return cls(
            source_box=source_box,
            target_width=target_width,
            target_height=target_height,
            scale=scale,
            offset_x=(target_width - resized_width) / 2.0,
            offset_y=(target_height - resized_height) / 2.0,
            horizontal_flip=horizontal_flip,
        )

    def point(self, x: float, y: float) -> tuple[float, float]:
        left, top, _right, _bottom = self.source_box
        target_x = (x - left) * self.scale + self.offset_x
        target_y = (y - top) * self.scale + self.offset_y
        if self.horizontal_flip:
            target_x = self.target_width - target_x
        return target_x, target_y

    def image(self, image: Image.Image) -> Image.Image:
        left, top, right, bottom = self.source_box
        crop = image.crop((round(left), round(top), round(right), round(bottom)))
        resized = crop.resize(
            (
                round((right - left) * self.scale),
                round((bottom - top) * self.scale),
            ),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (self.target_width, self.target_height))
        canvas.paste(resized, (round(self.offset_x), round(self.offset_y)))
        if self.horizontal_flip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return canvas

    def document(self) -> dict[str, Any]:
        return {
            "source_box": list(self.source_box),
            "target": [self.target_width, self.target_height],
            "scale": self.scale,
            "offset": [self.offset_x, self.offset_y],
            "horizontal_flip": self.horizontal_flip,
        }


def choose_bucket(width: int, height: int, buckets: Iterable[tuple[int, int]] = DEFAULT_BUCKETS):
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ratio = width / height
    return min(buckets, key=lambda bucket: abs(math.log(bucket[0] / bucket[1]) - math.log(ratio)))


def normalized_caption(captions: Iterable[str], *, maximum_characters: int = 900) -> str:
    valid = [
        re.sub(r"\s+", " ", str(caption)).strip()
        for caption in captions
        if str(caption).strip()
    ]
    bounded = [caption for caption in valid if len(caption) <= maximum_characters]
    return max(bounded or valid or [""], key=lambda value: (len(value), value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
