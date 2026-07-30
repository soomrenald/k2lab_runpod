"""Evaluate generated images against a reference depth map.

This command is intentionally separate from inference.  It can run a monocular
depth estimator over already-generated images, so comparative validation does
not change the generation process or its memory measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _parse_image(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("images must use LABEL=PATH")
    return label, Path(path)


def _normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    low, high = np.percentile(array[finite], (1.0, 99.0))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        return np.zeros_like(array)
    result = np.clip((array - low) / (high - low), 0.0, 1.0)
    result[~finite] = 0.0
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    left_flat -= left_flat.mean()
    right_flat -= right_flat.mean()
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_flat, right_flat) / denominator)


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    vertical, horizontal = np.gradient(np.asarray(values, dtype=np.float32))
    return np.hypot(horizontal, vertical)


def _silhouette_iou(left: np.ndarray, right: np.ndarray, quantile: float = 0.9) -> float:
    left_threshold = float(np.quantile(left, quantile))
    right_threshold = float(np.quantile(right, quantile))
    left_edges = left >= left_threshold
    right_edges = right >= right_threshold
    union = np.logical_or(left_edges, right_edges).sum()
    if not union:
        return 0.0
    return float(np.logical_and(left_edges, right_edges).sum() / union)


def _load_reference(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return _normalized(np.asarray(image, dtype=np.float32))


def _estimate_depths(
    images: list[tuple[str, Path]],
    *,
    estimator: str,
    device: str,
    size: tuple[int, int],
) -> dict[str, np.ndarray]:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as error:
        raise RuntimeError(
            "depth evaluation requires the model dependencies: install the 'model' extra"
        ) from error

    processor = AutoImageProcessor.from_pretrained(estimator)
    model = AutoModelForDepthEstimation.from_pretrained(estimator).to(device).eval()
    estimates: dict[str, np.ndarray] = {}
    for label, path in images:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
        prediction = functional.interpolate(
            prediction.unsqueeze(1),
            size=(size[1], size[0]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        estimates[label] = _normalized(prediction.float().cpu().numpy())
    return estimates


def evaluate(
    reference_depth: Path,
    images: list[tuple[str, Path]],
    *,
    estimator: str,
    device: str,
) -> dict[str, Any]:
    reference = _load_reference(reference_depth)
    height, width = reference.shape
    reference_edges = _gradient_magnitude(reference)
    estimates = _estimate_depths(
        images,
        estimator=estimator,
        device=device,
        size=(width, height),
    )

    metrics: dict[str, dict[str, float]] = {}
    for label, estimate in estimates.items():
        estimate_edges = _gradient_magnitude(estimate)
        direct = _correlation(reference, estimate)
        inverted = _correlation(reference, 1.0 - estimate)
        direct_rank = _correlation(_rank(reference), _rank(estimate))
        inverted_rank = _correlation(_rank(reference), _rank(1.0 - estimate))
        metrics[label] = {
            "depth_correlation": max(direct, inverted),
            "depth_rank_correlation": max(direct_rank, inverted_rank),
            "edge_alignment": _correlation(reference_edges, estimate_edges),
            "silhouette_edge_iou": _silhouette_iou(reference_edges, estimate_edges),
            "selected_polarity": "direct" if direct >= inverted else "inverted",
        }

    return {
        "version": 1,
        "reference_depth": str(reference_depth),
        "estimator": estimator,
        "device": device,
        "reference_size": [width, height],
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-depth", type=Path, required=True)
    parser.add_argument(
        "--image",
        action="append",
        type=_parse_image,
        required=True,
        help="generated image as LABEL=PATH; repeat for every comparison",
    )
    parser.add_argument(
        "--estimator",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face monocular depth-estimation model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = evaluate(
        arguments.reference_depth,
        arguments.image,
        estimator=arguments.estimator,
        device=arguments.device,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
