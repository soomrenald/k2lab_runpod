from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from k2core.depth import (
    DepthControlSettings,
    DepthRegion,
    EffectiveDepthField,
    NormalizedDepth,
    compose_effective_depth_field,
    compose_override_depth,
    depth_histogram,
    inspect_depth_checkpoint,
    load_depth_image,
    normalize_depth,
    resize_depth,
)
from k2_region_lab.krea_control_lora import (
    KreaControlError,
    KreaControlLatentBundle,
    _process_control_latent_for_model,
)
from k2_region_lab.regions import PixelBox, RegionDefinition

try:
    import torch
except ModuleNotFoundError:  # Control-plane and documentation environments.
    torch = None


@dataclass(frozen=True, slots=True)
class DepthControlPreparation:
    normalized: NormalizedDepth
    resized_values: np.ndarray
    schedule: "DepthScheduleController"
    source_histogram: Mapping[str, Any]
    normalized_histogram: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    preprocess_seconds: float

    def document(self) -> dict[str, Any]:
        return {
            "source": self.normalized.metadata.get("source"),
            "preprocessing": self.normalized.report.document(),
            "source_histogram": dict(self.source_histogram),
            "normalized_histogram": dict(self.normalized_histogram),
            "schedule": self.schedule.document(),
            "checkpoint": dict(self.checkpoint),
            "preprocess_seconds": self.preprocess_seconds,
        }


class DepthScheduleController:
    """Own one immutable image-token strength field per sampler transition."""

    def __init__(self, fields: tuple[EffectiveDepthField, ...]) -> None:
        if not fields:
            raise ValueError("depth schedule requires at least one field")
        shape = fields[0].image_token_values.shape
        if any(field.image_token_values.shape != shape for field in fields):
            raise ValueError("all depth schedule fields must share one token layout")
        self._fields = fields
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def transition_count(self) -> int:
        return len(self._fields)

    def current_values(self) -> np.ndarray:
        return self._fields[self._step].image_token_values.reshape(-1)

    def advance_after(self, completed_step: int, total_steps: int) -> None:
        if total_steps != len(self._fields):
            raise RuntimeError("depth schedule and sampler step counts diverged")
        if completed_step != self._step:
            raise RuntimeError("depth schedule callback and sampler step diverged")
        self._step = min(completed_step + 1, len(self._fields) - 1)

    def document(self) -> dict[str, Any]:
        return {
            "transition_count": len(self._fields),
            "token_shape": list(self._fields[0].image_token_values.shape),
            "overlap_policy": self._fields[0].overlap_policy,
            "steps": [
                {
                    "minimum": float(field.image_token_values.min()),
                    "maximum": float(field.image_token_values.max()),
                    "mean": float(field.image_token_values.mean()),
                    "region_multipliers": dict(field.region_multipliers),
                }
                for field in self._fields
            ],
        }


def _depth_regions(
    settings: DepthControlSettings,
    regions: tuple[RegionDefinition, ...],
) -> tuple[DepthRegion, ...]:
    geometry = {region.region_id: region for region in regions}
    missing = sorted(
        region.region_id
        for region in settings.regions
        if region.region_id not in geometry
    )
    if missing:
        raise ValueError(
            "depth settings reference regions missing from the generation: "
            + ", ".join(missing)
        )
    return tuple(
        DepthRegion(
            settings=region,
            box=PixelBox(
                geometry[region.region_id].box.x0,
                geometry[region.region_id].box.y0,
                geometry[region.region_id].box.x1,
                geometry[region.region_id].box.y1,
            ),
            priority=geometry[region.region_id].priority,
        )
        for region in settings.regions
    )


def prepare_depth_control(
    settings: DepthControlSettings,
    *,
    regions: tuple[RegionDefinition, ...],
    width: int,
    height: int,
    steps: int,
    allow_override: bool = False,
) -> DepthControlPreparation:
    if not settings.enabled or settings.depth_image is None or settings.checkpoint is None:
        raise ValueError("depth preparation requires enabled depth settings")
    if steps <= 0:
        raise ValueError("depth preparation requires at least one denoising step")
    started = time.monotonic()
    compatibility = inspect_depth_checkpoint(settings.checkpoint)
    if not compatibility.compatible or compatibility.checkpoint is None:
        raise KreaControlError(
            "depth_checkpoint_incompatible",
            "The selected depth adapter is incompatible with Krea 2.",
            private_detail="; ".join(compatibility.errors),
        )
    source = load_depth_image(settings.depth_image)
    normalized = normalize_depth(
        source,
        settings.normalization,
        metadata={"source": source.info.document()},
    )
    resized = resize_depth(normalized.values, width, height)
    configured_regions = _depth_regions(settings, regions)
    override_depths: dict[str, np.ndarray] = {}
    for region in settings.regions:
        if region.override_image is None:
            continue
        override_source = load_depth_image(region.override_image)
        override_normalized = normalize_depth(
            override_source,
            settings.normalization,
        )
        override_depths[region.region_id] = resize_depth(
            override_normalized.values,
            width,
            height,
        )
    if override_depths:
        if not allow_override:
            raise ValueError("override depth mode is disabled by the active feature flags")
        resized = compose_override_depth(
            resized,
            configured_regions,
            override_depths,
            feather_pixels=settings.feather_pixels,
        )
    fields = tuple(
        compose_effective_depth_field(
            settings,
            configured_regions,
            width=width,
            height=height,
            progress=(step / max(steps - 1, 1)),
            allow_override=allow_override,
        )
        for step in range(steps)
    )
    return DepthControlPreparation(
        normalized=normalized,
        resized_values=resized,
        schedule=DepthScheduleController(fields),
        source_histogram=MappingProxyType(depth_histogram(source.values)),
        normalized_histogram=MappingProxyType(depth_histogram(resized)),
        checkpoint=MappingProxyType(compatibility.document()),
        preprocess_seconds=time.monotonic() - started,
    )


def encode_depth_control(
    vae,
    model_patcher,
    preparation: DepthControlPreparation,
) -> KreaControlLatentBundle:
    if torch is None:
        raise KreaControlError(
            "depth_encode_failed",
            "Torch is unavailable for depth-control preprocessing.",
        )
    started = time.monotonic()
    rgb = np.repeat(preparation.resized_values[..., None], 3, axis=-1)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(dtype=torch.float32)
    try:
        latent = vae.encode(image.unsqueeze(0))
    except Exception as error:
        raise KreaControlError(
            "depth_encode_failed",
            "The Krea/Qwen VAE could not encode the depth image.",
            private_detail=f"depth VAE encode failed: {type(error).__name__}: {error}",
        ) from error
    processed = _process_control_latent_for_model(model_patcher, latent)
    digest = hashlib.sha256(preparation.resized_values.tobytes()).hexdigest()
    return KreaControlLatentBundle(
        full=processed,
        subjects=MappingProxyType({}),
        source_hashes=MappingProxyType({"full": digest}),
        encode_seconds=time.monotonic() - started,
    )
