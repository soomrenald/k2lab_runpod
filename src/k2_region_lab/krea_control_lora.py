from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from safetensors import safe_open

from k2_region_lab.semantic_conditioning import CURRENT_CONDITIONING_CONTEXT
from k2_region_lab.volumetric_control import (
    K2_VOLUMETRIC_CONTROL_FORMAT,
    K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
    KreaVolumetricControlBundle,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # Lightweight control-plane and training-data environments.
    torch = None
    nn = None
    functional = None


CONTROL_ATTACHMENT_KEY = "k2_krea_volumetric_control_lora"
CONTROL_LATENT_MAPPING_KEY = "k2_krea_volumetric_control_latents"
CONTROL_TOKEN_STRENGTH_KEY = "k2_krea_control_token_strength"
EXPECTED_BLOCKS = 28
EXPECTED_TARGETS = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.wo",
    "attn.gate",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)
REQUIRED_METADATA = (
    "k2lab_adapter_kind",
    "k2lab_adapter_version",
    "k2lab_control_format",
    "k2lab_control_format_sha256",
    "k2lab_renderer_version",
    "k2lab_base_model",
    "k2lab_inference_targets",
    "k2lab_rank",
    "k2lab_expanded_input_projection",
    "k2lab_expected_transformer_blocks",
    "k2lab_control_channel_mode",
    "k2lab_control_normalize",
    "k2lab_control_invert",
    "k2lab_dataset_manifest_sha256",
    "k2lab_trainer_repository",
    "k2lab_trainer_commit",
    "k2lab_training_commit",
    "k2lab_created_at",
)


class KreaControlError(RuntimeError):
    def __init__(self, code: str, safe_message: str, *, private_detail: str = "") -> None:
        super().__init__(private_detail or safe_message)
        self.code = code
        self.safe_message = safe_message
        self.private_detail = private_detail or safe_message


@dataclass(frozen=True, slots=True)
class KreaControlCheckpointInfo:
    path: Path
    sha256: str
    metadata: Mapping[str, str]
    rank: int
    expanded_projection_key: str
    compatible_block_pairs: int
    format_id: str
    verified: bool

    def document(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
            "rank": self.rank,
            "expanded_projection_key": self.expanded_projection_key,
            "compatible_block_pairs": self.compatible_block_pairs,
            "format_id": self.format_id,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class KreaControlCompatibilityReport:
    compatible: bool
    verified: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checkpoint: KreaControlCheckpointInfo | None

    def document(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "verified": self.verified,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checkpoint": self.checkpoint.document() if self.checkpoint else None,
        }


@dataclass(frozen=True, slots=True)
class KreaControlLatentBundle:
    full: Any
    subjects: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    encode_seconds: float


@dataclass(frozen=True, slots=True)
class KreaControlRuntimeReport:
    checkpoint_sha256: str
    format_id: str
    strength: float
    loaded_lora_keys: int
    patched_model_keys: int
    scope_calls: Mapping[str, int]
    control_latent_shapes: Mapping[str, tuple[int, ...]]

    def document(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "format_id": self.format_id,
            "strength": self.strength,
            "loaded_lora_keys": self.loaded_lora_keys,
            "patched_model_keys": self.patched_model_keys,
            "scope_calls": dict(self.scope_calls),
            "control_latent_shapes": {
                key: list(shape) for key, shape in self.control_latent_shapes.items()
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_prefixes(base: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in (
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "model.",
        ):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                changed = True
    return base


def _lora_pairs(keys: tuple[str, ...] | list[str]) -> tuple[tuple[str, str, str], ...]:
    available = set(keys)
    suffixes = (
        (".A", ".B"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_down", ".lora_up"),
        ("_lora.down.weight", "_lora.up.weight"),
    )
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for down_suffix, up_suffix in suffixes:
        for down_key in keys:
            if not down_key.endswith(down_suffix):
                continue
            base = down_key[: -len(down_suffix)]
            up_key = base + up_suffix
            identity = (down_key, up_key)
            if up_key in available and identity not in seen:
                seen.add(identity)
                pairs.append((base, down_key, up_key))
    return tuple(pairs)


def _block_target(base: str) -> tuple[int, str] | None:
    normalized = _strip_prefixes(base)
    if not normalized.startswith("blocks."):
        return None
    parts = normalized.split(".")
    if len(parts) < 4 or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def inspect_krea_control_checkpoint(
    path: Path,
    *,
    allow_unverified_legacy: bool = False,
) -> KreaControlCompatibilityReport:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.casefold() != ".safetensors":
        return KreaControlCompatibilityReport(
            compatible=False,
            verified=False,
            errors=("checkpoint must be a readable .safetensors file",),
            warnings=(),
            checkpoint=None,
        )
    try:
        with safe_open(resolved, framework="numpy") as handle:
            metadata = {str(key): str(value) for key, value in (handle.metadata() or {}).items()}
            keys = tuple(handle.keys())
            shapes = {key: tuple(int(value) for value in handle.get_slice(key).get_shape()) for key in keys}
    except Exception as error:
        return KreaControlCompatibilityReport(
            compatible=False,
            verified=False,
            errors=(f"invalid safetensors checkpoint: {type(error).__name__}",),
            warnings=(),
            checkpoint=None,
        )

    missing_metadata = tuple(key for key in REQUIRED_METADATA if key not in metadata)
    verified = not missing_metadata
    errors: list[str] = []
    warnings: list[str] = []
    if missing_metadata:
        if allow_unverified_legacy:
            warnings.append("Unverified legacy Krea control checkpoint")
        else:
            errors.append("required K2Lab checkpoint metadata is missing")
    elif metadata.get("k2lab_adapter_kind") != "krea2_control_lora":
        errors.append("checkpoint is not a Krea Control-LoRA")

    format_id = metadata.get("k2lab_control_format", "")
    if verified and format_id != K2_VOLUMETRIC_CONTROL_FORMAT:
        errors.append("control format does not match this K2Lab renderer")
    if (
        verified
        and metadata.get("k2lab_control_format_sha256")
        != K2_VOLUMETRIC_CONTROL_FORMAT_SHA256
    ):
        errors.append("renderer/palette hash does not match this K2Lab build")
    if verified and metadata.get("k2lab_base_model") != "krea/Krea-2-Raw":
        errors.append("checkpoint was not trained against Krea-2-Raw")
    if verified and metadata.get("k2lab_expanded_input_projection") != "true":
        errors.append("checkpoint does not declare an expanded input projection")
    if verified and metadata.get("k2lab_expected_transformer_blocks") != str(EXPECTED_BLOCKS):
        errors.append("checkpoint transformer block count is incompatible")
    if verified and metadata.get("k2lab_control_channel_mode") != "rgb":
        errors.append("checkpoint does not use RGB control channels")
    if verified and metadata.get("k2lab_control_normalize") != "none":
        errors.append("checkpoint uses an incompatible control normalization")
    if verified and metadata.get("k2lab_control_invert") != "false":
        errors.append("checkpoint uses an incompatible control inversion")

    projection_candidates = [
        key
        for key, shape in shapes.items()
        if (
            len(shape) == 2
            and shape[1] == 128
            and (key.endswith("first.weight") or key.endswith("img_in.weight"))
        )
    ]
    if len(projection_candidates) != 1:
        errors.append("exactly one doubled-width Krea input projection is required")
    projection_key = projection_candidates[0] if len(projection_candidates) == 1 else ""

    rank_values: set[int] = set()
    found_targets: set[tuple[int, str]] = set()
    for base, down_key, up_key in _lora_pairs(list(keys)):
        target = _block_target(base)
        if target is None:
            continue
        down_shape, up_shape = shapes[down_key], shapes[up_key]
        if len(down_shape) != 2 or len(up_shape) != 2:
            continue
        rank = down_shape[0]
        if up_shape[1] != rank:
            continue
        rank_values.add(rank)
        found_targets.add(target)
    declared_rank = int(metadata.get("k2lab_rank", next(iter(rank_values), 0)) or 0)
    if not rank_values or rank_values != {declared_rank}:
        errors.append("LoRA tensor rank does not match the declared rank")
    required_targets = {
        (block, target)
        for block in range(EXPECTED_BLOCKS)
        for target in EXPECTED_TARGETS
    }
    missing_targets = required_targets - found_targets
    if missing_targets:
        errors.append(
            f"checkpoint is missing {len(missing_targets)} required Krea block LoRA pairs"
        )

    info = KreaControlCheckpointInfo(
        path=resolved,
        sha256=_sha256_file(resolved),
        metadata=MappingProxyType(metadata),
        rank=declared_rank,
        expanded_projection_key=projection_key,
        compatible_block_pairs=len(found_targets),
        format_id=format_id or K2_VOLUMETRIC_CONTROL_FORMAT,
        verified=verified,
    )
    return KreaControlCompatibilityReport(
        compatible=not errors,
        verified=verified,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checkpoint=info,
    )


if nn is not None:

    class KreaControlInputProjection(nn.Module):
        def __init__(
            self,
            weight,
            *,
            image_features: int,
            original_first: Any,
            strength: float,
        ) -> None:
            super().__init__()
            if weight.ndim != 2 or weight.shape[1] != image_features * 2:
                raise KreaControlError(
                    "krea_control_projection_missing",
                    "The selected pose adapter has an incompatible input projection.",
                )
            self.image_features = int(image_features)
            self.control_features = int(image_features)
            self.out_features = int(weight.shape[0])
            self.strength = float(strength)
            self.weight = nn.Parameter(weight.detach().cpu().clone(), requires_grad=False)
            self.control_tokens = None
            self.token_strength = None
            object.__setattr__(self, "_original_first", original_first)

        @property
        def original_first(self):
            return object.__getattribute__(self, "_original_first")

        def set_original_first(self, value: Any) -> None:
            object.__setattr__(self, "_original_first", value)

        def forward(self, image_tokens):
            if image_tokens.shape[-1] != self.image_features:
                raise KreaControlError(
                    "krea_control_latent_shape_invalid",
                    "The pose adapter received an incompatible Krea latent shape.",
                )
            if self.control_tokens is None:
                raise KreaControlError(
                    "krea_control_scope_missing",
                    "The pose adapter has no control for the active conditioning scope.",
                )
            control = self.control_tokens.to(
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )
            if control.shape[1] != image_tokens.shape[1]:
                raise KreaControlError(
                    "krea_control_latent_shape_invalid",
                    "The pose control token count does not match the Krea image tokens.",
                )
            import comfy.model_management
            import comfy.utils

            control = comfy.utils.repeat_to_batch_size(control, image_tokens.shape[0])
            control_weight = comfy.model_management.cast_to_device(
                self.weight[:, self.image_features :],
                image_tokens.device,
                image_tokens.dtype,
            )
            contribution = functional.linear(control, control_weight, None)
            if self.token_strength is not None:
                token_strength = self.token_strength.to(
                    device=image_tokens.device,
                    dtype=image_tokens.dtype,
                )
                if token_strength.ndim == 1:
                    token_strength = token_strength.view(1, -1, 1)
                elif token_strength.ndim == 2:
                    token_strength = token_strength.unsqueeze(-1)
                if token_strength.ndim != 3 or token_strength.shape[1] != image_tokens.shape[1]:
                    raise KreaControlError(
                        "krea_control_latent_shape_invalid",
                        "The control-strength field does not match the Krea image tokens.",
                    )
                contribution = contribution * token_strength
            return self.original_first(image_tokens) + self.strength * contribution

else:

    class KreaControlInputProjection:  # pragma: no cover - only a helpful lightweight error.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Torch is required for Krea Control-LoRA inference")


def select_control_latent(mapping: Mapping[str, Any]) -> tuple[str, Any]:
    context = CURRENT_CONDITIONING_CONTEXT.get()
    region_id = None if context is None else context.scope.region_id
    if region_id is None:
        try:
            return "full", mapping["full"]
        except KeyError as error:
            raise KreaControlError(
                "krea_control_scope_missing",
                "The pose adapter has no full-canvas control latent.",
            ) from error
    subjects = mapping.get("subjects")
    if not isinstance(subjects, Mapping) or region_id not in subjects:
        raise KreaControlError(
            "krea_control_scope_missing",
            "The pose adapter has no control latent for an active subject.",
            private_detail=f"missing Krea control latent for subject scope {region_id!r}",
        )
    return f"subject:{region_id}", subjects[region_id]


def _process_control_latent_for_model(model_patcher, latent):
    if torch is None or not torch.is_tensor(latent) or latent.ndim not in (4, 5):
        raise KreaControlError(
            "krea_control_latent_shape_invalid",
            "The pose adapter control latent has an invalid shape.",
        )
    try:
        latent_format = model_patcher.get_model_object("latent_format")
    except Exception as error:
        raise KreaControlError(
            "krea_control_hook_incompatible",
            "The selected model does not expose native Krea latent formatting.",
        ) from error
    expected = getattr(latent_format, "latent_channels", None)
    if expected is not None and int(latent.shape[1]) != int(expected):
        raise KreaControlError(
            "krea_control_vae_incompatible",
            "Select the Krea/Qwen image VAE for the pose adapter.",
        )
    added_time = getattr(latent_format, "latent_dimensions", 2) == 3 and latent.ndim == 4
    processed = latent.unsqueeze(2) if added_time else latent
    processed = model_patcher.model.process_latent_in(processed)
    if added_time and processed.ndim == 5 and processed.shape[2] == 1:
        processed = processed[:, :, 0]
    return processed


def encode_control_bundle(
    vae,
    model_patcher,
    controls: KreaVolumetricControlBundle,
) -> KreaControlLatentBundle:
    if torch is None:
        raise KreaControlError(
            "krea_control_encode_failed",
            "Torch is unavailable for pose-adapter preprocessing.",
        )
    started = time.monotonic()

    def encode(image):
        tensor = torch.from_numpy(image.rgb.copy()).to(dtype=torch.float32).div_(255.0)
        tensor = tensor.unsqueeze(0)
        try:
            latent = vae.encode(tensor)
        except Exception as error:
            raise KreaControlError(
                "krea_control_encode_failed",
                "The Krea/Qwen VAE could not encode the pose control image.",
                private_detail=f"control VAE encode failed: {type(error).__name__}: {error}",
            ) from error
        return _process_control_latent_for_model(model_patcher, latent)

    full = encode(controls.full)
    subjects = {region_id: encode(image) for region_id, image in controls.subjects.items()}
    hashes = {"full": controls.full.sha256}
    hashes.update(
        {f"subject:{region_id}": image.sha256 for region_id, image in controls.subjects.items()}
    )
    return KreaControlLatentBundle(
        full=full,
        subjects=MappingProxyType(subjects),
        source_hashes=MappingProxyType(hashes),
        encode_seconds=time.monotonic() - started,
    )


def _shape_from_weight(weight) -> tuple[int, ...] | None:
    tensor_shape = getattr(weight, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(int(value) for value in tensor_shape)
    data = getattr(weight, "data", None)
    tensor_shape = getattr(data, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(int(value) for value in tensor_shape)
    shape = getattr(weight, "shape", None)
    return tuple(int(value) for value in shape) if shape is not None else None


def _nested_attr(root, path: str):
    current = root
    for component in path.split("."):
        if component.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


def _target_key(base: str) -> str | None:
    normalized = _strip_prefixes(base)
    return f"diffusion_model.{normalized}.weight" if normalized.startswith("blocks.") else None


def _model_target_shape(model_patcher, target: str) -> tuple[int, ...] | None:
    try:
        return _shape_from_weight(_nested_attr(model_patcher.model, target))
    except Exception:
        value = model_patcher.model.state_dict().get(target)
        return _shape_from_weight(value) if value is not None else None


def _runtime_lora_patches(state_dict: Mapping[str, Any], model_patcher):
    from comfy.weight_adapter.lora import LoRAAdapter

    patches: dict[str, Any] = {}
    loaded: set[str] = set()
    skipped: dict[str, str] = {}
    for base, down_key, up_key in _lora_pairs(list(state_dict)):
        target = _target_key(base)
        if target is None:
            continue
        shape = _model_target_shape(model_patcher, target)
        if shape is None or len(shape) < 2:
            skipped[down_key] = f"target {target!r} is unavailable"
            continue
        down, up = state_dict[down_key], state_dict[up_key]
        if (
            not torch.is_tensor(down)
            or not torch.is_tensor(up)
            or down.ndim != 2
            or up.ndim != 2
        ):
            skipped[down_key] = "adapter tensors are not 2D"
            continue
        out_features, in_features = shape[:2]
        if (
            up.shape[0] == out_features
            and down.shape[1] == in_features
            and up.shape[1] == down.shape[0]
        ):
            rank = int(down.shape[0])
        elif (
            down.shape[0] == in_features
            and up.shape[1] == out_features
            and down.shape[1] == up.shape[0]
        ):
            down = down.t().contiguous()
            up = up.t().contiguous()
            rank = int(down.shape[0])
        else:
            skipped[down_key] = f"adapter does not match live target shape {shape}"
            continue
        alpha = rank
        alpha_key = None
        for suffix in (".alpha", ".network_alpha", ".scale"):
            candidate = base + suffix
            if candidate in state_dict:
                alpha_key = candidate
                value = state_dict[candidate]
                alpha = float(value.detach().cpu().reshape(-1)[0])
                break
        keys = {down_key, up_key}
        if alpha_key is not None:
            keys.add(alpha_key)
        patches[target] = LoRAAdapter(keys, (up, down, alpha, None, None, None))
        loaded.update(keys)
    return patches, loaded, skipped


def _first_module(model_patcher):
    try:
        return model_patcher.get_model_object("diffusion_model.first")
    except Exception as error:
        raise KreaControlError(
            "krea_control_hook_incompatible",
            "Select a native ComfyUI Krea 2 model for the pose adapter.",
        ) from error


def _runtime_projection(
    model_patcher,
    state_dict: Mapping[str, Any],
    info: KreaControlCheckpointInfo,
    strength: float,
) -> KreaControlInputProjection:
    first = _first_module(model_patcher)
    shape = _shape_from_weight(getattr(first, "weight", None))
    if shape is None or len(shape) != 2:
        raise KreaControlError(
            "krea_control_projection_missing",
            "The selected Krea model has no compatible native input projection.",
        )
    weight = state_dict.get(info.expanded_projection_key)
    if (
        not torch.is_tensor(weight)
        or weight.ndim != 2
        or tuple(weight.shape) != (shape[0], shape[1] * 2)
    ):
        raise KreaControlError(
            "krea_control_checkpoint_incompatible",
            "The pose adapter input projection does not match the selected Krea model.",
            private_detail=(
                f"checkpoint projection {getattr(weight, 'shape', None)} does not match "
                f"live projection {shape}"
            ),
        )
    return KreaControlInputProjection(
        weight,
        image_features=shape[1],
        original_first=first,
        strength=strength,
    )


def _flatten_control_latent(latent):
    if latent.ndim == 4:
        return latent
    if latent.ndim == 5:
        batch, channels, frames, height, width = latent.shape
        return latent.reshape(batch * frames, channels, height, width)
    raise KreaControlError(
        "krea_control_latent_shape_invalid",
        "The pose adapter control latent has an invalid shape.",
    )


def _control_tokens(latent, x, patch: int, expected_features: int):
    import comfy.ldm.common_dit
    import comfy.model_management
    import comfy.utils

    target_batch = int(x.shape[0] * x.shape[2]) if x.ndim == 5 else int(x.shape[0])
    control = comfy.utils.repeat_to_batch_size(
        _flatten_control_latent(latent),
        target_batch,
    )
    control = comfy.model_management.cast_to_device(control, x.device, x.dtype)
    target_height, target_width = int(x.shape[-2]), int(x.shape[-1])
    if tuple(control.shape[-2:]) != (target_height, target_width):
        control = comfy.utils.common_upscale(
            control,
            target_width,
            target_height,
            "bilinear",
            "disabled",
        )
    control = comfy.ldm.common_dit.pad_to_patch_size(control, (patch, patch))
    batch, channels, height, width = control.shape
    features = channels * patch * patch
    if features != expected_features or height % patch or width % patch:
        raise KreaControlError(
            "krea_control_latent_shape_invalid",
            "The pose control latent does not match the Krea image-token layout.",
        )
    return (
        control.reshape(
            batch,
            channels,
            height // patch,
            patch,
            width // patch,
            patch,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, (height // patch) * (width // patch), features)
    )


def _transformer_options(args, kwargs) -> dict[str, Any] | None:
    candidate = kwargs.get("transformer_options")
    if isinstance(candidate, dict):
        return candidate
    if len(args) >= 5 and isinstance(args[4], dict):
        return args[4]
    if args and isinstance(args[-1], dict):
        return args[-1]
    return None


def _restore_projection(diffusion_model, projection: KreaControlInputProjection) -> None:
    projection.control_tokens = None
    projection.token_strength = None
    if getattr(diffusion_model, "first", None) is projection:
        diffusion_model.first = projection.original_first


def _control_wrapper(projection: KreaControlInputProjection, scope_calls: dict[str, int]):
    def wrapper(executor, *args, **kwargs):
        options = _transformer_options(args, kwargs)
        if options is None:
            raise KreaControlError(
                "krea_control_hook_incompatible",
                "The current sampler cannot provide Krea pose-adapter options.",
            )
        mapping = options.get(CONTROL_LATENT_MAPPING_KEY)
        if not isinstance(mapping, Mapping):
            raise KreaControlError(
                "krea_control_scope_missing",
                "The pose adapter control latents were not attached to this generation.",
            )
        scope_name, latent = select_control_latent(mapping)
        scope_calls[scope_name] = scope_calls.get(scope_name, 0) + 1
        diffusion_model = executor.class_obj
        previous_first = getattr(diffusion_model, "first", None)
        previous_tokens = projection.control_tokens
        previous_strength = projection.token_strength
        try:
            projection.control_tokens = _control_tokens(
                latent,
                args[0],
                int(diffusion_model.patch),
                projection.control_features,
            )
            token_strength = options.get(CONTROL_TOKEN_STRENGTH_KEY)
            if callable(getattr(token_strength, "current_values", None)):
                token_strength = token_strength.current_values()
            if token_strength is not None:
                projection.token_strength = (
                    token_strength
                    if torch.is_tensor(token_strength)
                    else torch.tensor(token_strength)
                )
            diffusion_model.first = projection
            return executor(*args, **kwargs)
        finally:
            projection.control_tokens = previous_tokens
            projection.token_strength = previous_strength
            if getattr(diffusion_model, "first", None) is projection:
                diffusion_model.first = projection.original_first or previous_first

    return wrapper


def _projection_injections(projection: KreaControlInputProjection):
    import comfy.patcher_extension

    def inject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is None:
            return
        current = getattr(diffusion_model, "first", None)
        if current is not None and current is not projection:
            projection.set_original_first(current)
        diffusion_model.first = projection.original_first
        projection.control_tokens = None
        projection.token_strength = None

    def eject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is not None:
            _restore_projection(diffusion_model, projection)

    return [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)]


def _projection_cleanup(model_patcher, *_args) -> None:
    attachment = model_patcher.get_attachment(CONTROL_ATTACHMENT_KEY)
    if not isinstance(attachment, Mapping):
        return
    projection = attachment.get("control_projection")
    diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
    if isinstance(projection, KreaControlInputProjection) and diffusion_model is not None:
        _restore_projection(diffusion_model, projection)


def install_krea_control_lora(
    model_patcher,
    checkpoint_path: Path,
    *,
    strength: float,
    allow_unverified_legacy: bool = False,
    expected_sha256: str | None = None,
    adapter_format_id: str | None = None,
):
    if torch is None:
        raise KreaControlError(
            "krea_control_hook_incompatible",
            "Torch is unavailable for Krea pose-adapter inference.",
        )
    if not 0.0 <= strength <= 2.0:
        raise ValueError("Krea pose-adapter strength must be between 0 and 2")
    compatibility = inspect_krea_control_checkpoint(
        checkpoint_path,
        allow_unverified_legacy=allow_unverified_legacy,
    )
    if not compatibility.compatible or compatibility.checkpoint is None:
        raise KreaControlError(
            "krea_control_checkpoint_incompatible",
            "The selected Krea pose adapter is incompatible with this runtime.",
            private_detail="; ".join(compatibility.errors),
        )
    if (
        expected_sha256 is not None
        and compatibility.checkpoint.sha256.casefold() != expected_sha256.casefold()
    ):
        raise KreaControlError(
            "krea_control_checkpoint_incompatible",
            "The selected Krea control adapter does not match the trusted artifact.",
        )
    if expected_sha256 is not None or adapter_format_id is not None:
        checkpoint = replace(
            compatibility.checkpoint,
            format_id=adapter_format_id or compatibility.checkpoint.format_id,
            verified=expected_sha256 is not None,
        )
        compatibility = replace(
            compatibility,
            verified=checkpoint.verified,
            warnings=(
                tuple(
                    warning
                    for warning in compatibility.warnings
                    if warning != "Unverified legacy Krea control checkpoint"
                )
                if checkpoint.verified
                else compatibility.warnings
            ),
            checkpoint=checkpoint,
        )
    if model_patcher.get_attachment(CONTROL_ATTACHMENT_KEY) is not None:
        raise KreaControlError(
            "krea_control_lora_target_conflict",
            "Only one Krea pose adapter can be active in a generation.",
        )
    import comfy.patcher_extension
    import comfy.utils

    try:
        state_dict = comfy.utils.load_torch_file(str(checkpoint_path), safe_load=True)
    except Exception as error:
        raise KreaControlError(
            "krea_control_checkpoint_invalid",
            "The selected Krea pose adapter could not be read safely.",
        ) from error
    generation_model = model_patcher.clone()
    projection = _runtime_projection(
        generation_model,
        state_dict,
        compatibility.checkpoint,
        strength,
    )
    patches, loaded_keys, skipped = _runtime_lora_patches(state_dict, generation_model)
    if not patches:
        raise KreaControlError(
            "krea_control_block_weights_missing",
            "The selected pose adapter has no compatible Krea block weights.",
            private_detail=f"skipped Control-LoRA targets: {skipped}",
        )
    patched_keys = generation_model.add_patches(
        patches,
        strength_patch=strength,
        strength_model=1.0,
    )
    if not patched_keys:
        raise KreaControlError(
            "krea_control_block_weights_missing",
            "The selected Krea model rejected every pose-adapter block weight.",
        )
    scope_calls: dict[str, int] = {}
    generation_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        CONTROL_ATTACHMENT_KEY,
        _control_wrapper(projection, scope_calls),
    )
    generation_model.set_injections(
        CONTROL_ATTACHMENT_KEY,
        _projection_injections(projection),
    )
    generation_model.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_DETACH,
        CONTROL_ATTACHMENT_KEY,
        _projection_cleanup,
    )
    generation_model.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
        CONTROL_ATTACHMENT_KEY,
        _projection_cleanup,
    )
    generation_model.set_attachments(
        CONTROL_ATTACHMENT_KEY,
        {
            "checkpoint": compatibility.checkpoint,
            "strength": strength,
            "loaded_lora_keys": len(loaded_keys),
            "patched_model_keys": len(patched_keys),
            "skipped_targets": skipped,
            "control_projection": projection,
            "scope_calls": scope_calls,
        },
    )
    return generation_model, compatibility


def attach_krea_control_latents(
    model_patcher,
    bundle: KreaControlLatentBundle,
    *,
    token_strength: Any | None = None,
):
    if model_patcher.get_attachment(CONTROL_ATTACHMENT_KEY) is None:
        raise KreaControlError(
            "krea_control_hook_incompatible",
            "The Krea pose adapter must be installed before control latents are attached.",
        )
    generation_model = model_patcher.clone()
    options = generation_model.model_options.setdefault("transformer_options", {})
    options[CONTROL_LATENT_MAPPING_KEY] = {
        "full": bundle.full,
        "subjects": dict(bundle.subjects),
    }
    if token_strength is not None:
        options[CONTROL_TOKEN_STRENGTH_KEY] = token_strength
    return generation_model


def krea_control_runtime_report(model_patcher) -> KreaControlRuntimeReport:
    attachment = model_patcher.get_attachment(CONTROL_ATTACHMENT_KEY)
    if not isinstance(attachment, Mapping):
        raise KreaControlError(
            "krea_control_hook_incompatible",
            "The generation model has no active Krea pose adapter.",
        )
    checkpoint = attachment["checkpoint"]
    options = model_patcher.model_options.get("transformer_options", {})
    mapping = options.get(CONTROL_LATENT_MAPPING_KEY, {})
    shapes: dict[str, tuple[int, ...]] = {}
    if isinstance(mapping, Mapping) and "full" in mapping:
        shapes["full"] = tuple(int(value) for value in mapping["full"].shape)
        for region_id, latent in mapping.get("subjects", {}).items():
            shapes[f"subject:{region_id}"] = tuple(int(value) for value in latent.shape)
    return KreaControlRuntimeReport(
        checkpoint_sha256=checkpoint.sha256,
        format_id=checkpoint.format_id,
        strength=float(attachment["strength"]),
        loaded_lora_keys=int(attachment["loaded_lora_keys"]),
        patched_model_keys=int(attachment["patched_model_keys"]),
        scope_calls=MappingProxyType(dict(attachment["scope_calls"])),
        control_latent_shapes=MappingProxyType(shapes),
    )
