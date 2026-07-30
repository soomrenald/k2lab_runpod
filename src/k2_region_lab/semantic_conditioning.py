from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


SEMANTIC_RUNTIME_VERSION = 1
SEMANTIC_ATTENTION_PENALTY = 20.0
SUBJECT_ISLAND_THRESHOLD = 0.50


class PoseSemanticError(RuntimeError):
    code = "pose_semantic_failed"


class SubjectConditioningCompileError(PoseSemanticError):
    code = "subject_conditioning_compile_failed"


class ConditioningScopeMismatchError(PoseSemanticError):
    code = "conditioning_scope_mismatch"


class SemanticSamplerHookError(PoseSemanticError):
    code = "semantic_sampler_hook_incompatible"


class SemanticPredictionShapeError(PoseSemanticError):
    code = "semantic_prediction_shape_invalid"


class SemanticAttentionError(PoseSemanticError):
    code = "semantic_attention_failed"


class SemanticLoraRoutingError(PoseSemanticError):
    code = "semantic_lora_routing_failed"


class SemanticMultiGpuUnsupportedError(PoseSemanticError):
    code = "semantic_prediction_composite_multigpu_unsupported"


class PoseSemanticMode(StrEnum):
    SPATIAL_ONLY = "spatial_only"
    ATTENTION_ISOLATION = "attention_isolation"
    PREDICTION_COMPOSITE = "prediction_composite"


class ConditioningScopeKind(StrEnum):
    FULL = "full"
    SUBJECT = "subject"


@dataclass(frozen=True, slots=True)
class ConditioningScope:
    kind: ConditioningScopeKind
    region_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == ConditioningScopeKind.FULL and self.region_id is not None:
            raise ValueError("full conditioning scope cannot have a region ID")
        if self.kind == ConditioningScopeKind.SUBJECT and not self.region_id:
            raise ValueError("subject conditioning scope requires a region ID")

    @classmethod
    def full(cls) -> ConditioningScope:
        return cls(ConditioningScopeKind.FULL)

    @classmethod
    def subject(cls, region_id: str) -> ConditioningScope:
        return cls(ConditioningScopeKind.SUBJECT, region_id)

    @property
    def cache_key(self) -> tuple[str, str | None]:
        return self.kind.value, self.region_id


@dataclass(frozen=True, slots=True)
class ConditioningExecutionContext:
    scope: ConditioningScope
    text_token_count: int

    def __post_init__(self) -> None:
        if self.text_token_count <= 0:
            raise ValueError("conditioning execution token count must be positive")


CURRENT_CONDITIONING_CONTEXT: ContextVar[ConditioningExecutionContext | None] = (
    ContextVar("k2_current_conditioning_context", default=None)
)


@contextmanager
def conditioning_execution_scope(
    scope: ConditioningScope,
    text_token_count: int,
) -> Iterator[ConditioningExecutionContext]:
    context = ConditioningExecutionContext(scope, text_token_count)
    token = CURRENT_CONDITIONING_CONTEXT.set(context)
    try:
        yield context
    finally:
        CURRENT_CONDITIONING_CONTEXT.reset(token)


@dataclass(frozen=True, slots=True)
class BoundSubjectPrompt:
    region_id: str
    region_name: str
    prompt: str
    text_token_count: int
    shared_visual_span: tuple[int, int] | None
    subject_span: tuple[int, int]
    face_identity_span: tuple[int, int] | None
    character_trigger_spans: Mapping[str, tuple[tuple[int, int], ...]]
    emphasis_spans: tuple[Any, ...]
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class SubjectSemanticConditioning:
    scope: ConditioningScope
    bound_prompt: BoundSubjectPrompt
    conditioning: tuple[Any, ...]
    ownership_mask: object
    ownership_coverage: float


@dataclass(frozen=True, slots=True)
class PoseSemanticPlan:
    mode: PoseSemanticMode
    full_scope: ConditioningScope
    full_conditioning: tuple[Any, ...]
    subjects: tuple[SubjectSemanticConditioning, ...]
    shared_visual_prompt_sha256: str
    estimated_forwards_per_gated_evaluation: int

    def __post_init__(self) -> None:
        region_ids = [subject.scope.region_id for subject in self.subjects]
        if len(region_ids) != len(set(region_ids)):
            raise ConditioningScopeMismatchError(
                "semantic subject conditioning region IDs must be unique"
            )
        if self.full_scope.kind != ConditioningScopeKind.FULL:
            raise ConditioningScopeMismatchError("semantic plan full scope is invalid")
        expected = 1 + len(self.subjects)
        if self.estimated_forwards_per_gated_evaluation != expected:
            raise ValueError(
                "estimated gated forwards must equal one full plus every subject"
            )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def annotate_conditioning(
    conditioning: Sequence[Sequence[Any]],
    *,
    scope: ConditioningScope,
    prompt_hash: str,
    text_token_count: int,
) -> tuple[list[Any], ...]:
    """Copy Comfy conditioning records and attach stable K2 scope metadata."""
    if text_token_count <= 0:
        raise ConditioningScopeMismatchError(
            "conditioning metadata requires a positive token count"
        )
    annotated: list[list[Any]] = []
    for record in conditioning:
        if len(record) < 2 or not isinstance(record[1], Mapping):
            raise ConditioningScopeMismatchError(
                "Krea conditioning record has an unsupported shape"
            )
        metadata = dict(record[1])
        metadata.update(
            {
                "k2_conditioning_scope": scope.kind.value,
                "k2_conditioning_region_id": scope.region_id,
                "k2_conditioning_prompt_sha256": prompt_hash,
                "k2_conditioning_text_token_count": text_token_count,
            }
        )
        annotated.append([record[0], metadata, *record[2:]])
    if not annotated:
        raise ConditioningScopeMismatchError("conditioning scope has no records")
    return tuple(annotated)


@dataclass(frozen=True, slots=True)
class ConditioningRecordGroup:
    scope: ConditioningScope
    text_token_count: int
    prompt_sha256: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ConditioningGroups:
    full: ConditioningRecordGroup
    subjects: Mapping[str, ConditioningRecordGroup]

    def require_subject(self, region_id: str) -> ConditioningRecordGroup:
        try:
            return self.subjects[region_id]
        except KeyError as error:
            raise ConditioningScopeMismatchError(
                f"missing subject conditioning scope for region {region_id!r}"
            ) from error


def group_conditioning_by_scope(
    records: Sequence[Mapping[str, Any]],
) -> ConditioningGroups:
    grouped: dict[ConditioningScope, list[dict[str, Any]]] = {}
    details: dict[ConditioningScope, tuple[int, str]] = {}
    for source in records:
        record = dict(source)
        try:
            kind = ConditioningScopeKind(str(record["k2_conditioning_scope"]))
            region_id_value = record.get("k2_conditioning_region_id")
            region_id = str(region_id_value) if region_id_value is not None else None
            scope = ConditioningScope(kind, region_id)
            token_count = int(record["k2_conditioning_text_token_count"])
            digest = str(record["k2_conditioning_prompt_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ConditioningScopeMismatchError(
                "every positive conditioning record must have one valid K2 scope"
            ) from error
        if token_count <= 0 or len(digest) != 64:
            raise ConditioningScopeMismatchError(
                "conditioning scope metadata is incomplete"
            )
        previous = details.setdefault(scope, (token_count, digest))
        if previous != (token_count, digest):
            raise ConditioningScopeMismatchError(
                f"conditioning records disagree within scope {scope.cache_key!r}"
            )
        grouped.setdefault(scope, []).append(record)

    full_scope = ConditioningScope.full()
    if full_scope not in grouped:
        raise ConditioningScopeMismatchError(
            "prediction composite requires exactly one full conditioning scope"
        )
    full_details = details[full_scope]
    full = ConditioningRecordGroup(
        full_scope,
        full_details[0],
        full_details[1],
        tuple(grouped.pop(full_scope)),
    )
    subjects: dict[str, ConditioningRecordGroup] = {}
    for scope, scope_records in grouped.items():
        if scope.kind != ConditioningScopeKind.SUBJECT or scope.region_id is None:
            raise ConditioningScopeMismatchError("unknown conditioning scope")
        token_count, digest = details[scope]
        subjects[scope.region_id] = ConditioningRecordGroup(
            scope, token_count, digest, tuple(scope_records)
        )
    return ConditioningGroups(full, subjects)


class SemanticMaskCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], Any] = {}

    def for_prediction(
        self,
        prediction: Any,
        ownership_masks: Mapping[str, object],
        *,
        region_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional

        if prediction.ndim not in {4, 5}:
            raise SemanticPredictionShapeError(
                "semantic prediction must be [B,C,H,W] or [B,C,T,H,W], "
                f"got {tuple(prediction.shape)}"
            )
        selected = tuple(region_ids or ownership_masks.keys())
        prepared: dict[str, Any] = {}
        for region_id in selected:
            if region_id not in ownership_masks:
                raise ConditioningScopeMismatchError(
                    f"missing ownership mask for subject {region_id!r}"
                )
            key = (
                region_id,
                prediction.device.type,
                prediction.device.index,
                str(prediction.dtype),
                int(prediction.ndim),
                int(prediction.shape[-2]),
                int(prediction.shape[-1]),
            )
            mask = self._cache.get(key)
            if mask is None:
                source = ownership_masks[region_id]
                mask = (
                    source.detach()
                    if isinstance(source, torch.Tensor)
                    else torch.as_tensor(source)
                )
                while mask.ndim > 2 and mask.shape[0] == 1:
                    mask = mask[0]
                if mask.ndim != 2:
                    raise SemanticPredictionShapeError(
                        f"ownership mask for {region_id!r} must be two-dimensional"
                    )
                mask = mask.to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                ).reshape(1, 1, *mask.shape)
                if tuple(mask.shape[-2:]) != tuple(prediction.shape[-2:]):
                    mask = functional.interpolate(
                        mask,
                        size=prediction.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if prediction.ndim == 5:
                    # Krea 2 uses Wan's spatiotemporal latent layout even for a
                    # still image, so its prediction is [B,C,1,H,W]. Ownership
                    # is spatial and intentionally broadcasts across that axis.
                    mask = mask.unsqueeze(2)
                mask = mask.clamp(0.0, 1.0)
                self._cache[key] = mask
            prepared[region_id] = mask

        if prepared:
            summed = torch.stack(tuple(prepared.values()), dim=0).sum(dim=0)
            normalizer = summed.clamp_min(1.0)
            prepared = {
                region_id: mask / normalizer
                for region_id, mask in prepared.items()
            }
        return prepared

    def clear(self) -> None:
        self._cache.clear()


def fuse_subject_predictions(
    *,
    full: Any,
    subjects: Mapping[str, Any],
    ownership_masks: Mapping[str, Any],
    gate_strength: float,
) -> Any:
    import torch

    gate = float(gate_strength)
    if not 0.0 <= gate <= 1.0:
        raise SemanticPredictionShapeError(
            "semantic prediction gate strength must be between zero and one"
        )
    if gate <= 0.0 or not subjects:
        return full
    weights: dict[str, Any] = {}
    for region_id, prediction in subjects.items():
        if (
            prediction.shape != full.shape
            or prediction.dtype != full.dtype
            or prediction.device != full.device
        ):
            raise SemanticPredictionShapeError(
                f"subject prediction {region_id!r} does not match the full prediction"
            )
        try:
            mask = ownership_masks[region_id]
        except KeyError as error:
            raise ConditioningScopeMismatchError(
                f"subject {region_id!r} has no ownership mask"
            ) from error
        if (
            mask.ndim != full.ndim
            or mask.shape[0] not in {1, full.shape[0]}
            or mask.shape[1] != 1
            or any(
                mask_size not in {1, prediction_size}
                for mask_size, prediction_size in zip(
                    mask.shape[2:],
                    full.shape[2:],
                    strict=True,
                )
            )
            or mask.dtype != full.dtype
            or mask.device != full.device
        ):
            raise SemanticPredictionShapeError(
                f"ownership mask {region_id!r} is not broadcastable to the prediction"
            )
        weights[region_id] = mask * gate
    weight_sum = torch.stack(tuple(weights.values()), dim=0).sum(dim=0).clamp(0.0, 1.0)
    fused = full * (1.0 - weight_sum)
    for region_id, prediction in subjects.items():
        fused = fused + prediction * weights[region_id]
    return fused


@dataclass(slots=True)
class SubjectPredictionDiagnostic:
    forward_calls: int = 0
    inside_delta_energy: Any = None
    inside_weight: Any = None


@dataclass(slots=True)
class SemanticPredictionDiagnostics:
    full_forward_calls: int = 0
    subject_forward_calls: int = 0
    maximum_summed_ownership_weight: float = 0.0
    subjects: dict[str, SubjectPredictionDiagnostic] = field(default_factory=dict)

    def observe_full(self) -> None:
        self.full_forward_calls += 1

    def observe_subject(self, region_id: str, full: Any, subject: Any, mask: Any) -> None:
        state = self.subjects.setdefault(region_id, SubjectPredictionDiagnostic())
        state.forward_calls += 1
        self.subject_forward_calls += 1
        delta = (subject.detach().float() - full.detach().float()).square()
        weight = mask.detach().float()
        energy = (delta * weight).sum()
        count = weight.sum() * delta.shape[1] * delta.shape[0]
        state.inside_delta_energy = (
            energy
            if state.inside_delta_energy is None
            else state.inside_delta_energy + energy
        )
        state.inside_weight = (
            count if state.inside_weight is None else state.inside_weight + count
        )

    def document(self, plan: PoseSemanticPlan) -> dict[str, Any]:
        subjects = []
        for subject in plan.subjects:
            region_id = subject.bound_prompt.region_id
            observed = self.subjects.get(region_id, SubjectPredictionDiagnostic())
            rms = 0.0
            if observed.inside_delta_energy is not None and observed.inside_weight is not None:
                rms = float(
                    (
                        observed.inside_delta_energy
                        / observed.inside_weight.clamp_min(1e-12)
                    )
                    .sqrt()
                    .item()
                )
            subjects.append(
                {
                    "region_id": region_id,
                    "region_name": subject.bound_prompt.region_name,
                    "prompt_sha256": subject.bound_prompt.prompt_sha256,
                    "text_token_count": subject.bound_prompt.text_token_count,
                    "ownership_coverage": subject.ownership_coverage,
                    "forward_calls": observed.forward_calls,
                    "prediction_delta_rms_inside_ownership": rms,
                }
            )
        return {
            "version": SEMANTIC_RUNTIME_VERSION,
            "mode": plan.mode.value,
            "shared_visual_prompt_sha256": plan.shared_visual_prompt_sha256,
            "subjects": subjects,
            "full_forward_calls": self.full_forward_calls,
            "subject_forward_calls": self.subject_forward_calls,
            "total_conditional_forward_calls": (
                self.full_forward_calls + self.subject_forward_calls
            ),
            "maximum_summed_ownership_weight": self.maximum_summed_ownership_weight,
            "multigpu": False,
        }


@dataclass(slots=True)
class PoseSemanticRuntime:
    mode: PoseSemanticMode
    pose_controller: Any
    plan: PoseSemanticPlan
    mask_cache: SemanticMaskCache = field(default_factory=SemanticMaskCache)
    diagnostics: SemanticPredictionDiagnostics = field(
        default_factory=SemanticPredictionDiagnostics
    )
    cancellation_check: Callable[[], None] | None = None
    progress: Callable[[dict[str, Any]], None] | None = None


def estimated_model_forwards(
    normal_steps: int,
    hard_steps: int,
    soft_steps: int,
    subject_count: int,
) -> int:
    if min(normal_steps, hard_steps, soft_steps, subject_count) < 0:
        raise ValueError("forward estimate inputs must not be negative")
    return normal_steps + (hard_steps + soft_steps) * (1 + subject_count)


@contextmanager
def installed_semantic_prediction_hook(
    model_options: dict[str, Any],
    semantic_runtime: PoseSemanticRuntime,
    comfy_samplers: Any,
) -> Iterator[Callable[[dict[str, Any]], list[Any]]]:
    key = "sampler_calc_cond_batch_function"
    if key in model_options:
        raise SemanticSamplerHookError(
            "a pre-existing sampler conditional-batch hook is incompatible with "
            "Prediction composite"
        )
    if "multigpu_clones" in model_options:
        raise SemanticMultiGpuUnsupportedError(
            "Prediction composite does not support multi-GPU execution"
        )
    ownership_sources = {
        subject.bound_prompt.region_id: subject.ownership_mask
        for subject in semantic_runtime.plan.subjects
    }

    def calculate(args: dict[str, Any]) -> list[Any]:
        import torch

        cond_sets = args["conds"]
        x = args["input"]
        sigma = args["sigma"]
        model = args["model"]
        current_options = args["model_options"]
        if "multigpu_clones" in current_options:
            raise SemanticMultiGpuUnsupportedError(
                "Prediction composite does not support multi-GPU execution"
            )
        if not cond_sets or cond_sets[0] is None:
            raise ConditioningScopeMismatchError(
                "prediction composite received no positive conditioning"
            )
        groups = group_conditioning_by_scope(cond_sets[0])
        expected_ids = {
            subject.bound_prompt.region_id
            for subject in semantic_runtime.plan.subjects
        }
        if set(groups.subjects) != expected_ids:
            raise ConditioningScopeMismatchError(
                "positive conditioning scopes do not match the semantic plan"
            )
        semantic_runtime.pose_controller.observe_sigma(sigma)
        gate = float(semantic_runtime.pose_controller.gate_strength)
        with conditioning_execution_scope(
            groups.full.scope, groups.full.text_token_count
        ):
            full = comfy_samplers.calc_cond_batch(
                model,
                [list(groups.full.records)],
                x,
                sigma,
                current_options,
            )[0]
        semantic_runtime.diagnostics.observe_full()
        fused = full
        if gate > 0.0 and expected_ids:
            masks = semantic_runtime.mask_cache.for_prediction(
                full,
                ownership_sources,
                region_ids=tuple(
                    subject.bound_prompt.region_id
                    for subject in semantic_runtime.plan.subjects
                ),
            )
            if masks:
                summed = torch.stack(tuple(masks.values()), dim=0).sum(dim=0)
                semantic_runtime.diagnostics.maximum_summed_ownership_weight = max(
                    semantic_runtime.diagnostics.maximum_summed_ownership_weight,
                    float(summed.max().item()),
                )
            weight_sum = (
                torch.stack(tuple(masks.values()), dim=0)
                .sum(dim=0)
                .mul(gate)
                .clamp(0.0, 1.0)
            )
            fused = full * (1.0 - weight_sum)
            for index, subject in enumerate(semantic_runtime.plan.subjects, start=1):
                if semantic_runtime.cancellation_check is not None:
                    semantic_runtime.cancellation_check()
                region_id = subject.bound_prompt.region_id
                group = groups.require_subject(region_id)
                if semantic_runtime.progress is not None:
                    semantic_runtime.progress(
                        {
                            "phase": semantic_runtime.pose_controller.phase,
                            "gate_strength": gate,
                            "semantic_prediction_index": index,
                            "semantic_prediction_total": len(expected_ids),
                            "region_id": region_id,
                        }
                    )
                with conditioning_execution_scope(group.scope, group.text_token_count):
                    prediction = comfy_samplers.calc_cond_batch(
                        model,
                        [list(group.records)],
                        x,
                        sigma,
                        current_options,
                    )[0]
                semantic_runtime.diagnostics.observe_subject(
                    region_id, full, prediction, masks[region_id]
                )
                if (
                    prediction.shape != full.shape
                    or prediction.dtype != full.dtype
                    or prediction.device != full.device
                ):
                    raise SemanticPredictionShapeError(
                        f"subject prediction {region_id!r} does not match "
                        "the full prediction"
                    )
                fused = fused + prediction * masks[region_id] * gate
                del prediction

        negative = cond_sets[1] if len(cond_sets) > 1 else None
        if negative:
            negative_prediction = comfy_samplers.calc_cond_batch(
                model, [negative], x, sigma, current_options
            )[0]
        else:
            # The pinned cfg_function returns the conditional result exactly at CFG 1.
            negative_prediction = torch.zeros_like(fused)
        return [fused, negative_prediction]

    model_options[key] = calculate
    try:
        yield calculate
    finally:
        semantic_runtime.mask_cache.clear()
        model_options.pop(key, None)
