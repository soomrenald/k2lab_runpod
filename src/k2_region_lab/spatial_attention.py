from __future__ import annotations

from typing import Any

from k2_region_lab.pose_gating import PoseGateRegionBinding
from k2_region_lab.regional_prompting import BoundRegionalPromptPlan
from k2_region_lab.regional_prompting import PromptTokenRoleKind
from k2_region_lab.semantic_conditioning import (
    CURRENT_CONDITIONING_CONTEXT,
    ConditioningScopeKind,
    PoseSemanticMode,
    SEMANTIC_ATTENTION_PENALTY,
    SUBJECT_ISLAND_THRESHOLD,
    SemanticAttentionError,
)


def text_region_ownership(plan: BoundRegionalPromptPlan) -> tuple[int, ...]:
    """Return zero for shared text and a distinct owner for each subject clause."""
    owners = [0] * plan.text_token_count
    owner = 0
    for span in plan.spans:
        if span.spatial_role != "subject":
            continue
        owner += 1
        owners[span.start : span.end] = [owner] * (span.end - span.start)
    return tuple(owners)


def image_region_ownership(plan: BoundRegionalPromptPlan) -> tuple[int, ...]:
    """Assign each subject-box image token to the first/highest-priority subject."""
    owners = [0] * plan.image_token_count
    owner = 0
    for span in plan.spans:
        if span.spatial_role != "subject":
            continue
        owner += 1
        for index, weight in enumerate(span.image_token_mask):
            if weight > 0.0 and owners[index] == 0:
                owners[index] = owner
    return tuple(owners)


def spatial_pair_bias(
    image_token_field: tuple[float, ...],
    strength: float,
    *,
    outside_penalty_ratio: float = 0.25,
    outside_penalty: float | None = None,
) -> tuple[float, ...]:
    """Convert a soft spatial field into additive attention-logit values."""
    if not 0.0 <= outside_penalty_ratio <= 1.0:
        raise ValueError("outside penalty ratio must be between zero and one")
    penalty = (
        strength * outside_penalty_ratio
        if outside_penalty is None
        else outside_penalty
    )
    if not 0.0 <= penalty <= 10.0:
        raise ValueError("outside penalty must be between zero and ten")
    return tuple((strength + penalty) * weight - penalty for weight in image_token_field)


class KreaSpatialAttentionOverride:
    """Partition regional text and route it to image tokens in one model pass.

    Some CUDA/ROCm SDPA kernels cannot use a dense additive mask efficiently.
    Query chunking computes the exact biased softmax without ever materializing
    the complete per-head score matrix.
    """

    def __init__(
        self,
        plan: BoundRegionalPromptPlan,
        *,
        outside_penalty_ratio: float = 0.25,
        query_chunk_size: int = 256,
        lora_delta_adaptation: bool = False,
        lora_delta_adaptation_gain: float = 0.35,
        pose_gate_binding: PoseGateRegionBinding | None = None,
        semantic_mode: PoseSemanticMode | str = PoseSemanticMode.SPATIAL_ONLY,
    ) -> None:
        self.plan = plan
        self.outside_penalty_ratio = outside_penalty_ratio
        if query_chunk_size <= 0:
            raise ValueError("attention query chunk size must be positive")
        if not 0.0 <= lora_delta_adaptation_gain <= 1.0:
            raise ValueError("LoRA delta adaptation gain must be between zero and one")
        self.query_chunk_size = query_chunk_size
        self.lora_delta_adaptation = lora_delta_adaptation
        self.lora_delta_adaptation_gain = lora_delta_adaptation_gain
        self.pose_gate_binding = pose_gate_binding
        self.semantic_mode = PoseSemanticMode(semantic_mode)
        self.expected_sequence_length = (
            plan.text_token_count + plan.image_token_count
        )
        self.matched_calls = 0
        self.text_refiner_calls = 0
        self.text_owners = text_region_ownership(plan)
        self.image_owners = image_region_ownership(plan)
        self.step_scale = 1.0
        self.region_scales: dict[str, float] = {}
        self.semantic_hard_blocks = 0
        self.semantic_soft_blocks = 0
        self._cache: dict[tuple[Any, ...], Any] = {}

    def __call__(self, original, *args, **kwargs):
        q = args[0]
        k = args[1]
        query_length = int(q.shape[-2])
        key_length = int(k.shape[-2])
        execution_context = CURRENT_CONDITIONING_CONTEXT.get()
        scoped_text_count = (
            execution_context.text_token_count
            if execution_context is not None
            and execution_context.scope.kind == ConditioningScopeKind.SUBJECT
            else self.plan.text_token_count
        )
        main_stream = (
            query_length == scoped_text_count + self.plan.image_token_count
            and key_length == scoped_text_count + self.plan.image_token_count
        )
        # Krea folds prompt tokens into the batch while its first two text-fusion
        # blocks attend over the checkpoint's 12 Qwen layer states.
        folded_layerwise_text = (
            query_length == 12
            and query_length == self.plan.text_token_count
            and int(q.shape[0]) >= self.plan.text_token_count
            and int(q.shape[0]) % self.plan.text_token_count == 0
        )
        text_refiner = (
            query_length == scoped_text_count
            and key_length == scoped_text_count
            and not folded_layerwise_text
        )
        if not main_stream and not text_refiner:
            return original(*args, **kwargs)

        if kwargs.get("mask") is not None:
            error_type = (
                SemanticAttentionError
                if self.semantic_mode != PoseSemanticMode.SPATIAL_ONLY
                else RuntimeError
            )
            raise error_type(
                "Krea chunked regional attention requires an unmasked stream"
            )
        if not kwargs.get("skip_reshape", False) or q.ndim != 4:
            error_type = (
                SemanticAttentionError
                if self.semantic_mode != PoseSemanticMode.SPATIAL_ONLY
                else RuntimeError
            )
            raise error_type(
                "Krea chunked spatial attention expected head-shaped query tensors"
            )

        v = args[2]
        original_head_dim = q.shape[-1]
        scale = float(kwargs.get("scale", original_head_dim**-0.5))
        output = self._chunked_attention(
            q,
            k,
            v,
            scale,
            main_stream=main_stream,
            text_count=scoped_text_count,
            execution_context=execution_context,
        )
        if main_stream:
            self.matched_calls += 1
        else:
            self.text_refiner_calls += 1
        if kwargs.get("skip_output_reshape", False):
            return output
        return output.transpose(1, 2).reshape(output.shape[0], output.shape[2], -1)

    def _chunked_attention(
        self,
        q,
        k,
        v,
        scale: float,
        *,
        main_stream: bool,
        text_count: int,
        execution_context,
    ):
        import torch

        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
            dtype=v.dtype,
            device=v.device,
        )
        key_transposed = k.transpose(-2, -1)
        full_scope = (
            execution_context is None
            or execution_context.scope.kind == ConditioningScopeKind.FULL
        )
        if full_scope:
            pair_fields, emphasis_fields, text_owners, image_owners = (
                self._pair_fields(q)
            )
        else:
            pair_fields, emphasis_fields = (), ()
            text_owners = image_owners = None
        for start in range(0, q.shape[-2], self.query_chunk_size):
            end = min(q.shape[-2], start + self.query_chunk_size)
            scores = torch.matmul(q[:, :, start:end], key_transposed) * scale
            scores = scores.float()
            self._apply_semantic_partition(
                scores,
                start,
                end,
                main_stream=main_stream,
                text_count=text_count,
                execution_context=execution_context,
                reference=q,
            )
            if main_stream and full_scope:
                self._partition_regional_stream(
                    scores, start, end, text_owners, image_owners
                )
                self._add_spatial_bias(scores, start, end, pair_fields, emphasis_fields)
            elif not main_stream and full_scope:
                self._partition_regional_text(scores, start, end, text_owners)
            probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
            output[:, :, start:end] = torch.matmul(probabilities, v)
            del scores, probabilities
        return output

    def _apply_semantic_partition(
        self,
        scores,
        start: int,
        end: int,
        *,
        main_stream: bool,
        text_count: int,
        execution_context,
        reference,
    ) -> None:
        import torch

        gate = (
            self.pose_gate_binding.gate_strength
            if self.pose_gate_binding is not None
            else 0.0
        )
        subject_scope = (
            execution_context is not None
            and execution_context.scope.kind == ConditioningScopeKind.SUBJECT
        )
        if gate <= 0.0:
            return
        if subject_scope:
            if self.semantic_mode != PoseSemanticMode.PREDICTION_COMPOSITE:
                return
            region_id = execution_context.scope.region_id
            hard_field = self.pose_gate_binding.hard_image_fields.get(region_id)
            if hard_field is None:
                raise SemanticAttentionError(
                    f"semantic attention has no ownership field for {region_id!r}"
                )
            text_domains = torch.full(
                (text_count,), 10, dtype=torch.int16, device=reference.device
            )
            if main_stream:
                image_domains = torch.tensor(
                    [
                        10 if value >= SUBJECT_ISLAND_THRESHOLD else 2
                        for value in hard_field
                    ],
                    dtype=torch.int16,
                    device=reference.device,
                )
                domains = torch.cat((text_domains, image_domains))
            else:
                domains = text_domains
            query_domains = domains[start:end]
            allowed = query_domains.reshape(-1, 1) == domains.reshape(1, -1)
        else:
            if self.semantic_mode != PoseSemanticMode.ATTENTION_ISOLATION:
                return
            owner_by_region: dict[str, int] = {}
            owner = 0
            for span in self.plan.spans:
                if span.spatial_role == "subject":
                    owner += 1
                    owner_by_region[span.region_id] = owner
            role_codes = {
                PromptTokenRoleKind.SPECIAL: 0,
                PromptTokenRoleKind.SHARED_VISUAL: 1,
                PromptTokenRoleKind.SCENE: 2,
                PromptTokenRoleKind.RELATIONSHIP: 3,
                PromptTokenRoleKind.OTHER_REGION: 4,
            }
            text_domains = torch.tensor(
                [
                    (
                        10 + owner_by_region.get(role.region_id, 0)
                        if role.kind == PromptTokenRoleKind.SUBJECT
                        else role_codes[role.kind]
                    )
                    for role in self.plan.token_roles
                ],
                dtype=torch.int16,
                device=reference.device,
            )
            if len(text_domains) != text_count:
                raise SemanticAttentionError(
                    "semantic attention token roles do not match full conditioning"
                )
            if main_stream:
                current_image_owners = self._current_image_owners()
                image_domains = torch.tensor(
                    [10 + value if value > 0 else 2 for value in current_image_owners],
                    dtype=torch.int16,
                    device=reference.device,
                )
                domains = torch.cat((text_domains, image_domains))
            else:
                domains = text_domains
            query_domains = domains[start:end]
            subject_queries = query_domains >= 10
            same_subject = query_domains.reshape(-1, 1) == domains.reshape(1, -1)
            shared_or_special = domains.reshape(1, -1) <= 1
            non_subject = domains.reshape(1, -1) < 10
            allowed = torch.where(
                subject_queries.reshape(-1, 1),
                same_subject | shared_or_special,
                non_subject,
            )
            shared_queries = query_domains <= 1
            allowed = torch.where(
                shared_queries.reshape(-1, 1),
                shared_or_special,
                allowed,
            )
        blocked = ~allowed
        blocked_count = int(blocked.sum().item())
        if gate >= 1.0:
            scores.masked_fill_(
                blocked.reshape(1, 1, end - start, -1),
                float("-inf"),
            )
            self.semantic_hard_blocks += blocked_count
        else:
            scores.add_(
                blocked.to(scores.dtype).reshape(1, 1, end - start, -1),
                alpha=-SEMANTIC_ATTENTION_PENALTY * gate,
            )
            self.semantic_soft_blocks += blocked_count

    def _pair_fields(self, reference):
        import torch

        device = reference.device
        key = (device.type, device.index, str(reference.dtype))
        cached = self._cache.get(key)
        if cached is None:
            normal_fields = []
            hard_fields = []
            for span in self.plan.spans:
                outside_penalty = self.plan.outside_penalty * (
                    1.0 if span.spatial_role in {"subject", "edit"} else 0.25
                )
                normal = span.image_token_field
                hard = (
                    self.pose_gate_binding.hard_field(span.region_id, normal)
                    if self.pose_gate_binding is not None
                    else normal
                )
                normal_fields.append(
                    torch.tensor(
                        spatial_pair_bias(
                            normal,
                            self.plan.strength,
                            outside_penalty_ratio=self.outside_penalty_ratio,
                            outside_penalty=outside_penalty,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                hard_fields.append(
                    torch.tensor(
                        spatial_pair_bias(
                            hard,
                            self.plan.strength,
                            outside_penalty_ratio=self.outside_penalty_ratio,
                            outside_penalty=outside_penalty,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            emphasis_fields = tuple(
                torch.tensor(
                    emphasis.image_token_field,
                    dtype=torch.float32,
                    device=device,
                )
                for emphasis in self.plan.emphases
            )
            text_owners = torch.tensor(
                self.text_owners, dtype=torch.int16, device=device
            )
            cached = (
                tuple(normal_fields),
                tuple(hard_fields),
                emphasis_fields,
                text_owners,
            )
            self._cache[key] = cached
        normal_fields, hard_fields, emphasis_fields, text_owners = cached
        gate = (
            self.pose_gate_binding.gate_strength
            if self.pose_gate_binding is not None
            else 0.0
        )
        fields = (
            normal_fields
            if gate <= 0.0
            else hard_fields
            if gate >= 1.0
            else tuple(
                normal + gate * (hard - normal)
                for normal, hard in zip(normal_fields, hard_fields, strict=True)
            )
        )
        owner_key = (*key, f"owners:{gate:.12g}")
        image_owners = self._cache.get(owner_key)
        if image_owners is None:
            image_owners = torch.tensor(
                self._current_image_owners(),
                dtype=torch.int16,
                device=device,
            )
            self._cache[owner_key] = image_owners
        return fields, emphasis_fields, text_owners, image_owners

    def _current_image_owners(self) -> tuple[int, ...]:
        if self.pose_gate_binding is None or self.pose_gate_binding.gate_strength <= 0.0:
            return self.image_owners
        owners = [0] * self.plan.image_token_count
        owner = 0
        for span in self.plan.spans:
            if span.spatial_role != "subject":
                continue
            owner += 1
            field = self.pose_gate_binding.effective_field(
                span.region_id, span.image_token_field
            )
            for index, weight in enumerate(field):
                if weight > 0.0 and owners[index] == 0:
                    owners[index] = owner
        return tuple(owners)

    def _partition_regional_text(self, scores, start, end, text_owners) -> None:
        """Keep subject-owned keys private to that subject in both text stages."""
        self._partition_owned_keys(scores, start, end, text_owners)

    def _partition_regional_stream(
        self, scores, start, end, text_owners, image_owners
    ) -> None:
        """Partition cross-modal subject attention without masking image-to-image."""
        text_count = self.plan.text_token_count
        text_end = min(end, text_count)
        if start < text_end:
            query_text_owners = text_owners[start:text_end]
            blocked_text = (text_owners.reshape(1, -1) > 0) & (
                query_text_owners.reshape(-1, 1) != text_owners.reshape(1, -1)
            )
            scores[
                :, :, : text_end - start, :text_count
            ].masked_fill_(
                blocked_text.reshape(1, 1, text_end - start, -1),
                float("-inf"),
            )
            blocked_images = (image_owners.reshape(1, -1) > 0) & (
                query_text_owners.reshape(-1, 1) != image_owners.reshape(1, -1)
            )
            scores[
                :, :, : text_end - start, text_count:
            ].masked_fill_(
                blocked_images.reshape(1, 1, text_end - start, -1),
                float("-inf"),
            )

        image_start = max(start, text_count)
        if image_start < end:
            query_image_owners = image_owners[
                image_start - text_count : end - text_count
            ]
            blocked_text = (text_owners.reshape(1, -1) > 0) & (
                query_image_owners.reshape(-1, 1) != text_owners.reshape(1, -1)
            )
            scores[
                :, :, image_start - start : end - start, :text_count
            ].masked_fill_(
                blocked_text.reshape(1, 1, end - image_start, -1),
                float("-inf"),
            )

    @staticmethod
    def _partition_owned_keys(scores, start, end, owners) -> None:
        query_owners = owners[start:end]
        blocked = (owners.reshape(1, -1) > 0) & (
            query_owners.reshape(-1, 1) != owners.reshape(1, -1)
        )
        scores.masked_fill_(
            blocked.reshape(1, 1, end - start, -1), float("-inf")
        )

    def _add_spatial_bias(self, scores, start, end, pair_fields, emphasis_fields) -> None:
        text_count = self.plan.text_token_count
        for span, pair in zip(self.plan.spans, pair_fields, strict=True):
            text_start = max(start, span.start)
            text_end = min(end, span.end)
            if text_start < text_end:
                scores[
                    :, :, text_start - start : text_end - start, text_count:
                ].add_(
                    pair.reshape(1, 1, 1, -1),
                    alpha=self.step_scale * self.region_scales.get(span.region_id, 1.0),
                )

            image_start = max(start, text_count)
            image_end = end
            if image_start < image_end:
                image_pair = pair[
                    image_start - text_count : image_end - text_count
                ]
                scores[
                    :, :, image_start - start : image_end - start, span.start : span.end
                ].add_(
                    image_pair.reshape(1, 1, -1, 1),
                    alpha=self.step_scale * self.region_scales.get(span.region_id, 1.0),
                )
        for emphasis, image_field in zip(
            self.plan.emphases, emphasis_fields, strict=True
        ):
            text_start = max(start, emphasis.start)
            text_end = min(end, emphasis.end)
            image_start = max(start, text_count)
            image_end = end
            if text_start >= text_end or image_start >= image_end:
                continue
            scores[
                :, :, image_start - start : image_end - start, text_start : text_end
            ].add_(
                image_field[
                    image_start - text_count : image_end - text_count
                ].reshape(1, 1, -1, 1),
                alpha=self.step_scale * emphasis.strength,
            )

    def set_lora_delta_scales(self, scales: dict[str, float]) -> None:
        """Set bounded, per-region multipliers for the next attention calls."""
        if not self.lora_delta_adaptation:
            return
        known_regions = {span.region_id for span in self.plan.spans}
        self.region_scales = {
            region_id: min(1.5, max(0.5, float(scale)))
            for region_id, scale in scales.items()
            if region_id in known_regions
        }

    def set_denoising_progress(self, completed_steps: int, total_steps: int) -> None:
        """Keep placement strong early, then relax it for late detail refinement."""
        if total_steps <= 0:
            raise ValueError("total denoising steps must be positive")
        progress = min(1.0, max(0.0, completed_steps / total_steps))
        relaxation_start = 0.55
        if progress <= relaxation_start:
            self.step_scale = 1.0
            return
        fraction = (progress - relaxation_start) / (1.0 - relaxation_start)
        self.step_scale = 1.0 + fraction * (self.plan.late_step_scale - 1.0)

    def clear(self) -> None:
        self._cache.clear()

    def summary(self) -> dict[str, object]:
        return {
            "text_refiner_attention_calls": self.text_refiner_calls,
            "text_partition": "subject_keys_private_to_region",
            "subject_box_exclusion": True,
            "cross_modal_partition": "subject_text_private_to_box",
            "image_to_image_attention": (
                "semantic_islands"
                if self.semantic_mode
                in {
                    PoseSemanticMode.ATTENTION_ISOLATION,
                    PoseSemanticMode.PREDICTION_COMPOSITE,
                }
                else "unmodified"
            ),
            "semantic_mode": self.semantic_mode.value,
            "semantic_hard_blocks": self.semantic_hard_blocks,
            "semantic_soft_blocks": self.semantic_soft_blocks,
            "semantic_attention_penalty": SEMANTIC_ATTENTION_PENALTY,
            "image_to_image_isolation": self.semantic_mode
            in {
                PoseSemanticMode.ATTENTION_ISOLATION,
                PoseSemanticMode.PREDICTION_COMPOSITE,
            },
            "lora_delta_adaptation": self.lora_delta_adaptation,
            "lora_delta_adaptation_gain": self.lora_delta_adaptation_gain,
            "final_region_scales": dict(self.region_scales),
            "volumetric_pose_gating": self.pose_gate_binding is not None,
        }
