from __future__ import annotations

from types import SimpleNamespace

import pytest

from k2_region_lab.pose import default_volumetric_subject_pose
from k2_region_lab.regional_prompting import (
    PromptEmphasis,
    compile_regional_prompt_plan,
    compile_subject_conditioning_prompt,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.semantic_conditioning import (
    CURRENT_CONDITIONING_CONTEXT,
    BoundSubjectPrompt,
    ConditioningScope,
    ConditioningScopeMismatchError,
    PoseSemanticMode,
    PoseSemanticPlan,
    PoseSemanticRuntime,
    SemanticMaskCache,
    SemanticSamplerHookError,
    SubjectSemanticConditioning,
    annotate_conditioning,
    conditioning_execution_scope,
    estimated_model_forwards,
    fuse_subject_predictions,
    group_conditioning_by_scope,
    installed_semantic_prediction_hook,
)


def subject(region_id: str, name: str, prompt: str) -> RegionDefinition:
    return RegionDefinition(
        region_id,
        name,
        PixelBox(0, 0, 512, 1024),
        prompt,
        face_identity_prompt=f"{name} face identity",
        region_type="subject",
        spatial_role="subject",
        pose=default_volumetric_subject_pose(),
    )


def test_subject_prompt_excludes_scene_relationship_and_other_subject() -> None:
    astronaut = subject(
        "astronaut",
        "Astronaut",
        "A white astronaut raising the right arm while holding a silver wrench",
    )
    knight = subject(
        "knight",
        "Knight",
        "A medieval knight crouching in black plate armor",
    )
    shared = "Cinematic realistic photography, 35 mm lens"
    full = compile_regional_prompt_plan(
        1024,
        1024,
        "A greenhouse crowded with enormous tropical leaves and red sofas",
        (astronaut, knight),
        shared_visual_prompt=shared,
    )
    plan = compile_subject_conditioning_prompt(
        shared_visual_prompt=shared,
        region=astronaut,
        identity_triggers=("astro-id",),
        emphases=(
            PromptEmphasis("astronaut", "silver wrench", 0.7),
            PromptEmphasis("knight", "black plate", 0.8),
        ),
    )

    assert full.prompt.startswith(f"{shared}.")
    assert "greenhouse" in full.prompt
    assert "greenhouse" not in plan.prompt
    assert "leaves" not in plan.prompt
    assert "sofas" not in plan.prompt
    assert "knight" not in plan.prompt.casefold()
    assert "silver wrench" in plan.prompt
    assert "astro-id" in plan.prompt
    assert len(plan.emphases) == 1
    assert plan.emphases[0].phrase == "silver wrench"


def test_subject_prompt_binding_is_deterministic_and_scope_local() -> None:
    region = subject("a", "Subject A", "A person in a blue coat")
    plan = compile_subject_conditioning_prompt(
        shared_visual_prompt="Soft window light",
        region=region,
        identity_triggers=("person-a",),
    )
    def token_count(prefix: str) -> int:
        return len(prefix.split())

    first = plan.bind_tokens(token_count)
    second = plan.bind_tokens(token_count)

    assert first.prompt_sha256 == second.prompt_sha256
    assert first.text_token_count == len(plan.prompt.split())
    assert first.shared_visual_span is not None
    assert first.face_identity_span is not None
    assert first.character_trigger_spans["person-a"]


def test_conditioning_context_restores_nested_and_exception_states() -> None:
    assert CURRENT_CONDITIONING_CONTEXT.get() is None
    with conditioning_execution_scope(ConditioningScope.full(), 20):
        assert CURRENT_CONDITIONING_CONTEXT.get().scope == ConditioningScope.full()
        with pytest.raises(RuntimeError):
            with conditioning_execution_scope(ConditioningScope.subject("a"), 8):
                assert CURRENT_CONDITIONING_CONTEXT.get().scope.region_id == "a"
                raise RuntimeError("stop")
        assert CURRENT_CONDITIONING_CONTEXT.get().scope == ConditioningScope.full()
    assert CURRENT_CONDITIONING_CONTEXT.get() is None


def test_conditioning_annotation_and_grouping_support_multiple_records() -> None:
    full = annotate_conditioning(
        ([SimpleNamespace(shape=(1, 12)), {"pooled": "full"}],),
        scope=ConditioningScope.full(),
        prompt_hash="a" * 64,
        text_token_count=12,
    )
    subject_records = annotate_conditioning(
        (
            [SimpleNamespace(shape=(1, 7)), {"part": 1}],
            [SimpleNamespace(shape=(1, 7)), {"part": 2}],
        ),
        scope=ConditioningScope.subject("a"),
        prompt_hash="b" * 64,
        text_token_count=7,
    )
    converted = [record[1] for record in (*full, *subject_records)]

    groups = group_conditioning_by_scope(converted)

    assert len(groups.full.records) == 1
    assert len(groups.require_subject("a").records) == 2
    with pytest.raises(ConditioningScopeMismatchError):
        groups.require_subject("missing")


@pytest.mark.parametrize(
    ("normal", "hard", "soft", "subjects", "expected"),
    [(8, 2, 2, 2, 20), (8, 0, 0, 3, 8), (1, 1, 0, 1, 3)],
)
def test_estimated_model_forward_equivalents(
    normal: int,
    hard: int,
    soft: int,
    subjects: int,
    expected: int,
) -> None:
    assert estimated_model_forwards(normal, hard, soft, subjects) == expected
    assert PoseSemanticMode("prediction_composite") == PoseSemanticMode.PREDICTION_COMPOSITE


def test_prediction_fusion_hard_soft_and_normal_math() -> None:
    torch = pytest.importorskip("torch")
    full = torch.ones((1, 2, 2, 2))
    subject_a = torch.full_like(full, 10.0)
    subject_b = torch.full_like(full, 20.0)
    mask_a = torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]])
    mask_b = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])

    hard = fuse_subject_predictions(
        full=full,
        subjects={"a": subject_a, "b": subject_b},
        ownership_masks={"a": mask_a, "b": mask_b},
        gate_strength=1.0,
    )
    soft = fuse_subject_predictions(
        full=full,
        subjects={"a": subject_a, "b": subject_b},
        ownership_masks={"a": mask_a, "b": mask_b},
        gate_strength=0.5,
    )
    normal = fuse_subject_predictions(
        full=full,
        subjects={"a": subject_a, "b": subject_b},
        ownership_masks={"a": mask_a, "b": mask_b},
        gate_strength=0.0,
    )

    assert hard[0, 0].tolist() == [[10.0, 5.5], [20.0, 1.0]]
    assert soft[0, 0].tolist() == [[5.5, 3.25], [10.5, 1.0]]
    assert torch.equal(normal, full)


def test_semantic_masks_and_fusion_support_krea_spatiotemporal_latents() -> None:
    torch = pytest.importorskip("torch")
    full = torch.ones((1, 16, 1, 2, 3))
    subject_prediction = torch.full_like(full, 7.0)
    ownership = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])

    masks = SemanticMaskCache().for_prediction(
        full,
        {"subject": ownership},
    )
    fused = fuse_subject_predictions(
        full=full,
        subjects={"subject": subject_prediction},
        ownership_masks=masks,
        gate_strength=1.0,
    )

    assert masks["subject"].shape == (1, 1, 1, 2, 3)
    assert fused.shape == full.shape
    assert fused[0, 0, 0].tolist() == [[7.0, 4.0, 1.0], [1.0, 7.0, 1.0]]


def _bound_subject(region_id: str, token_count: int) -> BoundSubjectPrompt:
    return BoundSubjectPrompt(
        region_id=region_id,
        region_name=f"Subject {region_id}",
        prompt=f"subject {region_id}",
        text_token_count=token_count,
        shared_visual_span=None,
        subject_span=(0, token_count),
        face_identity_span=None,
        character_trigger_spans={},
        emphasis_spans=(),
        prompt_sha256=region_id * 64,
    )


@pytest.mark.parametrize("latent_shape", [(1, 4, 2, 2), (1, 16, 1, 2, 2)])
def test_semantic_hook_evaluates_scopes_without_advancing_transition(
    latent_shape: tuple[int, ...],
) -> None:
    torch = pytest.importorskip("torch")
    controller = SimpleNamespace(
        gate_strength=1.0,
        phase="hard",
        current_transition=0,
        observe_sigma=lambda _sigma: None,
    )
    subject_bound = _bound_subject("a", 7)
    subject = SubjectSemanticConditioning(
        scope=ConditioningScope.subject("a"),
        bound_prompt=subject_bound,
        conditioning=(),
        ownership_mask=torch.ones((2, 2)),
        ownership_coverage=1.0,
    )
    plan = PoseSemanticPlan(
        mode=PoseSemanticMode.PREDICTION_COMPOSITE,
        full_scope=ConditioningScope.full(),
        full_conditioning=(),
        subjects=(subject,),
        shared_visual_prompt_sha256="c" * 64,
        estimated_forwards_per_gated_evaluation=2,
    )
    runtime = PoseSemanticRuntime(
        mode=PoseSemanticMode.PREDICTION_COMPOSITE,
        pose_controller=controller,
        plan=plan,
    )
    calls: list[str] = []

    def calculate(_model, condition_sets, x, _sigma, _options):
        scope = condition_sets[0][0]["k2_conditioning_scope"]
        calls.append(scope)
        value = 1.0 if scope == "full" else 10.0
        return [torch.full_like(x, value)]

    comfy = SimpleNamespace(calc_cond_batch=calculate)
    options: dict[str, object] = {}
    full_record = {
        "k2_conditioning_scope": "full",
        "k2_conditioning_region_id": None,
        "k2_conditioning_prompt_sha256": "f" * 64,
        "k2_conditioning_text_token_count": 12,
    }
    subject_record = {
        "k2_conditioning_scope": "subject",
        "k2_conditioning_region_id": "a",
        "k2_conditioning_prompt_sha256": "a" * 64,
        "k2_conditioning_text_token_count": 7,
    }
    x = torch.zeros(latent_shape)

    with installed_semantic_prediction_hook(options, runtime, comfy) as hook:
        result = hook(
            {
                "conds": [[full_record, subject_record], None],
                "input": x,
                "sigma": torch.tensor([1.0]),
                "model": object(),
                "model_options": options,
            }
        )

    assert calls == ["full", "subject"]
    assert torch.equal(result[0], torch.full_like(x, 10.0))
    assert torch.equal(result[1], torch.zeros_like(x))
    assert controller.current_transition == 0
    assert CURRENT_CONDITIONING_CONTEXT.get() is None
    assert "sampler_calc_cond_batch_function" not in options


def test_semantic_hook_rejects_existing_hook_without_replacing_it() -> None:
    existing = object()
    options = {"sampler_calc_cond_batch_function": existing}
    runtime = SimpleNamespace()

    with pytest.raises(SemanticSamplerHookError):
        with installed_semantic_prediction_hook(options, runtime, SimpleNamespace()):
            pass

    assert options["sampler_calc_cond_batch_function"] is existing
