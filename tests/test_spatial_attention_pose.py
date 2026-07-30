from k2_region_lab.pose_gating import (
    PoseGateController,
    PoseGatePhases,
    PoseGateRegionBinding,
    SoftGateSchedule,
)
from k2_region_lab.regional_prompting import compile_regional_prompt_plan
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.spatial_attention import KreaSpatialAttentionOverride


def test_attention_ownership_uses_hard_volumes_then_exact_normal_boxes() -> None:
    regions = (
        RegionDefinition(
            "left",
            "Left",
            PixelBox(0, 0, 16, 16),
            "left person",
            priority=10,
            spatial_role="subject",
        ),
        RegionDefinition(
            "right",
            "Right",
            PixelBox(16, 0, 32, 16),
            "right person",
            priority=5,
            spatial_role="subject",
        ),
    )
    plan = compile_regional_prompt_plan(32, 16, "two people", regions)
    bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
    controller = PoseGateController(
        phases=PoseGatePhases(1, 0, 1),
        soft_schedule=SoftGateSchedule.COSINE,
    )
    binding = PoseGateRegionBinding(
        controller=controller,
        hard_image_fields={
            "left": (0.0, 1.0),
            "right": (1.0, 0.0),
        },
    )
    override = KreaSpatialAttentionOverride(
        bound,
        pose_gate_binding=binding,
    )

    assert override._current_image_owners() == (2, 1)
    controller.mark_transition_complete(0)
    assert override._current_image_owners() == override.image_owners == (1, 2)
