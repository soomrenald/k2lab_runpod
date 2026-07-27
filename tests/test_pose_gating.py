from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from k2_region_lab.pose_gating import (
    PoseGateController,
    PoseGatePhases,
    PoseGateRuntimeError,
    PoseGateRegionBinding,
    SigmaScheduleError,
    SigmaScheduleMode,
    SigmaScheduleRequest,
    SoftGateSchedule,
    gate_strengths,
    phase_weighted_positions,
    resample_advanced_positions,
    resolve_sigma_schedule,
)
from k2_region_lab.worker.runtime import build_comfy_baseline_sigmas


@pytest.mark.parametrize("schedule", list(SoftGateSchedule))
def test_gate_strength_vectors_have_exact_phase_values(schedule: SoftGateSchedule) -> None:
    phases = PoseGatePhases(hard_steps=2, soft_steps=3, normal_steps=4)
    values = gate_strengths(phases, schedule)

    assert len(values) == phases.effective_steps
    assert values[:2] == (1.0, 1.0)
    assert all(0.0 < value < 1.0 for value in values[2:5])
    assert all(left >= right for left, right in zip(values[2:5], values[3:5]))
    assert values[5:] == (0.0,) * 4


def test_phase_weighted_positions_respect_phase_boundaries() -> None:
    phases = PoseGatePhases(hard_steps=2, soft_steps=2, normal_steps=8)
    positions = phase_weighted_positions(
        phases, hard_share=0.20, soft_share=0.35
    )

    assert len(positions) == 13
    assert positions[0] == 0.0
    assert positions[2] == pytest.approx(0.20)
    assert positions[4] == pytest.approx(0.55)
    assert positions[-1] == 1.0


def test_zero_step_phase_rejects_nonzero_share() -> None:
    phases = PoseGatePhases(hard_steps=0, soft_steps=2, normal_steps=8)
    with pytest.raises(SigmaScheduleError, match="zero-step hard"):
        phase_weighted_positions(phases, hard_share=0.1, soft_share=0.3)


def test_resample_advanced_positions_preserves_monotone_endpoints() -> None:
    values = resample_advanced_positions((0.0, 0.1, 0.65, 1.0), 7)

    assert len(values) == 8
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(left < right for left, right in zip(values, values[1:]))


def test_sigma_resolution_uses_baseline_curve_and_exact_endpoints() -> None:
    phases = PoseGatePhases(hard_steps=1, soft_steps=1, normal_steps=2)
    resolved = resolve_sigma_schedule(
        baseline_sigmas=(10.0, 8.0, 5.0, 2.0, 0.0),
        phases=phases,
        request=SigmaScheduleRequest(
            mode=SigmaScheduleMode.PHASE_WEIGHTED,
            hard_share=0.15,
            soft_share=0.25,
        ),
    )

    assert resolved.resolved_sigmas[0] == 10.0
    assert resolved.resolved_sigmas[-1] == 0.0
    assert all(
        left > right
        for left, right in zip(
            resolved.resolved_sigmas, resolved.resolved_sigmas[1:]
        )
    )
    assert resolved.normalized_positions[2] == pytest.approx(0.40)


def test_automatic_sigma_mode_is_exact_baseline_parity() -> None:
    phases = PoseGatePhases(hard_steps=2, soft_steps=0, normal_steps=2)
    baseline = (12.0, 9.0, 5.0, 1.0, 0.0)
    resolved = resolve_sigma_schedule(
        baseline_sigmas=baseline,
        phases=phases,
        request=SigmaScheduleRequest(),
    )

    assert resolved.resolved_sigmas == baseline


def test_controller_stays_on_transition_across_repeated_model_evaluations() -> None:
    controller = PoseGateController(
        phases=PoseGatePhases(1, 1, 1),
        soft_schedule=SoftGateSchedule.COSINE,
    )
    support = 0.25

    assert controller.denoise_mask(support) == pytest.approx(0.25)
    assert controller.denoise_mask(support) == pytest.approx(0.25)
    assert controller.current_transition == 0
    controller.mark_transition_complete(0)
    assert 0.25 < controller.denoise_mask(support) < 1.0
    with pytest.raises(PoseGateRuntimeError, match="out of sequence"):
        controller.mark_transition_complete(0)


def test_schedule_rejects_nonfinite_knots() -> None:
    phases = PoseGatePhases(1, 1, 1)
    with pytest.raises(SigmaScheduleError, match="finite"):
        resolve_sigma_schedule(
            baseline_sigmas=(3.0, 2.0, 1.0, 0.0),
            phases=phases,
            request=SigmaScheduleRequest(
                mode=SigmaScheduleMode.ADVANCED,
                normalized_knots=(0.0, math.nan, 0.8, 1.0),
            ),
        )


def test_regional_binding_blends_ownership_back_to_normal_field() -> None:
    controller = PoseGateController(
        phases=PoseGatePhases(1, 1, 1),
        soft_schedule=SoftGateSchedule.LINEAR,
    )
    binding = PoseGateRegionBinding(
        controller=controller,
        hard_image_fields={"subject": (1.0, 0.0, 0.0)},
    )
    normal = (1.0, 1.0, 0.5)

    assert binding.effective_field("subject", normal) == (1.0, 0.0, 0.0)
    assert binding.effective_field("ordinary-region", normal) == normal
    controller.mark_transition_complete(0)
    assert binding.effective_field("subject", normal) == pytest.approx(
        (1.0, 0.5, 0.25)
    )
    controller.mark_transition_complete(1)
    assert binding.effective_field("subject", normal) == normal


def test_gate_strengths_match_shared_browser_fixtures() -> None:
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures" / "pose_gate_strengths.json").read_text()
    )
    for fixture in fixtures:
        phases = PoseGatePhases(
            fixture["hard"], fixture["soft"], fixture["normal"]
        )
        assert gate_strengths(phases, fixture["schedule"]) == pytest.approx(
            fixture["values"]
        )


def test_comfy_baseline_sigma_helper_uses_installed_ksampler_path() -> None:
    calls = []

    class FakeModel:
        load_device = "cuda"
        model_options = {"example": True}

    class FakeKSampler:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))
            self.sigmas = "installed-comfy-sigmas"

    class FakeSamplers:
        KSampler = FakeKSampler

    result = build_comfy_baseline_sigmas(
        model=FakeModel(),
        steps=12,
        sampler_name="dpm_2",
        scheduler="simple",
        comfy_samplers=FakeSamplers,
    )

    assert result == "installed-comfy-sigmas"
    assert calls[0][1] == {
        "steps": 12,
        "device": "cuda",
        "sampler": "dpm_2",
        "scheduler": "simple",
        "denoise": 1.0,
        "model_options": {"example": True},
    }
