from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from k2core.depth import EffectiveDepthField
from k2_region_lab.depth import runtime as depth_runtime
from k2_region_lab.depth.runtime import DepthScheduleController


def _field(value: float) -> EffectiveDepthField:
    return EffectiveDepthField(
        pixel_values=np.full((16, 16), value, dtype=np.float32),
        image_token_values=np.full((1, 1), value, dtype=np.float32),
        region_multipliers={},
    )


def test_depth_schedule_advances_after_each_completed_sampler_transition() -> None:
    controller = DepthScheduleController((_field(1.0), _field(0.5), _field(0.0)))

    assert controller.current_values().tolist() == [1.0]
    controller.advance_after(0, 3)
    assert controller.current_values().tolist() == [0.5]
    controller.advance_after(1, 3)
    assert controller.current_values().tolist() == [0.0]
    controller.advance_after(2, 3)
    assert controller.current_values().tolist() == [0.0]


def test_depth_schedule_rejects_callback_divergence() -> None:
    controller = DepthScheduleController((_field(1.0), _field(0.0)))
    with pytest.raises(RuntimeError, match="step counts"):
        controller.advance_after(0, 3)
    with pytest.raises(RuntimeError, match="callback"):
        controller.advance_after(1, 2)


def test_depth_encode_uses_inference_lifecycle(monkeypatch) -> None:
    state = {"inference": False}

    class FakeTensor:
        def to(self, **_kwargs):
            return self

        def unsqueeze(self, _dimension):
            return self

    class InferenceMode:
        def __enter__(self):
            state["inference"] = True

        def __exit__(self, *_args):
            state["inference"] = False

    class FakeVae:
        def encode(self, _pixels):
            assert state["inference"]
            return "latent"

    fake_torch = SimpleNamespace(
        float32="float32",
        from_numpy=lambda _values: FakeTensor(),
        inference_mode=InferenceMode,
    )
    monkeypatch.setattr(depth_runtime, "torch", fake_torch)
    monkeypatch.setattr(
        depth_runtime,
        "_process_control_latent_for_model",
        lambda _model, latent: latent,
    )

    result = depth_runtime.encode_depth_control(
        FakeVae(),
        object(),
        SimpleNamespace(resized_values=np.ones((16, 16), dtype=np.float32)),
    )

    assert result.full == "latent"
    assert state["inference"] is False
