from __future__ import annotations

import numpy as np
import pytest

from k2core.depth import EffectiveDepthField
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
