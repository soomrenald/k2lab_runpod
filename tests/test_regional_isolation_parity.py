from __future__ import annotations

import importlib.util
import unittest

from k2_region_lab.regional_prompting import compile_regional_prompt_plan
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.spatial_attention import KreaSpatialAttentionOverride


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is tested in the ComfyUI worker environment")
class RegionalIsolationParityTests(unittest.TestCase):
    def _override(self) -> KreaSpatialAttentionOverride:
        plan = compile_regional_prompt_plan(
            32,
            16,
            "a unified portrait",
            (
                RegionDefinition("left", "Left", PixelBox(0, 0, 16, 16), "red coat"),
                RegionDefinition("right", "Right", PixelBox(16, 0, 32, 16), "blue coat"),
            ),
            falloff_pixels=0.0,
        )
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
        return KreaSpatialAttentionOverride(bound)

    def test_main_stream_never_changes_image_to_image_scores(self) -> None:
        override = self._override()
        text_count = override.plan.text_token_count
        total = text_count + override.plan.image_token_count
        reference = torch.zeros((1, 1, total, 1))
        _fields, _emphases, text_owners, image_owners = override._pair_fields(reference)
        scores = torch.arange(total * total, dtype=torch.float32).reshape(1, 1, total, total)
        before = scores[:, :, text_count:, text_count:].clone()

        override._partition_regional_stream(
            scores,
            0,
            total,
            text_owners,
            image_owners,
        )

        self.assertTrue(torch.equal(scores[:, :, text_count:, text_count:], before))

    def test_subject_text_is_private_without_cutting_global_coherence(self) -> None:
        override = self._override()
        text_count = override.plan.text_token_count
        total = text_count + override.plan.image_token_count
        reference = torch.zeros((1, 1, total, 1))
        _fields, _emphases, text_owners, image_owners = override._pair_fields(reference)
        scores = torch.zeros((1, 1, total, total))

        override._partition_regional_stream(
            scores,
            0,
            total,
            text_owners,
            image_owners,
        )

        left, right = override.plan.spans
        left_image = text_count
        right_image = text_count + 1
        self.assertTrue(torch.isneginf(scores[0, 0, left.start, right_image]))
        self.assertTrue(torch.isneginf(scores[0, 0, right_image, left.start]))
        self.assertEqual(float(scores[0, 0, left_image, right_image]), 0.0)
        self.assertEqual(float(scores[0, 0, right_image, left_image]), 0.0)
        self.assertEqual(float(scores[0, 0, left_image, 0]), 0.0)

    def test_runtime_summary_declares_unmodified_image_attention(self) -> None:
        summary = self._override().summary()
        self.assertEqual(summary["cross_modal_partition"], "subject_text_private_to_box")
        self.assertEqual(summary["image_to_image_attention"], "unmodified")
        self.assertNotIn("lora_influence_attention", summary)
