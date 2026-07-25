from __future__ import annotations

import unittest

from k2_region_lab.sampling import (
    COMFYUI_SAMPLERS,
    COMFYUI_SCHEDULERS,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    bong_tangent_sigmas,
    register_bong_tangent_scheduler,
    validate_sampler,
    validate_scheduler,
)


class SamplingOptionsTests(unittest.TestCase):
    def test_current_comfyui_standard_lists_include_recent_options(self) -> None:
        self.assertEqual(COMFYUI_SAMPLERS[0], "euler")
        self.assertIn("euler_cfg_pp", COMFYUI_SAMPLERS)
        self.assertIn("exp_heun_2_x0_sde", COMFYUI_SAMPLERS)
        self.assertIn("res_multistep_ancestral_cfg_pp", COMFYUI_SAMPLERS)
        self.assertIn("sa_solver_pece", COMFYUI_SAMPLERS)
        self.assertEqual(COMFYUI_SAMPLERS[-3:], ("ddim", "uni_pc", "uni_pc_bh2"))
        self.assertEqual(
            COMFYUI_SCHEDULERS,
            (
                "simple",
                "sgm_uniform",
                "karras",
                "exponential",
                "ddim_uniform",
                "beta",
                "normal",
                "linear_quadratic",
                "kl_optimal",
                "bong_tangent",
            ),
        )

    def test_defaults_and_validation(self) -> None:
        self.assertEqual(validate_sampler(DEFAULT_SAMPLER), "euler")
        self.assertEqual(validate_scheduler(DEFAULT_SCHEDULER), "simple")
        with self.assertRaisesRegex(ValueError, "unsupported ComfyUI sampler"):
            validate_sampler("not-a-sampler")
        with self.assertRaisesRegex(ValueError, "unsupported ComfyUI scheduler"):
            validate_scheduler("not-a-scheduler")

    def test_bong_tangent_schedule_is_complete_and_monotonic(self) -> None:
        sigmas = bong_tangent_sigmas(8, sigma_min=0.01, sigma_max=1.0)

        self.assertEqual(len(sigmas), 9)
        self.assertAlmostEqual(sigmas[0], 1.0)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertTrue(
            all(current > following for current, following in zip(sigmas, sigmas[1:]))
        )
        self.assertEqual(bong_tangent_sigmas(1), (1.0, 0.0))

    def test_bong_tangent_registers_with_comfyui_once(self) -> None:
        class Handler:
            def __init__(self, handler):
                self.handler = handler
                self.use_ms = True

        class KSampler:
            SCHEDULERS: list[str] = []

        class ComfySamplers:
            SchedulerHandler = Handler
            SCHEDULER_HANDLERS: dict[str, Handler] = {}
            SCHEDULER_NAMES: list[str] = []

        ComfySamplers.KSampler = KSampler

        class Torch:
            float32 = "float32"

            @staticmethod
            def tensor(values, *, dtype):
                return tuple(values), dtype

        class ModelSampling:
            sigma_min = 0.01
            sigma_max = 1.0

        register_bong_tangent_scheduler(ComfySamplers, Torch)
        register_bong_tangent_scheduler(ComfySamplers, Torch)

        self.assertEqual(ComfySamplers.SCHEDULER_NAMES, ["bong_tangent"])
        self.assertEqual(KSampler.SCHEDULERS, ["bong_tangent"])
        values, dtype = ComfySamplers.SCHEDULER_HANDLERS[
            "bong_tangent"
        ].handler(ModelSampling(), 8)
        self.assertEqual(len(values), 9)
        self.assertEqual(dtype, "float32")


if __name__ == "__main__":
    unittest.main()
