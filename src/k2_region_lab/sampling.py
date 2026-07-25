from __future__ import annotations

import math
from typing import Any


# Ordered to match ComfyUI's KSampler registries as of 2026-07-18. The worker
# validates the selected values against the installed ComfyUI runtime again
# before generation, so an environment mismatch produces a clear error.
COMFYUI_SAMPLERS = (
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)

COMFYUI_SCHEDULERS = (
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
)

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"


def validate_sampler(value: str) -> str:
    sampler = str(value)
    if sampler not in COMFYUI_SAMPLERS:
        raise ValueError(f"unsupported ComfyUI sampler: {sampler!r}")
    return sampler


def validate_scheduler(value: str) -> str:
    scheduler = str(value)
    if scheduler not in COMFYUI_SCHEDULERS:
        raise ValueError(f"unsupported ComfyUI scheduler: {scheduler!r}")
    return scheduler


def bong_tangent_sigmas(
    steps: int,
    *,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
) -> tuple[float, ...]:
    """Return the two-stage tangent sigma schedule used by bong_tangent."""

    if steps < 1:
        raise ValueError("bong_tangent steps must be positive")
    if not math.isfinite(sigma_min) or not math.isfinite(sigma_max):
        raise ValueError("bong_tangent sigma bounds must be finite")
    if sigma_min < 0.0 or sigma_max <= sigma_min:
        raise ValueError("bong_tangent requires 0 <= sigma_min < sigma_max")
    if steps == 1:
        return (sigma_max, 0.0)

    # Independently adapted from stable-diffusion.cpp's MIT-licensed
    # BongTangentScheduler (Copyright 2023 leejet):
    # https://github.com/leejet/stable-diffusion.cpp
    # Its defaults reproduce RES4LYF's named schedule.
    expanded_steps = steps + 2
    pivot_fraction = 0.6
    midpoint = int(expanded_steps * pivot_fraction)
    pivot = int(expanded_steps * pivot_fraction)
    slope = 0.2 / (expanded_steps / 40.0)
    stage_one_length = midpoint
    stage_two_length = expanded_steps - midpoint
    middle = sigma_min + (sigma_max - sigma_min) * 0.5

    def tangent_stage(
        length: int,
        stage_pivot: float,
        start: float,
        end: float,
    ) -> list[float]:
        maximum = (
            (2.0 / math.pi) * math.atan(-slope * (0.0 - stage_pivot)) + 1.0
        ) * 0.5
        minimum = (
            (2.0 / math.pi)
            * math.atan(-slope * (float(length - 1) - stage_pivot))
            + 1.0
        ) * 0.5
        value_range = maximum - minimum
        if math.isclose(value_range, 0.0, abs_tol=1e-8):
            return [
                start + (end - start) * index / max(1, length - 1)
                for index in range(length)
            ]
        scale = start - end
        return [
            (
                (
                    (
                        (2.0 / math.pi)
                        * math.atan(-slope * (float(index) - stage_pivot))
                        + 1.0
                    )
                    * 0.5
                    - minimum
                )
                / value_range
            )
            * scale
            + end
            for index in range(length)
        ]

    first = tangent_stage(stage_one_length, float(pivot), sigma_max, middle)
    second = tangent_stage(
        stage_two_length,
        float(pivot - stage_one_length),
        middle,
        sigma_min,
    )
    result = (first[:-1] + second)[: steps + 1]
    if len(result) < steps + 1:
        result.extend([sigma_min] * (steps + 1 - len(result)))
    result[-1] = 0.0
    return tuple(result)


def register_bong_tangent_scheduler(comfy_samplers: Any, torch_module: Any) -> None:
    """Install bong_tangent into the pinned ComfyUI scheduler registry once."""

    name = "bong_tangent"
    handlers = comfy_samplers.SCHEDULER_HANDLERS
    if name not in handlers:

        def schedule(model_sampling: Any, steps: int):
            values = bong_tangent_sigmas(
                steps,
                sigma_min=float(model_sampling.sigma_min),
                sigma_max=float(model_sampling.sigma_max),
            )
            return torch_module.tensor(values, dtype=torch_module.float32)

        handlers[name] = comfy_samplers.SchedulerHandler(schedule)

    names = comfy_samplers.SCHEDULER_NAMES
    if name not in names:
        names.append(name)
    runtime_names = comfy_samplers.KSampler.SCHEDULERS
    if name not in runtime_names:
        runtime_names.append(name)
