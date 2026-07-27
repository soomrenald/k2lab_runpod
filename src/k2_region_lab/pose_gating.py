from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Sequence


class PoseGatingError(RuntimeError):
    pass


class PoseMaskBuildError(PoseGatingError):
    pass


class PoseGateScheduleError(PoseGatingError):
    pass


class SigmaScheduleError(PoseGatingError):
    pass


class PoseGateRuntimeError(PoseGatingError):
    pass


class SoftGateSchedule(StrEnum):
    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEPPED = "stepped"


class SigmaScheduleMode(StrEnum):
    AUTOMATIC = "automatic"
    PHASE_WEIGHTED = "phase_weighted"
    ADVANCED = "advanced"


@dataclass(frozen=True, slots=True)
class PoseGatePhases:
    hard_steps: int
    soft_steps: int
    normal_steps: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int)
            for value in (self.hard_steps, self.soft_steps, self.normal_steps)
        ):
            raise PoseGateScheduleError("pose gate step counts must be integers")
        if self.hard_steps < 0 or self.soft_steps < 0 or self.normal_steps < 0:
            raise PoseGateScheduleError("pose gate step counts must not be negative")
        if not 1 <= self.effective_steps <= 100:
            raise PoseGateScheduleError("effective step count must be between 1 and 100")

    @property
    def effective_steps(self) -> int:
        return self.hard_steps + self.soft_steps + self.normal_steps

    def phase(self, transition: int) -> str:
        if not 0 <= transition < self.effective_steps:
            raise PoseGateScheduleError("transition index is outside the pose gate schedule")
        if transition < self.hard_steps:
            return "hard"
        if transition < self.hard_steps + self.soft_steps:
            return "soft"
        return "normal"


@dataclass(frozen=True, slots=True)
class SigmaScheduleRequest:
    mode: SigmaScheduleMode = SigmaScheduleMode.AUTOMATIC
    hard_share: float = 0.20
    soft_share: float = 0.30
    normalized_knots: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedSigmaSchedule:
    baseline_sigmas: tuple[float, ...]
    normalized_positions: tuple[float, ...]
    resolved_sigmas: tuple[float, ...]
    mode: SigmaScheduleMode
    phase_shares: dict[str, float]


def soft_gate_strength(
    soft_index: int,
    soft_steps: int,
    schedule: SoftGateSchedule | str,
) -> float:
    if soft_steps <= 0 or not 0 <= soft_index < soft_steps:
        raise PoseGateScheduleError("soft transition index is outside the soft phase")
    schedule = SoftGateSchedule(schedule)
    t = (soft_index + 1) / (soft_steps + 1)
    if schedule == SoftGateSchedule.COSINE:
        value = 0.5 * (1.0 + math.cos(math.pi * t))
    elif schedule == SoftGateSchedule.LINEAR:
        value = 1.0 - t
    elif schedule == SoftGateSchedule.EXPONENTIAL:
        value = 1.0 - (math.exp(4.0 * t) - 1.0) / (math.exp(4.0) - 1.0)
    else:
        value = 0.75 if t < 1 / 3 else 0.50 if t < 2 / 3 else 0.25
    if not 0.0 < value < 1.0:
        raise PoseGateScheduleError("soft gate strength must remain strictly between zero and one")
    return value


def gate_strengths(
    phases: PoseGatePhases,
    schedule: SoftGateSchedule | str = SoftGateSchedule.COSINE,
) -> tuple[float, ...]:
    return (
        (1.0,) * phases.hard_steps
        + tuple(
            soft_gate_strength(index, phases.soft_steps, schedule)
            for index in range(phases.soft_steps)
        )
        + (0.0,) * phases.normal_steps
    )


def automatic_positions(phases: PoseGatePhases) -> tuple[float, ...]:
    total = phases.effective_steps
    return tuple(index / total for index in range(total + 1))


def _validate_share(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SigmaScheduleError(f"{name} trajectory share must be finite and between 0 and 1")
    return value


def phase_weighted_positions(
    phases: PoseGatePhases,
    *,
    hard_share: float,
    soft_share: float,
) -> tuple[float, ...]:
    hard_share = _validate_share(hard_share, "hard")
    soft_share = _validate_share(soft_share, "soft")
    if phases.hard_steps == 0 and hard_share != 0.0:
        raise SigmaScheduleError("a zero-step hard phase must have zero trajectory share")
    if phases.soft_steps == 0 and soft_share != 0.0:
        raise SigmaScheduleError("a zero-step soft phase must have zero trajectory share")
    normal_share = 1.0 - hard_share - soft_share
    if phases.normal_steps and normal_share <= 0.0:
        raise SigmaScheduleError("normal trajectory share must be positive")
    if phases.hard_steps and hard_share <= 0.0:
        raise SigmaScheduleError("a non-empty hard phase needs positive trajectory share")
    if phases.soft_steps and soft_share <= 0.0:
        raise SigmaScheduleError("a non-empty soft phase needs positive trajectory share")

    values = [0.0]
    phase_specs = (
        (phases.hard_steps, 0.0, hard_share),
        (phases.soft_steps, hard_share, hard_share + soft_share),
        (phases.normal_steps, hard_share + soft_share, 1.0),
    )
    for count, start, end in phase_specs:
        if count == 0:
            continue
        values.extend(start + (end - start) * index / count for index in range(1, count + 1))
    positions = tuple(values)
    validate_normalized_positions(positions, phases.effective_steps)
    return positions


def validate_normalized_positions(
    positions: Sequence[float],
    effective_steps: int,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in positions)
    if len(values) != effective_steps + 1:
        raise SigmaScheduleError("sigma schedule must have one knot per transition boundary")
    if not all(math.isfinite(value) for value in values):
        raise SigmaScheduleError("sigma schedule knots must be finite")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise SigmaScheduleError("sigma schedule endpoints must be exactly zero and one")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise SigmaScheduleError("sigma schedule knots must be strictly increasing")
    return values


def resample_advanced_positions(
    positions: Sequence[float],
    effective_steps: int,
) -> tuple[float, ...]:
    if effective_steps < 1:
        raise SigmaScheduleError("effective step count must be positive")
    source = tuple(float(value) for value in positions)
    if len(source) < 2:
        return tuple(index / effective_steps for index in range(effective_steps + 1))
    validate_normalized_positions(source, len(source) - 1)
    source_steps = len(source) - 1
    result = []
    for index in range(effective_steps + 1):
        position = index / effective_steps
        scaled = position * source_steps
        lower = min(int(math.floor(scaled)), source_steps - 1)
        fraction = scaled - lower
        value = source[lower] + fraction * (source[lower + 1] - source[lower])
        result.append(value)
    result[0] = 0.0
    result[-1] = 1.0
    return validate_normalized_positions(result, effective_steps)


def _interpolate_baseline(
    baseline: tuple[float, ...],
    positions: tuple[float, ...],
) -> tuple[float, ...]:
    steps = len(baseline) - 1
    result: list[float] = []
    for position in positions:
        scaled = position * steps
        lower = min(int(math.floor(scaled)), steps - 1)
        fraction = scaled - lower
        result.append(baseline[lower] + fraction * (baseline[lower + 1] - baseline[lower]))
    result[0] = baseline[0]
    result[-1] = 0.0
    return tuple(result)


def validate_sigma_schedule(
    baseline_sigmas: Sequence[float],
    normalized_positions: Sequence[float],
    resolved_sigmas: Sequence[float],
    phases: PoseGatePhases,
) -> None:
    baseline = tuple(float(value) for value in baseline_sigmas)
    resolved = tuple(float(value) for value in resolved_sigmas)
    positions = validate_normalized_positions(
        normalized_positions, phases.effective_steps
    )
    expected = phases.effective_steps + 1
    if len(baseline) != expected or len(resolved) != expected:
        raise SigmaScheduleError("baseline and resolved sigma arrays must contain N + 1 values")
    if not all(math.isfinite(value) for value in (*baseline, *resolved, *positions)):
        raise SigmaScheduleError("sigma schedule contains a non-finite value")
    if baseline[-1] != 0.0:
        raise SigmaScheduleError("baseline sigma schedule must end at exactly zero")
    if resolved[0] != baseline[0] or resolved[-1] != 0.0:
        raise SigmaScheduleError("resolved sigma endpoints do not match the baseline")
    if any(left <= right for left, right in zip(resolved, resolved[1:])):
        raise SigmaScheduleError("resolved sigmas must be strictly decreasing")
    low, high = min(baseline), max(baseline)
    if any(not low <= value <= high for value in resolved):
        raise SigmaScheduleError("resolved sigma lies outside baseline bounds")


def resolve_sigma_schedule(
    *,
    baseline_sigmas: Sequence[float],
    phases: PoseGatePhases,
    request: SigmaScheduleRequest,
) -> ResolvedSigmaSchedule:
    baseline = tuple(float(value) for value in baseline_sigmas)
    if len(baseline) != phases.effective_steps + 1:
        raise SigmaScheduleError("baseline sigma schedule must contain N + 1 values")
    mode = SigmaScheduleMode(request.mode)
    if mode == SigmaScheduleMode.AUTOMATIC:
        positions = automatic_positions(phases)
        resolved = baseline
        hard_share = phases.hard_steps / phases.effective_steps
        soft_share = phases.soft_steps / phases.effective_steps
    elif mode == SigmaScheduleMode.PHASE_WEIGHTED:
        positions = phase_weighted_positions(
            phases,
            hard_share=request.hard_share,
            soft_share=request.soft_share,
        )
        resolved = _interpolate_baseline(baseline, positions)
        hard_share = request.hard_share
        soft_share = request.soft_share
    else:
        positions = validate_normalized_positions(
            request.normalized_knots, phases.effective_steps
        )
        resolved = _interpolate_baseline(baseline, positions)
        hard_share = positions[phases.hard_steps]
        soft_share = (
            positions[phases.hard_steps + phases.soft_steps] - hard_share
        )
    validate_sigma_schedule(baseline, positions, resolved, phases)
    return ResolvedSigmaSchedule(
        baseline_sigmas=baseline,
        normalized_positions=positions,
        resolved_sigmas=resolved,
        mode=mode,
        phase_shares={
            "hard": float(hard_share),
            "soft": float(soft_share),
            "normal": float(1.0 - hard_share - soft_share),
        },
    )


class PoseGateController:
    def __init__(
        self,
        *,
        phases: PoseGatePhases,
        soft_schedule: SoftGateSchedule | str,
        resolved_sigmas: ResolvedSigmaSchedule | None = None,
    ) -> None:
        self.phases = phases
        self.soft_schedule = SoftGateSchedule(soft_schedule)
        self.strengths = gate_strengths(phases, self.soft_schedule)
        self.resolved_sigmas = resolved_sigmas
        self.current_transition = 0
        self.current_sigma: float | None = None

    @property
    def gate_strength(self) -> float:
        if self.current_transition >= self.phases.effective_steps:
            return 0.0
        return self.strengths[self.current_transition]

    @property
    def phase(self) -> str:
        if self.current_transition >= self.phases.effective_steps:
            return "complete"
        return self.phases.phase(self.current_transition)

    def observe_sigma(self, sigma: Any) -> None:
        try:
            value = float(sigma.flatten()[0].item())
        except AttributeError:
            value = float(sigma)
        if math.isfinite(value):
            self.current_sigma = value

    def denoise_mask(self, support_mask: Any) -> Any:
        strength = self.gate_strength
        return 1.0 - strength * (1.0 - support_mask)

    def mark_transition_complete(self, transition: int) -> None:
        if transition != self.current_transition:
            raise PoseGateRuntimeError(
                "pose gate transition callback arrived out of sequence"
            )
        self.current_transition += 1
