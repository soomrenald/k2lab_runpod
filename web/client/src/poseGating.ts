import type { PoseSoftRelease } from "./studioProject";

export function poseGateStrengths(
  hardSteps: number,
  softSteps: number,
  normalSteps: number,
  schedule: PoseSoftRelease,
): number[] {
  const values = Array(Math.max(0, hardSteps)).fill(1);
  for (let index = 0; index < softSteps; index += 1) {
    const t = (index + 1) / (softSteps + 1);
    let value: number;
    if (schedule === "linear") value = 1 - t;
    else if (schedule === "exponential") value = 1 - (Math.exp(4 * t) - 1) / (Math.exp(4) - 1);
    else if (schedule === "stepped") value = t < 1 / 3 ? 0.75 : t < 2 / 3 ? 0.5 : 0.25;
    else value = 0.5 * (1 + Math.cos(Math.PI * t));
    values.push(value);
  }
  return [...values, ...Array(Math.max(0, normalSteps)).fill(0)];
}

export function automaticPoseKnots(transitions: number): number[] {
  return Array.from({ length: transitions + 1 }, (_, index) => index / transitions);
}

export function resamplePoseKnots(values: number[], transitions: number): number[] {
  if (transitions < 1) return [0, 1];
  if (values.length < 2) return automaticPoseKnots(transitions);
  const sourceSteps = values.length - 1;
  const result = Array.from({ length: transitions + 1 }, (_, index) => {
    const scaled = index / transitions * sourceSteps;
    const lower = Math.min(Math.floor(scaled), sourceSteps - 1);
    const fraction = scaled - lower;
    return values[lower] + fraction * (values[lower + 1] - values[lower]);
  });
  result[0] = 0;
  result[result.length - 1] = 1;
  return result;
}

export function poseKnotPhase(
  boundary: number,
  hardSteps: number,
  softSteps: number,
): "hard" | "soft" | "normal" | "complete" {
  if (boundary < hardSteps) return "hard";
  if (boundary < hardSteps + softSteps) return "soft";
  return "normal";
}
