from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    name: str
    prompt: str
    category: str


CASES = (
    EvaluationCase("standing", "a person", "standing"),
    EvaluationCase("seated", "a person", "seated"),
    EvaluationCase("crouching", "a person", "crouching"),
    EvaluationCase("leaning", "a person", "leaning"),
    EvaluationCase("asymmetric-arms", "a person", "asymmetric_arms"),
    EvaluationCase("arms-high-low", "a person", "arms_above_below_torso"),
    EvaluationCase("split-legs", "a person", "split_legs"),
    EvaluationCase("back-facing", "a person seen from behind", "back_facing"),
    EvaluationCase("cropped-subject", "a portrait of a person", "cropped"),
    EvaluationCase("two-separated", "two people", "two_people"),
    EvaluationCase("crossed-limbs", "a person", "crossed_limbs"),
    EvaluationCase("vague-prompt", "photograph", "vague_prompt"),
    EvaluationCase("empty-prompt", "", "empty_prompt"),
    EvaluationCase("unrelated-prompt", "an abstract blue composition", "unrelated_prompt"),
    EvaluationCase("pose-conflict", "a seated person", "pose_conflicting_prompt"),
    EvaluationCase("prompt-swap", "a different person", "prompt_swap"),
)
STRENGTHS = (0.0, 0.4, 0.6, 0.8, 1.0, 1.2)
MODELS = ("krea_raw", "krea_turbo")


def evaluation_plan(checkpoint: Path, controls: Path, output: Path) -> list[dict[str, Any]]:
    tasks = []
    for model in MODELS:
        for case in CASES:
            control = controls / f"{case.name}.png"
            for strength in STRENGTHS:
                tasks.append(
                    {
                        "case": case.name,
                        "category": case.category,
                        "prompt": case.prompt,
                        "control": str(control),
                        "checkpoint": str(checkpoint),
                        "model": model,
                        "adapter_enabled": strength > 0.0,
                        "strength": strength,
                        "seed": 42000,
                        "output": str(
                            output / f"{model}-{case.name}-strength-{strength:.1f}.png"
                        ),
                    }
                )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        help="Optional executable accepting one evaluation-task JSON object on stdin.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = evaluation_plan(args.checkpoint, args.controls, args.output)
    (args.output / "evaluation-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.runner:
        for task in plan:
            subprocess.run(
                [str(args.runner)],
                input=json.dumps(task),
                text=True,
                check=True,
            )
    print(
        "Evaluation plan written. Score head/torso centers, shoulder/hip orientation, "
        "visible limb-joint error, missing/extra people, prompt identity, and image quality."
    )


if __name__ == "__main__":
    main()
