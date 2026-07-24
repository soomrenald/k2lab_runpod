from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import torch


REFERENCE_REVISION = "9170f1fadffa4d380601e3c88ac0e982c09e88d8"


def git_show(repository: Path, revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{revision}:{path}"],
        text=True,
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RunPod regional behavior with the approved PySide6 backend."
    )
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "krea_region_project",
    )
    parser.add_argument("--reference-revision", default=REFERENCE_REVISION)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reference_root = arguments.reference_repo.expanduser().resolve()
    sys.path.insert(0, str(project_root / "src"))

    from k2_region_lab.regional_prompting import compile_regional_prompt_plan
    from k2_region_lab.regions import PixelBox, RegionDefinition
    from k2_region_lab.spatial_attention import KreaSpatialAttentionOverride

    legacy_source = git_show(
        reference_root,
        arguments.reference_revision,
        "src/k2_region_lab/spatial_attention.py",
    )
    legacy_module = types.ModuleType("k2_legacy_spatial_attention")
    exec(compile(legacy_source, "legacy/spatial_attention.py", "exec"), legacy_module.__dict__)

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
    current = KreaSpatialAttentionOverride(bound)
    legacy = legacy_module.KreaSpatialAttentionOverride(bound)
    total = bound.text_token_count + bound.image_token_count
    reference_tensor = torch.zeros((1, 1, total, 1))
    current_fields = current._pair_fields(reference_tensor)
    legacy_fields = legacy._pair_fields(reference_tensor)

    seed_scores = torch.arange(total * total, dtype=torch.float32).reshape(
        1, 1, total, total
    )
    current_scores = seed_scores.clone()
    legacy_scores = seed_scores.clone()
    current._partition_regional_stream(
        current_scores,
        0,
        total,
        current_fields[-2],
        current_fields[-1],
    )
    legacy._partition_regional_stream(
        legacy_scores,
        0,
        total,
        legacy_fields[-2],
        legacy_fields[-1],
    )
    text_count = bound.text_token_count
    image_scores_unchanged = torch.equal(
        current_scores[:, :, text_count:, text_count:],
        seed_scores[:, :, text_count:, text_count:],
    )

    source_matches = {}
    for filename in ("regional_prompting.py", "regional_lora.py"):
        relative = f"src/k2_region_lab/{filename}"
        current_source = (project_root / relative).read_text(encoding="utf-8")
        reference_source = git_show(
            reference_root,
            arguments.reference_revision,
            relative,
        )
        source_matches[filename] = {
            "exact": current_source == reference_source,
            "current_sha256": digest(current_source),
            "reference_sha256": digest(reference_source),
        }

    report = {
        "reference_repository": str(reference_root),
        "reference_revision": arguments.reference_revision,
        "regional_sources": source_matches,
        "ownership_exact": (
            current.text_owners == legacy.text_owners
            and current.image_owners == legacy.image_owners
        ),
        "main_stream_partition_exact": bool(torch.equal(current_scores, legacy_scores)),
        "image_to_image_scores_unchanged": bool(image_scores_unchanged),
        "current_summary": current.summary(),
    }
    report["passed"] = bool(
        all(item["exact"] for item in source_matches.values())
        and report["ownership_exact"]
        and report["main_stream_partition_exact"]
        and report["image_to_image_scores_unchanged"]
        and report["current_summary"]["image_to_image_attention"] == "unmodified"
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit("regional reference comparison failed")


if __name__ == "__main__":
    main()
