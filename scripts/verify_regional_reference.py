from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import torch


def repository_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare restored regional isolation against the original implementation."
    )
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "krea_reg_lora",
    )
    parser.add_argument(
        "--desktop-reference-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "krea_region_project",
    )
    parser.add_argument(
        "--desktop-reference-revision",
        default="e183b89^",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reference_root = arguments.reference_repo.expanduser().resolve()
    desktop_reference_root = arguments.desktop_reference_repo.expanduser().resolve()
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(reference_root))

    from k2_region_lab.regional_lora import LoraDeltaRoute
    from k2_region_lab.regional_prompting import compile_regional_prompt_plan
    from k2_region_lab.regions import PixelBox, RegionDefinition
    from k2_region_lab.spatial_attention import KreaSpatialAttentionOverride
    from k2_region_lab.worker.runtime import LoraDeltaStatistics
    from krea_region_lora.masks import region_from_bbox
    from krea_region_lora.tracking import RegionalRuntimeState
    from krea_region_lora.types import KreaRegionalLora, KreaRegionalLoraStack

    text_count = 4
    sequence_length = 20
    hidden = 4
    reference = torch.ones((1, sequence_length, hidden))
    regions = (
        region_from_bbox(
            [[{"x": 0, "y": 0, "width": 16, "height": 16}]],
            width=64,
            height=64,
            feather_px=0,
            snap_to_krea_token_grid=False,
        ),
        region_from_bbox(
            [[{"x": 32, "y": 32, "width": 16, "height": 16}]],
            width=64,
            height=64,
            feather_px=0,
            snap_to_krea_token_grid=False,
        ),
    )
    reference_loras = tuple(
        KreaRegionalLora(
            region=region,
            lora_name=name,
            threshold=0.05,
        )
        for region, name in zip(regions, ("left", "right"), strict=True)
    )
    reference_state = RegionalRuntimeState(
        KreaRegionalLoraStack(
            reference_loras,
            attention_isolation_strength=5.0,
            cross_lora_mode="penalize",
            cross_lora_strength=3.0,
        )
    )
    restored_routes = tuple(
        LoraDeltaRoute(
            lora_id=name,
            display_name=name,
            strength=1.0,
            global_scope=False,
            region_ids=(name,),
            region_names=(name,),
            text_token_mask=(0.0,) * text_count,
            image_token_mask=tuple(
                float(value) for value in region.token_mask.reshape(-1).tolist()
            ),
        )
        for region, name in zip(regions, ("left", "right"), strict=True)
    )
    restored_state = LoraDeltaStatistics(restored_routes)

    flag_matches = []
    for reference_lora, restored_route in zip(
        reference_loras,
        restored_routes,
        strict=True,
    ):
        delta = torch.zeros_like(reference)
        image_mask = reference_lora.region.token_mask.reshape(1, -1, 1)
        delta[:, text_count:] = image_mask * 0.20
        reference_flags = reference_state.update_from_delta(
            reference_lora,
            delta,
            reference,
        )
        restored_state.observe(
            restored_route,
            torch.linalg.vector_norm(delta, dim=-1),
            route_kind="combined",
            reference_norms=torch.linalg.vector_norm(reference, dim=-1),
        )
        restored_flags = restored_state.values[restored_route.lora_id]["modified_image_flags"]
        flag_matches.append(bool(torch.equal(reference_flags, restored_flags)))

    reference_bias = reference_state.build_attention_bias(
        batch=1,
        q_len=sequence_length,
        k_len=sequence_length,
        heads=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    restored_bias = torch.zeros((1, 1, sequence_length, sequence_length))
    restored_state.apply_asymmetric_attention(
        restored_bias,
        start=0,
        end=sequence_length,
        text_token_count=text_count,
    )
    reference_bias = reference_bias.reshape_as(restored_bias)
    difference = (reference_bias - restored_bias).abs()
    desktop_source = subprocess.check_output(
        [
            "git",
            "-C",
            str(desktop_reference_root),
            "show",
            (f"{arguments.desktop_reference_revision}:src/k2_region_lab/spatial_attention.py"),
        ],
        text=True,
    )
    desktop_module = types.ModuleType("k2_desktop_spatial_attention_reference")
    exec(
        compile(desktop_source, "desktop_reference/spatial_attention.py", "exec"),
        desktop_module.__dict__,
    )
    plan = compile_regional_prompt_plan(
        32,
        16,
        "portrait",
        (
            RegionDefinition(
                "left",
                "Left",
                PixelBox(0, 0, 16, 16),
                "red coat",
            ),
            RegionDefinition(
                "right",
                "Right",
                PixelBox(16, 0, 32, 16),
                "blue coat",
            ),
        ),
        falloff_pixels=0,
    )
    bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
    current_override = KreaSpatialAttentionOverride(bound)
    desktop_override = desktop_module.KreaSpatialAttentionOverride(bound)
    attention_reference = torch.zeros(
        (1, 1, bound.text_token_count + 2, bound.text_token_count + 2)
    )
    attention_restored = attention_reference.clone()
    reference_tensor = torch.zeros((1, 1, bound.text_token_count + 2, 1))
    *_, desktop_owners = desktop_override._pair_fields(reference_tensor)
    *_, current_owners = current_override._pair_fields(reference_tensor)
    desktop_override._partition_regional_stream(
        attention_reference,
        0,
        bound.text_token_count + 2,
        desktop_owners,
    )
    current_override._partition_regional_stream(
        attention_restored,
        0,
        bound.text_token_count + 2,
        current_owners,
    )
    desktop_revision = subprocess.check_output(
        [
            "git",
            "-C",
            str(desktop_reference_root),
            "rev-parse",
            arguments.desktop_reference_revision,
        ],
        text=True,
    ).strip()
    report = {
        "reference_repository": str(reference_root),
        "reference_commit": repository_commit(reference_root),
        "restored_repository": str(project_root),
        "restored_commit": repository_commit(project_root),
        "modified_token_flags_exact": all(flag_matches),
        "per_route_flag_matches": flag_matches,
        "attention_bias_exact": bool(torch.equal(reference_bias, restored_bias)),
        "attention_bias_max_abs_difference": float(difference.max().item()),
        "reference_nonzero_bias_entries": int(torch.count_nonzero(reference_bias)),
        "restored_nonzero_bias_entries": int(torch.count_nonzero(restored_bias)),
        "desktop_attention_reference_repository": str(desktop_reference_root),
        "desktop_attention_reference_commit": desktop_revision,
        "desktop_hard_partition_exact": bool(torch.equal(attention_reference, attention_restored)),
    }
    report["passed"] = bool(
        report["modified_token_flags_exact"]
        and report["attention_bias_exact"]
        and report["desktop_hard_partition_exact"]
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
