from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from PIL import Image

from k2_region_lab.agent.domain import (
    FileKind,
    JobKind,
    JobState,
    JobSubmitRequest,
)
from k2_region_lab.agent.jobs import JobManager
from k2_region_lab.agent.storage import WorkspaceLayout
from k2_region_lab.agent.transfers import TransferManager
from k2_region_lab.project import PROJECT_SCHEMA, PROJECT_VERSION


TERMINAL_STATES = {
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.CANCELLED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gate 11 through the persistent RunPod job service."
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "fixtures"
            / "gate11"
            / "native_clean_generation.json"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def project_from_fixture(fixture: dict) -> dict:
    return {
        "schema": PROJECT_SCHEMA,
        "version": PROJECT_VERSION,
        "canvas": {
            "width": fixture["width"],
            "height": fixture["height"],
        },
        "generation": {
            "global_prompt": fixture["prompt"],
            "shared_visual_prompt": "",
            "steps": fixture["steps"],
            "sampler": fixture["sampler"],
            "scheduler": fixture["scheduler"],
            "seed": fixture["seed"],
            "regional_prompting": fixture["regional_prompting"],
            "regional_prompt_strength": fixture["regional_prompt_strength"],
            "regional_outside_penalty": fixture["regional_outside_penalty"],
            "regional_feather_pixels": int(
                fixture["regional_feather_pixels"]
            ),
            "regional_subject_competition": (
                fixture["regional_subject_competition"]
            ),
            "regional_subject_fill": fixture["regional_subject_fill"],
            "regional_relaxation": True,
            "regional_late_step_scale": fixture["regional_late_step_scale"],
            "regional_lora_delta_adaptation": (
                fixture["regional_lora_delta_adaptation"]
            ),
            "regional_lora_delta_adaptation_gain": (
                fixture["regional_lora_delta_adaptation_gain"]
            ),
            "prompt_emphases": fixture["prompt_emphases"],
            "pose_gating_enabled": False,
            "pose_control_lora_enabled": False,
            "projector_enabled": False,
            "post_upscale": False,
        },
        "regions": [
            {
                **region,
                "region_type": "subject",
                "pose": None,
            }
            for region in fixture["regions"]
        ],
        "loras": [],
        "image_edit": {},
        "runtime": {
            "vram_mode": "auto",
            "reserve_vram_gb": 4.0,
            "keep_model_loaded": False,
            "system_ram_guard_enabled": True,
        },
    }


def manager(
    layout: WorkspaceLayout,
    transfers: TransferManager,
    args: argparse.Namespace,
) -> JobManager:
    return JobManager(
        layout,
        transfers,
        worker_python=Path(sys.executable),
        comfyui_root=Path("/opt/ComfyUI"),
        inference_backend="native",
        native_tokenizer_path=args.tokenizer,
        native_tokenizer_sha256=args.tokenizer_sha256,
        worker_startup_timeout_seconds=args.timeout_seconds,
        generation_timeout_seconds=args.timeout_seconds,
    )


async def run_probe(args: argparse.Namespace) -> dict:
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    layout = WorkspaceLayout(args.workspace_root.expanduser().resolve())
    layout.initialize()
    transfers = TransferManager(layout)
    records = {}
    for kind, path in (
        (FileKind.DIFFUSION_MODELS, args.transformer),
        (FileKind.TEXT_ENCODERS, args.text_encoder),
        (FileKind.VAE, args.vae),
    ):
        records[kind] = await transfers.index_existing_file(kind, path)
    request = JobSubmitRequest(
        command_id="gate11-agent-native-generation",
        kind=JobKind.GENERATE,
        project_id="gate11-native-project",
        project=project_from_fixture(fixture),
        diffusion_model_file_id=records[FileKind.DIFFUSION_MODELS].id,
        text_encoder_file_id=records[FileKind.TEXT_ENCODERS].id,
        vae_file_id=records[FileKind.VAE].id,
        filename_prefix=fixture["filename_prefix"],
    )
    active_manager = manager(layout, transfers, args)
    started_at = time.monotonic()
    submitted = await active_manager.submit(request)
    duplicate = await active_manager.submit(request)
    while True:
        job = await active_manager.get(submitted.id)
        if job.state in TERMINAL_STATES:
            break
        if time.monotonic() - started_at > args.timeout_seconds:
            await active_manager.cancel(submitted.id)
            raise TimeoutError("Gate 11 agent job exceeded its deadline")
        await asyncio.sleep(0.25)
    page = await active_manager.events(job.id)
    resumed_page = await active_manager.events(
        job.id,
        cursor=page.next_cursor,
    )
    await active_manager.close()

    reconnected_manager = manager(layout, transfers, args)
    recovered = await reconnected_manager.get(job.id)
    recovered_events = await reconnected_manager.events(job.id)
    if not recovered.output_file_ids:
        raise RuntimeError(
            f"Gate 11 job ended as {recovered.state}: {recovered.error_code}"
        )
    output_record, output_path = await transfers.resolve_file(
        recovered.output_file_ids[0],
        required_kind=FileKind.OUTPUTS,
    )
    with Image.open(output_path) as image:
        metadata = {
            key: image.info.get(key)
            for key in (
                "backend",
                "correlation_id",
                "seed",
                "sampler",
                "scheduler",
            )
        }
    await reconnected_manager.close()
    return {
        "status": "passed",
        "duration_seconds": time.monotonic() - started_at,
        "job_id": job.id,
        "command_id": job.command_id,
        "backend": job.backend,
        "state": job.state,
        "duplicate_job_id": duplicate.id,
        "event_count": len(page.items),
        "resumed_event_count": len(resumed_page.items),
        "reconnected_state": recovered.state,
        "reconnected_event_count": len(recovered_events.items),
        "output_file_id": output_record.id,
        "output_sha256": output_record.sha256,
        "output_path": str(output_path),
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    try:
        summary = asyncio.run(run_probe(args))
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
