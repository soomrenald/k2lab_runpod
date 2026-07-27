from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CommandKind(StrEnum):
    PROBE = "probe"
    DIAGNOSE_ACCELERATOR = "diagnose_accelerator"
    DISCOVER_MODELS = "discover_models"
    VALIDATE_MODELS = "validate_models"
    VALIDATE_LORAS = "validate_loras"
    LOAD_MODEL = "load_model"
    INITIALIZE_RUN = "initialize_run"
    GENERATE_BASELINE = "generate_baseline"
    EDIT_IMAGE = "edit_image"
    REFINE_FACES = "refine_faces"
    NEXT_BLOCK = "next_block"
    NEXT_STEP = "next_step"
    CONTINUE = "continue"
    PAUSE = "pause"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class WorkerState(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    PROBING = "probing"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


WORKER_ERROR_MESSAGES = {
    "worker_oom": (
        "The GPU ran out of memory. Reduce canvas size, disable post-upscale, "
        "or use a GPU with more VRAM."
    ),
    "worker_ram_low": (
        "The Pod does not have enough allocatable system RAM for this operation. "
        "Release memory or use a Pod with more RAM."
    ),
    "worker_probe_failed": (
        "The GPU runtime probe failed. Check the Pod GPU assignment and worker environment."
    ),
    "model_load_failed": (
        "Krea 2 model loading failed. Verify the selected transformer, text encoder, "
        "and VAE files and available GPU memory."
    ),
    "lora_validation_failed": (
        "LoRA validation failed. Verify that every selected LoRA targets Krea 2."
    ),
    "pose_mask_build_failed": (
        "The volumetric mannequin masks could not be built. Check the subject boxes "
        "and mannequin geometry."
    ),
    "pose_gate_schedule_invalid": (
        "The volumetric pose gate or sigma schedule is invalid. Check phase steps "
        "and trajectory settings."
    ),
    "pose_gate_runtime_failed": (
        "Volumetric pose gating failed during sampling. Review the pose-gating "
        "diagnostics and try again."
    ),
    "pose_gate_hook_incompatible": (
        "Another sampling mask hook conflicts with volumetric pose gating."
    ),
    "generation_failed": (
        "Generation failed while applying the selected LoRA or sampling settings. "
        "Verify that the LoRA targets Krea 2 and try again."
    ),
    "image_edit_failed": (
        "Image editing failed while applying the selected models or edit settings."
    ),
    "face_refinement_failed": (
        "Face refinement failed while applying the selected detector, LoRA, or crop settings."
    ),
    "worker_failed": "The GPU worker could not complete this job.",
}


def classify_worker_error(
    error: BaseException,
    command_kind: CommandKind | None,
) -> tuple[str, str]:
    combined = f"{type(error).__name__} {error}".casefold()
    if "outofmemory" in combined or "out_of_memory" in combined or "out of memory" in combined:
        code = "worker_oom"
    elif isinstance(error, MemoryError):
        code = "worker_ram_low"
    elif command_kind == CommandKind.PROBE:
        code = "worker_probe_failed"
    elif command_kind == CommandKind.LOAD_MODEL:
        code = "model_load_failed"
    elif command_kind == CommandKind.VALIDATE_LORAS:
        code = "lora_validation_failed"
    elif command_kind == CommandKind.GENERATE_BASELINE and "posemaskbuild" in combined:
        code = "pose_mask_build_failed"
    elif command_kind == CommandKind.GENERATE_BASELINE and (
        "posegateschedule" in combined or "sigmaschedule" in combined
    ):
        code = "pose_gate_schedule_invalid"
    elif command_kind == CommandKind.GENERATE_BASELINE and (
        "pre-existing denoise-mask" in combined or "posegatehook" in combined
    ):
        code = "pose_gate_hook_incompatible"
    elif command_kind == CommandKind.GENERATE_BASELINE and (
        "pose gate" in combined or "volumetric pose" in combined
    ):
        code = "pose_gate_runtime_failed"
    elif command_kind == CommandKind.GENERATE_BASELINE:
        code = "generation_failed"
    elif command_kind == CommandKind.EDIT_IMAGE:
        code = "image_edit_failed"
    elif command_kind == CommandKind.REFINE_FACES:
        code = "face_refinement_failed"
    else:
        code = "worker_failed"
    return code, WORKER_ERROR_MESSAGES[code]


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    command_id: str
    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    command_id: str | None
    state: WorkerState
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
