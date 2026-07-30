from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from k2_region_lab.semantic_conditioning import PoseSemanticError
from k2_region_lab.krea_control_lora import KreaControlError


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
    "subject_conditioning_compile_failed": (
        "A subject-only conditioning prompt could not be compiled."
    ),
    "conditioning_scope_mismatch": (
        "Semantic conditioning scopes did not match the posed subjects."
    ),
    "semantic_sampler_hook_incompatible": (
        "Another sampler hook conflicts with Prediction composite."
    ),
    "semantic_prediction_shape_invalid": (
        "Semantic prediction tensors or ownership masks have incompatible shapes."
    ),
    "semantic_attention_failed": (
        "Subject-semantic attention isolation failed."
    ),
    "semantic_lora_routing_failed": (
        "A regional LoRA could not be routed to its subject conditioning scope."
    ),
    "semantic_prediction_composite_multigpu_unsupported": (
        "Prediction composite currently supports one GPU only."
    ),
    "krea_control_checkpoint_invalid": (
        "The selected Krea pose adapter is not a valid verified safetensors checkpoint."
    ),
    "krea_control_checkpoint_incompatible": (
        "The selected pose adapter is incompatible with this Krea model."
    ),
    "krea_control_format_mismatch": (
        "The selected pose adapter uses a different volumetric control format."
    ),
    "krea_control_projection_missing": (
        "The pose adapter is missing its expanded Krea input projection."
    ),
    "krea_control_block_weights_missing": (
        "The pose adapter is missing one or more required Krea block weights."
    ),
    "krea_control_vae_incompatible": (
        "The selected VAE is incompatible with the Krea pose adapter."
    ),
    "krea_control_encode_failed": (
        "The volumetric control image could not be encoded with the selected VAE."
    ),
    "krea_control_latent_shape_invalid": (
        "The encoded volumetric control latent has an incompatible shape."
    ),
    "krea_control_scope_missing": (
        "A required subject control scope was not prepared."
    ),
    "krea_control_lora_target_conflict": (
        "The pose adapter conflicts with another model patch target."
    ),
    "krea_control_hook_incompatible": (
        "Another sampling hook conflicts with Krea volumetric pose control."
    ),
    "depth_checkpoint_incompatible": (
        "The selected depth adapter is incompatible with this Krea 2 runtime."
    ),
    "depth_encode_failed": (
        "The depth image could not be encoded with the selected Krea/Qwen VAE."
    ),
    "depth_feature_disabled": "Depth control is disabled on this worker.",
    "depth_regions_disabled": "Regional depth weighting is disabled on this worker.",
    "depth_override_disabled": "Regional depth override is disabled on this worker.",
    "depth_checkpoint_invalid": "Select the verified Krea 2 depth adapter checkpoint.",
    "depth_image_invalid": "Select a supported grayscale depth image.",
    "generation_failed": (
        "Generation failed during sampling. Review the detailed worker diagnostic "
        "and verify the selected generation settings."
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
    elif isinstance(error, PoseSemanticError):
        code = error.code
    elif isinstance(error, KreaControlError):
        code = error.code
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
        "pre-existing denoise-mask" in combined
        or "posegatehookincompatible" in combined
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
