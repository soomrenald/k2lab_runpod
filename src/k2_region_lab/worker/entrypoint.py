from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from k2core.backends import NativeK2Backend
from k2core.inference import (
    ConfigurationError,
    DTypePolicy,
    DevicePolicy,
    GenerationRequest,
    ImageEditRequest,
    K2InferenceError,
    OutOfMemoryError,
    PipelineConfig,
    ProgressEvent,
    convert_error,
)
from k2core.model import (
    ComponentReference,
    RegisteredModel,
    TokenizerReference,
    discover_model_artifacts as discover_native_model_artifacts,
)

from k2_region_lab.config import ModelDirectories
from k2_region_lab.debug import configure_debug_logging
from k2_region_lab.model import discover_model_artifacts
from k2_region_lab.pose_gating import (
    PoseGatingSettings,
    SigmaScheduleMode,
    SigmaScheduleRequest,
    SoftGateSchedule,
)
from k2_region_lab.projector import DEFAULT_PROJECTOR_PRESET
from k2_region_lab.regional_prompting import (
    prompt_emphases_from_payload,
    region_definitions_from_payload,
)
from k2_region_lab.worker.protocol import CommandKind, WorkerState, classify_worker_error
from k2_region_lab.worker.runtime import (
    ComfyBaselineRuntime,
    diagnose_accelerator,
    probe_runtime,
    validate_model_artifacts,
)


def emit(
    state: WorkerState,
    message: str,
    *,
    command_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    print(
        json.dumps(
            {
                "command_id": command_id,
                "state": state.value,
                "message": message,
                "payload": payload or {},
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def model_directories(payload: dict[str, Any]) -> ModelDirectories:
    return ModelDirectories(
        diffusion_models=Path(payload["diffusion_models"]),
        text_encoders=Path(payload["text_encoders"]),
        vae=Path(payload["vae"]),
        loras=Path(
            payload.get("lora_directory", "~/ComfyUI/models/loras")
        ).expanduser(),
        upscale_models=Path(
            payload.get("upscale_models", "~/ComfyUI/models/upscale_models")
        ).expanduser(),
        diffusion_model_file=(
            Path(payload["diffusion_model_file"])
            if payload.get("diffusion_model_file") else None
        ),
        text_encoder_file=(
            Path(payload["text_encoder_file"])
            if payload.get("text_encoder_file") else None
        ),
        vae_file=Path(payload["vae_file"]) if payload.get("vae_file") else None,
    )


def selected_inference_backend(payload: dict[str, Any]) -> str:
    selected = str(payload.get("inference_backend", "comfyui")).strip().casefold()
    if selected not in {"comfyui", "native"}:
        raise ConfigurationError(
            f"Unsupported inference backend {selected!r}.",
            backend_name=selected,
            phase="worker_startup",
            remediation="Set K2LAB_INFERENCE_BACKEND to 'comfyui' or 'native'.",
        )
    return selected


def registered_native_model(payload: dict[str, Any]) -> RegisteredModel:
    components: dict[str, ComponentReference] = {}
    for name, path_key, hash_key in (
        ("transformer", "diffusion_model_file", "diffusion_model_sha256"),
        ("text_encoder", "text_encoder_file", "text_encoder_sha256"),
        ("vae", "vae_file", "vae_sha256"),
    ):
        supplied_path = str(payload.get(path_key) or "").strip()
        supplied_hash = str(payload.get(hash_key) or "").strip()
        if not supplied_path or not supplied_hash:
            raise ConfigurationError(
                f"Native loading requires an explicit {name} path and SHA-256.",
                backend_name="native",
                phase="model_loading",
                remediation="Select all three content-addressed model components.",
            )
        components[name] = ComponentReference(Path(supplied_path), supplied_hash)
    tokenizer_path = str(payload.get("tokenizer_path") or "").strip()
    tokenizer_hash = str(payload.get("tokenizer_sha256") or "").strip()
    if not tokenizer_path or not tokenizer_hash:
        raise ConfigurationError(
            "Native loading requires an explicit tokenizer path and SHA-256.",
            backend_name="native",
            phase="model_loading",
            remediation="Configure K2LAB_NATIVE_TOKENIZER_PATH and its SHA-256.",
        )
    return RegisteredModel(
        name=str(payload.get("registered_model_name", "krea2-runpod")),
        architecture="krea2",
        transformer=components["transformer"],
        text_encoder=components["text_encoder"],
        vae=components["vae"],
        tokenizer=TokenizerReference(Path(tokenizer_path), tokenizer_hash),
        default_dtype=str(payload.get("model_default_dtype", "bfloat16")),
    )


def native_backend_loaded(backend: NativeK2Backend | None) -> bool:
    return (
        backend is not None
        and backend.pipeline is not None
        and backend.pipeline.loaded
    )


def emit_native_progress(command_id: str | None, event: ProgressEvent) -> None:
    detail = dict(event.detail)
    nested_memory = detail.pop("memory", {})
    memory = (
        {**dict(nested_memory), **detail}
        if isinstance(nested_memory, dict)
        else detail
    )
    if event.phase == "diffusion":
        message = (
            f"Denoising step {int(event.step or 0)}/{int(event.total_steps or 0)}"
        )
    else:
        message = event.phase.replace("_", " ").title()
        if event.fraction == 0.0:
            message += " started"
        elif event.fraction == 1.0:
            message += " complete"
    emit(
        WorkerState.RUNNING,
        message,
        command_id=command_id,
        payload={
            "backend": "native",
            "phase": event.phase,
            "step": int(event.step or 0),
            "total_steps": int(event.total_steps or 0),
            "fraction": event.fraction,
            "memory": memory,
        },
    )


def finish_native_job(
    backend: NativeK2Backend,
    payload: dict[str, Any],
    *,
    command_id: str | None,
    completed_label: str,
) -> bool:
    resident = bool(payload.get("keep_model_loaded", False))
    if not resident:
        backend.unload()
    emit(
        WorkerState.COMPLETE,
        (
            f"{completed_label}; native model retained in VRAM"
            if resident
            else f"{completed_label}; worker releasing GPU and system RAM"
        ),
        command_id=command_id,
        payload={"backend": "native", "resident": resident},
    )
    return resident


def finish_job(
    runtime: ComfyBaselineRuntime,
    *,
    command_id: str | None,
    completed_label: str,
) -> bool:
    retention = runtime.retain_baseline_model_if_safe()
    resident = bool(retention["resident"])
    emit(
        WorkerState.COMPLETE,
        (
            f"{completed_label}; baseline model retained in VRAM"
            if resident
            else f"{completed_label}; worker releasing GPU and system RAM"
        ),
        command_id=command_id,
        payload=retention,
    )
    return resident


def main() -> int:
    configure_debug_logging("worker")
    logger = logging.getLogger("k2_region_lab.worker.entrypoint")
    logger.debug("worker starting with executable=%s argv=%r", sys.executable, sys.argv)
    runtime: ComfyBaselineRuntime | None = None
    native_backend: NativeK2Backend | None = None
    active_backend: str | None = None
    artifacts = None
    emit(WorkerState.UNLOADED, "GPU worker started")
    for encoded in sys.stdin:
        command_id: str | None = None
        kind: CommandKind | None = None
        try:
            command = json.loads(encoded)
            command_id = command.get("command_id")
            kind = CommandKind(command["kind"])
            payload = command.get("payload", {})
            command_backend = selected_inference_backend(payload)
            if active_backend is None:
                active_backend = command_backend
            elif active_backend != command_backend:
                raise ConfigurationError(
                    "A resident worker cannot switch inference backends.",
                    backend_name=active_backend,
                    phase="worker_command",
                )
            logger.debug("received worker command id=%s kind=%s", command_id, kind.value)
            comfyui_root = Path(payload.get("comfyui_root", "~/ComfyUI")).expanduser()
            if kind == CommandKind.PROBE:
                emit(WorkerState.PROBING, "Probing worker runtime", command_id=command_id)
                result = probe_runtime(comfyui_root)
                emit(
                    WorkerState.UNLOADED,
                    "Worker runtime probe complete",
                    command_id=command_id,
                    payload={"backend": active_backend, **result},
                )
            elif kind == CommandKind.DIAGNOSE_ACCELERATOR:
                emit(WorkerState.PROBING, "Running accelerator diagnostics", command_id=command_id)
                result = diagnose_accelerator(comfyui_root)
                logger.debug("accelerator diagnostics: %r", result)
                emit(
                    WorkerState.READY if result.get("accelerator_available") else WorkerState.ERROR,
                    "Accelerator diagnostics complete",
                    command_id=command_id,
                    payload=result,
                )
            elif kind in (CommandKind.DISCOVER_MODELS, CommandKind.VALIDATE_MODELS):
                emit(WorkerState.VALIDATING, "Validating model artifacts", command_id=command_id)
                directories = model_directories(payload)
                artifacts, manifests = validate_model_artifacts(
                    directories, Path(payload["manifest_directory"])
                )
                compatible = artifacts.complete and all(item["compatible"] for item in manifests)
                emit(
                    WorkerState.READY if compatible else WorkerState.ERROR,
                    "Model artifacts validated" if compatible else "Model validation failed",
                    command_id=command_id,
                    payload={"complete": artifacts.complete, "manifests": manifests},
                )
            elif kind == CommandKind.LOAD_MODEL:
                already_loaded = (
                    runtime is not None and runtime.loaded
                    if active_backend == "comfyui"
                    else native_backend_loaded(native_backend)
                )
                if already_loaded:
                    emit(
                        WorkerState.READY,
                        "Krea 2 baseline already loaded",
                        command_id=command_id,
                        payload={"backend": active_backend, "reused": True},
                    )
                    continue
                if active_backend == "native":
                    directories = model_directories(payload)
                    native_artifacts = discover_native_model_artifacts(directories)
                    emit(
                        WorkerState.LOADING,
                        "Loading native Krea 2 baseline components",
                        command_id=command_id,
                        payload={"backend": "native"},
                    )
                    native_backend = native_backend or NativeK2Backend()
                    loaded_pipeline = native_backend.load(
                        PipelineConfig(
                            artifacts=native_artifacts,
                            registered_model=registered_native_model(payload),
                            memory_policy=str(
                                payload.get("memory_policy", "safe_16gb")
                            ),
                            reserve_vram_gb=float(
                                payload.get("reserve_vram_gb", 4.0)
                            ),
                            minimum_system_ram_gb=float(
                                payload.get("minimum_system_ram_gb", 14.0)
                            ),
                            cpu_vae=bool(payload.get("cpu_vae", False)),
                            oom_recovery=bool(payload.get("oom_recovery", True)),
                            strict_loading=bool(payload.get("strict_loading", True)),
                            device_policy=DevicePolicy(
                                transformer_device=str(
                                    payload.get("transformer_device", "auto")
                                ),
                                text_encoder_device=str(
                                    payload.get("text_encoder_device", "auto")
                                ),
                                vae_device=str(payload.get("vae_device", "auto")),
                                compute_dtype=DTypePolicy(
                                    str(payload.get("compute_dtype", "auto"))
                                ),
                                weight_dtype=DTypePolicy(
                                    str(payload.get("weight_dtype", "auto"))
                                ),
                                cpu_offload=bool(
                                    payload.get("cpu_offload", False)
                                ),
                                vae_tiling=bool(payload.get("vae_tiling", False)),
                            ),
                        )
                    )
                    emit(
                        WorkerState.READY,
                        "Native Krea 2 baseline components loaded",
                        command_id=command_id,
                        payload={
                            "backend": "native",
                            **dict(loaded_pipeline.metadata),
                        },
                    )
                    continue
                if artifacts is None:
                    directories = model_directories(payload)
                    artifacts = discover_model_artifacts(directories)
                emit(
                    WorkerState.LOADING,
                    "Loading Krea 2 baseline components",
                    command_id=command_id,
                )
                runtime = runtime or ComfyBaselineRuntime(
                    comfyui_root,
                    face_detector_path=(
                        Path(payload["face_detector_path"])
                        if payload.get("face_detector_path") else None
                    ),
                )
                loaded = runtime.load(
                    artifacts,
                    memory_policy_key=str(payload.get("memory_policy", "safe_16gb")),
                    vram_mode=str(payload.get("vram_mode", "auto")),
                    reserve_vram_gb=float(payload.get("reserve_vram_gb", 4.0)),
                    keep_model_loaded=bool(payload.get("keep_model_loaded", False)),
                    minimum_system_ram_gb=float(
                        payload.get("minimum_system_ram_gb", 14.0)
                    ),
                    system_ram_guard_enabled=bool(
                        payload.get("system_ram_guard_enabled", True)
                    ),
                    cpu_vae=bool(payload.get("cpu_vae", False)),
                    oom_recovery=bool(payload.get("oom_recovery", True)),
                )
                emit(
                    WorkerState.READY,
                    "Krea 2 baseline components loaded",
                    command_id=command_id,
                    payload=loaded,
                )
            elif kind == CommandKind.VALIDATE_LORAS:
                if runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before validating LoRAs")
                emit(
                    WorkerState.VALIDATING,
                    "Validating LoRA compatibility",
                    command_id=command_id,
                )
                reports = runtime.diagnose_loras(list(payload.get("loras", [])))
                compatible = bool(reports) and all(report["compatible"] for report in reports)
                emit(
                    WorkerState.READY if compatible else WorkerState.ERROR,
                    "LoRA diagnostics complete",
                    command_id=command_id,
                    payload={"compatible": compatible, "loras": reports},
                )
            elif kind == CommandKind.GENERATE_BASELINE:
                if active_backend == "native":
                    if not native_backend_loaded(native_backend):
                        raise RuntimeError(
                            "load the native Krea 2 baseline before generating"
                        )
                    assert native_backend is not None
                    generation_started_at = time.monotonic()
                    emit(
                        WorkerState.RUNNING,
                        "Native generation started",
                        command_id=command_id,
                        payload={"backend": "native"},
                    )

                    def native_event(
                        message: str, event_payload: dict[str, Any]
                    ) -> None:
                        emit(
                            WorkerState.RUNNING,
                            message,
                            command_id=command_id,
                            payload={"backend": "native", **event_payload},
                        )

                    native_request = GenerationRequest.from_payload(
                        payload, correlation_id=str(command_id or "")
                    )
                    generated = native_backend.generate(
                        native_request,
                        progress=lambda event: emit_native_progress(
                            command_id, event
                        ),
                        diagnostic=native_event,
                    ).to_payload()
                    emit(
                        WorkerState.RUNNING,
                        (
                            "Native generation run finished in "
                            f"{time.monotonic() - generation_started_at:.2f} seconds"
                        ),
                        command_id=command_id,
                        payload={"backend": "native"},
                    )
                    emit(
                        WorkerState.READY,
                        "Native generation complete",
                        command_id=command_id,
                        payload={"backend": "native", **generated},
                    )
                    if finish_native_job(
                        native_backend,
                        payload,
                        command_id=command_id,
                        completed_label="Native generation complete",
                    ):
                        continue
                    return 0
                if runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before generating")
                generation_started_at = time.monotonic()
                emit(
                    WorkerState.RUNNING,
                    "Generation started",
                    command_id=command_id,
                )

                def progress(step: int, total: int, memory: dict[str, Any]) -> None:
                    phase = memory.get("pose_gate_phase")
                    gate = memory.get("pose_gate_strength")
                    message = f"Denoising step {step}/{total}"
                    if isinstance(phase, str) and isinstance(gate, (int, float)):
                        message += f" — {phase} gate, strength {float(gate):.2f}"
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload={
                            "step": step,
                            "total_steps": total,
                            "memory": memory,
                            **(
                                {
                                    "phase": phase,
                                    "gate_strength": float(gate),
                                    "sigma": memory.get("sigma"),
                                    "next_sigma": memory.get("next_sigma"),
                                    "normalized_trajectory_progress": memory.get(
                                        "normalized_trajectory_progress"
                                    ),
                                }
                                if isinstance(phase, str)
                                and isinstance(gate, (int, float))
                                else {}
                            ),
                        },
                    )

                def runtime_event(message: str, event_payload: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload=event_payload,
                    )

                generated = runtime.generate(
                    prompt=str(payload.get("prompt", "")),
                    shared_visual_prompt=str(
                        payload.get("shared_visual_prompt", "")
                    ),
                    width=int(payload.get("width", 1024)),
                    height=int(payload.get("height", 1024)),
                    steps=int(payload.get("steps", 8)),
                    sampler=str(payload.get("sampler", "euler")),
                    scheduler=str(payload.get("scheduler", "simple")),
                    seed=int(payload.get("seed", 0)),
                    output_directory=Path(payload["output_directory"]),
                    filename_prefix=str(payload.get("filename_prefix", "baseline")),
                    regions=region_definitions_from_payload(payload.get("regions", [])),
                    emphases=prompt_emphases_from_payload(
                        payload.get("prompt_emphases", [])
                    ),
                    regional_prompting=bool(payload.get("regional_prompting", True)),
                    regional_prompt_strength=float(
                        payload.get("regional_prompt_strength", 1.0)
                    ),
                    regional_outside_penalty=float(
                        payload.get("regional_outside_penalty", 1.0)
                    ),
                    regional_feather_pixels=float(
                        payload.get("regional_feather_pixels", 128.0)
                    ),
                    regional_subject_competition=bool(
                        payload.get("regional_subject_competition", True)
                    ),
                    regional_subject_fill=bool(
                        payload.get("regional_subject_fill", True)
                    ),
                    regional_late_step_scale=float(
                        payload.get("regional_late_step_scale", 0.35)
                    ),
                    regional_lora_delta_adaptation=bool(
                        payload.get("regional_lora_delta_adaptation", False)
                    ),
                    regional_lora_delta_adaptation_gain=float(
                        payload.get("regional_lora_delta_adaptation_gain", 0.35)
                    ),
                    pose_gating=PoseGatingSettings(
                        enabled=bool(payload.get("pose_gating_enabled", False)),
                        hard_steps=int(payload.get("pose_hard_gate_steps", 2)),
                        soft_steps=int(payload.get("pose_soft_gate_steps", 2)),
                        soft_schedule=SoftGateSchedule(
                            payload.get("pose_soft_gate_schedule", "cosine")
                        ),
                        sigma_request=SigmaScheduleRequest(
                            mode=SigmaScheduleMode(
                                payload.get("pose_sigma_schedule_mode", "automatic")
                            ),
                            hard_share=float(
                                payload.get("pose_sigma_hard_share", 0.20)
                            ),
                            soft_share=float(
                                payload.get("pose_sigma_soft_share", 0.30)
                            ),
                            normalized_knots=tuple(
                                float(value)
                                for value in payload.get("pose_sigma_knots", ())
                            ),
                        ),
                    ),
                    pose_semantic_mode=str(
                        payload.get("pose_semantic_mode", "prediction_composite")
                    ),
                    pose_control_lora_enabled=bool(
                        payload.get("pose_control_lora_enabled", False)
                    ),
                    pose_control_lora_path=(
                        Path(payload["pose_control_lora_file"])
                        if payload.get("pose_control_lora_file")
                        else None
                    ),
                    pose_control_lora_file_id=(
                        str(payload["pose_control_lora_file_id"])
                        if payload.get("pose_control_lora_file_id")
                        else None
                    ),
                    pose_control_lora_strength=float(
                        payload.get("pose_control_lora_strength", 1.0)
                    ),
                    pose_control_format=str(
                        payload.get(
                            "pose_control_format",
                            "k2-volumetric-pose-control-v1",
                        )
                    ),
                    pose_control_allow_unverified_legacy=bool(
                        payload.get("pose_control_allow_unverified_legacy", False)
                    ),
                    projector_enabled=bool(payload.get("projector_enabled", False)),
                    projector_preset=str(
                        payload.get("projector_preset", "filter_bypass2")
                    ),
                    projector_values=tuple(payload.get("projector_values", ())),
                    projector_multiplier=float(payload.get("projector_multiplier", 1.0)),
                    projector_identity_protection=float(
                        payload.get("projector_identity_protection", 1.0)
                    ),
                    post_upscale=bool(payload.get("post_upscale", False)),
                    upscale_scale=int(payload.get("upscale_scale", 2)),
                    upscale_method=str(payload.get("upscale_method", "lanczos")),
                    upscale_model_path=(
                        Path(payload["upscale_model_path"])
                        if payload.get("upscale_model_path")
                        else None
                    ),
                    loras=list(payload.get("loras", [])),
                    project_json=(
                        dict(payload["project_json"])
                        if isinstance(payload.get("project_json"), dict)
                        else None
                    ),
                    progress=progress,
                    event=runtime_event,
                )
                duration_seconds = time.monotonic() - generation_started_at
                emit(
                    WorkerState.RUNNING,
                    f"Generation run finished in {duration_seconds:.2f} seconds",
                    command_id=command_id,
                    payload={"duration_seconds": duration_seconds},
                )
                emit(
                    WorkerState.READY,
                    "Generation complete",
                    command_id=command_id,
                    payload=generated,
                )
                if finish_job(
                    runtime,
                    command_id=command_id,
                    completed_label="Generation complete",
                ):
                    continue
                return 0
            elif kind == CommandKind.EDIT_IMAGE:
                if active_backend == "native":
                    if not native_backend_loaded(native_backend):
                        raise RuntimeError(
                            "load the native Krea 2 baseline before image editing"
                        )
                    assert native_backend is not None
                    emit(
                        WorkerState.RUNNING,
                        "Native image editing started",
                        command_id=command_id,
                        payload={"backend": "native"},
                    )

                    def native_edit_event(
                        message: str, event_payload: dict[str, Any]
                    ) -> None:
                        emit(
                            WorkerState.RUNNING,
                            message,
                            command_id=command_id,
                            payload={"backend": "native", **event_payload},
                        )

                    native_edit_request = ImageEditRequest.from_payload(
                        payload, correlation_id=str(command_id or "")
                    )
                    edited = native_backend.generate(
                        native_edit_request,
                        progress=lambda event: emit_native_progress(
                            command_id, event
                        ),
                        diagnostic=native_edit_event,
                    ).to_payload()
                    emit(
                        WorkerState.READY,
                        "Native image editing complete",
                        command_id=command_id,
                        payload={"backend": "native", **edited},
                    )
                    if finish_native_job(
                        native_backend,
                        payload,
                        command_id=command_id,
                        completed_label="Native image edit complete",
                    ):
                        continue
                    return 0
                if runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before image editing")
                emit(
                    WorkerState.RUNNING,
                    "Image editing started",
                    command_id=command_id,
                )

                def edit_progress(step: int, total: int, memory: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        f"Image-edit denoising step {step}/{total}",
                        command_id=command_id,
                        payload={"step": step, "total_steps": total, "memory": memory},
                    )

                def edit_event(message: str, event_payload: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload=event_payload,
                    )

                edited = runtime.edit_image(
                    image_path=Path(payload["image_path"]),
                    output_directory=(
                        Path(payload["output_directory"])
                        if payload.get("output_directory")
                        else None
                    ),
                    prompt=str(payload.get("prompt", "")),
                    regions=region_definitions_from_payload(payload.get("regions", [])),
                    reference_prompt=str(payload.get("reference_prompt", "")),
                    reference_regions=region_definitions_from_payload(
                        payload.get("reference_regions", [])
                    ),
                    prompt_emphases=prompt_emphases_from_payload(
                        payload.get("prompt_emphases", [])
                    ),
                    loras=list(payload.get("loras", [])),
                    seed=int(payload.get("seed", 0)),
                    steps=int(payload.get("steps", 8)),
                    sampler=str(payload.get("sampler", "euler")),
                    scheduler=str(payload.get("scheduler", "simple")),
                    denoise=float(payload.get("denoise", 0.15)),
                    latent_feather_pixels=int(
                        payload.get("latent_feather_pixels", 64)
                    ),
                    composite_feather_pixels=int(
                        payload.get("composite_feather_pixels", 48)
                    ),
                    edit_entire_image=bool(payload.get("edit_entire_image", False)),
                    preserve_identity=bool(payload.get("preserve_identity", True)),
                    reference_description_retention=float(
                        payload.get("reference_description_retention", 1.0)
                    ),
                    regional_prompt_strength=float(
                        payload.get("regional_prompt_strength", 1.0)
                    ),
                    regional_outside_penalty=float(
                        payload.get("regional_outside_penalty", 1.0)
                    ),
                    regional_feather_pixels=float(
                        payload.get("regional_feather_pixels", 128.0)
                    ),
                    regional_subject_competition=bool(
                        payload.get("regional_subject_competition", True)
                    ),
                    regional_subject_fill=bool(
                        payload.get("regional_subject_fill", True)
                    ),
                    regional_late_step_scale=float(
                        payload.get("regional_late_step_scale", 0.35)
                    ),
                    regional_lora_delta_adaptation=bool(
                        payload.get("regional_lora_delta_adaptation", False)
                    ),
                    regional_lora_delta_adaptation_gain=float(
                        payload.get("regional_lora_delta_adaptation_gain", 0.35)
                    ),
                    projector_enabled=bool(payload.get("projector_enabled", False)),
                    projector_preset=str(
                        payload.get("projector_preset", DEFAULT_PROJECTOR_PRESET)
                    ),
                    projector_values=(
                        tuple(float(value) for value in payload["projector_values"])
                        if payload.get("projector_values") is not None
                        else None
                    ),
                    projector_multiplier=float(
                        payload.get("projector_multiplier", 1.0)
                    ),
                    projector_identity_protection=float(
                        payload.get("projector_identity_protection", 1.0)
                    ),
                    project_json=(
                        dict(payload["project_json"])
                        if isinstance(payload.get("project_json"), dict)
                        else None
                    ),
                    progress=edit_progress,
                    event=edit_event,
                )
                emit(
                    WorkerState.READY,
                    "Image editing complete",
                    command_id=command_id,
                    payload=edited,
                )
                if finish_job(
                    runtime,
                    command_id=command_id,
                    completed_label="Image edit complete",
                ):
                    continue
                return 0
            elif kind == CommandKind.REFINE_FACES:
                if active_backend == "native":
                    raise ConfigurationError(
                        "Face refinement is not implemented by the native K2 backend.",
                        backend_name="native",
                        phase="face_refinement",
                        correlation_id=str(command_id or ""),
                        remediation="Set K2LAB_INFERENCE_BACKEND=comfyui.",
                    )
                if runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before refining faces")
                emit(
                    WorkerState.RUNNING,
                    "Face refinement started",
                    command_id=command_id,
                )

                def refinement_event(
                    message: str, event_payload: dict[str, Any]
                ) -> None:
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload=event_payload,
                    )

                refined = runtime.refine_faces(
                    image_path=Path(payload["image_path"]),
                    output_directory=(
                        Path(payload["output_directory"])
                        if payload.get("output_directory")
                        else None
                    ),
                    regions=region_definitions_from_payload(payload.get("regions", [])),
                    loras=list(payload.get("loras", [])),
                    seed=int(payload.get("seed", 0)),
                    steps=int(payload.get("steps", 8)),
                    denoise=float(payload.get("denoise", 0.15)),
                    crop_size=int(payload.get("crop_size", 512)),
                    padding=float(payload.get("padding", 2.0)),
                    feather=float(payload.get("feather", 0.12)),
                    blend=float(payload.get("blend", 0.5)),
                    lora_scale=float(payload.get("lora_scale", 0.5)),
                    detector_threshold=float(
                        payload.get("detector_threshold", 0.15)
                    ),
                    detector_provider=str(payload.get("detector_provider", "auto")),
                    selected_face_indices=(
                        tuple(int(index) for index in payload["selected_face_indices"])
                        if payload.get("selected_face_indices") is not None
                        else None
                    ),
                    manual_face_paths=tuple(
                        tuple((float(point[0]), float(point[1])) for point in path)
                        for path in payload.get("manual_face_paths", ())
                    ),
                    project_json=(
                        dict(payload["project_json"])
                        if isinstance(payload.get("project_json"), dict)
                        else None
                    ),
                    event=refinement_event,
                )
                emit(
                    WorkerState.READY,
                    "Face refinement complete",
                    command_id=command_id,
                    payload=refined,
                )
                if finish_job(
                    runtime,
                    command_id=command_id,
                    completed_label="Face refinement complete",
                ):
                    continue
                return 0
            elif kind == CommandKind.SHUTDOWN:
                emit(WorkerState.COMPLETE, "GPU worker stopped", command_id=command_id)
                return 0
            else:
                raise ValueError(f"unsupported worker command: {kind.value}")
        except Exception as error:
            logger.exception("worker command failed")
            traceback.print_exc(file=sys.stderr)
            structured = (
                error
                if isinstance(error, K2InferenceError)
                else convert_error(
                    error,
                    backend_name=active_backend or "comfyui",
                    phase=kind.value if kind is not None else "worker_command",
                    correlation_id=str(command_id or ""),
                    gpu_work_started=kind
                    in {
                        CommandKind.LOAD_MODEL,
                        CommandKind.GENERATE_BASELINE,
                        CommandKind.EDIT_IMAGE,
                        CommandKind.REFINE_FACES,
                    },
                )
            )
            error_code, error_detail = classify_worker_error(error, kind)
            if isinstance(structured, OutOfMemoryError):
                error_code = "worker_oom"
            emit(
                WorkerState.ERROR,
                error_detail,
                command_id=command_id,
                payload={
                    "exception_type": type(error).__name__,
                    "error_code": error_code,
                    "command_kind": kind.value if kind is not None else None,
                    "backend": active_backend or "comfyui",
                    "error": structured.to_payload(),
                },
            )
            if kind in {
                CommandKind.LOAD_MODEL,
                CommandKind.GENERATE_BASELINE,
                CommandKind.EDIT_IMAGE,
                CommandKind.REFINE_FACES,
            }:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
