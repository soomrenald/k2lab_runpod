"""Single-reference instruction editing, isolated from regional img2img."""

from contextlib import redirect_stdout
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

from k2_region_lab.image2image import Image2ImageSettings
from k2_region_lab.image_edit import load_source_image
from k2_region_lab.output import validate_filename_prefix

UPSTREAM_COMMIT = "86f886dac23013d88996e3a2e99093ba44d322fb"


def run_image2image(runtime, payload, *, progress=None, event=None):
    if not runtime.loaded:
        raise RuntimeError("load Krea 2 before image2image")
    options = Image2ImageSettings.model_validate(
        {key: value for key, value in payload.items() if key in Image2ImageSettings.model_fields}
    )
    if not options.prompt.strip():
        raise ValueError("image2image requires an edit instruction")
    prefix = validate_filename_prefix(payload.get("filename_prefix", "image2image"))

    import numpy as np
    import torch
    from PIL import Image, PngImagePlugin
    import comfy.sample
    import comfy.samplers
    import comfy.model_management
    # The upstream module also prints a banner during its first import.
    with redirect_stdout(sys.stderr):
        from k2_region_lab.worker.krea2edit import Krea2EditGroundedEncode, Krea2EditModelPatch
    from k2_region_lab.sampling import register_bong_tangent_scheduler

    register_bong_tangent_scheduler(comfy.samplers, torch)
    if options.sampler not in comfy.samplers.KSampler.SAMPLERS:
        raise ValueError("image2image sampler is unavailable in this runtime")
    if options.scheduler not in comfy.samplers.KSampler.SCHEDULERS:
        raise ValueError("image2image scheduler is unavailable in this runtime")
    source, _ = load_source_image(Path(payload["image_path"]))
    pixels = torch.from_numpy(np.asarray(source, dtype=np.float32).copy() / 255.0)[None]

    protocol_stdout = sys.stdout
    original_progress, original_event = progress, event

    def forward_progress(*args):
        with redirect_stdout(protocol_stdout):
            if original_progress:
                original_progress(*args)

    def forward_event(*args):
        with redirect_stdout(protocol_stdout):
            if original_event:
                original_event(*args)

    progress, event = forward_progress, forward_event
    # Upstream prints diagnostics; keep the worker's stdout strictly JSON events.
    with redirect_stdout(sys.stderr), torch.no_grad():
        runtime._ensure_memory("before image2image grounded encoding", event)
        encoder = Krea2EditGroundedEncode()
        positive = encoder.encode(
            runtime.clip, options.prompt, pixels, grounding_px=options.grounding_px
        )[0]
        negative = (
            encoder.encode(runtime.clip, "", pixels, grounding_px=options.grounding_px)[0]
            if options.cfg > 1
            else positive
        )
        patches, _, _ = runtime._load_lora_patches(
            {
                "path": payload["identity_lora_path"],
                "strength": options.lora_strength,
            }
        )
        if not patches:
            raise ValueError("Identity Edit LoRA has no compatible Krea 2 targets")
        model = runtime.model.clone()
        applied = model.add_patches(patches, options.lora_strength)
        if not applied:
            raise ValueError("Identity Edit LoRA could not be applied")
        # Match the pinned ComfyUI EmptySD3LatentImage geometry directly.
        target = {
            "samples": torch.zeros(
                [1, 16, options.height // 8, options.width // 8],
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype(),
            ),
            "downscale_ratio_spacial": 8,
        }
        runtime._ensure_memory("before image2image source encoding", event)
        source_latent = {"samples": runtime._encode_vae(pixels)}
        model = Krea2EditModelPatch().patch(
            model,
            source_latent,
            vae=runtime.vae,
            source_image=pixels,
            target_latent=target,
            fit_mode=options.fit_mode,
            ref_boost=options.ref_boost,
        )[0]

        def callback(step, denoised, current, total):
            del denoised, current
            snapshot = runtime.memory_snapshot(f"image2image step {step + 1}/{total}")
            if progress:
                progress(step + 1, total, snapshot)
            if snapshot["gpu_free_bytes"] < snapshot["critical_free_bytes"]:
                raise RuntimeError("GPU out of memory guard triggered during image2image")

        runtime._ensure_memory("before image2image sampling", event)
        samples = comfy.sample.sample(
            model,
            comfy.sample.prepare_noise(target["samples"], options.seed),
            options.steps,
            options.cfg,
            options.sampler,
            options.scheduler,
            positive,
            negative,
            target["samples"],
            denoise=1.0,
            callback=callback,
            disable_pbar=True,
            seed=options.seed,
        )
        runtime._prepare_vae_handoff(model, event)
        decoded = runtime._decode_vae(samples)[0]
    while decoded.ndim > 3 and decoded.shape[0] == 1:
        decoded = decoded[0]
    if decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise RuntimeError(f"unexpected image2image decoded shape: {tuple(decoded.shape)}")
    array = (
        (decoded.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1).numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    image = Image.fromarray(array)
    destination = Path(payload["output_directory"]).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = destination / f"{prefix}_image2image_{stamp}_{uuid4().hex[:8]}_seed-{options.seed}.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("k2lab_mode", "krea2_identity_image2image")
    metadata.add_text(
        "image2image", json.dumps({**options.model_dump(), "upstream_commit": UPSTREAM_COMMIT})
    )
    if isinstance(payload.get("project_json"), dict):
        metadata.add_text("k2lab_project", json.dumps(payload["project_json"]))
    image.save(output, pnginfo=metadata)
    return {
        "image_path": str(output),
        "width": image.width,
        "height": image.height,
        "seed": options.seed,
        "memory": runtime.memory_snapshot("image2image complete"),
    }
