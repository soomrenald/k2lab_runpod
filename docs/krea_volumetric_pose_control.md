# Krea-native volumetric pose control

K2Lab project schema 23 adds an optional trained Control-LoRA for the existing
volumetric subject mannequin. The adapter is additive: normal generation, regional
prompting, regional LoRAs, Prediction composite, pose gating, image editing, and face
refinement keep their existing behavior when it is disabled.

## Control format

The canonical format is `k2-volumetric-pose-control-v1`. It is a full-canvas opaque
RGB PNG with a black background and fixed semantic colors for the head, neck, torso,
and anatomically left/right limb segments. K2Lab renders it at 4× resolution and
downsamples with Lanczos. The format document, palette, draw order, geometry constants,
renderer version, and format SHA-256 live in
`src/k2_region_lab/volumetric_control.py`.

The browser preview is returned by the backend canonical renderer. Its PNG, dimensions,
coverage, and SHA-256 are therefore the exact control image that inference encodes.
Subject tabs show isolated subject controls; they are not pose-gating support masks.

## Training

The reproducible toolchain is in `training/krea2_volumetric_control`. It pins the
trainer, native ComfyUI reference, and official Krea repositories by immutable commit.
The external trainer and native reference currently have no declared license, so their
source is not vendored into K2Lab.

Typical flow:

1. Build deterministic COCO image/control/caption pairs with `build_coco_pairs.py`.
2. Encode final aligned target and control images independently with the Qwen/Krea VAE
   using `prepare_latent_shards.py`.
3. Run `train.py` against the exact external trainer checkout recorded in
   `upstream.json`.
4. Evaluate Raw, Turbo, required cases, seeds, and the fixed strength sweep with
   `evaluate.py`.
5. Inspect the selected safetensors contract with `inspect_checkpoint.py`.

The upstream trainer only resumes weights and the step number. K2Lab labels that path
as approximate resume; it does not claim optimizer, scheduler, RNG, dataloader, or
gradient-accumulation continuity.

## Checkpoint contract and assets

Upload selected checkpoints to **Krea pose adapters**, which maps to
`models/krea_control_loras`. Do not upload them as ordinary regional LoRAs.

Verified checkpoints must be safetensors with the required K2 metadata, an expanded
Krea input projection, the complete 28-block LoRA target set, the exact control-format
and palette hash, `krea/Krea-2-Raw` as the base, RGB/no-normalization/no-inversion
settings, and a consistent rank. Incompatible or incomplete checkpoints fail before
sampling; K2Lab never silently falls back to an ordinary LoRA or generic ControlNet.

## GUI and inference

In **Advanced → Volumetric pose gating**:

- enable **trained pose adapter**;
- choose a checkpoint from the dedicated asset kind;
- set constant all-step strength from 0.00 to 2.00;
- preview the full or subject-specific canonical control.

At least one enabled subject mannequin and a selected checkpoint are required. The
adapter may run without hard/soft gating. With Prediction composite, the full
conditioning forward receives the full control and each subject forward receives its
subject-only control. Attention isolation and spatial-only operation use the full
control. Regional identity LoRAs remain independently routed after the adapter.

The worker validates the checkpoint, model projection, block targets, VAE latent shape,
scope controls, and hook compatibility. Output PNG metadata includes
`pose_control_lora_runtime`; raw control PNG bytes are not embedded.

## RunPod status warnings

Passive RunPod status polling uses the last known durable workspace state while a
provider refresh is slow or unavailable. A transient provider timeout is shown as an
amber `PROVIDER` warning, never as a worker or generation error. Completed jobs remain
terminal and their output IDs remain available.

Successful provider status is cached for 30 seconds. Failures back off for 15, 30, 60,
then 120 seconds. Only one status refresh is active per workspace, and passive reads use
a separate lock from explicit start/stop/delete operations. Explicit mutation timeouts
are reconciled with an idempotent provider read and are not blindly replayed.
