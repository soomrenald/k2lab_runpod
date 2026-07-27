# K2 Region Lab User Guide

This guide explains the K2 Region Lab RunPod application from first launch through advanced
regional generation, pose control, memory diagnostics, and recovery. The labels in this guide
match the `k2lab_pose` browser interface and workspace image `0.2.0-pose.2`.

## Contents

- [The K2 mental model](#the-k2-mental-model)
- [Installation and first launch](#installation-and-first-launch)
- [Creating a cloud workspace](#creating-a-cloud-workspace)
- [Studio layout](#studio-layout)
- [Installing and selecting models](#installing-and-selecting-models)
- [Projects and PNG metadata](#projects-and-png-metadata)
- [Generating an image](#generating-an-image)
- [Regions and subject boxes](#regions-and-subject-boxes)
- [Volumetric mannequin controls](#volumetric-mannequin-controls)
- [LoRA routing](#lora-routing)
- [Image editing](#image-editing)
- [Face refinement](#face-refinement)
- [Advanced generation settings](#advanced-generation-settings)
- [Projector preset defaults and source provenance](#projector-preset-defaults-and-source-provenance)
- [Advanced image-edit settings](#advanced-image-edit-settings)
- [Face-refinement settings](#face-refinement-settings)
- [Assets, uploads, and downloads](#assets-uploads-and-downloads)
- [Jobs, batches, events, and timeout recovery](#jobs-batches-events-and-timeout-recovery)
- [RAM and VRAM management](#ram-and-vram-management)
- [Pod lifecycle and storage safety](#pod-lifecycle-and-storage-safety)
- [Updating K2 and a Pod without losing data](#updating-k2-and-a-pod-without-losing-data)
- [Troubleshooting](#troubleshooting)

## The K2 mental model

The browser interface and control plane run locally. The expensive work runs on your RunPod
GPU. Persistent files live on RunPod storage.

```text
Browser
  ↕ local loopback
K2 control plane on your computer
  ↕ authenticated RunPod/agent requests
RunPod workspace agent
  ↕ sequential jobs
GPU worker + persistent workspace files
```

The control plane records which immutable image and agent version belong to a workspace. This
is why an arbitrary Pod or a Pod using a different image is rejected instead of silently
running incompatible code.

### What persists where

On your computer:

- encrypted RunPod credentials;
- workspace and provider-resource records;
- the selected image digest and version;
- the local control-plane log.

On RunPod storage:

- diffusion models, text encoders, VAEs, LoRAs, upscalers, and the optional face detector;
- input images and generated outputs;
- saved K2 project files;
- upload/download progress;
- job summaries and redacted job events.

## Installation and first launch

### Prerequisites

Install `git` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then clone
the current branch:

```bash
git clone --branch k2lab_pose https://github.com/soomrenald/k2lab_runpod.git
cd k2lab_runpod
```

Start K2 with the matching immutable image and version:

```bash
./scripts/k2lab-runpod --image 'ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:71d034c346a5a2c1bb21a90df507d9a0b2dfb3f3e718e5380a9526e29b65b2c5' --image-version '0.2.0-pose.2'
```

The entire command must be one shell command. A line beginning only with `--image-version`
causes `bash: --image-version: command not found`.

The browser normally opens automatically. If it does not, open:

```text
http://127.0.0.1:8000
```

Subsequent launches use the saved configuration:

```bash
./scripts/k2lab-runpod
```

Useful launcher options:

| Option | Effect |
| --- | --- |
| `--image IMAGE@sha256:DIGEST` | Saves a public immutable workspace image. |
| `--image-version VERSION` | Saves the human-readable version expected from the agent. |
| `--state-dir PATH` | Uses a different private local state directory. |
| `--port PORT` | Uses a different loopback port; allowed range is 1024–65535. |
| `--no-open` | Starts without opening a browser. |
| `--no-follow` | If K2 is already running, report it and exit instead of following its log. |

### RunPod API key

Use a restricted user-owned key with only the Pod, volume, inventory, and billing-read
permissions needed by K2. The key is encrypted using the local `credential.key`; it is not
stored in project files or images.

## Creating a cloud workspace

### Workspace type

**Persistent Pod**

- The regular persistent volume belongs to one Pod.
- Stop retains the Pod and volume.
- Starting again may wait for the selected GPU type.
- Deleting the workspace permanently deletes that regular volume.

**Portable workspace**

- Files live on an independent RunPod network volume.
- Each start can create a new compatible Pod in the volume's datacenter.
- Stop terminates the temporary Pod but retains the network volume.
- Deleting the application workspace also retains the network volume for safety.
- Network-volume storage continues billing until it is separately deleted in RunPod.

Choose a Portable workspace when long-term data portability matters more than the simplicity
of one persistent Pod.

### GPU priority

Select GPUs in the order K2 should try them. The displayed VRAM and current hourly price help
compare candidates.

- **Secure Cloud** uses RunPod Secure Cloud inventory.
- **Community Cloud** may cost less but is not available for portable network-volume
  workspaces.
- **Use interruptible compute** can reduce cost, but the Pod may stop without notice.

Large Krea 2 workflows generally benefit from high-VRAM GPUs. K2 can use lower-memory GPUs
with Dynamic or Low VRAM mode, but model loading and each generation will be slower.

### Storage and safety controls

| Control | Meaning |
| --- | --- |
| **Container disk** | Temporary/runtime disk used by the image and environment. |
| **Workspace volume** | Persistent capacity for models, projects, inputs, and outputs. |
| **Network volume** | Existing portable storage to attach, or create a new one. |
| **Datacenter** | Location for a new portable volume and its compatible Pods. |
| **Idle stop** | Stops compute after the configured idle period. |
| **Hard session limit** | Maximum duration of a limited lease. |
| **No time limit** | Disables the lease deadline; billing continues until manual stop. |

Review availability and cost before creating the workspace. Compute is billed while the Pod is
running. Persistent storage is billed even while compute is stopped.

## Studio layout

### Top bar

- **New** resets the current project.
- **Open** loads a local `.json` or `.k2lab.json` project.
- **Import PNG** reads embedded K2 project metadata from a generated PNG.
- **Save** writes the project into persistent **Assets → Projects**.
- **Save as** asks for a new persistent project name.
- The workspace status button opens readiness, cost, lease, start/stop, migration, deletion,
  and worker-memory controls.
- **Stop GPU now** stops compute immediately.

Saving does not automatically download a copy to your computer. Download saved projects from
**Assets → Projects** when you want a local backup.

### Left rail

Modes:

- **Generate** — text-to-image, regions, LoRAs, mannequins, and post-upscale.
- **Edit** — source-based image editing with reference and target layers.
- **Faces** — optional detection and selected-face refinement.

Utilities:

- **Assets** — uploads and persistent workspace files.
- **Transfers** — sequential provider-side Civitai/Hugging Face downloads.
- **Events** — resizable job and operation history.
- **Setup** — deterministic model selections and output filename prefix.

### Canvas toolbar

- **Load image / Replace image** loads and uploads a local source.
- **Download image** downloads the image currently shown on the canvas.
- **Clear canvas** removes the displayed source/result from the current view.
- **Draw region** creates an ordinary prompt region.
- **Draw subject** creates a person-oriented subject box with a mannequin.

Dragging the body of a selected box moves the box. Eight handles resize it. Non-selected boxes
do not intercept pointer input.

### Inspector

The right inspector has four tabs:

- **Prompt** — global prompt, selected regional prompt, identity prompt, phrase emphasis.
- **Regions** — ordering, enable/disable state, selection, and deletion.
- **LoRAs** — strength and routing for each current layer.
- **Advanced** — sampler, scheduler, guidance, pose, memory, upscale, and expert settings.

## Installing and selecting models

Generation requires a compatible:

1. diffusion/transformer model;
2. text encoder;
3. VAE.

Open **Transfers** to download from a provider or **Assets** to upload a local file. Then open
**Setup** and select the exact file for each role.

The **Face detector (face tools only)** is optional. Missing face detection must never block
Generate or Edit. When you first use **Detect faces**, K2 can install the small pinned detector
model for you.

**Generation output filename prefix** controls the beginning of output filenames. It must be
1–128 characters and cannot contain `/`, `\`, or a null character.

Project files remember model filenames. When reopening a project, K2 resolves those names to
opaque files in the current workspace. A missing selection must be resolved in Setup before
submission.

## Projects and PNG metadata

The project format stores:

- prompts and phrase emphasis;
- regions, order, roles, and mannequin geometry;
- LoRA names, strengths, routing, and character trigger phrases;
- generation/edit/face settings;
- selected model filenames;
- canvas and source references.

**Open** loads JSON directly from your computer. **Import PNG** reads the embedded project
document from a K2 output PNG. LoRAs are rebound by normalized filename so imported metadata
does not show opaque internal file IDs when the file exists in the workspace.

Keep local backups of important `.k2lab.json` files and outputs. A project does not contain the
actual model weights.

## Generating an image

A reliable first generation:

1. Select the baseline model, text encoder, and VAE in **Setup**.
2. Leave regional guidance at its defaults.
3. Enter a concise complete scene in **Global prompt**.
4. If using regions, include the exact intended number of people in the global prompt.
5. Draw boxes large enough to contain each intended subject and action.
6. Add one regional prompt per box.
7. Route LoRAs deliberately.
8. Use **Preview unified prompt** to check subject count and front-to-back order.
9. Generate with the baseline sampler/scheduler recommended for your model.

The current output remains visible while a replacement job runs. A completed output is
selected on the canvas and appears first in **Assets → Outputs**.

### Global and regional prompts

The global prompt should describe the unified scene: setting, total subject count, overall
composition, lighting, and relationships.

Regional prompts should describe what belongs in each box. Avoid independently restating a
complete scene in every box; doing so encourages duplication. For character LoRAs, keep the
training trigger in the LoRA's **Training trigger** field rather than repeating it throughout
visible prompts.

### Phrase emphasis

Select exact text in the global or selected regional prompt, set **Selected phrase boost**, and
choose the matching emphasis action. Emphasis is stored by phrase and occurrence. Editing away
the selected text removes or marks the stale emphasis instead of silently applying it elsewhere.

## Regions and subject boxes

### Ordinary region box

Use for:

- non-person objects;
- scenery and architecture;
- foreground/background bands;
- localized edit targets;
- areas that should receive a standard regional LoRA.

**Spatial role**:

- **Auto** chooses behavior from the box width.
- **Subject target** treats the region as a localized subject.
- **Background band** treats it as scene/background context.

### Subject box

A subject box adds:

- a person-oriented regional prompt;
- a separate **Face identity prompt**;
- a filled volumetric mannequin;
- optional pose-gating occupancy and exclusive subject ownership.

The subject box itself still participates in unified regional prompting when global pose gating
is off. **Enable this mannequin** controls whether that subject contributes to pose gating.

### Front-to-back order

The Regions list is the authoritative front-to-back order. **Move forward** moves a region
toward the front of overlaps; **Move backward** moves it behind peers. This ordering affects the
unified prompt and overlap ownership.

### Avoiding extra people and prompt leakage

- State the exact person count in the global prompt.
- Give each person one subject box.
- Make boxes tall/wide enough to contain the intended body and action.
- Do not use a small face-only box for a full-body subject.
- Use **Separate overlapping subject targets** when boxes overlap.
- Keep identity details in that subject's identity prompt and character LoRA route.
- Raise guidance gradually; extreme values can reduce image coherence.
- Confirm the unified prompt before changing many controls at once.

## Volumetric mannequin controls

Subject mannequins are filled body volumes, not a ControlNet input. Their geometry creates
early denoising support and ownership masks inside the existing regional pipeline.

### Direct canvas controls

- Circular joint handles move individual joints.
- The center head handle moves the head ellipse.
- The right and bottom head handles resize its horizontal and vertical radii.
- The gold torso-center handle moves the neck, shoulders, hips, and attached head together.
- The cream rotation handle above the torso rotates the entire figure around the torso center.
- The four pink diamond handles move a complete arm or leg as a rigid three-joint group.
- Individual joints remain available after a group move for local articulation.
- A visible neck volume connects the head to the torso.

Use **Standing**, **Squatting**, and **Mirror** as starting points. Limbs may extend outside the
subject box for interactions.

### Enabling the gate

Open **Advanced → Volumetric pose gating** and enable
**Constrain generation to subject mannequins**.

The effective transitions are:

```text
hard gate steps + soft gate steps + normal Steps
```

- **Hard gate steps** restrict early denoising updates to mannequin support.
- **Soft gate steps** gradually release that restriction.
- **Normal steps** are the ordinary unrestricted regional transitions.
- **Effective total** is the actual total number of sampler transitions.

Start with `2 hard + 2 soft` and **Scheduler default**. Adding gate steps without adjusting
normal Steps increases total computation.

### Soft release

- **Cosine** — smooth default transition.
- **Linear** — releases at a constant rate.
- **Exponential** — remains stronger earlier, then releases faster.
- **Stepped** — releases in discrete levels.

### Sigma schedule

- **Scheduler default** preserves the selected scheduler's trajectory over all effective steps.
- **Phase weighted** assigns percentages of the denoising trajectory to hard, soft, and normal
  phases. Balanced, Pose lock, and Gentle presets are available.
- **Advanced curve** exposes every normalized sigma boundary.

Nondefault sigma allocation is experimental, especially with Turbo/distilled models. If a
pose-gated output becomes grainy or low quality, return to **Scheduler default** before changing
prompts or LoRAs.

See [Volumetric subject pose gating](subject_pose_control.md) for the implementation model.

## LoRA routing

Each LoRA has:

- an enable toggle;
- **Strength**;
- independent bindings for generation, edit-reference, and edit-target layers;
- global or selected-region assignment;
- a routing mode.

### Standard regional

Use for style, clothing, concepts, objects, or ordinary character LoRAs that should affect the
assigned box through standard regional conditioning.

### Character identity (face)

Use for a trained character identity. Assign it to one or more subject regions. K2 inserts the
**Training trigger** into the subject's identity anchor and spatially gates the LoRA contribution
to reduce global identity leakage.

Do not duplicate the trigger in every visible prompt. A LoRA can still leak if its training is
strongly entangled with pose, outfit, framing, or background; spatial routing limits inference
scope but cannot remove bias learned into the weights.

### Diagnosing a weak or incompatible LoRA

1. Generate once with only the baseline/distillation LoRA.
2. Add one identity LoRA globally to confirm that the model can respond to it.
3. Route that LoRA to one large subject box.
4. Use its exact training trigger and a simple portrait prompt.
5. Confirm that the LoRA targets Krea 2 transformer modules.
6. Compare the event-log LoRA delta measurement.
7. If identity remains absent globally, the issue is likely the LoRA/training rather than
   spatial gating.

## Image editing

Edit mode requires a cloud source image.

### Reference layer

The reference layer describes the original layout and identities. Draw boxes around important
source subjects/areas and enter their original prompts. Reference LoRAs can be routed on this
layer.

### Edit targets

Switch to **Edit targets** and draw boxes where changes should happen. The global field becomes
the overall edit instruction; target prompts describe local changes.

### Common workflow

1. Load/upload or select a source from Assets.
2. Describe the original image on the Reference layer.
3. Create target boxes and local instructions.
4. Start with low Denoise and **Preserve reference identity** enabled.
5. Increase Denoise only when the requested change is too weak.

## Face refinement

Face refinement is optional and separate from normal generation/editing.

1. Use a PNG source or **Use latest first pass**.
2. Choose **Detect faces** or draw one or more manual lassos.
3. Select face indices to refine.
4. Assign at least one active character LoRA to a subject region.
5. Run **Refine faces**.

If the detector model is missing, K2 offers to install the pinned detector. Choosing **Not now**
does not affect Generate or Edit.

Manual lasso controls:

- **Draw lasso** enables freehand selection.
- **Undo lasso** removes the last manual path.
- **Clear lassos** removes all manual paths.
- **Select all / Select none** changes detected-face selection.

## Advanced generation settings

### Sampling and image

| Setting | Practical effect |
| --- | --- |
| **Sampler** | Numerical method used for each denoising transition. Use model-recommended defaults before experimenting. |
| **Scheduler** | Distributes noise/sigma across steps. Includes `bong_tangent`; scheduler choice can materially change distilled-model quality. |
| **Steps** | Normal generation transitions. Pose hard/soft steps are additional. |
| **Seed value** | Reproduces initial noise when all other inputs are unchanged. |
| **Fixed** | Reuses the entered seed. Disabled for multi-run batches. |
| **Random** | Chooses a fresh seed for each run. |
| **Increment** | Uses consecutive seeds and advances the displayed seed. |
| **Width / Height** | Output size, in pixels; values are constrained to multiples of 16. |

### Spatial guidance

| Setting | Practical effect |
| --- | --- |
| **Inside boost** | Strengthens a region's prompt/attention contribution inside its target. Raise gradually when a regional concept is ignored. |
| **Outside penalty** | Suppresses that regional contribution outside its target. Raise gradually to reduce leakage. Excessive values can make subjects disconnected or rigid. |
| **Spatial falloff** | Width in pixels of the soft transition around a region. Larger values blend more naturally but permit more crossover; smaller values isolate more sharply. |
| **Late-step scale** | Fraction of regional spatial guidance retained near the end of denoising. Lower values let the final image unify; higher values preserve placement but can retain boundaries. |
| **Separate overlapping subject targets** | Creates competition/ownership where subject regions overlap, reducing one subject's prompt or LoRA entering another. |
| **Make subjects fill their boxes** | Encourages subject occupancy across the region rather than collapsing into a small part of it. |
| **Relax spatial guidance during late steps** | Reduces spatial restrictions late so lighting, texture, edges, and interactions can become one coherent scene. |
| **Adapt spatial guidance from regional LoRA delta** | Measures how strongly regional LoRAs change the model and adjusts spatial guidance accordingly. |
| **LoRA delta response** | Controls adaptation sensitivity. At `0`, measured LoRA delta has no effect; at `1`, adaptation responds fully. |

Hypothetical LoRA-delta example: two characters have the same configured LoRA strength, but
Character A's LoRA produces a small model delta while Character B's produces a large delta.
With adaptation enabled, K2 applies more containment pressure to B, which has more leakage risk,
while avoiding unnecessary pressure on A. The response control scales how strongly that measured
difference changes guidance.

Start at:

```text
Inside boost: 1
Outside penalty: 1
Spatial falloff: 128
Late-step scale: 0.35
Relax late guidance: enabled
```

Change one value at a time.

### Batch mode

**Run generation in batch mode** creates the selected number of independent runs. Jobs are
submitted and executed sequentially. Fixed seed is not allowed because identical settings would
create identical initial noise.

### Unified spatial prompting

**Use unified spatial prompting** compiles global and ordered regional clauses into one scene
prompt while spatial fields separately guide latent cells, attention, and LoRA scope.

**Preview unified prompt** displays the exact compiled text and resolved front-to-back order.
Use it when people are duplicated, actions are ignored, or a background prompt behaves as a
subject.

### GPU memory

See [RAM and VRAM management](#ram-and-vram-management).

### Post-upscale

- **Post-upscale after releasing Krea VRAM** unloads the Krea generation allocation before
  upscaling.
- **Output scale** selects 2× or 4×.
- **CPU Lanczos** is deterministic and requires no model.
- **Neural model (tiled GPU)** requires an uploaded/selected compatible upscaler.

### Projector

The projector is an expert global model-vector adjustment.

- **Preset** chooses a known vector or Custom.
- The 12 numeric vector entries directly edit the custom vector.
- **Global multiplier** scales the vector.
- **Face identity protection** reduces projector influence on protected identity conditioning.

Leave the projector disabled unless you understand the target model and the intended preset.

#### Projector preset defaults and source provenance

K2's actual default state is:

| Setting | Default |
| --- | --- |
| **Enable projector** | Off |
| **Preset shown while off** | FilterBypass2 |
| **Global multiplier** | `1.0` |
| **Face identity protection** | `1.0` |

Consequently, FilterBypass2 being the *default preset selection* does not mean that any
projector delta is applied by default. The vector has no effect until **Enable projector** is
turned on. The same off-by-default behavior applies to the image-edit reference projector.

The preset table in K2 is:

| Preset | 12 projector-column deltas |
| --- | --- |
| **FilterBypass2** | `0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, 0, 0` |
| **FilterBypass3** | `0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, -0.6094, 0` |
| **skc3vo** | `-5.44, -16.11, -37.11, -50.39, -70.70, -39.45, -39.84, -143.7511, -51.17, -89.06, -60.94, -11.28` |
| **z0jglf** | `-13.60, -40.275, -92.775, -159.75, -176.75, -98.625, -99.60, -359.3778, -127.925, -222.65, -152.35, -28.20` |

These are community-derived reference weights, not official Krea defaults, guarantees, or
recommended settings. Their provenance is recorded here so the numbers can be audited:

- **FilterBypass2 and FilterBypass3:** the
  [Sentinel7/krea2 model-card revision](https://huggingface.co/Sentinel7/krea2/commit/8a6c6313e1e34e1e7e26aac30ec2d35cee75b6ea)
  records the sparse layer values and links the
  [original FilterBypass2 Civitai model/version](https://civitai.com/models/2746817?modelVersionId=3089754).
  The FilterBypass3 reference file is preserved at a
  [pinned Sentinel7/krea2 revision](https://huggingface.co/Sentinel7/krea2/tree/41a18fe8d1826b09c6d53d3ca4204afd9e96dbef/2728234/3067151).
- **skc3vo:** the 268-byte reference file is available in the
  [Comfy-Org/Krea-2 repository at a pinned revision](https://huggingface.co/Comfy-Org/Krea-2/blob/7b75ff3c61d88257ab29630be389af9adace3fd3/loras/skc3vo.safetensors).
- **z0jglf:** both `skc3vo.safetensors` and `z0jglf.safetensors` are preserved together in the
  [andrewwe/kr2 repository at a pinned revision](https://huggingface.co/andrewwe/kr2/tree/a096071125550d7d021adc19b5dd863a31d8aeaf).
  The z0jglf vector is the skc3vo vector scaled by `2.5`, subject to the displayed decimal
  precision.

The executable source of truth is
[`src/k2_region_lab/projector.py`](https://github.com/soomrenald/k2lab_runpod/blob/k2lab_pose/src/k2_region_lab/projector.py).
The feature entered the original desktop K2 implementation in
[commit `bf82ac2`](https://github.com/soomrenald/k2lab/commit/bf82ac2d9466e1f679993a5b9a50c9389dd5ad9e).
When documentation and behavior disagree, use the checked-out source and its tests to determine
what the running revision applies.

## Advanced image-edit settings

| Setting | Practical effect |
| --- | --- |
| **Steps** | Number of edit denoising transitions. |
| **Seed · fixed** | Reproducible edit noise. |
| **Denoise** | Overall change strength. Low values preserve the source; high values redraw more. |
| **Reference retention** | Strength of source/reference conditioning. |
| **Latent feather** | Softens the edit mask in latent space, reducing hard boundaries. |
| **Composite feather** | Softens the final pixel-space blend into the source. |
| **Inside boost / Outside penalty / Spatial falloff / Late-step scale** | Same regional principles as generation, applied to edit layers. |
| **Preserve reference identity** | Protects subject identity from unnecessary change. |
| **Edit entire image** | Applies edit conditioning globally rather than only to target regions. |

The LoRA delta adaptation, competition, fill, memory diagnostics, and reference projector are
also available in Edit mode where applicable.

## Face-refinement settings

| Setting | Practical effect |
| --- | --- |
| **Steps** | Refinement transitions for each selected face crop. |
| **Seed** | Fixed refinement noise. |
| **Denoise** | How much the face may change. |
| **Padding** | Context included around the detected/lasso face. |
| **Edge feather** | Softness of the face-mask edge. |
| **Blend** | Strength of the refined crop in the final composite. |
| **Regional LoRA scale** | LoRA strength used for face refinement. |
| **Detector threshold** | Minimum detection confidence; lower values find more candidates. |
| **Crop working resolution** | 256, 512, 768, or 1024-pixel face working size. |
| **Detector device** | Auto, CPU, or NVIDIA CUDA. |

Start with low Denoise and moderate Blend. High Denoise can improve likeness but also change
expression, age, lighting, or head geometry.

## Assets, uploads, and downloads

### Assets

Asset tabs separate inputs, outputs, projects, baseline models, text encoders, VAEs, LoRAs,
upscalers, ControlNet inventory, and face-detection files.

Local uploads:

- are SHA-256 hashed in the browser;
- are queued sequentially;
- can be paused, resumed, retried, or cancelled;
- resume from accepted chunks after interruption;
- detect duplicates by digest.

Every file has a checkbox and Delete action. **Select all** plus selective unchecking supports
batch cleanup. Deletion is permanent for those workspace files.

Outputs default to newest first and can be sorted by newest, oldest, name, or size. Outputs have
thumbnail and enlarged previews. The canvas **Download image** button downloads the currently
displayed result without opening Assets.

### Provider transfers

Transfers support Civitai and Hugging Face.

1. Add an optional download-only/read-only token.
2. Paste a canonical model/repository/file URL.
3. Select the destination category.
4. Choose **Inspect before download**.
5. Review the resolved filename/size.
6. Start the provider download.

Provider downloads are queued and executed sequentially. Large files move directly from the
provider to the Pod.

Safetensors is preferred. Pickle-based formats require explicit unsafe-format confirmation.
K2 rejects embedded URL credentials and redirects to unapproved hosts.

## Jobs, batches, events, and timeout recovery

Generation, edits, face refinement, uploads, and provider downloads are serialized within their
controlled queues to avoid worker and network concurrency failures.

The Events dock shows:

- model loading;
- memory checks;
- LoRA application and measured deltas;
- pose phase and denoising progress;
- output completion;
- safe error messages.

### Job receipt timeout

A POST timeout does not prove that the Pod rejected the job. It may have accepted the command
while the response was lost.

When this happens, K2 displays **Job receipt was not received** and saves the exact command ID
and request in local browser storage.

- **Recover submission** resends the same idempotent command ID. The agent returns the existing
  job or creates it once; it does not create a duplicate.
- **Cancel all remote work** cancels running/queued jobs and unloads the worker. Models,
  projects, inputs, and completed outputs remain.

Recovery remains available after a page refresh. Do not dismiss the situation by starting a
new job with a new command ID.

## RAM and VRAM management

RAM and VRAM are different:

- **VRAM** is GPU memory used by model weights, activations, latent tensors, and LoRAs.
- **RAM** is system memory in the Pod container.
- Linux **file cache** may appear in raw RAM usage but is reclaimable and is not a memory leak.

### Execution mode

- **Auto** resolves to High VRAM on GPUs with at least 40 GiB and Dynamic VRAM below that.
- **High VRAM** keeps more model data on the GPU for performance.
- **Dynamic VRAM** balances GPU placement and offload.
- **Low VRAM** maximizes offload and is slowest.

### VRAM reserve

**VRAM reserve · GiB** is free GPU memory K2 should preserve. Retention and allocation decisions
use the actual Pod GPU, not host-system memory or Linux file cache.

### Keep baseline model loaded

When enabled, K2 retains only the resolved baseline transformer between runs, and only in
resolved High VRAM mode while the reserve remains free. Regional/character LoRAs are released.

The baseline may occupy roughly 25 GiB depending on the model. This is expected VRAM residency,
not a leak. Model changes, pressure, OOM recovery, other execution modes, and **Release worker
memory** discard it.

### System RAM safeguard

**Enable system RAM safeguard** performs preflight checks against non-reclaimable memory.
Disabling it allows jobs to proceed without K2's guard, but the Linux cgroup and kernel OOM
killer still apply.

Diagnostics:

- **Allocatable now** — practical memory available after accounting for real use and reclaimable
  cache.
- **Actual non-cache use** — non-reclaimable memory used by processes/anonymous allocations.
- **Clean reclaimable cache** — filesystem cache Linux can evict.
- **Raw cgroup charge** — total accounting value, including cache.
- **Anonymous/process RAM** — process-backed memory.
- **Shared + dirty files** — shared memory and non-clean file pages.
- **Worker** — Released, Active, or Baseline resident.

Use **Refresh RAM** for current telemetry and **Release worker memory** to cancel active jobs and
discard the resident worker/model.

## Pod lifecycle and storage safety

The workspace popover shows compute price, storage price, lease time, readiness, and the RunPod
Pod ID.

- **Extend session** extends a limited lease.
- **Start GPU** starts stopped compute.
- **Start GPU without time limit** continues billing until manual stop.
- **Stop GPU now** stops/terminates compute according to workspace type.
- **Release worker memory** unloads the model without stopping the Pod.
- **Migrate to portable storage** performs a verified SHA-256 copy.
- **Connect migrated Pod** verifies a manually migrated persistent Pod before changing the saved
  provider ID.
- **Delete cloud workspace** has storage consequences described in its confirmation.

Migration can temporarily bill both source and target compute/storage. The original Pod remains
stopped and retained until you test the portable copy and separately confirm deletion.

Read [RunPod workspace operations](runpod_workspace_operations.md) before migration.

## Updating K2 and a Pod without losing data

An update has three identities that must agree:

1. local source/UI commit;
2. immutable `image@sha256:digest`;
3. `K2LAB_IMAGE_VERSION`.

### Update the local application

Stop the original K2 control-plane process, then:

```bash
cd ~/k2lab_runpod
git switch k2lab_pose
git pull --ff-only
./scripts/k2lab-runpod --image 'ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:71d034c346a5a2c1bb21a90df507d9a0b2dfb3f3e718e5380a9526e29b65b2c5' --image-version '0.2.0-pose.2'
```

If another K2 instance is still listening on port 8000, the launcher follows its log and does
not replace its in-memory configuration. Stop that original process first.

### Update a workspace Pod

Changing local configuration affects newly created Pods; it does not mutate an existing Pod.

Before any Pod replacement:

1. Save/download important project files.
2. Confirm models and outputs are on the persistent workspace volume.
3. Record the workspace ID, Pod ID, GPU type, volume size, and current image digest/version.
4. Stop generation and transfers.
5. Use the supported workspace migration/reconnection flow or edit/recreate the Pod while
   retaining the same persistent volume and K2 environment identity.
6. Set both the exact immutable image digest and matching `K2LAB_IMAGE_VERSION`.
7. Use **Connect migrated Pod** when a new persistent Pod ID must replace the saved ID.

K2 verifies workspace identity, agent credential, immutable image, GPU type, mount path, and
volume size. Do not weaken these checks to force an incompatible Pod to connect.

For a Portable workspace, stop and start after updating the local configured image; the
network volume remains and the newly created ephemeral Pod uses the configured immutable image.

Never delete the old persistent volume merely to update software.

## Troubleshooting

### `address already in use` or port 8000 is busy

The current launcher detects an already-running K2 control plane and follows
`control-plane.log`. If another application owns port 8000, stop it or run:

```bash
./scripts/k2lab-runpod --port 8001
```

Check listeners with:

```bash
ss -ltnp | grep ':8000'
```

### The Pod agent image version does not match the workspace

The agent's `K2LAB_IMAGE_VERSION` differs from the version recorded by the local controller.
Use the exact matching digest and version pair. Restarting only the browser does not change
either process.

### The migrated Pod does not use this workspace's immutable image

The new Pod's container image does not resolve to the digest recorded for the workspace. Edit
or recreate it using the exact immutable digest, retain the persistent volume and workspace
environment, then verify again.

### Workspace agent not responding

1. Open the workspace popover and check readiness.
2. Open the RunPod Pod page and inspect Docker startup/logs.
3. Confirm the Pod is running and port 8080 health checks succeed.
4. If a job POST timed out, use **Recover submission** rather than submitting another job.
5. If recovery cannot obtain a receipt, use **Cancel all remote work** and retry after readiness
   returns.

Repeated HTTP 200 polling means the controller is reachable; it does not mean a queued GPU job
has started.

### A generation stayed queued for minutes, then ran normally

A previous timed-out POST may have created an unobserved job that held the single worker queue.
Current versions preserve the command ID and show the recovery dialog. Recover or cancel all
remote work instead of creating another request.

### Krea 2 model loading failed

- Verify the selected diffusion model, text encoder, and VAE in Setup.
- Confirm the files completed download and pass safetensors validation.
- Confirm they are compatible with Krea 2.
- Check actual VRAM and system RAM diagnostics.
- Use Dynamic/Low VRAM or a larger GPU.
- Release worker memory before changing the baseline model.

### Generation failed while applying LoRA or sampling settings

- Disable all optional LoRAs and run the baseline.
- Add LoRAs one at a time.
- Verify each LoRA targets Krea 2 transformer modules.
- Reduce LoRA strength.
- Return sampler/scheduler and pose sigma schedule to known-good defaults.
- Check Events for the specific worker error and measured LoRA delta.
- Release worker memory after an OOM or incompatible-model failure.

### RAM appears stuck at 85–95%

Read **Actual non-cache use** and **Clean reclaimable cache**, not only Raw cgroup charge.
Downloaded/model files often remain as clean Linux file cache and are evicted automatically.
Use **Release worker memory** if the Worker says Active or Baseline resident and you want to
discard real resident allocations.

### VRAM remains high while idle

If **Keep baseline model loaded between runs** is enabled in High VRAM mode, baseline weights
remain resident intentionally. LoRAs should not remain loaded. Disable retention or use
**Release worker memory** to return to a released state.

### The output does not appear on the canvas

- Wait for the job to reach Completed in Events.
- Open **Assets → Outputs** and preview the newest file.
- Reload the studio; K2 restores the newest verified output.
- Use the output's Download action to confirm the file is valid.
- Transient output delivery is retried automatically; a permanently blank image indicates an
  agent/output access issue rather than sampling.

### Additional people appear

- State the exact number of people globally.
- Ensure each person has one sufficiently large subject box.
- Avoid multiple full-scene descriptions in regional prompts.
- Check front-to-back order and **Preview unified prompt**.
- Enable subject competition for overlaps.
- Use character-identity routing only on the intended subject.
- Make action boxes large enough to include the entire interaction.

### Regions look like disconnected rectangles or have boundary artifacts

- Return Inside boost and Outside penalty toward `1`.
- Increase Spatial falloff.
- Keep Relax late spatial guidance enabled.
- Reduce Late-step scale.
- Use one coherent global scene prompt.
- Avoid treating every box as an independent complete image.
- Return pose sigma scheduling to Scheduler default.

### Prompt or LoRA leakage remains high

- Confirm routing in the currently active layer.
- Use Character identity routing for identity LoRAs.
- Increase Outside penalty gradually.
- Reduce Spatial falloff cautiously.
- Enable subject competition.
- Make boxes match the full latent area the subject/action needs.
- Inspect whether the LoRA itself is globally entangled by testing it alone.

### Pose output is grainy or poor quality

- Use Scheduler default sigma allocation.
- Start with 2 hard and 2 soft gate steps.
- Remember gate steps are added to normal Steps.
- Reduce hard/soft steps before changing the global prompt.
- Use Cosine release.
- Ensure mannequin coverage is neither tiny nor nearly the whole canvas.

### Face detector is missing

Use **Faces → Detect faces** and accept the pinned detector installation, or upload/select a
detector in Setup. Normal Generate and Edit remain available without it.

### Provider download is stuck

- Downloads are intentionally sequential; inspect the head of the queue.
- Check the transfer state and bytes/throughput.
- Cancel while retaining resumable data, then Retry/resume.
- Verify the provider token and canonical URL.
- For Hugging Face, verify repository allow patterns.
- Do not embed credentials in URLs.
- Inspect the Pod's network and available workspace capacity.

### `unsafe URL` during a Civitai download

Use a canonical Civitai model/download URL and a saved download-only token. K2 validates every
redirect and refuses credentials embedded in URLs. If the provider returns a redirect to an
unapproved host, inspect the resolved file and report the event/error code rather than disabling
URL safety globally.

### Where to find logs

Local control-plane log:

```text
~/.local/state/k2-region-lab/control-plane.log
```

RunPod container/agent logs are available from the Pod page in the RunPod console. The studio
**Events** dock provides the generation-specific, redacted event stream.

When reporting a problem, include:

- local commit (`git rev-parse HEAD`);
- image digest and image version;
- Pod GPU and VRAM;
- selected model filenames;
- job ID and command ID if shown;
- the relevant Events lines;
- the local control-plane error message;
- whether the issue reproduces with optional LoRAs/pose gating disabled.
