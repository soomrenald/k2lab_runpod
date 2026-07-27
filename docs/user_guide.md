# K2 Region Lab User Guide

This guide explains K2 Region Lab from first launch through regional generation, LoRA routing,
image editing, face refinement, memory diagnostics, and RunPod maintenance. It describes the
`main` release `0.1.17` and project schema version 19.

## Contents

- [How K2 is organized](#how-k2-is-organized)
- [Installation and first launch](#installation-and-first-launch)
- [Creating a cloud workspace](#creating-a-cloud-workspace)
- [Studio layout](#studio-layout)
- [Installing and selecting models](#installing-and-selecting-models)
- [Projects and PNG metadata](#projects-and-png-metadata)
- [Generating an image](#generating-an-image)
- [Regions and spatial roles](#regions-and-spatial-roles)
- [LoRA routing](#lora-routing)
- [Image editing](#image-editing)
- [Face refinement](#face-refinement)
- [Advanced generation settings](#advanced-generation-settings)
- [Projector preset defaults and source provenance](#projector-preset-defaults-and-source-provenance)
- [Advanced image-edit settings](#advanced-image-edit-settings)
- [Face-refinement settings](#face-refinement-settings)
- [Assets, uploads, and downloads](#assets-uploads-and-downloads)
- [Jobs, batches, events, and request sequencing](#jobs-batches-events-and-request-sequencing)
- [RAM and VRAM management](#ram-and-vram-management)
- [Pod lifecycle and storage safety](#pod-lifecycle-and-storage-safety)
- [Updating K2 and a Pod without losing data](#updating-k2-and-a-pod-without-losing-data)
- [Troubleshooting](#troubleshooting)

## How K2 is organized

The browser interface and control plane run locally. The expensive work runs on your RunPod
GPU. Persistent files live on RunPod storage.

```text
Browser
  ↕ local loopback
K2 control plane on your computer
  ↕ authenticated RunPod and agent requests
RunPod workspace agent
  ↕ sequential operations
GPU worker and persistent workspace files
```

The control plane records which immutable image and agent version belong to a workspace. An
arbitrary Pod or one using a different image is rejected instead of silently running
incompatible code.

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
the repository:

```bash
git clone https://github.com/soomrenald/k2lab_runpod.git
cd k2lab_runpod
```

Start K2 with the signed release image and matching version:

```bash
./scripts/k2lab-runpod --image 'ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:f1dbf619c615bb55be781eada7b11d28a70818d1dba4139814e1ea41645f8717' --image-version '0.1.17'
```

The entire line must be one shell command. A separate line beginning with
`--image-version` produces `bash: --image-version: command not found`.

The browser normally opens automatically. If it does not, open:

```text
http://127.0.0.1:8000
```

Subsequent launches use the saved image configuration:

```bash
./scripts/k2lab-runpod
```

Useful launcher options:

| Option | Effect |
| --- | --- |
| `--image IMAGE@sha256:DIGEST` | Saves a public immutable workspace image. |
| `--image-version VERSION` | Saves the human-readable version expected from the agent. |
| `--state-dir PATH` | Uses a different private local state directory. |
| `--port PORT` | Uses another loopback port; allowed range is 1024–65535. |
| `--no-open` | Starts without opening a browser. |
| `--no-follow` | If K2 is already running, report it and exit instead of following its log. |

### RunPod API key

Use a restricted user-owned key with only the Pod, volume, inventory, and billing-read
permissions needed by K2. The key is encrypted using the local `credential.key`; it is not
stored in project files or generated images.

## Creating a cloud workspace

### Workspace type

**Persistent Pod**

- The regular persistent volume belongs to one Pod.
- Stop retains the Pod and volume.
- Starting again may wait for the selected GPU type.
- Deleting the workspace permanently deletes that regular volume.

**Portable workspace**

- Files live on an independent RunPod network volume.
- Each start can create a compatible temporary Pod in the volume's datacenter.
- Stop terminates the temporary Pod but retains the network volume.
- Deleting the application workspace also retains the network volume for safety.
- Network-volume storage continues billing until separately deleted in RunPod.

Choose a Portable workspace when data portability matters more than the simplicity of one
long-lived Pod.

### GPU priority

Select GPUs in the order K2 should try them. The interface shows VRAM and current hourly price.

- **Secure Cloud** uses RunPod Secure Cloud inventory.
- **Community Cloud** may cost less but is unavailable for portable network-volume workspaces.
- **Use interruptible compute** can reduce cost, but the Pod may stop without notice.

Large Krea 2 workflows benefit from high-VRAM GPUs. Lower-memory GPUs can use Dynamic or Low
VRAM execution at the cost of additional model movement and longer generations.

### Storage and safety controls

| Control | Meaning |
| --- | --- |
| **Container disk** | Runtime disk used by the image and environment. |
| **Workspace volume** | Persistent capacity for models, projects, inputs, and outputs. |
| **Network volume** | Existing portable storage to attach, or a new volume to create. |
| **Datacenter** | Location for a new portable volume and its compatible Pods. |
| **Idle stop** | Stops compute after the configured idle period. |
| **Hard session limit** | Maximum duration of a limited lease. |
| **No time limit** | Disables the lease deadline; billing continues until manual stop. |

Compute is billed while the Pod is running. Persistent storage is billed while compute is
stopped.

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

Saving does not automatically download a copy to your computer. Download important projects
from **Assets → Projects** for a local backup.

### Left rail

Modes:

- **Generate** — text-to-image, regional prompting, LoRAs, and post-upscale.
- **Edit** — source-based editing with reference-layout and edit-target layers.
- **Faces** — optional detection, manual lassos, and selected-face refinement.

Utilities:

- **Assets** — uploads and persistent workspace files.
- **Transfers** — sequential Civitai/Hugging Face downloads.
- **Events** — resizable job and operation history.
- **Setup** — deterministic model selections and output filename prefix.

### Canvas toolbar

- **Load image / Replace image** uploads a local source image.
- **Download image** downloads the image currently shown on the canvas.
- **Clear canvas** removes the displayed source/result from the current view.
- **Draw region** lets you drag a new regional box.

Dragging the body of a selected box moves it. Eight handles resize it. Select a box before
moving or resizing it.

### Inspector

The right inspector has four tabs:

- **Prompt** — global prompt, regional prompt, identity prompt, and phrase emphasis.
- **Regions** — ordering, enable/disable state, selection, and removal.
- **LoRAs** — strength and routing for each current layer.
- **Advanced** — sampler, scheduler, spatial guidance, memory, upscale, and expert settings.

## Installing and selecting models

Generation requires a compatible:

1. diffusion/transformer model;
2. text encoder;
3. VAE.

Open **Transfers** to download from a provider or **Assets** to upload a local file. Then open
**Setup** and select the exact file for each role.

The **Face detector (face tools only)** is optional. Missing face detection does not block
Generate or Edit. When you first use **Detect faces**, K2 can install the small pinned detector
model and select it automatically.

**Generation output filename prefix** controls the beginning of output filenames. It must be
1–128 characters and cannot contain `/`, `\`, or a null character.

Projects remember model filenames. When opening a project, K2 resolves those names to opaque
files in the current workspace. Resolve any missing selection in Setup before submission.

## Projects and PNG metadata

The project format stores:

- prompts and phrase emphasis;
- regions, ordering, spatial roles, and enabled states;
- LoRA names, strengths, layer bindings, routing, and character trigger phrases;
- generation, edit, face, projector, upscale, and memory settings;
- selected model filenames;
- canvas and source references.

**Open** loads JSON directly from your computer. **Import PNG** reads the embedded project
document from a K2 output PNG. LoRAs are rebound by normalized filename so imported metadata
uses the real LoRA name when that file exists in the workspace.

A project document does not contain model weights. Keep separate backups of important projects,
inputs, outputs, and custom models.

## Generating an image

A reliable first generation:

1. Select the baseline model, text encoder, and VAE in **Setup**.
2. Leave spatial guidance at its defaults.
3. Enter a concise, complete scene in **Global prompt**.
4. If using regions, state the exact intended number of people globally.
5. Draw boxes large enough to contain each subject, object, or interaction.
6. Enter one focused prompt per region.
7. Set spatial roles and front-to-back order.
8. Route LoRAs deliberately.
9. Use **Preview unified prompt** to inspect the compiled text and order.
10. Generate with baseline sampler/scheduler settings known to work with the model.

The current output remains visible while a replacement job runs. A completed output is selected
on the canvas and appears first in **Assets → Outputs**.

### Global and regional prompts

The global prompt describes the unified scene: setting, total subject count, composition,
lighting, camera, and relationships.

Regional prompts describe what belongs in each box. Avoid repeating a complete standalone scene
inside every region. Repetition can increase duplicate people and disconnected compositions.

### Phrase emphasis

Select text in the global or regional prompt, set **Selected phrase boost**, and choose the
matching emphasis button. K2 stores the phrase, occurrence, scope, and strength. Editing the
source prompt can invalidate an emphasis; invalid entries are shown and should be recreated.

## Regions and spatial roles

Choose **Draw region**, then drag a rectangle on the canvas. A region stores:

- a unique name;
- its rectangle;
- a regional prompt;
- an optional face-identity prompt;
- a spatial role;
- front-to-back priority;
- enabled state.

### Spatial roles

**Subject target**

Use for a person, character, animal, or distinct foreground subject. Subject competition and
fill logic can apply to it.

**Background band**

Use for scenery, sky, walls, floor, architecture, or broad environmental areas. It participates
in unified prompting without being treated as a subject.

**Auto**

K2 infers the role from box width. Explicit roles are easier to reason about in complex scenes.

### Ordering

The Regions tab is ordered front to back. Use **Move forward** and **Move backward** to describe
occlusion and ownership. The same order appears in **Preview unified prompt**.

### Box sizing

A box should cover the area needed by the complete subject or interaction, not only the face or
torso. Boxes that are too small encourage cropping, duplication, or ignored actions. Large
overlaps require careful ordering and subject competition.

## LoRA routing

Each LoRA has one strength and independent bindings for:

- generation;
- edit reference;
- edit targets.

### Global

The LoRA affects the whole image. Use this for styles, baseline adapters, and concepts intended
to influence every region.

### Standard regional

The LoRA is assigned to one or more boxes and spatially gated with those regional targets. Use
this for regional styles, garments, objects, or non-identity concepts.

### Character identity

Character-identity routing requires regional scope and a training trigger. K2 inserts the
trigger into that region's hidden identity anchor. Do not duplicate it in the visible prompt.

Use the region's **Face identity prompt** for stable identity characteristics: person class,
face shape, hair, eyes, and other persistent facial traits. Keep clothing and scene context in
the regional prompt unless they are truly part of identity.

### Diagnosing a LoRA

Start with the baseline and add one LoRA at a time. If a LoRA changes unrelated regions:

- verify that **Global** is off;
- verify the assigned box;
- select the intended routing mode;
- reduce strength;
- increase the outside penalty gradually;
- test whether the LoRA is intrinsically entangled by running it alone.

## Image editing

Edit mode has two layers.

### Reference layout

This describes what the source contains. Reference prompts, regions, phrase emphasis, LoRAs,
and the reference projector condition source understanding and preservation.

### Edit targets

This describes what should change. Draw boxes around target areas and enter an instruction for
each one. The global edit instruction is combined with target prompts.

### Suggested workflow

1. Load or select the source image.
2. On **Reference layout**, describe the original image and optionally add reference regions.
3. Switch to **Edit targets**.
4. Draw boxes around areas to change.
5. Enter a concise edit instruction globally or per target.
6. Start with low Denoise.
7. Increase Denoise only when the requested edit is too weak.
8. Adjust latent/composite feathering if boundaries are visible.

## Face refinement

Face refinement operates on a cloud source image.

1. Load an image or choose **Use latest first pass**.
2. Open **Regions** and choose **Detect faces**.
3. If needed, approve installation of the pinned detector.
4. Select detected faces.
5. Use **Draw lasso** for an additional face that detection missed.
6. Confirm the generation-reference prompt and LoRAs.
7. Start with low Denoise and moderate Blend.
8. Choose **Refine faces**.

Detection and manual lassos can be combined. The detector model is required only for automatic
detection; normal generation, editing, and manual source use do not require it.

## Advanced generation settings

### Sampling and seed

| Setting | Practical effect |
| --- | --- |
| **Sampler** | Numerical method used for each denoising transition. Start with a setting known to work with the baseline. |
| **Scheduler** | Distributes noise/sigma across steps. Includes `bong_tangent`; scheduler choice can materially change distilled-model quality. |
| **Steps** | Number of denoising transitions. Distilled models often expect a small range. |
| **Seed value** | Reproduces initial noise when all other inputs are unchanged. |
| **Fixed** | Reuses the entered seed. Unavailable for multi-run batches. |
| **Random** | Chooses a new seed for each run. |
| **Increment** | Uses consecutive seeds and advances the displayed seed. |
| **Width / Height** | Output dimensions, constrained to multiples of 16. |

### Spatial guidance

| Setting | Practical effect |
| --- | --- |
| **Inside boost** | Strengthens a region's prompt/attention contribution inside its target. Raise gradually when a concept is ignored. |
| **Outside penalty** | Suppresses that regional contribution outside its target. Raise gradually to reduce leakage. Excessive values can isolate subjects unnaturally. |
| **Spatial falloff** | Width in pixels of the soft transition around a box. Larger values blend more naturally but allow more crossover; smaller values isolate more sharply. |
| **Late-step scale** | Fraction of regional guidance retained near the end. Lower values help final unification; higher values preserve placement more strictly. |
| **Separate overlapping subject targets** | Creates competition/ownership where subject boxes overlap. |
| **Make subjects fill their boxes** | Encourages subject occupancy across the target instead of collapsing into a small portion. |
| **Relax spatial guidance during late steps** | Reduces restrictions late so lighting, texture, edges, and interactions can unify. |
| **Adapt spatial guidance from regional LoRA delta** | Measures how strongly regional LoRAs change the model and adjusts containment. |
| **LoRA delta response** | Controls sensitivity to that measured delta. `0` ignores it; `1` applies the full response. |

Hypothetical LoRA-delta example: Character A's LoRA produces a small model delta while Character
B's produces a large delta at the same configured strength. With adaptation enabled, K2 applies
more containment pressure to B because it has greater leakage risk, while avoiding unnecessary
pressure on A.

Start with:

```text
Inside boost: 1
Outside penalty: 1
Spatial falloff: 128
Late-step scale: 0.35
Relax late guidance: enabled
```

Change one value at a time.

### Batch mode

**Run generation in batch mode** creates the selected number of independent runs. Runs are
submitted sequentially. Fixed seed is unavailable because identical settings would reuse the
same initial noise.

### Unified spatial prompting

**Use unified spatial prompting** compiles global and ordered regional clauses into one scene
prompt while spatial fields guide latent cells, attention, and LoRA scope.

**Preview unified prompt** displays the compiled text and resolved front-to-back order. Use it
when people are duplicated, actions are ignored, or a background prompt behaves as a subject.

### GPU memory

See [RAM and VRAM management](#ram-and-vram-management).

### Post-upscale

- **Post-upscale after releasing Krea VRAM** unloads the generation allocation first.
- **Output scale** selects 2× or 4×.
- **CPU Lanczos** is deterministic and requires no model.
- **Neural model (tiled GPU)** requires a compatible uploaded upscaler.

### Projector

The projector is an expert global model-vector adjustment.

- **Preset** chooses a known vector or Custom.
- The 12 numeric entries directly edit the custom vector.
- **Global multiplier** scales the vector.
- **Face identity protection** reduces projector influence on identity-conditioning tokens.

Leave it disabled unless you understand the target model and intended vector.

#### Projector preset defaults and source provenance

K2's default state is:

| Setting | Default |
| --- | --- |
| **Apply global projector vector** | Off |
| **Preset stored while off** | FilterBypass2 |
| **Global multiplier** | `1.0` |
| **Face identity protection** | `1.0` |

FilterBypass2 being the default preset selection does not mean a projector delta is applied.
The vector has no effect until the projector is enabled. The image-edit reference projector is
also disabled by default.

The preset table is:

| Preset | 12 projector-column deltas |
| --- | --- |
| **FilterBypass2** | `0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, 0, 0` |
| **FilterBypass3** | `0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, -0.6094, 0` |
| **skc3vo** | `-5.44, -16.11, -37.11, -50.39, -70.70, -39.45, -39.84, -143.7511, -51.17, -89.06, -60.94, -11.28` |
| **z0jglf** | `-13.60, -40.275, -92.775, -159.75, -176.75, -98.625, -99.60, -359.3778, -127.925, -222.65, -152.35, -28.20` |

These are community-derived reference weights, not official Krea defaults, guarantees, or
recommended settings. Their provenance is recorded so the values can be audited:

- **FilterBypass2 and FilterBypass3:** the
  [Sentinel7/krea2 model-card revision](https://huggingface.co/Sentinel7/krea2/commit/8a6c6313e1e34e1e7e26aac30ec2d35cee75b6ea)
  records the sparse values and links the
  [original FilterBypass2 Civitai model/version](https://civitai.com/models/2746817?modelVersionId=3089754).
  The FilterBypass3 file is preserved at a
  [pinned Sentinel7/krea2 revision](https://huggingface.co/Sentinel7/krea2/tree/41a18fe8d1826b09c6d53d3ca4204afd9e96dbef/2728234/3067151).
- **skc3vo:** the reference file is available in the
  [Comfy-Org/Krea-2 repository at a pinned revision](https://huggingface.co/Comfy-Org/Krea-2/blob/7b75ff3c61d88257ab29630be389af9adace3fd3/loras/skc3vo.safetensors).
- **z0jglf:** both reference files are preserved together in the
  [andrewwe/kr2 repository at a pinned revision](https://huggingface.co/andrewwe/kr2/tree/a096071125550d7d021adc19b5dd863a31d8aeaf).
  The z0jglf vector is the skc3vo vector scaled by `2.5`, subject to displayed precision.

The executable source of truth is
[`src/k2_region_lab/projector.py`](https://github.com/soomrenald/k2lab_runpod/blob/main/src/k2_region_lab/projector.py).
The feature entered the original desktop implementation in
[commit `bf82ac2`](https://github.com/soomrenald/k2lab/commit/bf82ac2d9466e1f679993a5b9a50c9389dd5ad9e).
When documentation and behavior disagree, use the checked-out source and tests to determine
what the running revision applies.

## Advanced image-edit settings

| Setting | Practical effect |
| --- | --- |
| **Sampler / Scheduler** | Sampling method and sigma schedule for the edit. |
| **Steps** | Number of edit denoising transitions. |
| **Seed · fixed** | Reproducible edit noise. |
| **Denoise** | Overall change strength. Low values preserve the source; high values redraw more. |
| **Reference retention** | Strength of source/reference conditioning. |
| **Latent feather** | Softens the edit mask in latent space. |
| **Composite feather** | Softens the final pixel-space blend into the source. |
| **Inside boost / Outside penalty / Spatial falloff / Late-step scale** | The same regional principles used by generation, applied to edit layers. |
| **Preserve reference identity** | Protects subject identity from unnecessary change. |
| **Edit entire image** | Applies edit conditioning globally instead of only inside targets. |

LoRA-delta adaptation, subject competition, fill, memory diagnostics, and the reference
projector are also available where applicable.

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

Start with low Denoise and moderate Blend. High Denoise can improve likeness but may also
change expression, age, lighting, or head geometry.

## Assets, uploads, and downloads

### Assets

Asset tabs separate inputs, projects, outputs, diffusion models, text encoders, VAEs, LoRAs,
upscalers, and face-detection files.

Local uploads:

- are SHA-256 hashed in the browser;
- are queued sequentially;
- can be paused, resumed, retried, or cancelled;
- resume from accepted chunks after interruption;
- detect duplicates by digest.

Every file has a checkbox and Delete action. **Select all** plus selective unchecking supports
batch cleanup. Deletion is permanent for those workspace files.

Outputs default to newest first and can be sorted by newest, oldest, name, or size. Outputs
have thumbnail and enlarged previews. **Download image** downloads the canvas image without
opening Assets.

### Provider transfers

Transfers support Civitai and Hugging Face:

1. Add an optional download-only/read-only token.
2. Paste a canonical model, repository, or file URL.
3. Select the destination category.
4. Choose **Inspect before download**.
5. Review the resolved filename and size.
6. Start the provider download.

Provider transfers are queued and executed sequentially. Large files move directly from the
provider to the Pod. K2 validates redirects, provider hosts, workspace capacity, and supported
file formats. Do not place credentials inside a URL.

## Jobs, batches, events, and request sequencing

The workspace serializes control-plane requests and GPU work to avoid overlapping operations.
Uploads, provider downloads, generation batches, and other agent requests wait their turn.

The Events dock retains up to 1,000 entries and shows submission, queue, worker progress,
completion, cancellation, and redacted errors. Drag its top handle to resize it.

If the control plane reports that the workspace agent did not finish before its timeout:

1. Do not repeatedly press the action button.
2. Check Events to see whether a remote job was accepted or is still progressing.
3. Check the Pod and agent logs in RunPod.
4. Wait for the current sequential operation to finish.
5. Refresh the workspace state before retrying.
6. Use **Cancel remote job** when K2 shows a recognized active job.

A network timeout does not prove the remote operation failed. Repeated submissions can add more
work behind the request that is already running.

## RAM and VRAM management

### VRAM execution modes

| Mode | Behavior |
| --- | --- |
| **Auto** | Resolves to High VRAM on devices with at least 40 GiB; otherwise Dynamic. |
| **High VRAM** | Prioritizes performance and keeps more model state on the GPU. |
| **Dynamic VRAM** | Balances GPU residency and offloading. |
| **Low VRAM** | Maximizes offloading for constrained GPUs; slowest option. |

**VRAM reserve** is the amount K2 must leave free. It uses the actual Pod GPU capacity, not a
hard-coded GPU size.

**Keep baseline model loaded between runs** works only when execution resolves to High VRAM and
the reserve remains available. It retains the baseline transformer for faster subsequent
generations. LoRAs are not retained. Model changes, memory pressure, OOM recovery, other
execution modes, or **Release worker memory** discard the resident worker.

### System RAM safeguard

The safeguard uses non-reclaimable usage rather than mistaking clean Linux file cache for real
memory pressure.

- **Allocatable now** — practical memory available after real use and reclaimable cache.
- **Actual non-cache use** — process/anonymous and other non-reclaimable allocations.
- **Clean reclaimable cache** — filesystem cache Linux can evict.
- **Raw cgroup charge** — total cgroup accounting value, including cache.
- **Anonymous/process RAM** — process-backed memory.
- **Shared + dirty files** — shared memory and non-clean file pages.
- **Worker** — Released, Active, or Baseline resident.

**Enable system RAM safeguard** can be turned off for diagnosis. Linux cgroup limits and the
kernel OOM killer still apply, so disabling it can cause the worker or Pod to be terminated.

Use **Refresh RAM** for current telemetry. **Release worker memory** cancels active worker work
and unloads resident model allocations.

## Pod lifecycle and storage safety

The workspace popover shows compute price, storage price, lease time, readiness, and the RunPod
Pod ID.

- **Extend session** extends a limited lease.
- **Start GPU** starts stopped compute.
- **Start GPU without time limit** continues billing until manually stopped.
- **Stop GPU now** stops or terminates compute according to workspace type.
- **Release worker memory** unloads the model without stopping the Pod.
- **Migrate to portable storage** performs a verified SHA-256 copy.
- **Connect migrated Pod** verifies a manually replaced persistent Pod before changing the
  saved provider ID.
- **Delete cloud workspace** has the storage consequences shown in its confirmation.

Migration can temporarily bill both source and target resources. The original Pod remains
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
git switch main
git pull --ff-only
./scripts/k2lab-runpod --image 'ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:f1dbf619c615bb55be781eada7b11d28a70818d1dba4139814e1ea41645f8717' --image-version '0.1.17'
```

If another K2 instance still listens on port 8000, the launcher follows its log and does not
replace its in-memory configuration. Stop that original process first.

### Update a workspace Pod

Changing local configuration affects newly created Pods; it does not mutate a running Pod.

Before replacement:

1. Save and download important project files.
2. Confirm models and outputs are on the persistent workspace volume.
3. Record workspace ID, Pod ID, GPU type, volume size, image digest, and version.
4. Stop generation and transfers.
5. Use the supported migration/reconnection flow or replace the Pod while retaining the same
   persistent volume and K2 workspace environment.
6. Set both the exact immutable image digest and matching `K2LAB_IMAGE_VERSION`.
7. Use **Connect migrated Pod** when a new persistent Pod ID must replace the saved ID.

K2 verifies workspace identity, agent credential, immutable image, GPU type, mount path, and
volume size. Do not weaken those checks to force an incompatible Pod to connect.

For a Portable workspace, the network volume remains when its temporary Pod stops. A subsequent
start creates a Pod using the configured immutable image.

Never delete a persistent volume merely to update software.

## Troubleshooting

### `address already in use` or port 8000 is busy

The launcher detects an already-running K2 control plane and follows `control-plane.log`. If
another application owns port 8000, stop it or run:

```bash
./scripts/k2lab-runpod --port 8001
```

Check listeners with:

```bash
ss -ltnp | grep ':8000'
```

### The Pod agent image version does not match the workspace

The agent's `K2LAB_IMAGE_VERSION` differs from the version expected by the local controller.
Use the exact matching digest/version pair. Restarting only the browser changes neither.

### The migrated Pod does not use this workspace's immutable image

The replacement Pod does not resolve to the digest recorded for the workspace. Replace it using
the exact immutable digest while retaining the persistent volume and workspace environment,
then verify again.

### Workspace agent not responding

1. Open the workspace popover and check readiness.
2. Inspect the Pod startup and agent logs in RunPod.
3. Confirm the Pod is running and agent health checks succeed.
4. Check whether a long sequential transfer or job is still active.
5. Avoid submitting duplicates while the operation outcome is uncertain.
6. Stop/start the workspace only after persistent file writes have finished.

### A generation remained queued, then started later

Another serialized request, transfer, or job was ahead of it. Check Events and Transfers.
Downloads and generation runs intentionally do not overlap.

### Krea 2 model loading failed

- Verify diffusion model, text encoder, and VAE selections in Setup.
- Confirm downloads completed and pass safetensors validation.
- Confirm the files are compatible with Krea 2.
- Check actual VRAM and system RAM diagnostics.
- Use Dynamic/Low VRAM or a larger GPU.
- Release worker memory before changing the baseline model.

### Generation failed while applying LoRA or sampling settings

- Disable all optional LoRAs and run the baseline.
- Add LoRAs one at a time.
- Verify each LoRA targets Krea 2 transformer modules.
- Reduce LoRA strength.
- Return sampler and scheduler to known-good baseline values.
- Check Events for the worker error and measured LoRA delta.
- Release worker memory after an OOM or incompatible-model failure.

### RAM appears stuck at 85–95%

Read **Actual non-cache use** and **Clean reclaimable cache**, not only Raw cgroup charge.
Downloaded and model files often remain as clean filesystem cache and are evicted automatically.
Use **Release worker memory** if the Worker says Active or Baseline resident and you want to
discard real resident allocations.

### VRAM remains high while idle

If **Keep baseline model loaded between runs** is enabled in High VRAM mode, baseline weights
remain resident intentionally. LoRAs should not remain loaded. Disable retention or use
**Release worker memory**.

### Output does not appear on the canvas

- Wait for the job to reach Completed in Events.
- Open **Assets → Outputs** and preview the newest file.
- Reload the studio and reopen the newest verified output.
- Download the output to confirm the file is valid.

### Additional people appear

- State the exact number of people globally.
- Give each person one sufficiently large subject-role box.
- Avoid multiple full-scene descriptions in regional prompts.
- Check front-to-back order and **Preview unified prompt**.
- Enable subject competition for overlaps.
- Use character-identity routing only on the intended subject.
- Make interaction boxes large enough to include the complete action.

### Regions look disconnected or have boundary artifacts

- Return Inside boost and Outside penalty toward `1`.
- Increase Spatial falloff.
- Keep late spatial relaxation enabled.
- Reduce Late-step scale.
- Use one coherent global scene prompt.
- Avoid treating every box as an independent complete image.

### Prompt or LoRA leakage remains high

- Confirm routing on the active layer.
- Use Character identity routing for identity LoRAs.
- Increase Outside penalty gradually.
- Reduce Spatial falloff cautiously.
- Enable subject competition.
- Make boxes cover the complete target area.
- Test whether the LoRA itself is globally entangled.

### Face detector is missing

Use **Faces → Detect faces** and accept installation of the pinned detector, or upload/select a
detector in Setup. Normal Generate and Edit remain available without it.

### Provider download is stuck

- Downloads are intentionally sequential; inspect the head of the queue.
- Check transfer state, bytes, and throughput.
- Cancel while retaining resumable data, then retry/resume.
- Verify the provider token and canonical URL.
- Confirm Pod networking and workspace capacity.

### `unsafe URL` during a Civitai download

Use a canonical Civitai model/download URL and a saved download-only token. K2 validates every
redirect and refuses credentials embedded in URLs. If the provider returns an unapproved host,
record the resolved file and event/error code rather than disabling URL checks globally.

### Logs and useful report details

Local log:

```text
~/.local/state/k2-region-lab/control-plane.log
```

RunPod container/agent logs are available on the Pod page. The studio **Events** dock provides
the generation-specific redacted event stream.

When reporting a problem, include:

- local commit (`git rev-parse HEAD`);
- image digest and image version;
- Pod GPU and VRAM;
- selected model filenames;
- job ID if shown;
- relevant Events lines;
- local control-plane error text;
- whether the issue reproduces with optional LoRAs disabled.
