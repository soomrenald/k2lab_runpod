# K2 Region Lab for RunPod

K2 Region Lab is a browser-based image generation workspace for people who want precise
control over where subjects, objects, prompts, and LoRAs affect an image. The interface runs
locally on your computer while generation runs on a GPU Pod in your RunPod account.

K2 supports ordinary text-to-image generation, unified regional prompting, regional and
character-identity LoRAs, image editing, face refinement, and post-upscaling. Models, projects,
inputs, and outputs live in persistent RunPod storage instead of disappearing when the browser
closes.

> **Preview software:** K2 can create billable RunPod resources and is under active
> development. Read the storage and deletion warnings before creating a workspace.

For a complete explanation of the interface and settings, see the
**[K2 Region Lab User Guide](docs/user_guide.md)**. It includes workflows, memory diagnostics,
Pod updates, and common troubleshooting.

## How K2 works

K2 has three parts:

1. **The local control plane** runs on your computer at `http://127.0.0.1:8000`. It stores
   your encrypted RunPod credential, workspace records, and the browser interface.
2. **The RunPod workspace agent** runs inside a signed, versioned GPU image. It manages
   persistent files, model downloads, generation jobs, and worker memory.
3. **The persistent workspace** stores models, LoRAs, projects, source images, and outputs.
   A Persistent Pod uses a volume attached to that Pod; a Portable workspace uses an
   independent RunPod network volume.

Your prompts and model files do not need to pass through a third-party K2 service. The local
application talks directly to RunPod and to the authenticated agent on your Pod.

## Feature overview

- Krea 2 text-to-image generation with selectable sampler, scheduler, seed, size, and steps.
- Unified global and regional prompting with soft latent-space and attention guidance.
- Subject and background region roles with explicit front-to-back ordering.
- Regional LoRA routing and dedicated character-identity LoRA routing.
- Optional volumetric mannequin pose conditioning and Krea-native pose Control LoRA routing.
- Feature-gated global and regional Krea depth conditioning with Blender authoring tools.
- Image editing with separate reference-layout and edit-target layers.
- Optional face detection and selected-face refinement. Face detection is not required for
  normal generation or image editing.
- Local uploads and sequential Civitai/Hugging Face downloads performed on the Pod.
- Persistent asset browsing, output previews, sorting, downloads, and batch deletion.
- JSON project files and project metadata import from generated PNG files.
- Fixed, random, and incrementing seeds plus sequential multi-run batches.
- High-, Dynamic-, and Low-VRAM modes with optional baseline-model retention.
- Pod RAM diagnostics that distinguish real memory use from reclaimable Linux file cache.
- Optional post-upscaling with CPU Lanczos or a tiled GPU upscaler.
- Persistent Pod and portable network-volume workspace lifecycles.

The regional implementation follows the unified latent-space prompting and LoRA-isolation
method inherited from the original desktop K2 Region Lab. See
[regional isolation parity](docs/regional_isolation_parity.md) for the technical comparison.

The optional projector is disabled by default. If enabled, the interface initially selects
**FilterBypass2**. The bundled projector vectors were transcribed from community reference
weights; they are not official Krea defaults or recommendations. See
[projector defaults and source provenance](docs/user_guide.md#projector-preset-defaults-and-source-provenance)
for exact values and pinned source links.

## Requirements

You need:

- a Linux computer with a current web browser;
- `git`;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/);
- a RunPod account with billing configured;
- a restricted RunPod API key that can manage Pods and read inventory/pricing;
- enough RunPod storage for the baseline model, text encoder, VAE, LoRAs, and outputs.

Node.js is needed only for frontend development. The normal launcher serves the bundled
browser interface.

## Install

Clone the repository:

```bash
git clone https://github.com/soomrenald/k2lab_runpod.git
cd k2lab_runpod
```

On the first launch, provide the signed `0.1.17` workspace image and matching version:

```bash
./scripts/k2lab-runpod --image 'ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:f1dbf619c615bb55be781eada7b11d28a70818d1dba4139814e1ea41645f8717' --image-version '0.1.17'
```

Keep that command on one line. The launcher:

- creates a private state directory;
- generates a local encryption key;
- creates the local database;
- saves the immutable image selection for later launches;
- opens `http://127.0.0.1:8000`;
- writes a persistent control-plane log.

Future launches need only:

```bash
./scripts/k2lab-runpod
```

If K2 is already running, launching it again follows the existing log instead of trying to
start a second server. Pressing `Ctrl+C` in that follower stops only the log view.

## First-time setup

1. Create a restricted user-owned API key in RunPod.
2. Paste it into **RunPod account** and choose **Validate and continue**.
3. Choose a **Persistent Pod** or **Portable workspace**.
4. Select one or more GPUs in preference order.
5. Set storage, idle-stop, and session-limit values.
6. Review the live cost estimate and create the workspace.
7. Wait until the workspace, agent, storage, and GPU indicators are ready.
8. Open **Transfers** or **Assets** to install the required model files.
9. Open **Setup** and select the diffusion model, text encoder, and VAE.
10. Enter a prompt and choose **Generate image**.

The face detector shown in Setup is optional and is used only by **Faces** tools.

## Required model files

Normal Krea 2 generation requires compatible files for:

- the Krea 2 diffusion/transformer model;
- the text encoder;
- the VAE.

LoRAs and upscalers are optional. Use **Transfers** to download files from Civitai or Hugging
Face directly on the Pod, or use **Assets** to upload local files. Select the exact baseline
files in **Setup** before generating.

K2 does not bundle copyrighted model weights and does not bypass provider access controls.
Only download files you are authorized to use.

## Basic generation workflow

1. Describe the complete image in **Prompt → Global prompt**.
2. Choose **Draw region**, then drag a box over a subject, object, or background area.
3. Select the box and enter its regional prompt.
4. Set its spatial role to **Subject target**, **Background band**, or **Auto**.
5. Add LoRAs in **LoRAs**, then choose global, standard regional, or character-identity
   routing.
6. Start with the default spatial-guidance values.
7. Open **Preview unified prompt** to verify subject count and front-to-back order.
8. Choose **Generate image**.

The newest completed output appears on the canvas and first in **Assets → Outputs**.

## Data, billing, and deletion

- **Stop GPU now** stops compute billing but does not stop persistent-storage billing.
- Stopping a Persistent Pod retains its attached volume.
- Stopping a Portable workspace terminates its temporary Pod and retains its network volume.
- Deleting a Persistent Pod workspace permanently deletes its regular volume.
- K2 deliberately retains portable network volumes when deleting an application workspace.
  Delete them separately in RunPod only after verifying a backup.
- **No time limit** keeps a Pod running and billing until you stop it manually.
- Interruptible compute may stop without notice.

Read [RunPod workspace operations](docs/runpod_workspace_operations.md) before migrating or
deleting storage.

## Local files and logs

The default local state directory is:

```text
~/.local/state/k2-region-lab
```

Important files include:

- `config.json` — saved immutable image digest and version;
- `credential.key` — encryption key for stored credentials;
- `state.sqlite3` — workspace and operation records;
- `control-plane.log` — local runtime and HTTP log.

Back up `credential.key` together with `state.sqlite3`. Losing the key makes encrypted
credentials in that database unreadable.

The launcher listens only on `127.0.0.1`. Do not port-forward or publicly proxy the local
single-user control plane.

## Updating

Updating has two separate parts:

1. Update the local UI/control plane with `git switch main` and `git pull --ff-only`.
2. Configure the immutable workspace image and matching `--image-version` for that release
   before creating or replacing a Pod.

Stop the original local control-plane process before launching with new image arguments.
Changing the saved image does not replace an already running Pod. Follow the
[safe update procedure](docs/user_guide.md#updating-k2-and-a-pod-without-losing-data) for an
existing workspace.

Never pair an image digest with a different image-version string. K2 intentionally rejects a
Pod whose agent version or immutable image does not match the workspace record.

## Documentation

- [Complete user guide](docs/user_guide.md)
- [Projector defaults and source provenance](docs/user_guide.md#projector-preset-defaults-and-source-provenance)
- [RunPod lifecycle and migration runbook](docs/runpod_workspace_operations.md)
- [Regional isolation parity](docs/regional_isolation_parity.md)
- [Volumetric pose control](docs/krea_volumetric_pose_control.md)
- [Subject pose controls](docs/subject_pose_control.md)
- [Depth control](docs/DEPTH_CONTROL.md)
- [Blender depth workflow](docs/BLENDER_DEPTH_WORKFLOW.md)
- [Web/desktop parity notes](docs/web_desktop_parity.md)
- [RunPod web workspace specification](docs/runpod_web_workspace_spec.md)

## Development

Install development dependencies:

```bash
uv sync --extra dev --extra web
cd web/client
npm install
```

Run the non-billing development backend:

```bash
uv run k2lab-web --reload
```

In another terminal:

```bash
cd web/client
npm run dev
```

Open `http://127.0.0.1:5173`. The development backend is labelled in the interface and does
not provision RunPod resources.

Run validation:

```bash
uv run pytest
cd web/client
npm run test:project
npm run test:ui
npm run typecheck
npm run build
```

The ordinary test suite does not create a Pod. The destructive live acceptance suite requires
an explicit billing/deletion sentinel and a dedicated disposable RunPod account.
