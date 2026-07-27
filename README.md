# K2 Region Lab for RunPod

K2 Region Lab is a browser-based image generation workspace for people who want precise
control over where subjects, objects, prompts, and LoRAs affect an image. The interface runs
locally on your computer while generation runs on a GPU Pod in your RunPod account.

K2 supports ordinary text-to-image generation, regional prompting, regional and
character-identity LoRAs, image editing, face refinement, post-upscaling, and pose-controlled
subject boxes. Models, projects, inputs, and outputs live in persistent RunPod storage rather
than disappearing when the browser closes.

> **Preview software:** K2 can create billable RunPod resources and is under active
> development. Read the storage and deletion warnings before creating a workspace.

For a complete explanation of every screen and setting, see the
**[K2 Region Lab User Guide](docs/user_guide.md)**. It also contains common workflows,
troubleshooting steps, memory diagnostics, and safe Pod-update instructions.

## How K2 works

K2 has three parts:

1. **The local control plane** runs on your computer at `http://127.0.0.1:8000`. It stores
   your encrypted RunPod credential, workspace records, and the browser interface.
2. **The RunPod workspace agent** runs inside a versioned GPU image. It manages persistent
   files, model downloads, generation jobs, and worker memory.
3. **The persistent workspace** stores models, LoRAs, projects, source images, and outputs.
   A Persistent Pod uses a volume attached to that Pod; a Portable workspace uses an
   independent RunPod network volume.

Your prompts and model files do not need to pass through a third-party K2 service. The local
application talks directly to RunPod and to the authenticated agent on your Pod.

## Feature overview

- Krea 2 text-to-image generation with selectable sampler, scheduler, seed, size, and steps.
- Unified global and regional prompting without hard rectangular image seams.
- Ordinary region boxes for objects, scenery, architecture, and background areas.
- Subject boxes with editable volumetric mannequins and early-denoising pose gating.
- Regional LoRA routing and dedicated character-identity LoRA routing.
- Image editing with separate reference-layout and edit-target layers.
- Optional face detection and selected-face refinement. Face detection is not required for
  normal generation or image editing.
- Local uploads plus sequential Civitai and Hugging Face downloads performed on the Pod.
- Persistent asset browsing, previews, sorting, downloads, and batch deletion.
- JSON project files and project metadata import from generated PNG files.
- Fixed, random, and incrementing seeds plus sequential multi-run batches.
- High-, dynamic-, and low-VRAM modes; optional baseline-model retention on large GPUs.
- Pod RAM diagnostics that distinguish real process memory from reclaimable Linux file cache.
- Recoverable, idempotent job submission when a network POST times out.
- Persistent Pod and portable network-volume workspace lifecycles.

The regional implementation follows the unified latent-space prompting and LoRA-isolation
method inherited from the original desktop K2 Region Lab. See
[regional isolation parity](docs/regional_isolation_parity.md) for the technical comparison.

The optional projector is disabled by default. If it is enabled, the interface initially
selects **FilterBypass2**. That preset and the other bundled projector vectors were transcribed
from community reference LoRA weights; they are not official Krea defaults or recommendations.
See [projector defaults and source provenance](docs/user_guide.md#projector-preset-defaults-and-source-provenance)
for the exact values and pinned source links.

## Requirements

You need:

- a Linux computer with a current web browser;
- `git`;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for the local Python
  environment;
- a RunPod account with billing configured;
- a restricted RunPod API key that can manage Pods and read inventory/pricing;
- enough RunPod storage for your baseline model, text encoder, VAE, LoRAs, and outputs.

Node.js is only required for frontend development. The normal launcher serves the already
built browser interface.

## Install

Clone the current pose-preview branch and enter the repository:

```bash
git clone --branch k2lab_pose https://github.com/soomrenald/k2lab_runpod.git
cd k2lab_runpod
```

On the first launch, provide both the immutable workspace image and its matching version:

```bash
./scripts/k2lab-runpod --image 'ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:71d034c346a5a2c1bb21a90df507d9a0b2dfb3f3e718e5380a9526e29b65b2c5' --image-version '0.2.0-pose.2'
```

Keep that command on one line. The launcher:

- creates a private state directory;
- generates a local encryption key;
- creates the local database;
- saves the immutable image selection for later launches;
- opens `http://127.0.0.1:8000`;
- writes a persistent control-plane log.

Future launches only need:

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
7. Wait until the workspace, agent, storage, and GPU readiness indicators are green.
8. Open **Transfers** or **Assets** to install the required model files.
9. Open **Setup** and select the diffusion model, text encoder, and VAE.
10. Enter a prompt and choose **Generate image**.

The face detector shown in Setup is optional and is used only by **Faces** tools.

## Required model files

Normal Krea 2 generation requires compatible files for:

- the Krea 2 diffusion/transformer model;
- the text encoder;
- the VAE.

LoRAs and upscalers are optional. Use **Transfers** to inspect and download files from Civitai
or Hugging Face directly on the Pod, or use **Assets** to upload local files. Select the exact
baseline files in **Setup** before generating.

K2 does not bundle copyrighted model weights and does not bypass provider access controls.
Only download files you are authorized to use.

## Basic generation workflow

1. Describe the whole image in **Prompt → Global prompt**.
2. Use **Draw region** for objects/backgrounds or **Draw subject** for a person with a
   mannequin.
3. Select each box and enter its regional prompt.
4. Add LoRAs in **LoRAs**, then choose global, regional, or character-identity routing.
5. Start with the default regional guidance values.
6. Open **Preview unified prompt** to verify subject count and front-to-back order.
7. Choose **Generate image**.
8. The newest completed output appears on the canvas and at the top of
   **Assets → Outputs**.

Pose gating is optional. A subject box may be used for character prompting without enabling
the global mannequin gate.

## Data, billing, and deletion

- **Stop GPU now** stops compute billing but keeps persistent storage billing.
- Stopping a Persistent Pod retains its attached volume.
- Stopping a Portable workspace terminates its temporary Pod and retains its network volume.
- Deleting a Persistent Pod workspace permanently deletes its regular volume.
- K2 deliberately retains portable network volumes when deleting an application workspace.
  Delete those separately in RunPod only after verifying a backup.
- **No time limit** keeps a Pod running and billing until you stop it manually.
- Interruptible compute may stop without notice.

See [RunPod workspace operations](docs/runpod_workspace_operations.md) before migrating or
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

Back up `credential.key` together with `state.sqlite3`. Losing the key makes the encrypted
credentials in that database unreadable.

The launcher listens only on `127.0.0.1`. Do not port-forward or publicly proxy the local
single-user control plane.

## Updating

Updating has two separate parts:

1. Update the local UI/control plane with `git pull --ff-only` while on `k2lab_pose`.
2. Configure the matching immutable workspace image and `--image-version` before creating or
   migrating a Pod.

Stop the local control-plane process before launching it with new image arguments. Changing
the saved image does not replace an already running Pod. Follow the
[safe update procedure](docs/user_guide.md#updating-k2-and-a-pod-without-losing-data) for an
existing workspace.

Never pair an image digest with a different image-version string. K2 intentionally rejects a
Pod whose agent version or immutable image does not match the workspace record.

## Documentation

- [Complete user guide](docs/user_guide.md)
- [Projector defaults and source provenance](docs/user_guide.md#projector-preset-defaults-and-source-provenance)
- [Volumetric subject pose control](docs/subject_pose_control.md)
- [RunPod lifecycle and migration runbook](docs/runpod_workspace_operations.md)
- [Regional isolation parity](docs/regional_isolation_parity.md)
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
