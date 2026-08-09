# K2Lab Krea-Native Volumetric Pose Control-LoRA

## Additive implementation specification for Codex 5.6 Sol High

**Repository:** `https://github.com/soomrenald/k2lab_runpod`  
**Starting branch:** the current working branch containing the completed subject-semantic pose-conditioning implementation  
**Expected branch name:** `k2lab_pose`  
**Expected current project schema:** 22  
**Expected current worker protocol:** 3  
**Existing mannequin format:** `k2-volumetric-pose-v1`  
**New control-image format:** `k2-volumetric-pose-control-v1`  
**Primary milestone:** add a Krea-native volumetric pose Control-LoRA training and inference path without breaking any currently functioning K2Lab behavior, and correct misleading passive RunPod timeout errors such as the one shown after a successfully completed job.

---

# 0. Non-negotiable directive to Codex

Implement this specification end to end on top of the current implementation.

Do not stop after producing a plan, isolated renderer, standalone trainer, UI mockup, or partial inference experiment. Continue through the first point at which:

1. the current application builds and passes all existing tests;
2. a compatible Krea volumetric pose Control-LoRA checkpoint can be selected in the RunPod web GUI;
3. the exact current mannequin is rendered into the hidden control format;
4. the control image is VAE-encoded and injected into native Krea 2 inference;
5. Prediction composite uses the correct full-scene or per-subject control latent for each conditioning scope;
6. a real RunPod generation can be launched with the adapter enabled;
7. the existing semantic-routing, regional prompt, regional LoRA, hard/soft gate, sigma, edit, face, asset, project, and workspace functionality still works;
8. a passive RunPod provider timeout no longer appears as a red generation error after a completed job.

Before modifying code:

```bash
git status --short
git rev-parse HEAD
git log -1 --oneline
```

Record:

- current branch;
- current commit;
- every uncommitted file;
- current `PROJECT_VERSION`;
- current worker protocol version;
- current agent/API/image versions;
- pinned ComfyUI commit in `Dockerfile.workspace`;
- current test results.

If the working tree contains uncommitted functionality, preserve it. Do not reset, clean, switch branches, restore files, or overwrite local work.

## 0.1 Explicit non-regression requirement

The implementation is additive. It must not break, remove, weaken, silently alter, or regress any current functionality, including:

- the volumetric mannequin editor;
- all 13 articulation handles and the head ellipse;
- standing, squatting, mirror, move, and resize behavior;
- hard, soft, and normal denoising phases;
- cosine, linear, exponential, and stepped release schedules;
- automatic, phase-weighted, and advanced sigma scheduling;
- Spatial only semantic routing;
- Attention isolation semantic routing;
- Prediction composite semantic routing;
- separate scene, shared-visual, and subject prompts;
- full-scene and subject-only prompt previews;
- scope-aware regional and identity LoRA routing;
- current regional attention and LoRA masking;
- current full/subject prediction compositing;
- current job progress, cancellation, output saving, PNG metadata, and retained-model behavior;
- normal generation with pose features disabled;
- image editing;
- face detection/refinement;
- post-upscaling;
- project open/save/import/cloud persistence;
- asset upload, resumable transfers, Civitai, and Hugging Face downloads;
- persistent-Pod and portable-workspace lifecycle behavior;
- worker-memory release;
- current security and redaction behavior.

The new adapter must be disabled by default for migrated projects. With the adapter disabled, the generation path must remain behaviorally equivalent to the current implementation.

Do not “simplify” the current semantic system to make the adapter easier to add.

## 0.2 No silent fallback

If the user enables the adapter and any required checkpoint, control format, VAE, projection, block weight, control latent, conditioning scope, or runtime hook is incompatible:

- fail before sampling when possible;
- return a typed, specific error;
- preserve the real private traceback;
- do not silently run without the adapter;
- do not silently switch semantic modes;
- do not silently substitute the wide denoising support mask for the semantic control image;
- do not classify the failure as a generic regional LoRA problem.

## 0.3 No generic ControlNet regression

Do not reintroduce the removed Qwen/InstantX generic ControlNet path.

This feature is a Krea-native Control-LoRA using:

- the Qwen-Image VAE;
- a control latent channel-concatenated with the noisy Krea latent;
- the expanded Krea input projection stored in the checkpoint;
- Control-LoRA weights over the Krea transformer;
- native ComfyUI Krea model loading and sampling.

Do not require a user-installed ComfyUI custom node at runtime. Community code may be studied and adapted, subject to license review and attribution, but K2Lab must own a tested native integration.

---

# 1. Verified current behavior and remaining limitation

A live completed job confirmed that the current Prediction composite implementation is functioning semantically:

```text
Job: af92c42fb0034419bc6bac046e2fd5a3

Full predictions:                18
Subject-only predictions:       10
Total conditional evaluations:  28
Subject delta RMS inside ownership:
    0.9286823868751526

Subject-only prompt contains scene terms:
    "room":          false
    "wooden floor":  false
    "wooden floors": false
```

For one subject using:

```text
Hard:   8
Soft:   2
Normal: 8
```

the expected count is:

```text
18 full-scene evaluations
+ 10 subject-only evaluations
= 28 semantic-scope evaluations
```

The current implementation has therefore solved:

> Generate the correct subject semantics in the subject-owned area rather than arbitrary scene content.

It has not solved:

> Interpret the head ellipse, torso polygon, and individual limb capsules as a specific human pose.

The current spatial system can decide:

- where denoising is accepted;
- which subject prompt and LoRAs produce the accepted prediction.

It does not provide Krea with a learned mapping from the hidden mannequin geometry to anatomy.

This implementation adds that missing learned mapping while retaining the current spatial and semantic systems.

---

# 2. Intended final architecture

```text
User edits k2-volumetric-pose-v1 mannequins
        ↓
K2Lab renders k2-volumetric-pose-control-v1 RGB control images
        ↓
Qwen-Image VAE encodes each control image once per job
        ↓
Krea-native expanded input projection receives:
    noisy image tokens + control tokens
        ↓
Krea volumetric pose Control-LoRA interprets:
    head / torso / left-right limb geometry
        ↓
Existing Prediction composite supplies:
    full scene or exact subject-only semantics
        ↓
Existing regional LoRA system supplies:
    correct subject identity/style in the correct scope
        ↓
Existing hard/soft gate reinforces:
    early location, containment, and ownership
        ↓
Existing normal phase integrates:
    anatomy, clothing, scene, lighting, and detail
```

Responsibilities:

| Component | Responsibility |
|---|---|
| Volumetric Control-LoRA | Anatomical interpretation of head, torso, and major limbs |
| Prediction composite | Correct subject semantics in each mannequin-owned area |
| Scope-aware regional LoRA | Correct identity/style assignment |
| Hard/soft spatial gate | Early subject location and containment |
| Sigma schedule | How much diffusion trajectory occurs in each gate phase |
| Prompt | Clothing, action semantics, facing, gaze, held items, overlap intent, fine details |
| Normal phase | Scene integration and final refinement |

The Control-LoRA must complement the current system, not replace it.

---

# 3. Implementation scope

Implement four connected deliverables:

1. **Canonical control-image format**
   - deterministic hidden RGB rendering from the current mannequin;
   - full-scene and per-subject variants;
   - browser/backend parity;
   - preview and metadata.

2. **Reproducible training toolchain**
   - COCO pair builder;
   - prebucketed VAE shard builder;
   - pinned Control-LoRA trainer integration;
   - checkpoint metadata;
   - evaluation tooling;
   - separate training environment.

3. **Native K2Lab inference integration**
   - checkpoint asset kind and validator;
   - expanded input projection;
   - native Krea block patches;
   - control-latent VAE encoding;
   - conditioning-scope-aware latent selection;
   - GUI controls;
   - output diagnostics.

4. **RunPod timeout correctness**
   - investigate the exact source of passive provider timeout messages;
   - prevent passive status-refresh failures from being displayed as generation errors;
   - retain last known workspace/job/output state;
   - distinguish warning, stale status, and actual failed user operations;
   - add retries, diagnostics, recovery, and tests.

---

# 4. Canonical hidden control format

## 4.1 Format identifier

Define:

```python
K2_VOLUMETRIC_CONTROL_FORMAT = "k2-volumetric-pose-control-v1"
```

The format identifier covers:

- geometry;
- colors;
- draw order;
- antialiasing;
- coordinate transform;
- crop/resize convention;
- background convention;
- left/right semantics.

Changing any of those requires a new format version.

## 4.2 Source state

The control renderer consumes the existing canonical:

```text
k2-volumetric-pose-v1
```

Do not create a second editable pose state.

The current editor remains the only user-authored pose representation.

## 4.3 Hidden control image

The hidden control image is not the visible editor overlay.

It contains:

- black background;
- fully opaque semantic body-part fills;
- antialiased edges;
- no translucency;
- no region box;
- no selection outline;
- no centerline;
- no articulation handle;
- no head resize handle;
- no labels;
- no subject name;
- no background image;
- no support-mask dilation;
- no feathered denoising halo.

Use exact core mannequin geometry, not the wide support mask.

## 4.4 Semantic palette

Use anatomical left/right, not image-left/image-right.

Centralize this exact initial palette:

| Part | RGB |
|---|---|
| Background | `(0, 0, 0)` |
| Head | `(255, 255, 255)` |
| Neck | `(255, 215, 0)` |
| Torso | `(255, 128, 0)` |
| Left upper arm | `(255, 32, 64)` |
| Left forearm | `(255, 32, 192)` |
| Right upper arm | `(32, 128, 255)` |
| Right forearm | `(32, 255, 255)` |
| Left thigh | `(128, 32, 255)` |
| Left calf | `(224, 32, 255)` |
| Right thigh | `(32, 255, 64)` |
| Right calf | `(160, 255, 32)` |

Expose the palette through one immutable backend object and one generated frontend artifact. Do not manually maintain two unrelated copies.

Calculate and publish a stable palette/format SHA-256 from canonical JSON.

## 4.5 Geometry

Use the current mannequin geometry exactly:

- head ellipse;
- neck capsule;
- shoulder/hip torso volume;
- tapered upper/lower arm capsules;
- tapered thigh/calf capsules;
- current scale-derived widths;
- current out-of-box behavior;
- current canvas clipping.

Do not use the visible editor's stylistic stroke widths if those differ from backend mask geometry.

## 4.6 Draw order

Use a fixed documented order:

1. torso;
2. left thigh;
3. left calf;
4. right thigh;
5. right calf;
6. left upper arm;
7. left forearm;
8. right upper arm;
9. right forearm;
10. neck;
11. head.

Draw each tapered capsule with its own endpoint disks before later parts overwrite it.

The current 2D mannequin has no limb-depth ordering. A deterministic draw order is acceptable for v1. Do not invent hidden 3D depth.

## 4.7 Raster quality

Render at 4× output resolution and downsample with Lanczos to the exact generation canvas.

Output:

```text
RGB uint8 image
exact generation width and height
black background
```

The VAE encoder receives normalized `[0,1]` RGB with:

```text
channel_mode = rgb
normalize = none
invert = false
```

## 4.8 Full and subject images

Generate:

```text
full_control
    all enabled posed subjects on one canvas

subject_control[region_id]
    only that subject on the same full-size canvas
```

All maps use the same canvas, coordinates, colors, and format.

Do not crop subject-only maps to the subject box.

## 4.9 Browser/backend parity

The backend renderer is authoritative for training and inference.

The browser may render a preview, but it must use generated/shared geometry definitions and golden fixtures.

Add a backend endpoint that returns the canonical PNG preview for the current project. The user-visible preview must be the exact backend image that will be encoded, not a merely similar SVG recreation.

## 4.10 Pure API

Create or extend a focused module:

```text
src/k2_region_lab/volumetric_control.py
```

Suggested API:

```python
@dataclass(frozen=True, slots=True)
class VolumetricControlFormat:
    identifier: str
    palette: Mapping[str, tuple[int, int, int]]
    palette_sha256: str
    renderer_version: int

@dataclass(frozen=True, slots=True)
class VolumetricControlImage:
    region_id: str | None
    width: int
    height: int
    rgb: np.ndarray
    sha256: str
    non_background_pixels: int
    coverage: float

@dataclass(frozen=True, slots=True)
class VolumetricControlBundle:
    format: VolumetricControlFormat
    full: VolumetricControlImage
    subjects: Mapping[str, VolumetricControlImage]

def render_volumetric_control_bundle(...) -> VolumetricControlBundle:
    ...
```

No ComfyUI imports in this pure renderer.

---

# 5. Training toolchain

Training is separate from the production control-plane and inference image.

Do not add large training dependencies to the normal web/control-plane environment.

## 5.1 Directory structure

Add:

```text
training/krea2_volumetric_control/
├── README.md
├── pyproject.toml or requirements.lock
├── upstream.json
├── build_coco_pairs.py
├── prepare_latent_shards.py
├── train.py
├── evaluate.py
├── inspect_checkpoint.py
├── dataset.py
├── coco_pose.py
├── checkpoint_metadata.py
├── Dockerfile
└── tests/
```

The reusable control renderer must be imported from the main K2Lab package.

Do not duplicate the geometry inside the training directory.

## 5.2 Upstream trainer

Use the control-agnostic training method from:

```text
Tanmaypatil123/Krea-2-controlnet
```

The known recipe:

- trains on Krea-2-Raw;
- freezes the base;
- trains the expanded input projection;
- applies rank-64 LoRA weights across all 28 blocks;
- channel-concatenates control and noisy latents;
- stores only trainable weights.

Before copying upstream code:

1. inspect and record its exact commit;
2. inspect its license;
3. record attribution;
4. do not vendor code if the license does not permit it.

If vendoring is not clearly permitted:

- require a separately checked-out pinned upstream path;
- validate the commit before training;
- place all K2Lab-specific dataset and metadata code in this repository;
- document the exact setup command.

Create:

```json
{
  "repository": "https://github.com/Tanmaypatil123/Krea-2-controlnet",
  "commit": "<exact pinned commit>",
  "verified_at": "<ISO date>",
  "license": "<detected license or external-checkout requirement>"
}
```

Do not follow mutable `main` during a production training run.

## 5.3 Training environment

Provide a separate container definition.

Do not assume an A40 limit.

Target:

- BF16;
- H100/H200/A100-80GB or larger;
- batch 8 at approximately one megapixel;
- gradient checkpointing;
- Krea-2-Raw base;
- Qwen3-VL text conditioning;
- Qwen-Image VAE.

The training container must not contain RunPod API credentials by default.

## 5.4 Data source

The streamlined initial data source is COCO 2017:

- train images;
- person-keypoint annotations;
- caption annotations;
- untouched validation split.

The scripts must require the user to supply local COCO paths or explicit download consent. Do not redistribute COCO data.

## 5.5 COCO to K2 mapping

COCO joints:

```text
nose
left/right eye
left/right ear
left/right shoulder
left/right elbow
left/right wrist
left/right hip
left/right knee
left/right ankle
```

K2 joints:

```text
neck
left/right shoulder
left/right elbow
left/right wrist
left/right hip
left/right knee
left/right ankle
```

Mapping:

```text
neck = midpoint(left_shoulder, right_shoulder)
```

Head ellipse:

1. use visible nose/eyes/ears to estimate face center and horizontal span;
2. use neck-to-face distance to estimate vertical radius;
3. if insufficient face points:
   - derive center above neck;
   - use shoulder width and current standing-pose proportions;
4. clamp to the current valid head-radius range.

Record which fallback was used per sample.

## 5.6 Initial sample filtering

Default first-run filters:

- `iscrowd == 0`;
- both shoulders visible/labeled;
- both hips visible/labeled;
- at least 8 usable body joints;
- target person bounding-box height at least 96 pixels after transform;
- exclude degenerate or nonfinite coordinates;
- exclude controls with less than 1% or greater than 75% canvas coverage;
- retain partially occluded joints when annotated;
- no requirement for hands/fingers.

Make thresholds CLI options and record them in the manifest.

## 5.7 Sample composition

Build a mixed dataset:

```text
70% full-image samples
30% subject-focused crops
```

Full-image samples:

- render every qualifying person;
- teach arbitrary placement and multiple bodies;
- retain scene captions.

Subject-focused crops:

- one qualifying person;
- crop with 15–40% context margin;
- teach strong anatomical interpretation at useful scale.

Make the mix configurable.

Do not create per-person duplicate controls against an unchanged full image without ensuring target/control alignment.

## 5.8 Bucketing and transform order

The training/inference control raster must match.

Required order:

1. select the final approximately-one-megapixel aspect bucket;
2. compute resize and crop;
3. transform target image and keypoints;
4. apply standard left/right semantic swaps if horizontally flipping;
5. render the control image directly at final bucket dimensions;
6. save the already aligned target/control pair;
7. VAE-encode without performing a second independent crop.

Do not render thin/small controls and then enlarge them.

## 5.9 Horizontal flip

If enabled:

- flip the target;
- transform all x coordinates;
- perform the standard left/right joint-identity swap;
- re-render from transformed semantic joints;
- do not merely flip an already-rendered color map;
- ensure the anatomical left palette remains attached to the transformed anatomical left joints.

Add a golden test.

## 5.10 Captions

Use one deterministic COCO caption per sample.

Default selection:

- prefer the longest valid caption within a reasonable token limit;
- normalize whitespace;
- preserve subject/scene content;
- do not automatically append detailed pose prose.

Rationale:

- the adapter must learn pose from the control image;
- captions should identify content and scene;
- captions should not be the only source of limb layout.

Keep upstream caption dropout at `0.10`.

Add evaluation cases with:

- generic person prompt;
- empty prompt;
- prompt unrelated to pose;
- pose-conflicting prompt.

## 5.11 Pair output

Output:

```text
dataset/
├── train/
│   ├── images/
│   ├── controls/
│   ├── metadata.jsonl
│   └── manifest.json
└── validation/
    ├── images/
    ├── controls/
    ├── metadata.jsonl
    └── manifest.json
```

Each metadata line includes:

```json
{
  "file_name": "000001.jpg",
  "control_file_name": "000001.png",
  "text": "caption",
  "source_image_id": 123,
  "person_annotation_ids": [456],
  "bucket": [1024, 1024],
  "control_format": "k2-volumetric-pose-control-v1",
  "control_sha256": "...",
  "transform": {},
  "head_fallback": "face_points"
}
```

The manifest includes:

- source annotation hashes;
- K2Lab commit;
- renderer format/hash;
- upstream trainer commit;
- filters;
- counts;
- bucket distribution;
- single/multi-person distribution;
- caption statistics;
- excluded-sample reasons;
- manifest SHA-256.

## 5.12 VAE shard preparation

Implement a K2-specific prebucketed shard preparer.

Requirements:

- load each final target/control pair;
- assert identical size;
- assert declared bucket matches actual size;
- encode target and control with the same Qwen-Image VAE;
- apply the VAE's expected latent mean/std normalization;
- assert equal latent shapes;
- write the upstream-compatible `.npz` shard format;
- support resume with `_DONE`;
- validate every shard index;
- include source/control hashes.

Do not rely on an upstream path that silently re-resizes already final controls.

## 5.13 Initial training settings

Provide these defaults:

```text
Base model:             Krea-2-Raw
Control type label:     k2_volumetric_pose_v1
Precision:              BF16
Rank:                   64
Learning rate:          1e-4
AdamW betas:            0.9, 0.99
Weight decay:           0
Per-step batch:         8
Gradient accumulation:  4
Effective batch:        32
Warmup:                 200
Caption dropout:        0.10
Initial max steps:      6000
Checkpoint interval:    500
Validation interval:    250
```

Training sequence:

```text
20-step synthetic smoke test
500-step small-data test
2,000-step capacity test
6,000-step first serious run
```

Do not claim 6,000 steps is optimal. Select by validation.

## 5.14 Resume correctness

Inspect the upstream resume implementation.

If it restores only weights/step but not:

- optimizer;
- scheduler;
- RNG;
- data position;
- gradient accumulation;

document that limitation and add complete resume state or clearly mark resume as approximate.

Do not claim exact continuation if it is not exact.

## 5.15 Evaluation

Implement a fixed evaluation set containing:

- standing;
- seated;
- crouching;
- leaning;
- asymmetric arms;
- arms above/below torso;
- split legs;
- back-facing examples;
- cropped subjects;
- two separated people;
- crossed limbs;
- vague prompts;
- prompt swaps.

Generate with:

- Krea Raw;
- Krea Turbo;
- adapter off;
- adapter strengths `0.4, 0.6, 0.8, 1.0, 1.2`;
- fixed seeds.

Measure:

- detected head-center error;
- torso-center error;
- shoulder/hip line orientation;
- elbow/wrist/knee/ankle normalized errors where detected;
- missing-person rate;
- extra-person rate;
- prompt/identity adherence review;
- image-quality review.

The primary goal is coarse placement, not exact fingers or facial direction.

## 5.16 Checkpoint selection rule

Proceed with rank 64 when it provides useful coarse pose control without unacceptable quality loss.

If results are weak:

1. verify data/control parity;
2. improve dataset and renderer;
3. train longer;
4. test rank 128;
5. only then consider a larger partial-ControlNet branch.

Do not indefinitely raise strength to compensate for a bad representation.

---

# 6. Checkpoint contract

## 6.1 Dedicated adapter kind

A Krea Control-LoRA is not an ordinary regional LoRA.

Add a dedicated asset kind:

```text
krea_control_loras
```

Do not present it in the ordinary LoRA list.

Do not reuse:

```text
controlnet_models
```

That legacy kind may remain for compatibility, but it is not the new adapter type.

## 6.2 Required safetensors metadata

Write:

```text
k2lab_adapter_kind = krea2_control_lora
k2lab_adapter_version = 1
k2lab_control_format = k2-volumetric-pose-control-v1
k2lab_control_format_sha256 = <hash>
k2lab_renderer_version = 1
k2lab_base_model = krea/Krea-2-Raw
k2lab_inference_targets = krea/Krea-2-Raw,krea/Krea-2-Turbo
k2lab_rank = 64
k2lab_expanded_input_projection = true
k2lab_expected_transformer_blocks = 28
k2lab_control_channel_mode = rgb
k2lab_control_normalize = none
k2lab_control_invert = false
k2lab_dataset_manifest_sha256 = <hash>
k2lab_trainer_repository = <repo>
k2lab_trainer_commit = <commit>
k2lab_training_commit = <K2Lab commit>
k2lab_created_at = <ISO timestamp>
```

Also retain upstream metadata.

## 6.3 Strict validator

Before GPU sampling, inspect:

- safetensors validity;
- metadata;
- expanded first-projection weight;
- expected doubled input width;
- rank;
- all required compatible block LoRA pairs;
- no unexpected pickle format;
- Krea architecture compatibility;
- control-format equality;
- palette/renderer hash;
- base model family.

Return a structured compatibility report.

Do not rely only on filename.

## 6.4 Older checkpoints

Allow an explicit advanced override only for checkpoints lacking K2 metadata after tensor-shape validation.

The GUI must label such a checkpoint:

```text
Unverified legacy Krea control checkpoint
```

Do not enable it silently.

---

# 7. Native inference module

Create:

```text
src/k2_region_lab/krea_control_lora.py
```

Use the community Krea2 Control-LoRA integration as a behavioral reference, but adapt it to:

- the pinned ComfyUI version;
- K2Lab model lifecycle;
- current scope-aware Prediction composite;
- current retained baseline cache;
- current regional LoRA implementation;
- current typed errors and diagnostics.

## 7.1 Core types

Suggested API:

```python
@dataclass(frozen=True, slots=True)
class KreaControlCheckpointInfo:
    path: Path
    sha256: str
    metadata: Mapping[str, str]
    rank: int
    expanded_projection_key: str
    compatible_block_pairs: int
    format_id: str
    verified: bool

@dataclass(frozen=True, slots=True)
class KreaControlLatentBundle:
    full: torch.Tensor
    subjects: Mapping[str, torch.Tensor]
    source_hashes: Mapping[str, str]
    encode_seconds: float

@dataclass(frozen=True, slots=True)
class KreaControlRuntimeReport:
    checkpoint_sha256: str
    format_id: str
    strength: float
    loaded_lora_keys: int
    patched_model_keys: int
    scope_calls: Mapping[str, int]
    control_latent_shapes: Mapping[str, tuple[int, ...]]
```

## 7.2 Model isolation

Current retained base-model behavior must remain safe.

Required sequence:

1. obtain the retained/cached baseline model exactly as current code does;
2. clone the model path for this generation;
3. install the Control-LoRA on the clone;
4. install current routed regional LoRAs without contaminating the baseline;
5. attach control-latent mapping to the clone;
6. sample;
7. restore/eject projection state on cleanup;
8. discard the generation clone or retain only according to current safe model-cache policy;
9. prove the baseline model's original input projection is unchanged.

Never patch the retained baseline object in place.

## 7.3 Expanded input projection

Implement an equivalent of:

```text
Krea2ControlInputProjection
```

Behavior:

- normal image tokens continue through the current native/original first projection;
- control tokens use only the checkpoint's control half;
- result is the sum of native image output and control contribution;
- ordinary first-projection LoRA patches must remain active;
- control projection state exists only during the diffusion forward;
- original projection is restored in `finally`, on detach, and on cleanup.

## 7.4 Block LoRA patches

Apply compatible Control-LoRA block weights through the current ComfyUI `ModelPatcher`.

Requirements:

- match live module shapes, including quantized storage wrappers;
- do not assume `state_dict()` storage shape alone;
- report loaded, patched, and skipped keys;
- reject zero compatible patches;
- apply one user strength;
- preserve current routed regional LoRAs.

## 7.5 Interaction with existing regional LoRAs

Test both:

- ordinary global LoRAs;
- scope-aware subject identity LoRAs.

The control projection must not bypass a regional/global LoRA attached to the native image projection.

If current routed LoRA code directly wraps a module also needed by the control adapter:

- define one explicit patch order;
- test it;
- keep both deltas;
- do not disable either feature.

If a rare target is genuinely incompatible, fail with:

```text
krea_control_lora_target_conflict
```

and identify the target privately.

## 7.6 VAE encoding

Use the already selected Krea/Qwen VAE.

For every job:

1. render full and required subject control images;
2. convert to `[B,H,W,C]` RGB float `[0,1]`;
3. encode each image independently;
4. do not let a 3D/video VAE interpret multiple subjects as frames;
5. process each latent through the selected model's latent format;
6. cache the processed result for the entire job;
7. release it at job cleanup.

Do not re-encode at each denoising step or each semantic forward.

## 7.7 Latent size

The control image is rendered at the generation canvas size.

After VAE encoding:

- match the main latent;
- if a small mismatch exists because of patch padding, use the same native Krea control resize/pad process;
- validate token feature count;
- do not independently center crop.

## 7.8 Conditioning-scope-aware selection

Reuse the existing exception-safe conditioning scope context.

Selection:

```text
FULL scope:
    full control latent

SUBJECT(region_id) scope:
    that region's subject-only control latent
```

Modes:

### Spatial only

One full prediction:

```text
use full control latent
```

### Attention isolation

One full prediction:

```text
use full control latent
```

### Prediction composite

Full prediction:

```text
use full control latent
```

Subject A prediction:

```text
use Subject A control latent
```

Subject B prediction:

```text
use Subject B control latent
```

Normal phase:

```text
only the full prediction runs
use full control latent
```

If a subject scope has no matching control latent, raise:

```text
krea_control_scope_missing
```

Do not fall back to the full map.

## 7.9 Context implementation

The diffusion wrapper must read the active conditioning scope at call time.

Do not mutate one process-global `control_latent` before sequential forwards without exception-safe restoration.

Recommended attachment:

```python
{
    "full": full_latent,
    "subjects": {region_id: latent},
}
```

The wrapper:

```python
scope = CURRENT_CONDITIONING_CONTEXT.get()
control = select_control_latent(scope, mapping)
```

The current single-GPU restriction for Prediction composite remains.

## 7.10 Control strength and schedule

First version UI:

```text
Enable Krea volumetric pose adapter
Adapter checkpoint
Strength
```

Default strength:

```text
1.0
```

Validation:

```text
0.0 <= strength <= 2.0
```

The adapter remains active through:

- hard phase;
- soft phase;
- normal phase.

Do not fade it with the spatial gate in v1.

Reason:

- the adapter is the learned pose interpretation;
- releasing it during normal steps would permit the normal phase to erase pose;
- current hard/soft release already controls spatial and semantic isolation.

A strength of zero is equivalent to disabled, but preserve explicit enable state.

Do not add start/end or per-phase strength controls until first tests justify them.

## 7.11 Adapter without hard gating

Permit:

```text
Control-LoRA enabled
Pose gating disabled
```

This is required for diagnostic Test A.

In that case:

- render full control;
- run ordinary full generation;
- adapter remains active;
- no hard/soft mask;
- no extra subject forwards unless current Prediction composite is independently enabled by valid settings.

## 7.12 Number of model forwards

The adapter adds no denoiser forward by itself.

Prediction composite counts remain unchanged.

It adds:

- one-time control rendering;
- one-time VAE encodes;
- projection/control-token processing inside existing forwards.

Preserve current forward accounting.

---

# 8. Project schema and protocol

Expected migration:

```text
22 → 23
```

If the actual current schema is later than 22, increment exactly once and document the actual numbers.

## 8.1 Project fields

Add:

```json
{
  "pose_control_lora_enabled": false,
  "pose_control_lora_model": null,
  "pose_control_lora_strength": 1.0,
  "pose_control_format": "k2-volumetric-pose-control-v1"
}
```

Do not overload ordinary saved LoRAs.

## 8.2 Migration

For all older projects:

```text
pose_control_lora_enabled = false
pose_control_lora_model = null
pose_control_lora_strength = 1.0
pose_control_format = current format
```

No existing project may become adapter-enabled because of migration.

Preserve all current schema-22 semantic settings exactly.

## 8.3 Job request

Add:

```text
pose_control_lora_file_id
```

Do not pass a browser-supplied raw path.

Resolve the opaque file ID on the workspace agent.

## 8.4 Worker protocol and capability

Bump the worker protocol.

Advertise:

```json
{
  "krea_volumetric_pose_control_lora": {
    "version": 1,
    "control_formats": ["k2-volumetric-pose-control-v1"],
    "scope_aware": true,
    "single_gpu_prediction_composite": true,
    "strength_schedule": "constant_all_steps"
  }
}
```

Reject old worker/new client combinations explicitly.

## 8.5 Asset storage

Add physical layout support for:

```text
models/krea_control_loras
```

Update:

- `FileKind`;
- storage layout;
- upload validation;
- Civitai/Hugging Face destination lists;
- manifests;
- portable migration allowlist;
- inventory;
- deletion;
- file picker;
- API TypeScript types;
- tests.

Do not place these files into the normal regional-LoRA picker.

---

# 9. Web GUI

## 9.1 Controls

Under Advanced → Volumetric pose gating, add:

```text
Krea volumetric pose adapter
[ ] Enable trained pose adapter
Checkpoint: [asset selector]
Strength:   [0.00–2.00, default 1.00]
[Preview control image]
```

Help text:

```text
The trained Krea Control-LoRA teaches the model that the hidden
head, torso, and left/right limb colors represent a human pose.
It complements Prediction composite and the hard/soft spatial gate.
```

## 9.2 Compatibility display

After selecting a checkpoint, show:

- verified/unverified;
- control format;
- rank;
- base model;
- expected blocks;
- checkpoint hash;
- renderer/palette hash;
- compatibility result.

Block generation for a verified incompatibility.

## 9.3 Control preview

Provide tabs:

```text
All subjects
Subject A
Subject B
...
```

The PNG must come from the backend canonical renderer.

Show:

- exact format ID;
- image dimensions;
- control hash;
- coverage.

Do not show the wide support mask as the control preview.

## 9.4 Warnings

When enabled:

- no posed subject → block;
- no checkpoint → block;
- wrong format → block;
- unverified legacy checkpoint → require explicit acknowledgement;
- ordinary Krea model not selected → block;
- non-Krea VAE → block;
- Prediction composite multi-GPU → retain current block.

## 9.5 Work estimate

Do not change model-forward count.

Add:

```text
Pose adapter preprocessing:
1 full + N subject control VAE encodes
```

Only subject maps needed by the selected semantic mode should be encoded.

---

# 10. Runtime metadata

Add:

```json
{
  "pose_control_lora_runtime": {
    "version": 1,
    "enabled": true,
    "checkpoint_file_id": "...",
    "checkpoint_sha256": "...",
    "checkpoint_metadata": {
      "rank": 64,
      "base_model": "krea/Krea-2-Raw",
      "control_format": "k2-volumetric-pose-control-v1",
      "verified": true
    },
    "strength": 1.0,
    "full_control_sha256": "...",
    "subject_controls": {
      "region-id": {
        "sha256": "...",
        "coverage": 0.14,
        "latent_shape": [1, 16, 128, 128]
      }
    },
    "vae_encode_seconds": 1.23,
    "loaded_lora_keys": 100,
    "patched_model_keys": 100,
    "scope_calls": {
      "full": 18,
      "subject:region-id": 10
    },
    "format_sha256": "...",
    "renderer_version": 1
  }
}
```

Do not embed raw control PNGs in normal output metadata.

Store them only under an explicit private debug flag.

---

# 11. Typed errors

Add:

```text
krea_control_checkpoint_invalid
krea_control_checkpoint_incompatible
krea_control_format_mismatch
krea_control_projection_missing
krea_control_block_weights_missing
krea_control_vae_incompatible
krea_control_encode_failed
krea_control_latent_shape_invalid
krea_control_scope_missing
krea_control_lora_target_conflict
krea_control_hook_incompatible
```

Safe user messages must distinguish:

- asset problem;
- VAE problem;
- model architecture problem;
- runtime scope problem;
- sampler problem.

Preserve private traceback and diagnostic ID.

---

# 12. RunPod provider-timeout investigation and correction

## 12.1 Observed defect

The event log showed:

```text
4:26:42 PM  WORKER  Generation complete; baseline model retained in VRAM
4:26:42 PM  WORKER  Remote job complete.
4:26:44 PM  WORKER  Remote job complete. The verified output is stored in cloud files.
5:00:31 PM  ERROR   RunPod did not respond before the provider timeout.
```

The job had already completed successfully and its output was stored.

The exact timeout text is currently produced by the RunPod provider API client when an `httpx.TimeoutException` occurs.

The current browser workspace refresh loop polls workspace status repeatedly and reports every caught refresh exception into the studio event log as an error.

The current backend workspace-status path calls RunPod `get_pod`.

This strongly suggests the red line is a passive workspace-provider refresh timeout, not a generation failure.

Codex must confirm this with operation-level diagnostics before implementing the final fix.

## 12.2 Required investigation

Instrument, reproduce, and identify:

- frontend caller;
- control-plane endpoint;
- backend method;
- provider endpoint;
- HTTP method;
- timeout type:
  - connect;
  - read;
  - write;
  - pool;
- elapsed time;
- retry count;
- whether agent health remained reachable;
- workspace state before/after;
- active/completed job state;
- whether output remained retrievable.

Use stable operation names, for example:

```text
workspace.passive_provider_refresh
workspace.explicit_start
workspace.explicit_stop
workspace.explicit_delete
workspace.startup_reconcile
workspace.lease_reaper
```

Log:

```text
operation
workspace_id
provider_resource_id hash or redacted suffix
attempt
elapsed_ms
exception_type
result
```

Never log API keys or complete provider responses containing secrets.

## 12.3 Correct semantic classification

A passive provider status refresh timeout is:

```text
transient provider-status warning
```

It is not:

- a generation error;
- a worker error;
- a job failure;
- proof the Pod stopped;
- proof the output is invalid;
- a reason to replace a terminal completed job state.

## 12.4 Terminal job monotonicity

Once a job is:

```text
completed
failed
cancelled
```

unrelated workspace/provider polling may not change that job's state or append a job-scoped error.

A completed job remains completed.

Its output IDs remain available.

Add explicit invariants/tests.

## 12.5 Stale-while-revalidate provider status

Stop making every five-second browser refresh depend on a fresh RunPod provider API response.

Implement provider-status caching with:

```text
success TTL: 30 seconds
failure backoff: 15, 30, 60, 120 seconds, capped
single in-flight refresh per workspace
```

The normal workspace endpoint should:

1. return the durable last-known workspace record promptly;
2. include provider freshness metadata;
3. trigger or await a provider refresh only according to TTL/operation requirements;
4. keep last-known state on transient provider timeout/unavailability;
5. surface definitive errors such as provider 404 separately.

Recommended model:

```python
class ProviderStatusFreshness(BaseModel):
    stale: bool
    refresh_in_flight: bool
    last_attempt_at: datetime | None
    last_success_at: datetime | None
    last_error_code: str | None
    last_error_message: str | None
    next_retry_at: datetime | None
```

Add it to `WorkspaceRecord` or a backward-compatible status envelope.

## 12.6 Passive refresh behavior

For passive status polling:

Catch only transient errors:

```text
provider_timeout
provider_unavailable
```

Then:

- return last-known workspace state;
- mark provider status stale;
- do not set workspace state to `error`;
- do not overwrite durable workspace error fields;
- do not throw HTTP 504 to the browser;
- add private/audit diagnostics;
- keep agent/job/output access available.

Do not suppress:

- invalid API key;
- insufficient permissions;
- provider resource not found;
- confirmed incompatible/deleted state.

## 12.7 Agent-first liveness

When the workspace is currently `ready` and the authenticated workspace agent responds:

- treat the agent as direct evidence that the runtime is reachable;
- do not tell the user the workspace failed merely because the separate RunPod control API timed out;
- display provider status as stale while retaining agent readiness.

The provider API remains authoritative for billable lifecycle operations, but it is not the only liveness signal.

## 12.8 Provider GET retries

For idempotent GET/HEAD provider requests:

- retry transient timeout/502/503/504;
- use bounded exponential backoff with jitter;
- use one total operation deadline;
- record attempts;
- do not retry indefinitely.

Suggested:

```text
attempts: 3
delays: 0.25s, 0.75s
total deadline: 30s
```

Tune after tests.

## 12.9 Explicit mutation timeouts

For start/stop/delete/resume requests, a timeout is ambiguous: RunPod may have applied the operation without returning the response.

Do not blindly retry a non-idempotent mutation.

After a mutation timeout:

1. poll `get_pod` with bounded reconciliation;
2. determine whether requested state was reached;
3. if reached, return success with a warning/audit note;
4. if definitively not reached, return the operation error;
5. if indeterminate, return:

```text
RunPod did not confirm the operation before timeout.
The request may have succeeded; K2Lab will continue status reconciliation.
```

Do not encourage repeated destructive clicks while state is unknown.

## 12.10 Frontend severity

Add event severity/source distinctions:

```text
info
worker
warning
error
```

Optional source:

```text
job
workspace
provider
transfer
```

Passive timeout display:

```text
WARNING  PROVIDER
RunPod status refresh timed out. Using the last known workspace
status; completed jobs and cloud outputs are unaffected.
```

Do not render it as red `ERROR`.

## 12.11 Dedupe and recovery

On transition:

```text
healthy → stale
```

append one warning.

While stale:

- do not append the same warning every poll;
- update a small status badge/banner.

On transition:

```text
stale → healthy
```

append one informational recovery event:

```text
RunPod provider status is reachable again.
```

Clear the badge.

## 12.12 Browser polling

Current periodic refresh must no longer call `report(..., "error")` for a passive transient provider timeout.

Use `ApiError.code`.

Behavior:

```typescript
if (isPassiveProviderTransient(error)) {
    updateProviderFreshness(...)
    appendDedupedWarning(...)
    keepCurrentWorkspace(...)
} else {
    handleActualWorkspaceError(...)
}
```

Do not use message-string matching.

## 12.13 Event-log separation

Worker job events remain worker events.

Workspace provider warnings must not be inserted as if emitted by the completed worker.

The event row should clearly identify:

```text
PROVIDER
```

rather than:

```text
WORKER
```

## 12.14 Output availability

When provider status is stale but agent/output proxy remains reachable:

- output preview/download continues;
- latest output remains displayed;
- project save/open continues where agent access works;
- worker-memory status may independently become unavailable without invalidating output;
- the UI does not clear the completed result.

## 12.15 Provider lock contention

Inspect `_SerializedRunPodApi`.

Ensure a slow passive refresh cannot unnecessarily block an explicit user lifecycle request behind a long status poll.

Options:

- separate read/status and mutation lanes;
- cancel/skip stale passive refresh when an explicit mutation arrives;
- short status-cache path;
- priority lock.

Implement and test one safe strategy.

---

# 13. Timeout tests

## 13.1 Backend provider timeout after completed job

Using `httpx.MockTransport`:

1. create ready workspace;
2. create completed job with output ID;
3. make `GET /pods/{id}` time out;
4. call passive workspace refresh.

Assert:

- HTTP response remains successful or uses the new stale envelope;
- workspace remains ready/last-known;
- provider freshness is stale;
- completed job remains completed;
- output ID remains;
- no job error is created;
- audit contains transient timeout.

## 13.2 Repeated timeout

Simulate ten polls.

Assert:

- one visible warning transition;
- no ten duplicate events;
- backoff increases;
- no provider request on every five-second UI tick while backoff is active.

## 13.3 Recovery

Next provider GET succeeds.

Assert:

- stale clears;
- one recovery event;
- workspace updates;
- no output/job regression.

## 13.4 Provider 404

Assert this is not suppressed.

The missing-Pod recovery UI must still work.

## 13.5 Agent healthy/provider stale

Assert:

- workspace remains operational;
- agent-backed job/output endpoints work;
- status shows provider stale, agent reachable.

## 13.6 Mutation timeout

Simulate stop POST timeout followed by provider GET showing stopped.

Assert:

- action resolves as stopped;
- no unsafe second stop;
- audit records ambiguous timeout and reconciliation.

Also test indeterminate state.

## 13.7 Frontend fake-timer test

With a completed job and loaded output:

- trigger passive refresh timeout;
- event dock shows warning/provider;
- no red error;
- output remains;
- job state remains completed;
- repeated ticks do not duplicate;
- recovery clears warning.

---

# 14. Non-regression tests for the current implementation

Before adding new behavior, capture current tests and add explicit regression coverage.

## 14.1 Adapter disabled

Same seed/settings with adapter disabled:

- same sampler call count;
- same semantic-scope evaluation count;
- same prompt compilation;
- same LoRA scope calls;
- no control render;
- no control VAE encode;
- no expanded projection;
- no new attachment;
- same output metadata except schema migration fields.

## 14.2 Live semantic baseline

Preserve the known count:

```text
1 subject
8 hard
2 soft
8 normal
Prediction composite
→ 18 full + 10 subject = 28 semantic-scope evaluations
```

The adapter must not change that count.

## 14.3 All semantic modes

Test adapter on/off under:

- Spatial only;
- Attention isolation;
- Prediction composite.

## 14.4 Regional LoRAs

Test:

- global LoRA;
- standard subject LoRA;
- identity LoRA;
- two distinct subject LoRAs;
- Control-LoRA plus each;
- adapter cleanup.

## 14.5 Gating and sigma

All current hard/soft/sigma tests must remain unchanged and pass.

## 14.6 Other job kinds

Run tests for:

- edit image;
- refine faces;
- detect faces;
- post-upscale;
- worker memory release.

The new adapter must not appear in those paths unless explicitly designed later.

## 14.7 Project and cloud behavior

Test:

- current schema exact round trip;
- migration to new schema;
- PNG metadata;
- cloud project;
- portable manifest;
- asset upload/download/delete;
- old projects adapter disabled.

---

# 15. Control-specific automated tests

## 15.1 Renderer

Test:

- exact format ID;
- exact palette;
- no editor handles;
- black background;
- full/subject maps;
- deterministic hash;
- left/right colors;
- head/torso/limb bounds;
- out-of-box limbs;
- 4× supersampling;
- frontend/backend preview parity.

## 15.2 Dataset builder

Test synthetic COCO fixtures:

- neck midpoint;
- head fallback;
- crop/resize;
- horizontal flip and left/right swap;
- multi-person map;
- manifest hashes;
- filtering reasons;
- exact image/control alignment.

## 15.3 Checkpoint validator

Use synthetic safetensors:

- valid rank-64;
- missing projection;
- wrong doubled width;
- missing block weights;
- wrong format hash;
- ordinary LoRA incorrectly selected;
- unsafe file rejected.

## 15.4 Control projection

With Torch:

- native image projection preserved;
- control half added;
- zero control contribution behavior;
- strength zero;
- token count;
- shape mismatch;
- projection restored after success/error;
- retained base unchanged.

## 15.5 Scope selection

Test:

- full → full latent;
- Subject A → A latent;
- Subject B → B latent;
- missing subject typed error;
- context restoration;
- normal Prediction-composite full path.

## 15.6 Regional LoRA coexistence

Verify numerically:

- native image projection delta remains;
- control delta remains;
- subject identity delta remains;
- other subject identity delta remains zero in the wrong subject scope.

## 15.7 VAE encode

Test independent images, not video frames.

Assert latent shapes and cache hits.

---

# 16. RunPod acceptance sequence

Use one GPU. Start with Euler and scheduler default.

## 16.1 Stage A: current behavior, adapter off

Repeat the existing working job.

Confirm:

- completion;
- output;
- exact semantic counts;
- no adapter runtime block;
- no regression.

## 16.2 Stage B: adapter alone

Settings:

```text
Adapter: enabled
Strength: 1.0
Pose gating: disabled
Prediction composite: not required
No character LoRA
One subject
```

Generate same prompt with two strongly different mannequins.

Success:

- output pose changes with mannequin;
- subject semantics remain;
- no gating required to observe control.

## 16.3 Stage C: adapter plus Prediction composite

Use adversarial scene/subject prompts.

Confirm:

- correct subject semantics;
- major pose follows mannequin better than the current adapter-off baseline;
- runtime scope counts remain expected;
- full and subject control hashes differ appropriately.

## 16.4 Stage D: light gate

Initial recommended settings:

```text
Hard:   1–2
Soft:   2–4
Normal: 8
Adapter: 1.0 throughout
Fill subject boxes: off for pose test
Late spatial relaxation: off for pose test
```

Do not begin by retaining the old 8-hard compensation setting.

## 16.5 Stage E: identity LoRA

Add one subject identity LoRA.

Confirm identity and pose coexist.

## 16.6 Stage F: two subjects

Use:

- distinct mannequins;
- distinct prompts;
- distinct identity LoRAs.

Confirm:

- full scope uses both mannequins;
- Subject A scope uses only A;
- Subject B scope uses only B;
- prompt/LoRA swap changes identity, not pose ownership.

## 16.7 Stage G: timeout reproduction

After a completed generation:

- force/mock provider `get_pod` timeout;
- keep agent/output reachable.

Confirm:

- completed output remains;
- event is warning/provider;
- no red generation error;
- no repeated warning spam;
- recovery message appears.

---

# 17. Documentation

Update:

```text
README.md
docs/subject_pose_control.md
docs/web_desktop_parity.md
docs/runpod_workspace_operations.md
training/krea2_volumetric_control/README.md
```

Document:

- current semantic composite remains required/recommended;
- what the adapter adds;
- control format;
- training commands;
- checkpoint metadata;
- inference controls;
- Raw training / Turbo inference;
- initial light-gating recommendation;
- known limitations;
- provider-stale warning semantics;
- completed jobs are not invalidated by passive provider timeouts.

Remove no current documentation unless it is factually obsolete.

---

# 18. Expected file-level changes

Codex must inspect current symbols, but expected areas include:

## Core

```text
src/k2_region_lab/volumetric_control.py
src/k2_region_lab/krea_control_lora.py
src/k2_region_lab/volumetric_pose.py
src/k2_region_lab/semantic_conditioning.py
src/k2_region_lab/regional_lora.py
src/k2_region_lab/worker/runtime.py
src/k2_region_lab/worker/protocol.py
src/k2_region_lab/project.py
```

## Agent/API/storage

```text
src/k2_region_lab/agent/domain.py
src/k2_region_lab/agent/storage.py
src/k2_region_lab/agent/jobs.py
src/k2_region_lab/web/agent_client.py
src/k2_region_lab/web/runpod_api.py
src/k2_region_lab/web/runpod_backend.py
src/k2_region_lab/web/domain.py
src/k2_region_lab/web/state_store.py
src/k2_region_lab/web/app.py
```

## Browser

```text
web/client/src/api.ts
web/client/src/studioProject.ts
web/client/src/components/WorkspaceStudio.tsx
web/client/src/components/PoseGatingControls.tsx
web/client/src/eventLog.ts
web/client/src/styles.css
```

## Training

```text
training/krea2_volumetric_control/*
```

## Tests/docs/image

```text
tests/*
web/client/scripts/*
Dockerfile.workspace
optional training Dockerfile
README.md
docs/*
```

Search:

```text
PROJECT_VERSION
WORKER_PROTOCOL_VERSION
FileKind
controlnet_models
pose_control_lora
CURRENT_CONDITIONING_CONTEXT
sampler_calc_cond_batch_function
RunPod did not respond before the provider timeout
provider_timeout
controlPlane.workspace
report(
setTimeout
get_workspace_status
get_pod
```

---

# 19. Commands Codex must run

At minimum:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

Browser:

```bash
cd web/client
npm run test:project
npm run test:ui
npm run typecheck
npm run build
```

Torch/Comfy tests must run inside the workspace image or another environment containing the pinned Torch and ComfyUI:

```bash
/opt/comfyui-venv/bin/python -m pytest \
  -q tests/test_krea_control_lora.py \
     tests/test_semantic_conditioning.py \
     tests/test_worker_runtime.py \
  -rs
```

No critical Control-LoRA tensor test may remain skipped merely because the local lightweight environment lacks Torch.

Training-tool tests use synthetic fixtures and must not require COCO download or an 80GB GPU.

---

# 20. Completion criteria

All are required.

## Existing functionality

- current full test suite still passes;
- current semantic live path still works;
- adapter-disabled path is unchanged;
- edit/faces/assets/projects/workspaces still work.

## Control format

- exact hidden RGB format exists;
- backend preview works;
- training and inference use the same renderer/hash.

## Training

- COCO pair builder works;
- shard preparer works;
- trainer is pinned/reproducible;
- checkpoint metadata is written;
- evaluator exists;
- smoke training can start.

## Inference

- checkpoint validator works;
- native expanded projection works;
- current LoRAs coexist;
- control latents are encoded once;
- conditioning scope selects correct latent;
- adapter stays active through normal phase;
- one sampler trajectory remains.

## RunPod timeout

- passive provider timeout no longer appears as a red job/generation error;
- completed job remains completed;
- output remains available;
- warning is deduplicated;
- status recovery is shown;
- explicit lifecycle failures remain actionable.

## Reporting

Codex reports truthfully what was and was not GPU-tested.

---

# 21. Explicitly deferred

Do not implement in this milestone:

- 3D mannequin;
- depth rendering;
- hand/finger rig;
- face landmarks;
- learned front/back limb ordering;
- multiple simultaneous Control-LoRAs;
- user-adjustable part palette;
- per-phase adapter strength;
- automatic pose reranking/retry;
- multi-GPU Prediction composite;
- hosted training UI;
- automatic COCO download without consent;
- background-first generation;
- img2img;
- second sampler trajectory.

---

# 22. Primary technical references Codex must verify

## Krea 2

- Official repository: `https://github.com/krea-ai/krea-2`
- Train LoRAs on Raw; apply on Turbo.
- Turbo recommended inference: 8 steps, CFG disabled.

## Krea Control-LoRA trainer

- `https://github.com/Tanmaypatil123/Krea-2-controlnet`
- Control-agnostic custom RGB path.
- Expanded input projection plus LoRA over 28 blocks.
- Qwen-Image VAE encoded control latent.
- Rank-64 released depth adapter demonstrates the architecture.

## Native ComfyUI reference

- `https://github.com/facok/comfyui-krea2-controlnet`
- Native first-projection preservation.
- Control half contribution.
- ModelPatcher block weights.
- VAE encode and latent normalization.
- Cleanup and attachment behavior.

## Current K2Lab

Use the checked-out current working tree as the source of truth, especially:

- subject-semantic Prediction composite;
- scope ContextVar;
- current regional LoRA router;
- current pose gate controller;
- current project schema;
- current worker protocol;
- current RunPod polling and event-log behavior.

Do not code against an old online snapshot when the working tree is newer.

---

# 23. Final Codex report template

```text
Starting branch:
Starting commit:
Starting working-tree status:
Ending commit:
Pinned ComfyUI commit:
Pinned Krea Control-LoRA trainer commit:
Pinned native ComfyUI reference commit:

Current-version verification:
- project schema:
- worker protocol:
- agent/image version:

Implemented:
- canonical control format:
- training data builder:
- VAE shard builder:
- trainer integration:
- checkpoint validator:
- runtime integration:
- scope-aware control selection:
- GUI/assets:
- metadata:
- timeout investigation:
- timeout fix:

Non-regression:
- existing semantic modes:
- regional LoRAs:
- gating/sigma:
- image edit:
- faces:
- project persistence:
- assets:
- workspace lifecycle:

Provider-timeout root cause:
- frontend caller:
- backend endpoint:
- provider operation:
- timeout type:
- corrected behavior:

Tests:
- command → result

Torch/Comfy tests:
- command → result

Browser:
- command → result

Training smoke test:
- command → result/not run

RunPod GPU tests:
- completed/not run
- results:

Known limitations:
- ...

Deviations:
- ...
```

Do not substitute another planning document for implementation.

---

# 24. Separate main-branch QOL workstream: Hide regions

This is a required, independently reviewable QOL update. It must be implemented for the normal `main` branch and then included in the pose branch containing this Control-LoRA work.

It is not a pose-only feature and must not depend on:

- pose state;
- mannequin geometry;
- semantic routing;
- Control-LoRA code;
- project schema 23;
- worker protocol changes;
- any backend or GPU feature.

The main-branch implementation should be a small isolated commit that can be merged or cherry-picked into the pose branch. The pose-branch integration may resolve expected `RegionCanvas.tsx` conflicts, but it must preserve the exact same user-facing behavior while additionally hiding pose/mannequin overlays.

## 24.1 User outcome

Add a toggle button in the canvas toolbar, alongside the current top-of-canvas image and drawing controls.

Default state:

```text
[eye-off icon] Hide regions
```

Hidden state:

```text
[eye icon] Show regions
```

When the user selects **Hide regions**, the canvas must display the source or generated image without any region-derived visual overlay.

The toggle must hide, as applicable:

### Main branch

- region fills;
- region outlines;
- region labels;
- selected-region styling;
- resize handles;
- region movement hit targets;
- edit reference/target region overlays.

### Pose branch

Everything above, plus:

- subject mannequin body fills;
- mannequin centerlines;
- head ellipse;
- articulation handles;
- head resize handles;
- subject-box labels;
- any pose-specific selection decorations;
- any future hidden-control diagnostic overlay if it is being drawn on the ordinary canvas.

The feature must not hide or alter:

- the loaded source image;
- the generated result image;
- the source/result comparison image;
- the comparison slider/control;
- the canvas toolbar;
- Inspector fields;
- region list/state in the Inspector;
- scene/subject prompts;
- LoRA assignments;
- generation settings;
- downloaded image content.

The button is for visual inspection only.

## 24.2 State semantics

Use a UI-only boolean such as:

```typescript
const [regionsHidden, setRegionsHidden] = useState(false);
```

Requirements:

- default is visible;
- toggling does not call `onRegions`;
- toggling does not mutate any `RegionBox`;
- toggling does not mutate pose joints or head geometry;
- toggling does not alter `enabled`;
- toggling does not alter priority/order;
- toggling does not alter `selectedId`;
- toggling does not alter prompt or LoRA assignment;
- toggling does not alter the generation payload;
- toggling does not alter output metadata;
- toggling does not alter backend masks or conditioning;
- toggling does not create a project-schema field;
- toggling does not require a project-version bump;
- toggling does not persist into the canonical project JSON.

It is a local canvas-view preference for the current mounted studio session.

Opening, saving, importing, or generating must not serialize the value.

The selected region may remain selected internally. When overlays are shown again, the same selection and handles should return.

## 24.3 Modes

Show the button in:

- Generate mode;
- Edit mode, for both Reference layout and Edit targets.

Do not show the **Hide regions** button in Face mode in this first version.

Face detections and manual face lassos are not region overlays and must remain unchanged.

If the user switched to Face mode while regions were hidden, face overlays must still render normally. Returning to Generate/Edit may retain the local hidden state.

## 24.4 Interaction safety while hidden

Invisible canvas editing is prohibited.

When hiding regions:

1. cancel an armed but not yet started draw mode;
2. clear any local drag state only if doing so is safe and no pointer operation is active;
3. never delete a completed region;
4. never commit a new region as a side effect;
5. do not alter selection.

Disable the toggle during an active pointer drag/resize/joint operation, or defer the toggle until pointer completion. Do not interrupt a captured pointer in a way that corrupts region geometry.

While regions are hidden:

- the region-overlay layer must not accept pointer input;
- clicking the image must not move/select/resize a hidden region;
- no hidden resize or joint handle may remain interactive;
- draw-region and draw-subject actions must not create invisible geometry.

Preferred drawing behavior:

- clicking **Draw region** while hidden first shows regions and then enters region drawing mode;
- on the pose branch, clicking **Draw subject** while hidden first shows regions and then enters subject drawing mode.

This is preferable to allowing invisible drawing.

## 24.5 Rendering implementation

The current main and pose branches both render the toolbar and SVG annotation layer in:

```text
web/client/src/components/RegionCanvas.tsx
```

Implement the feature there unless the checked-out code has since factored the toolbar/overlay into a dedicated component.

Do not remove the SVG element if Face mode still needs it.

Recommended structure:

```tsx
const regionOverlayHidden = regionsHidden && mode !== "face";

<button
  type="button"
  className={`quiet-button ${regionsHidden ? "active" : ""}`}
  aria-pressed={regionsHidden}
  aria-label={regionsHidden ? "Show regions" : "Hide regions"}
  title={regionsHidden
    ? "Show region and subject overlays"
    : "Hide region and subject overlays to inspect the image"}
  onClick={toggleRegionsHidden}
>
  <Icon name={regionsHidden ? "eye" : "eyeOff"} />
  {regionsHidden ? "Show regions" : "Hide regions"}
</button>
```

Then:

```tsx
<svg
  className={`region-overlay ${regionOverlayHidden ? "regions-hidden" : ""}`}
  ...
  onPointerDown={regionOverlayHidden ? undefined : beginDraw}
  onPointerMove={regionOverlayHidden ? undefined : movePointer}
  onPointerUp={regionOverlayHidden ? undefined : endPointer}
  onPointerCancel={regionOverlayHidden ? undefined : endPointer}
>
  {!regionOverlayHidden && orderedRegions.map(...)}
  {mode === "face" && ...existing face overlays...}
</svg>
```

The exact implementation may instead use conditional groups or CSS, provided it satisfies:

- no region-derived pixels are visible;
- no region-derived pointer targets remain active;
- face overlays remain functional;
- showing restores the overlays without reconstruction or data loss.

Do not merely set opacity to zero while leaving pointer interactions active.

## 24.6 Button placement and responsive behavior

Place the button in the existing `.canvas-actions` toolbar.

Recommended order:

```text
Load/Replace image
Download image
Hide/Show regions
Clear canvas
Draw region
Draw subject (pose branch)
```

The toolbar may wrap according to existing responsive CSS.

Requirements:

- no horizontal viewport overflow on the supported mobile layout;
- button remains reachable by keyboard;
- visible focus state;
- active/pressed state consistent with existing toolbar buttons;
- at least the existing button hit-target size;
- no icon-only behavior that hides the text label.

## 24.7 Icons

Add to the shared icon component:

```text
eye
eyeOff
```

Use conventional accessible line icons.

`Icon.tsx` currently uses a closed `IconName` union and path map; update both.

Do not use an unrelated existing icon merely to avoid adding the correct symbols.

## 24.8 Branch workflow

Implement this as a distinct QOL commit.

Preferred sequence:

```bash
# Preserve all current uncommitted work first.
git status --short

# From a clean main-based topic branch:
git switch main
git pull --ff-only
git switch -c qol/hide-regions

# Implement and test the main-branch feature.
git add web/client/src/components/RegionCanvas.tsx \
        web/client/src/components/Icon.tsx \
        web/client/src/styles.css \
        web/client/scripts
git commit -m "Add canvas region visibility toggle"

# Merge or cherry-pick the isolated commit into the pose branch.
git switch k2lab_pose
git cherry-pick <qol-commit>
```

If the working tree cannot safely switch branches:

- do not reset or discard work;
- first create a protective commit or stash explicitly with the user's work preserved;
- keep the QOL changes logically isolated;
- report the exact commit/patch needed on `main`;
- do not claim the main-branch work is complete until it exists on a main-based branch.

Expected conflict area:

```text
web/client/src/components/RegionCanvas.tsx
```

Resolve it by preserving:

- main behavior;
- pose branch `DrawMode`;
- Draw subject;
- volumetric mannequin rendering;
- all pose interactions;
- the same visibility toggle behavior.

## 24.9 Tests: main branch

Add browser/UI contract coverage proving:

1. **Hide regions** is visible in Generate mode.
2. Initial `aria-pressed` is false.
3. Region boxes and labels are initially rendered.
4. Clicking **Hide regions**:
   - changes label to **Show regions**;
   - sets `aria-pressed` true;
   - removes/hides every region group;
   - makes the overlay noninteractive;
   - does not call `onRegions`;
   - does not call selection mutation;
   - does not change the supplied `regions` data.
5. Clicking **Show regions** restores:
   - all regions;
   - prior selection;
   - resize handles.
6. Edit Reference and Edit targets behave the same.
7. Face mode does not show the button and face overlays still render.
8. Clicking Draw region while hidden:
   - shows overlays;
   - activates drawing;
   - does not create a region until the normal pointer gesture occurs.
9. No project JSON field is added.
10. Save/open round trips are byte-for-byte unaffected except for unrelated existing canonicalization.

## 24.10 Tests: pose branch

In addition to all main tests, prove:

1. a subject region's volumetric mannequin is initially visible;
2. Hide regions removes:
   - subject box;
   - label;
   - head;
   - torso;
   - limbs;
   - articulation handles;
   - resize handles;
3. the underlying pose object remains deeply equal;
4. showing restores the exact mannequin and selected handles;
5. Draw subject while hidden shows the overlays before drawing;
6. hidden state does not affect:
   - rendered backend control map;
   - pose mask bundle;
   - semantic ownership;
   - generation request;
   - Prediction composite;
   - Control-LoRA control image.

## 24.11 No backend/schema work

This QOL update must not require changes to:

- Python project schema;
- worker protocol;
- workspace agent;
- RunPod backend;
- model code;
- PNG generation metadata;
- cloud project format;
- database migrations.

If Codex finds itself changing those systems for this toggle, the implementation scope has drifted.

## 24.12 Completion criteria

This separate QOL workstream is complete only when:

- the main-branch topic commit exists and passes the normal browser tests/build;
- the pose branch includes the same feature;
- a generated image can be viewed directly on the canvas with no region or mannequin graphics over it;
- toggling back restores every region/mannequin without data loss;
- generation results are unchanged;
- project serialization is unchanged;
- Face mode remains functional;
- responsive toolbar behavior is verified.

Include in the final Codex report:

```text
Hide-regions main-based commit:
Hide-regions pose integration commit:
Main browser tests:
Pose browser tests:
Project schema unchanged: yes/no
Region data unchanged in toggle tests: yes/no
Pose data unchanged in toggle tests: yes/no
```
