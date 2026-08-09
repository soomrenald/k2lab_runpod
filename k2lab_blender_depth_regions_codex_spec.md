# Codex Specification: Blender Depth Workflow and Regional Depth Control for K2Lab

## 1. Repository strategy

### Decision

Do **not** create a completely fresh repository.

Implement this as a **separate feature branch of `k2lab_runpod`**, with the depth-control runtime isolated behind a clean internal interface so it does not alter existing behavior unless explicitly enabled.

Recommended branch name:

```text
feature/blender-depth-regions
```

If the shared `k2core` repository or package already exists and is actively used by both desktop and RunPod variants, place reusable depth-control logic in a corresponding `k2core` feature branch and keep only UI, request-schema, and deployment integration in `k2lab_runpod`.

Recommended paired branches in that case:

```text
k2core:        feature/depth-control
k2lab_runpod:  feature/blender-depth-regions
```

The implementation must not duplicate model-loading, latent preparation, regional masking, LoRA loading, scheduling, or inference code that already exists in `k2core` or the current K2Lab runtime.

### Why this approach

A fresh repository would unnecessarily duplicate:

- Krea model loading;
- RunPod deployment;
- region definitions;
- regional prompt compilation;
- regional LoRA containment;
- generation scheduling;
- canvas and image handling;
- job persistence;
- diagnostics;
- error handling.

A direct main-branch modification would create unnecessary regression risk.

The correct strategy is:

```text
existing K2Lab runtime
        |
optional depth-control subsystem
        |
feature flag / request-level enablement
```

The existing non-depth path must remain byte-for-byte or behaviorally unchanged wherever practical.

---

## 2. Primary objective

Add a Blender-authored depth workflow to K2Lab so that:

1. A user can construct a scene in Blender using poseable mannequins and editable everyday objects such as beds, chairs, tables, and custom meshes.
2. Blender can export a 2D depth map representing the camera view.
3. K2Lab can load that depth map or any compatible external depth map.
4. One integrated Krea denoising run uses the full depth map as global spatial conditioning.
5. Existing K2Lab regional prompts, regional LoRAs, ownership masks, and leakage controls continue to function.
6. Regions may independently control how strongly the global depth condition applies within them.
7. Regional depth behavior is implemented as masked weighting or blending of a single global depth condition, not as separate regional image generations.
8. The entire feature remains optional and does not modify existing output when disabled.

---

## 3. Core runtime model

The runtime must perform **one integrated denoising process**.

Do not implement separate per-region image generations and composite them afterward.

Conceptual pipeline:

```text
Blender or external depth map
        |
depth preprocessing
        |
depth-control latent/features
        |
global Krea denoising pass
        |
K2Lab regional prompt conditioning
K2Lab regional LoRA containment
K2Lab ownership/leakage controls
regional depth weighting
        |
final image
```

The full depth condition remains available globally.

Regional depth settings modify the local influence of that condition but do not create independent regional diffusion runs.

---

## 4. Initial model integration target

Use the existing public Krea 2 depth Control-LoRA implementation as the initial compatibility target.

The implementation must support configuration for:

- depth checkpoint path;
- adapter configuration;
- expanded input projection;
- control strength;
- Raw versus Turbo inference;
- depth preprocessing convention;
- VAE encoding of the control image;
- optional quantized base-model loading only if already supported safely.

Do not retrain a depth model during this project phase.

The first goal is to validate and integrate an existing working depth checkpoint.

---

## 5. Hard non-regression requirements

The implementation must explicitly preserve all current K2Lab functionality.

When depth control is disabled:

- existing prompts must compile identically;
- existing regional generation must behave identically;
- existing regional LoRA gating must behave identically;
- existing pose-branch or other experimental code must not be altered;
- model-loading behavior must remain unchanged;
- job schemas must remain backward-compatible;
- old saved jobs and configurations must continue to load;
- inference speed and VRAM use must not materially change;
- no depth model or checkpoint may load;
- no additional transformer forwards may occur;
- tests for current functionality must continue to pass.

Depth support must be opt-in through an explicit setting such as:

```json
{
  "depth_control": {
    "enabled": true
  }
}
```

Default:

```json
{
  "depth_control": {
    "enabled": false
  }
}
```

---

## 6. Feature flags and staged activation

Add feature flags:

```text
K2_DEPTH_CONTROL_ENABLED
K2_DEPTH_REGIONS_ENABLED
K2_DEPTH_OVERRIDE_ENABLED
K2_BLENDER_BUNDLE_IMPORT_ENABLED
```

Defaults:

```text
false
```

The flags allow development and testing without exposing incomplete behavior to normal users.

The ordinary depth-map upload path may be enabled before Blender bundle import.

---

## 7. Depth-region modes

Do not use only a binary depth/non-depth setting.

Each K2Lab region should support one of the following modes:

### 7.1 `inherit`

The region inherits the globally configured depth condition and strength.

This should be the default when global depth is enabled.

### 7.2 `emphasize`

The region uses the same global depth map but increases local depth influence.

Example:

```json
{
  "depth_mode": "emphasize",
  "depth_strength": 1.5
}
```

### 7.3 `relax`

The region reduces local depth influence while retaining continuity with the global scene.

Example:

```json
{
  "depth_mode": "relax",
  "depth_strength": 0.35
}
```

### 7.4 `ignore`

The region suppresses depth influence as much as safely possible.

This is useful for:

- sky;
- fog;
- abstract effects;
- text;
- regions intended to diverge from Blender geometry.

It must still feather into neighboring depth-controlled areas.

### 7.5 `override`

Optional later phase.

The region may supply a separate local depth map that is blended over the global depth map through the region mask.

Do not implement `override` in the first MVP unless the earlier modes are validated.

---

## 8. Effective depth weighting

For every image token, compute an effective depth influence from:

- global depth strength;
- region masks;
- per-region depth modes;
- per-region depth strengths;
- mask feathering;
- region overlap policy.

Conceptual form:

```text
effective_depth_strength(x, y)
    = global_depth_strength
      × regional_depth_multiplier(x, y)
```

For override mode:

```text
effective_depth(x, y)
    = blend(
        global_depth(x, y),
        local_override_depth(x, y),
        feathered_region_mask(x, y)
      )
```

Do not introduce hard rectangular cutoffs.

All region masks must be mapped consistently from canvas pixel space to:

- control-image space;
- VAE latent space;
- Krea image-token space.

Mask transforms must be deterministic and unit tested.

---

## 9. Overlapping-region policy

Depth-region overlap must be deterministic.

Recommended initial policy:

1. Regions retain existing K2Lab z-order or explicit priority.
2. For `emphasize` and `relax`, combine multipliers through weighted blending rather than simple addition.
3. Clamp final local depth strength to configured safe bounds.
4. `ignore` must not silently erase a higher-priority explicit override.
5. Invalid combinations must raise a clear validation error or warning.

Suggested bounds:

```text
minimum effective depth strength: 0.0
maximum effective depth strength: 3.0
```

Make bounds configurable.

---

## 10. Depth input support

Support these input formats:

### Required MVP formats

- PNG, 8-bit grayscale;
- PNG, 16-bit grayscale;
- TIFF, 16-bit grayscale if existing image libraries support it reliably.

### Optional later formats

- OpenEXR single-channel depth;
- Blender multi-layer EXR;
- packed Blender export bundle.

The loader must:

- preserve source bit depth where practical;
- avoid color-management transformations;
- reject unsupported alpha/color encodings clearly;
- record width, height, dtype, minimum, maximum, and normalization method;
- produce a visible preview;
- never silently reinterpret RGB artwork as depth without warning.

---

## 11. Depth convention and calibration

The public checkpoint may expect a particular depth convention.

Implement configurable preprocessing:

```yaml
depth:
  invert: false
  normalization: percentile
  percentile_near: 1.0
  percentile_far: 99.0
  gamma: 1.0
  clamp: true
  invalid_value_policy: far
```

Supported normalization modes:

- `none`
- `minmax`
- `percentile`
- `camera_range`
- `checkpoint_reference`

Support both:

- near = white, far = black;
- near = black, far = white.

The system must never guess silently.

A calibration command must generate comparison grids across:

- inversion;
- min/max versus percentile normalization;
- gamma values;
- depth strengths;
- Raw versus Turbo.

Example:

```bash
python -m k2lab.depth.calibrate \
  --image test_target.png \
  --depth blender_depth_16bit.png \
  --output reports/depth-calibration/
```

The calibration report must recommend the best preprocessing based on measured depth adherence and visual inspection.

---

## 12. Blender authoring workflow

### 12.1 Blender template

Create a versioned Blender template containing:

- one calibrated camera;
- configurable render dimensions;
- metric scene scale;
- near/far depth settings;
- one or more poseable mannequins;
- basic editable objects:
  - bed;
  - chair;
  - table;
  - sofa;
  - floor;
  - wall;
- collections for:
  - characters;
  - furniture;
  - environment;
  - lights;
  - cameras;
- object names suitable for export metadata.

Do not require final photorealistic materials.

Geometry matters more than appearance.

### 12.2 Poseable mannequins

The initial mannequin should support:

- full-body posing;
- selectable body proportions if practical;
- root translation;
- limb rotation;
- sitting, reclining, standing, crouching, and leaning;
- visible geometry sufficient to generate useful depth silhouettes.

Do not make mannequin sophistication a blocker for depth integration.

Use an existing permissively licensed mannequin or create a simple rigged model.

Record license and source.

### 12.3 Blender export script

Create a Blender Python add-on or command-line script that exports:

- high-precision depth map;
- 8-bit preview depth map;
- object-ID segmentation image;
- per-object masks where practical;
- camera metadata;
- render resolution;
- near/far clipping;
- object transforms;
- object names and IDs;
- scene scale;
- Blender version;
- template version;
- export timestamp;
- checksums.

Recommended bundle:

```text
scene_export/
├── depth_16bit.png
├── depth_preview.png
├── object_ids.png
├── masks/
│   ├── mannequin_001.png
│   ├── chair_001.png
│   └── bed_001.png
├── camera.json
├── objects.json
├── export.json
└── scene.blend
```

K2Lab must initially require only the depth image.

Masks and metadata are future-compatible optional inputs.

### 12.4 Camera consistency

The exporter must ensure:

- depth output uses the exact active camera;
- render aspect ratio matches K2Lab generation dimensions;
- camera clipping planes are recorded;
- depth normalization is deterministic;
- image origin and axis conventions are documented;
- depth image is not vertically flipped relative to K2Lab canvas space.

Add a calibration checkerboard or asymmetric test scene to verify orientation.

---

## 13. Suggested repository structure

If no shared `k2core` package is available:

```text
k2lab_runpod/
├── k2lab/
│   ├── depth/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── checkpoint.py
│   │   ├── control_lora.py
│   │   ├── masks.py
│   │   ├── regional.py
│   │   ├── calibration.py
│   │   ├── diagnostics.py
│   │   └── types.py
│   └── ...
├── blender/
│   ├── template/
│   ├── addon/
│   ├── export_scene.py
│   └── README.md
├── tests/
│   ├── depth/
│   └── ...
├── docs/
│   ├── DEPTH_CONTROL.md
│   ├── BLENDER_DEPTH_WORKFLOW.md
│   └── DEPTH_REGIONS.md
└── reports/
```

If `k2core` exists:

```text
k2core/
└── k2core/depth/
    ├── config.py
    ├── loader.py
    ├── preprocess.py
    ├── checkpoint.py
    ├── control_lora.py
    ├── masks.py
    ├── regional.py
    └── diagnostics.py
```

and:

```text
k2lab_runpod/
├── web/API integration
├── request schemas
├── upload handling
├── visual preview
├── deployment
└── Blender helper assets
```

Do not maintain two separate implementations.

---

## 14. Request schema

Add a backward-compatible schema extension.

Example:

```json
{
  "depth_control": {
    "enabled": true,
    "checkpoint": "Patil/Krea-2-depth-controlnet",
    "depth_image": "uploads/depth_16bit.png",
    "global_strength": 1.0,
    "start_percent": 0.0,
    "end_percent": 1.0,
    "invert": false,
    "normalization": {
      "mode": "percentile",
      "near_percentile": 1.0,
      "far_percentile": 99.0,
      "gamma": 1.0
    },
    "feather_pixels": 32
  },
  "regions": [
    {
      "id": "person-a",
      "prompt": "a woman in a red coat",
      "depth_mode": "emphasize",
      "depth_strength": 1.5
    },
    {
      "id": "sky",
      "prompt": "dramatic clouds",
      "depth_mode": "relax",
      "depth_strength": 0.25
    }
  ]
}
```

Old requests with no `depth_control` field must continue to work.

---

## 15. Model-loading design

Depth checkpoint loading must be isolated from normal Krea loading.

Required behavior:

```text
depth disabled:
    load normal Krea path only

depth enabled:
    load normal Krea
    patch or wrap input projection safely
    load depth Control-LoRA
    verify checkpoint tensor shapes
    verify base-model compatibility
```

When the depth checkpoint is unloaded or a normal job follows:

- restore or recreate the clean normal model state;
- do not leave modified input projections or hooks attached;
- do not leak depth adapter state into subsequent jobs.

Prefer immutable model variants or explicit cached model instances over repeatedly mutating one shared model in unsafe ways.

---

## 16. Interaction with ordinary LoRAs

Validate:

- global LoRA plus global depth;
- regional LoRA plus global depth;
- multiple regional LoRAs plus depth;
- Turbo LoRA plus depth;
- depth adapter plus character/style LoRAs.

The system must track:

- load order;
- merge order;
- runtime hook order;
- dtype;
- scale;
- adapter naming;
- conflicting target modules.

No LoRA may silently overwrite depth-control tensors.

---

## 17. Interaction with K2Lab regional inference

Preserve the current regional algorithm.

Depth adds one more spatially weighted condition.

At each relevant denoising step:

1. Prepare global prompt conditioning.
2. Prepare regional prompt conditioning.
3. Apply regional LoRA containment.
4. Prepare global depth-control input.
5. Compute effective regional depth weighting.
6. Apply depth control through the compatible depth-control pathway.
7. Apply ownership and leakage controls according to current K2Lab behavior.
8. Continue the normal denoising schedule.

The exact ordering must be measured and documented.

Implement an ablation mode allowing:

- depth only;
- regions only;
- depth plus regions;
- depth plus regional LoRAs;
- depth plus regional weighting.

---

## 18. Timestep scheduling

Support:

```text
global depth start percentage
global depth end percentage
per-region depth start percentage
per-region depth end percentage
```

Initial defaults:

```text
start: 0.0
end: 1.0
```

Later testing should determine whether depth should be strongest during early composition steps and reduced later.

Do not assume this without calibration.

---

## 19. Diagnostics

Add diagnostics for:

- source depth histogram;
- normalized depth histogram;
- decoded control latent preview;
- effective depth-strength mask;
- per-region depth multipliers;
- control residual RMS or equivalent;
- depth/no-depth prediction difference;
- correct/shuffled/inverted depth response;
- model-forward count;
- adapter load state;
- memory usage;
- runtime overhead.

Save diagnostics in each job report when debug mode is enabled.

---

## 20. Evaluation suite

Create a fixed evaluation suite before regional integration.

### 20.1 Global depth tests

Include Blender scenes with:

- one standing mannequin;
- two mannequins at different depths;
- person seated on chair;
- person lying on bed;
- table in foreground;
- chair behind person;
- overlapping objects;
- strong perspective;
- wide and portrait aspect ratios;
- sparse scene;
- dense room.

For each case generate:

- depth disabled;
- correct depth;
- shuffled depth;
- inverted depth;
- horizontally shifted depth;
- blank depth;
- several depth strengths.

### 20.2 Metrics

Measure:

- correlation between input depth and estimated output depth;
- rank-order depth consistency;
- edge alignment;
- silhouette overlap where applicable;
- object center and scale;
- blank/corrupt rate;
- prompt similarity;
- image quality;
- runtime;
- VRAM.

Do not use depth correlation alone.

### 20.3 Regional tests

For each scene compare:

- global depth only;
- normal K2Lab regions only;
- combined global depth plus regions;
- depth-emphasis region;
- depth-relaxed region;
- depth-ignore region;
- overlapping depth regions;
- regional character LoRA containment;
- subject interacting with furniture.

Required visual checks:

- global perspective preserved;
- regional identity remains contained;
- emphasized object follows depth more strongly;
- relaxed region may diverge without breaking the entire scene;
- no rectangular seams;
- no region-edge depth discontinuities;
- no cross-region LoRA leakage regression.

---

## 21. Approval gates

### Gate A: branch and architecture review

Before modifying inference code, Codex must report:

- current repository structure;
- whether `k2core` exists and is used;
- current model-loading path;
- current regional inference path;
- current LoRA hook path;
- proposed files and interfaces;
- non-regression plan.

### Gate B: community checkpoint validation

Before K2Lab integration:

- run the checkpoint in a standalone harness;
- verify correct depth beats zero and shuffled depth;
- verify Raw and Turbo behavior;
- determine expected preprocessing convention;
- report VRAM and runtime.

Stop if the checkpoint cannot be independently validated.

### Gate C: global K2Lab integration

Require:

- depth-enabled global generation works;
- depth-disabled generation is unchanged;
- ordinary LoRAs still work;
- no stale model mutation;
- tests pass.

### Gate D: regional depth weighting

Require:

- soft regional weighting works;
- no hard seams;
- global depth continuity remains;
- ordinary regional behavior is preserved.

### Gate E: Blender workflow

Require:

- deterministic export;
- camera alignment;
- 16-bit depth preservation;
- K2Lab import;
- reproducible reference scene.

### Gate F: merge readiness

Before merging:

- full test suite;
- benchmark report;
- migration notes;
- documentation;
- feature flags;
- rollback plan;
- explicit user approval.

---

## 22. Development phases

### Phase 0: repository audit

Codex must inspect the actual repositories before coding.

Determine:

- whether reusable runtime logic belongs in `k2core`;
- whether `k2lab_runpod` currently duplicates desktop logic;
- which branch should be based on;
- whether the pose branch contains useful generic control infrastructure;
- which code can be reused safely;
- which pose-specific code must not be reused.

Do not merge the failed pose architecture into depth work merely because both use control inputs.

### Phase 1: standalone depth validation harness

Build a CLI-only test harness outside the web request path.

Required command:

```bash
python -m k2lab.depth.validate \
  --base-model /models/krea2 \
  --depth-checkpoint /models/krea2-depth \
  --prompt "..." \
  --depth-image depth.png \
  --output reports/depth-validation/
```

No K2Lab regions yet.

### Phase 2: optional global depth path

Integrate global depth into K2Lab behind feature flag.

Depth disabled must remain unchanged.

### Phase 3: normal regions plus global depth

Run existing K2Lab regions while global depth is active.

Do not add regional weighting until this combination is stable.

### Phase 4: regional depth modes

Add `inherit`, `emphasize`, `relax`, and `ignore`.

Use feathered masks.

### Phase 5: Blender exporter and template

Build the Blender authoring assets and import workflow.

### Phase 6: optional override depth

Only after prior phases pass.

Allow a region-specific local depth image blended into global depth.

### Phase 7: production hardening

Add:

- persistence;
- job schema migration;
- diagnostics;
- documentation;
- deployment changes;
- recovery;
- model caching;
- memory controls;
- timeout handling.

---

## 23. Testing requirements

### Unit tests

- depth normalization;
- inversion;
- 8-bit and 16-bit loading;
- mask resizing;
- feathering;
- regional strength composition;
- overlap policy;
- schema defaults;
- old-job compatibility;
- zero-depth path;
- adapter tensor validation;
- Blender metadata parsing.

### Integration tests

- normal generation without depth;
- global depth generation;
- depth plus global LoRA;
- depth plus regional LoRA;
- multiple regions;
- region modes;
- sequential depth and non-depth jobs;
- model unload/reload;
- Raw;
- Turbo where supported;
- job cancellation;
- RunPod timeout handling.

### Regression tests

Capture fixed-seed outputs or latent fingerprints for the current non-depth path.

Depth-disabled behavior must remain within a documented numerical tolerance.

---

## 24. Performance requirements

Measure:

- model load time;
- added VRAM;
- generation runtime;
- control preprocessing time;
- VAE encoding time;
- effect of regional depth weighting;
- effect of multiple regions;
- model-cache behavior.

The depth path should encode the global depth map once per job, not once per region or denoising step unless the checkpoint architecture makes that unavoidable.

Reuse cached transformed masks.

---

## 25. Error handling

Return explicit errors for:

- missing depth checkpoint;
- incompatible checkpoint tensor shapes;
- unsupported image type;
- invalid depth range;
- fully constant depth image;
- NaN/Inf depth values;
- depth dimensions incompatible with output;
- invalid region depth mode;
- invalid overlap configuration;
- missing override image;
- failed Blender bundle parsing;
- unsupported Raw/Turbo combination.

Do not silently fall back to normal generation when the user explicitly enabled depth.

---

## 26. Documentation

Create:

```text
docs/DEPTH_CONTROL.md
docs/DEPTH_REGIONS.md
docs/BLENDER_DEPTH_WORKFLOW.md
docs/DEPTH_CALIBRATION.md
docs/DEPTH_TROUBLESHOOTING.md
```

Document:

- supported checkpoints;
- checkpoint licenses;
- depth conventions;
- Blender export steps;
- region modes;
- expected limitations;
- Raw/Turbo differences;
- VRAM/runtime overhead;
- interaction with LoRAs;
- feature flags;
- rollback procedure.

---

## 27. Expected limitations

State clearly:

- depth controls coarse geometry, silhouette, scale, and occlusion better than exact anatomical articulation;
- exact hands and fingers are not guaranteed;
- a mannequin depth silhouette may produce a person with similar volume but not exact joint angles;
- conflicting prompts and depth can reduce quality;
- strong LoRAs can fight depth;
- regional depth weighting can cause artifacts if masks are too hard;
- separate regions cannot imply mutually inconsistent camera geometry without degradation;
- Blender depth convention may require calibration for the chosen checkpoint.

---

## 28. Merge strategy

Do not merge directly into `main`.

Use:

```text
feature/blender-depth-regions
```

Recommended process:

1. Branch from the current stable K2Lab RunPod main branch.
2. Keep commits small and phase-specific.
3. Rebase or merge stable-main changes regularly.
4. Do not modify unrelated regional or UI code.
5. Add feature flags before runtime code.
6. Require all existing tests plus new depth tests.
7. Produce a release-candidate deployment.
8. Run a fixed regression suite.
9. Obtain explicit approval.
10. Merge only after all gates pass.

If `k2core` is active:

1. Implement depth primitives in `k2core`.
2. Tag or pin the tested `k2core` commit.
3. Update `k2lab_runpod` to consume that revision.
4. Avoid copying the implementation into both repositories.

---

## 29. Definition of done

The feature is complete when:

- K2Lab can load an external or Blender-generated depth map;
- the full depth map controls one global Krea denoising pass;
- normal K2Lab regions continue to work;
- regions can inherit, emphasize, relax, or ignore depth with feathered transitions;
- regional LoRA containment still works;
- depth-disabled output remains unchanged;
- sequential depth and non-depth jobs do not leak model state;
- Blender exports deterministic aligned depth maps;
- Raw support is validated;
- Turbo support is either validated or explicitly marked unsupported;
- diagnostics and calibration reports exist;
- all current and new tests pass;
- feature flags allow immediate rollback;
- merge requires explicit approval.

---

## 30. Immediate Codex instructions

Begin with a repository audit only.

Do not write integration code until Codex reports:

1. whether `k2core` exists and is the correct shared implementation location;
2. the exact existing Krea model-loading path;
3. the exact existing regional-conditioning path;
4. the exact existing regional LoRA containment path;
5. reusable components from prior control work;
6. risks of stale model mutation;
7. proposed branch topology;
8. exact files to add or modify;
9. expected test coverage;
10. standalone checkpoint-validation plan.

After the audit, stop for approval at Gate A.

Do not modify `main`, deploy a production endpoint, or alter the current default generation path before approval.
