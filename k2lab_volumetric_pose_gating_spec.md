# K2Lab Volumetric Mannequin Pose Gating

## End-to-end implementation specification for Codex 5.6 Sol High

**Repository:** `https://github.com/soomrenald/k2lab_runpod`  
**Starting branch:** `k2lab_pose`  
**Baseline inspected:** July 26, 2026  
**Target milestone:** the first version that can be exercised end-to-end through the real RunPod web GUI with Krea 2 generation  
**Project document target:** version 21  
**Pose format target:** `k2-volumetric-pose-v1`

---

## 0. Directive to Codex

Implement this specification end to end. Do not stop after producing a plan, partial backend, UI mockup, or isolated proof of concept.

Before changing code:

1. Check out `k2lab_pose`.
2. Record the exact starting commit with `git rev-parse HEAD`.
3. Run the existing Python and browser tests/builds.
4. Inspect the live branch before applying this specification; the branch is the source of truth for exact symbols and signatures.
5. Preserve all currently functional generation, regional prompting, regional LoRA, project, agent, RunPod, image-edit, face-refinement, and asset behavior except where this specification explicitly replaces the broken Qwen/InstantX pose path.

When implementation is complete:

1. Run the complete Python test suite.
2. Run the browser project/UI contract tests, typecheck, and production build.
3. Update the relevant documentation and version/capability declarations.
4. Leave the branch at the point where a new workspace image can be built and the feature tested from the RunPod web GUI.
5. Report:
   - starting and ending commit;
   - files changed;
   - migrations performed;
   - tests and builds run;
   - exact commands needed to build/publish the workspace image and launch it;
   - any test that could not be run without a real GPU;
   - known limitations.
6. Do not claim that mannequin adherence is visually successful until it has actually been tested on Krea 2 through RunPod. The implementation milestone is a correct, observable, end-to-end experimental system.

Do not:

- train or download a pose ControlNet or Control-LoRA;
- retain the InstantX/Qwen-Image ControlNet as a Krea pose option;
- implement image-to-image, background-first generation, a second denoising pass, re-noising, or a VAE round trip;
- replace the existing regional system with a second independent implementation;
- silently swallow sampler, mask, sigma, or worker errors;
- ask for ordinary implementation choices that this specification resolves.

Where a branch change makes a named file or symbol differ, preserve the required behavior and document the adjusted implementation.

---

## 1. Product objective

Add coarse, strongly constrained subject placement and posing to the existing K2Lab regional prompt/LoRA generation path by:

1. replacing the current 18-point ball-and-stick editor with a filled editable 2D-volumetric mannequin;
2. deriving per-subject occupancy and ownership masks from that mannequin;
3. using the union of all enabled mannequin support masks as a strict, time-varying denoising gate during early sampler steps;
4. using each subject's exclusive mannequin ownership mask to bind the existing regional prompt and regional LoRA to that subject during the gated phases;
5. gradually releasing the spatial restriction;
6. continuing with the existing unrestricted regional pipeline for the user's normal step count;
7. allowing the user to control both:
   - the number of model evaluations in each phase; and
   - how the sampler's sigma trajectory is distributed among those phases.

The intended control level is:

- subject location in the overall canvas;
- relative placement of multiple subjects;
- head volume and position;
- torso position and coarse orientation;
- upper/lower arm placement;
- upper/lower leg placement;
- broad stance and body footprint.

The first version does **not** promise:

- exact joint coordinates in the final image;
- hand or finger pose;
- facial expression or precise head facing;
- guaranteed front/back ordering for crossed limbs;
- mathematically immutable final silhouettes after the gate is released;
- production-quality pose obedience without empirical tuning.

Head direction, gaze, limb-overlap semantics, and finer pose interpretation remain prompt responsibilities.

---

## 2. Current branch state and defect to replace

The inspected `k2lab_pose` branch currently contains:

- version-20 project documents;
- subject regions with normalized 18-joint OpenPose-style poses;
- SVG ball-and-stick editing;
- standing, squatting, and mirror controls;
- a composed full-canvas OpenPose image;
- a generic ComfyUI ControlNet load/apply path;
- a Qwen/InstantX ControlNet selector and job asset;
- the already-functional regional prompt and regional LoRA implementation.

The current pose runtime is invalid for Krea 2 because it attempts to attach a Qwen-Image architecture ControlNet to Krea 2. The model can load far enough to appear attached but fails at sampling. The cleanup event can then look like a successful release, while the actual error is categorized as a generic generation/LoRA failure.

Replace that path. Do not patch around it.

Required removals from active generation:

- Qwen/InstantX pose-model selection;
- `pose_controlnet_file_id` as a requirement for generation;
- generic `load_controlnet` / `ControlNetApplyAdvanced` use for Krea pose control;
- pose-control strength/start/end settings that only apply to the removed path;
- success-like cleanup messaging after a failed sample;
- string-based misclassification of pose failures as LoRA compatibility failures.

`FileKind.CONTROLNET_MODELS` may remain for inventory/backward compatibility if it is used elsewhere or removing it would strand existing files. It must no longer be required or selectable for this pose feature.

---

## 3. Required generation model: one continuous text-to-image sampler call

Generation must remain one continuous text-to-image trajectory:

```text
initial noise
    ↓
hard-gated transitions
    ↓
soft-gated transitions
    ↓
normal unrestricted regional transitions
    ↓
decode/output
```

There must be:

- one initial noise tensor;
- one monotonically decreasing sigma array;
- one invocation of the existing ComfyUI sampling entry point;
- no decode/re-encode boundary;
- no second sampler run;
- no background-first pass;
- no image-to-image mode switch;
- no re-noising restart.

### 3.1 Step accounting

The existing `steps` field retains its name and becomes explicitly the number of **normal unrestricted steps**.

Add:

- `pose_hard_gate_steps`;
- `pose_soft_gate_steps`.

When pose gating is enabled:

```text
effective_steps =
    pose_hard_gate_steps
  + pose_soft_gate_steps
  + steps
```

Example:

```text
Hard:   2
Soft:   2
Normal: 8
Total: 12 sampler transitions
```

When pose gating is disabled:

```text
effective_steps = steps
```

The disabled path must preserve the previous generation behavior and must not install the dynamic mask hook.

### 3.2 Phase ownership

For zero-based sampler transition `i`:

```text
0 <= i < H             → hard phase
H <= i < H + S         → soft phase
H + S <= i < H + S + R → normal phase
```

Where:

- `H = pose_hard_gate_steps`;
- `S = pose_soft_gate_steps`;
- `R = steps`;
- `N = H + S + R`.

The sampler callback must identify and report the phase for every completed transition.

---

## 4. Denoising-gate behavior

### 4.1 Mask definitions

For each enabled subject `r`, generate:

- `C_r`: binary core mannequin mask;
- `S_r`: dilated and feathered support mask in `[0, 1]`;
- `O_r`: exclusive ownership mask for that subject, derived from `S_r` and current front-to-back region priority.

Define the global support mask:

```text
U = max_r(S_r)
```

`U` is the union used to control whether a latent location may accept denoising updates.

### 4.2 Gate-strength sequence

Define a scalar gate strength `g_i` for each transition:

```text
g_i = 1.0                    during hard transitions
0.0 < g_i < 1.0              during soft transitions
g_i = 0.0                    during normal transitions
```

The effective denoise mask for transition `i` is:

```text
D_i = 1 - g_i * (1 - U)
```

Equivalent interpretation:

```text
D_i = lerp(all_ones, U, g_i)
```

Therefore:

- hard phase: `D_i = U`;
- normal phase: `D_i = 1`;
- soft phase: the denoisable area and edge strength gradually open from `U` to the whole canvas.

The core of each mannequin remains fully denoisable during hard gating. Feathered support edges are partially denoisable. The rest of the canvas accepts no predicted denoising update during hard gating.

### 4.3 Scheduler-consistent state outside the gate

Do not freeze arbitrary stale latent values outside the mask.

Use ComfyUI's masked-sampling/inpainting mechanism so that latent locations outside `D_i` remain on the model sampler's scheduler-consistent noisy trajectory for the current sigma. When the gate opens, those locations must enter normal generation at the correct current noise level.

The current ComfyUI sampler supports this through:

- the `noise_mask` argument;
- `KSamplerX0Inpaint`;
- `model_options["denoise_mask_function"]`.

Install a dynamic `denoise_mask_function` for the single sampling call. It must return `D_i` at each model evaluation.

Use the project's existing empty latent as the latent reference; do not introduce a source image.

### 4.4 Hook lifecycle

Before sampling:

1. build and validate the mask bundle;
2. convert the global support mask to the same full-resolution tensor convention already used by the project's image-edit mask path;
3. snapshot any existing `denoise_mask_function`;
4. install the pose-gating callback;
5. bind the same `PoseGateController` to the regional prompt and LoRA runtime.

After sampling, whether it succeeds or fails:

1. restore the previous model option exactly;
2. detach the controller from all regional runtime objects;
3. release temporary mask tensors;
4. do not emit a success-like cleanup event from `finally`.

The ordinary generate path currently should not have another dynamic denoise-mask hook. If one is unexpectedly present, do not silently overwrite it. Either compose it through a tested combination rule or raise a typed compatibility error. For the first implementation, an explicit typed failure is preferable to an untested composition.

### 4.5 Controller timing

Use transition index as the source of truth for phase timing.

Recommended controller lifecycle:

```python
controller = PoseGateController(
    hard_steps=H,
    soft_steps=S,
    normal_steps=R,
    release_schedule=...,
)
controller.current_transition = 0
```

During all model evaluations belonging to a sampler transition, return the same gate strength. In the sampler's once-per-transition callback:

```python
controller.mark_transition_complete(step_index)
```

This avoids changing the mask between predictor/corrector evaluations in samplers that call the model more than once per transition.

Record the current sigma for diagnostics, but do not infer phase solely from floating-point sigma equality.

For the first real acceptance tests, recommend Euler because Krea's official open implementation uses Euler flow integration and because its one-evaluation-per-transition behavior makes phase analysis easiest. Other existing samplers remain selectable but custom gating/sigma behavior is experimental until tested.

---

## 5. Soft-release schedules

Add the following release schedules:

- `cosine` — default;
- `linear`;
- `exponential`;
- `stepped`.

For `S` soft transitions and zero-based soft index `k`:

```text
t = (k + 1) / (S + 1)
```

This deliberately excludes both endpoints:

- hard phase owns the exact `1.0` endpoint;
- normal phase owns the exact `0.0` endpoint.

Define:

### Cosine

```text
g = 0.5 * (1 + cos(pi * t))
```

### Linear

```text
g = 1 - t
```

### Exponential

Use a fixed exponent constant `a = 4`:

```text
g = 1 - (exp(a * t) - 1) / (exp(a) - 1)
```

This retains a strong restriction for more of the soft phase and releases rapidly near its end.

### Stepped

Use deterministic plateaus:

```text
t < 1/3       → 0.75
1/3 <= t < 2/3 → 0.50
t >= 2/3      → 0.25
```

For `S == 0`, return no soft values.

Create one pure function that produces the exact `N`-element gate-strength vector. Use the same function for:

- runtime;
- tests;
- GUI preview;
- metadata.

If duplicating this calculation in TypeScript for immediate preview, add shared golden fixtures so Python and TypeScript cannot drift.

---

## 6. User-modifiable sigma scheduling

Phase step count, sigma allocation, and gate release are different controls and must remain independent:

1. **Step counts:** number of model evaluations in hard, soft, and normal phases.
2. **Sigma allocation:** how much of the scheduler trajectory those evaluations span.
3. **Soft release:** how the spatial gate falls from strong to open during soft evaluations.

### 6.1 Terminology

Expose normalized trajectory progress:

```text
p = 0.0 → initial/high-noise endpoint
p = 1.0 → final/zero-sigma endpoint
```

Do not ask ordinary users to type raw model sigma values. Raw sigmas depend on model, sampler, scheduler, and current ComfyUI behavior.

The worker resolves normalized progress into exact sigmas and records both.

### 6.2 Sigma schedule modes

Add:

- `automatic`;
- `phase_weighted`;
- `advanced`.

#### Automatic

Use the selected ComfyUI sampler/scheduler's baseline sigma tensor for `effective_steps` without warping.

This must be the default and the parity baseline.

#### Phase weighted

Expose:

- hard trajectory share;
- soft trajectory share;
- normal trajectory share, calculated as the remainder.

The values define normalized progress at phase boundaries:

```text
p_0       = 0
p_H       = hard_share
p_H+S     = hard_share + soft_share
p_N       = 1
```

Within each non-empty phase, distribute normalized progress linearly among that phase's transitions.

Example:

```text
H = 2, S = 2, R = 8
hard_share = 0.20
soft_share = 0.35
normal_share = 0.45
```

The first four gated transitions then span 55% of the baseline scheduler trajectory, while the eight normal transitions span the remaining 45%.

Call these **trajectory shares**, not guaranteed percentages of semantic image formation.

Provide convenience presets:

| Preset | Hard | Soft | Normal |
|---|---:|---:|---:|
| Balanced | 20% | 30% | 50% |
| Pose lock | 25% | 40% | 35% |
| Gentle | 15% | 25% | 60% |

`Scheduler default` switches to `automatic`; it is not a weighted preset.

If a phase has zero steps:

- force its share to zero;
- disable its share control;
- reject a submitted nonzero share for it;
- leave normal share as the calculated remainder.

#### Advanced

Store one editable normalized progress knot for each sigma endpoint:

```text
N transitions → N + 1 knots
```

Requirements:

- first knot locked to `0.0`;
- last knot locked to `1.0`;
- all intermediate knots finite and strictly increasing;
- no duplicate knots;
- one knot for every phase boundary and transition boundary;
- editing available both through an SVG graph and exact numeric fields;
- phase background bands visible in the graph;
- soft-gate-strength curve visible as an overlay;
- the graph clearly distinguishes normalized trajectory progress from gate strength.

When the effective step count changes in advanced mode, resample the prior monotone knot curve to the new `N + 1` length. Re-lock endpoints and validate monotonicity. Do not silently truncate the array.

### 6.3 Resolving exact sigmas

Let the selected ComfyUI sampler/scheduler create its baseline sigma tensor for `N` effective steps. Use the same code path and sampler-specific adjustments that normal sampling uses; do not reimplement scheduler formulas.

The resolved baseline must contain `N + 1` values.

For automatic mode:

```text
resolved_sigmas = baseline_sigmas
```

For phase-weighted and advanced modes:

1. construct normalized positions `p_i`;
2. define baseline index positions:

   ```text
   u_j = j / N
   ```

3. evaluate the monotone baseline sigma curve at each `p_i` using piecewise-linear interpolation;
4. force:
   - first sigma to exact baseline first sigma;
   - final sigma to exact zero;
5. preserve device/dtype requirements expected by ComfyUI.

Conceptually:

```text
sigma_i = interpolate(
    x = p_i,
    xp = [0/N, 1/N, ..., N/N],
    fp = baseline_sigmas,
)
```

Do not linearly invent raw sigma values between the model's absolute endpoints without reference to the selected scheduler.

### 6.4 Validation

Before sampling, validate:

- `N >= 1`;
- exact length `N + 1`;
- all positions and sigmas finite;
- normalized positions begin at `0.0` and end at `1.0`;
- normalized positions strictly increase;
- sigmas monotonically and strictly decrease except for the required final zero behavior allowed by the baseline;
- first sigma equals the baseline start;
- final sigma equals exactly zero;
- all sigmas remain within baseline bounds;
- no phase with zero steps has nonzero phase share;
- hard and soft shares are each in `[0, 1]`;
- `hard_share + soft_share < 1` when normal steps are nonzero;
- no interval is too small to survive the selected dtype;
- no phase is assigned zero trajectory span when it has nonzero steps.

If the selected ComfyUI baseline itself contains a sampler-specific exceptional pattern, preserve its valid behavior and test against that sampler rather than applying a generic destructive normalization.

Pass the resulting tensor explicitly to the existing call:

```python
comfy.sample.sample(
    ...,
    steps=effective_steps,
    sigmas=resolved_sigmas,
)
```

### 6.5 Turbo warning

Krea 2 Turbo is officially intended for eight-step inference with CFG disabled. This application intentionally allows:

```text
hard + soft + normal > 8
```

and custom sigma placement. Treat this as experimental.

The GUI should display a concise non-blocking warning when:

- Krea 2 Turbo is selected; and
- effective steps differ from eight, or sigma mode is not automatic.

Do not prohibit the experiment.

---

## 7. Replace the 18-joint mannequin with `k2-volumetric-pose-v1`

### 7.1 Canonical state

Replace the current visible 18-joint state with:

```json
{
  "enabled": true,
  "format": "k2-volumetric-pose-v1",
  "joints": {
    "neck": {"x": 0.50, "y": 0.20},
    "left_shoulder": {"x": 0.39, "y": 0.24},
    "right_shoulder": {"x": 0.61, "y": 0.24},
    "left_elbow": {"x": 0.34, "y": 0.42},
    "right_elbow": {"x": 0.66, "y": 0.42},
    "left_wrist": {"x": 0.31, "y": 0.60},
    "right_wrist": {"x": 0.69, "y": 0.60},
    "left_hip": {"x": 0.44, "y": 0.52},
    "right_hip": {"x": 0.56, "y": 0.52},
    "left_knee": {"x": 0.43, "y": 0.72},
    "right_knee": {"x": 0.57, "y": 0.72},
    "left_ankle": {"x": 0.42, "y": 0.94},
    "right_ankle": {"x": 0.58, "y": 0.94}
  },
  "head": {
    "cx": 0.50,
    "cy": 0.105,
    "rx": 0.075,
    "ry": 0.105
  }
}
```

The example values are defaults, not migration assumptions. Calibrate them against the current standing pose.

The 13 articulation handles are:

- neck;
- left/right shoulder;
- left/right elbow;
- left/right wrist;
- left/right hip;
- left/right knee;
- left/right ankle.

Remove completely from the new format and editor:

- nose;
- left/right eye;
- left/right ear;
- facial/head skeleton connections.

All 13 body joints are required. Do not add per-joint enable/disable complexity in v1. The pose itself retains a subject-level `enabled` flag.

Coordinates remain normalized relative to the subject box and may extend beyond the box, matching current behavior. Continue allowing useful out-of-box values; validate against the current safe normalized range rather than clipping every joint to `[0, 1]`.

### 7.2 Head ellipse

The head is a filled ellipse with:

- center `cx`, `cy`;
- horizontal radius `rx`;
- vertical radius `ry`.

No face nodes exist inside it.

Interaction:

- dragging the ellipse body moves its center;
- selected ellipses show horizontal and vertical resize handles;
- resizing keeps the center fixed;
- enforce small positive minimum radii;
- do not implement head rotation in v1;
- head direction/facing remains prompt-controlled.

### 7.3 Body primitives

Render a filled body from these deterministic primitives:

#### Head

Filled ellipse from the head state.

#### Torso

A filled rounded quadrilateral based on:

```text
left shoulder
right shoulder
right hip
left hip
```

Expand the quadrilateral modestly about its centroid so joint centers are not the outer boundary. Use a centralized constant, initially approximately `1.08`.

Round/connect the four corners by drawing suitable joint disks or equivalent geometry.

#### Neck

Draw a filled capsule between the neck point and the nearest boundary point on the head ellipse in the neck-to-head direction.

#### Arms

Draw tapered capsules for:

```text
shoulder → elbow
elbow → wrist
```

#### Legs

Draw tapered capsules for:

```text
hip → knee
knee → ankle
```

Draw endpoint disks with the larger adjacent radius so connected segments cannot leave pinholes.

### 7.4 Initial geometry constants

Centralize all calibration constants in one immutable Python structure and an equivalent TypeScript constant object. Do not scatter magic numbers.

Use a body scale unit based on the selected subject box:

```text
u = max(1.5 px, 0.0125 * min(subject_box_width, subject_box_height))
```

Initial full widths:

| Primitive | Proximal width | Distal width |
|---|---:|---:|
| upper arm | `4.5u` | `4.0u` |
| forearm | `4.0u` | `3.0u` |
| thigh | `6.5u` | `5.5u` |
| calf | `5.5u` | `3.8u` |
| neck | `3.5u` | `3.5u` |

These are first-test calibration values. Keep them easy to adjust in one place, but do not expose them in the first GUI.

If actual preview proportions make these constants clearly unsuitable, Codex may calibrate them while preserving the centralized, tested design and documenting the final values.

### 7.5 Tapered-capsule construction

For endpoints `A`, `B` and radii `r_a`, `r_b`:

1. compute the segment unit vector;
2. compute its perpendicular;
3. draw the four-point tapered polygon;
4. draw an endpoint disk at `A` with `r_a`;
5. draw an endpoint disk at `B` with `r_b`;
6. if the segment is nearly zero length, draw a disk using the larger radius.

Use the same mathematical construction in browser SVG and backend rasterization.

### 7.6 Editor presentation

Selected subject:

- translucent filled volumetric body;
- small articulation handles;
- head move/resize handles;
- optional thin centerlines only as editing aids;
- clear selected-state outline.

Unselected subject:

- filled volumetric mannequin only;
- no ball-and-stick nodes;
- no centerlines;
- no resize handles.

The backend conditioning masks must never contain:

- editor handles;
- centerlines;
- selection outlines;
- labels;
- bounding-box outlines;
- resize affordances.

### 7.7 Presets and mirror

Preserve:

- standing preset;
- squatting preset;
- mirror horizontally.

Update the presets to set all 13 joints and the head ellipse.

Horizontal mirror must:

- reflect every x coordinate about the subject-local center;
- swap all left/right body joints;
- reflect the head center;
- preserve head radii.

Add tests for involution:

```text
mirror(mirror(pose)) == pose
```

within floating-point tolerance.

---

## 8. Backend mask generation

Create a pure backend module, recommended as:

```text
src/k2_region_lab/volumetric_pose.py
```

or split as:

```text
src/k2_region_lab/pose.py
src/k2_region_lab/pose_masks.py
```

The public API should make geometry, rendering, and mask summaries explicit and testable.

Suggested types:

```python
@dataclass(frozen=True)
class VolumetricPoseStyle:
    ...

@dataclass(frozen=True)
class SubjectPoseMasks:
    region_id: str
    core: np.ndarray
    support: np.ndarray
    ownership: np.ndarray | None
    ...

@dataclass(frozen=True)
class VolumetricMaskBundle:
    width: int
    height: int
    subjects: tuple[SubjectPoseMasks, ...]
    union_core: np.ndarray
    union_support: np.ndarray
    summary: PoseMaskSummary
```

### 8.1 Raster quality

Render masks at 2× or 4× supersampling and downsample with Lanczos. Use one defined factor consistently in tests. The current pose renderer already uses supersampling; retain that quality principle.

Outputs:

- grayscale float masks normalized to `[0, 1]`;
- exact canvas width/height;
- no clipping to subject boxes;
- clipping only at the overall output canvas boundary.

### 8.2 Core and support

For each subject:

1. render binary core geometry;
2. dilate it by an automatically derived margin;
3. feather the outer boundary;
4. restore core pixels to exactly `1.0`.

Initial automatic support values:

```text
dilation_radius = max(8 px, 2.5u)
feather_radius  = max(4 px, 1.0u)
```

Use existing dependencies only. Pillow `MaxFilter` plus `GaussianBlur` is acceptable if validated at supported output sizes.

Conceptual construction:

```python
dilated = max_filter(core, dilation_radius)
soft = gaussian_blur(dilated, feather_radius)
support = maximum(core, soft)
support = clip(support, 0, 1)
```

Keep these constants centralized and include them in the runtime mask summary.

### 8.3 Ownership

Use the same current front-to-back region priority as regional prompting.

For each subject in priority order:

1. take pixels where its support is nonzero;
2. remove pixels already claimed by a higher-priority subject;
3. preserve the subject's support alpha inside its exclusive area;
4. mark the area claimed for lower-priority subjects.

The global union remains the maximum of all subject support masks and is not reduced by ownership.

Required behavior:

- overlapping mannequins do not cause two character LoRAs to own the same hard-gated image token;
- higher-priority subject wins overlap;
- non-overlapping areas remain unchanged;
- a subject with pose disabled has no mannequin mask or ownership.

### 8.4 Mask summary

Calculate and retain:

- enabled subject count;
- per-subject core pixel count/coverage;
- per-subject support pixel count/coverage;
- union support coverage;
- overlap pixel count before exclusivity;
- chosen priority order;
- style/version identifier;
- dilation/feather values;
- warnings for extremely small or extremely large coverage.

Do not put raw masks in ordinary job events.

Optionally support a diagnostic environment flag such as:

```text
K2LAB_POSE_GATE_DEBUG=1
```

Under that flag, write core/support/ownership debug PNGs to the job's private temporary diagnostic directory. Do not expose raw filesystem paths through browser events.

---

## 9. Dynamic regional prompt and LoRA routing

The existing regional implementation is functional and remains authoritative. Extend it rather than replacing it.

### 9.1 Required effective mask

For subject region `r`, define:

- `B_r`: its existing normal regional image mask/field;
- `O_r`: its exclusive volumetric ownership mask/field;
- `g_i`: current pose-gate strength.

The effective regional image field during transition `i` is:

```text
R_r(i) = (1 - g_i) * B_r + g_i * O_r
```

Therefore:

- hard phase: subject prompt/LoRA is routed through its own mannequin volume;
- soft phase: routing expands gradually toward the existing region;
- normal phase: existing regional behavior is restored exactly.

Text-token masks do not change. Global LoRAs do not change.

### 9.2 Regional LoRA

Extend routed image/combined LoRA masks to support:

- cached normal field;
- cached hard ownership field;
- runtime scalar gate strength.

Do not rebuild large masks on every transformer call. Cache device/dtype-specific tensors and perform only the scalar blend needed for the current transition.

For a route targeting multiple subject regions, union the corresponding current effective fields using the existing route-union semantics.

For:

- global LoRA: unchanged;
- text-only route: unchanged;
- ordinary non-subject region without a mannequin: hard field defaults to its existing normal field unless current semantics require it to remain outside the subject gate;
- subject image/combined route: use the dynamic field above.

### 9.3 Regional prompt attention

Extend the current spatial-attention override to obtain current image fields from a small runtime provider/controller.

Requirements:

- no regeneration of the prompt plan per step;
- normal phase must match the existing attention behavior;
- hard phase must bind each subject's regional text to its exclusive volumetric ownership;
- overlap must respect current front-to-back priority;
- soft phase must transition without a discontinuous jump.

The implementation may compute current owners from the blended fields in current priority order. The image-token grid is small enough for this to be inexpensive, but cache every static component.

Continue using the current `CanvasGeometry` and model-specific image-token mapping. Do not hardcode a guessed latent or patch scale.

### 9.4 Existing regional relaxation

The branch already has its own regional denoising/attention progress behavior. Preserve it.

Adjust its progress denominator to the effective sampler transition count where appropriate, but do not conflate:

- existing regional relaxation;
- pose-gate release;
- sigma trajectory allocation.

Add tests showing that gating disabled reproduces the old plan and routing behavior.

---

## 10. Sigma and gate implementation modules

Create a pure module recommended as:

```text
src/k2_region_lab/pose_gating.py
```

Suggested public API:

```python
class SoftGateSchedule(StrEnum):
    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEPPED = "stepped"

class SigmaScheduleMode(StrEnum):
    AUTOMATIC = "automatic"
    PHASE_WEIGHTED = "phase_weighted"
    ADVANCED = "advanced"

@dataclass(frozen=True)
class PoseGatePhases:
    hard_steps: int
    soft_steps: int
    normal_steps: int

    @property
    def effective_steps(self) -> int: ...

@dataclass(frozen=True)
class SigmaScheduleRequest:
    mode: SigmaScheduleMode
    hard_share: float
    soft_share: float
    normalized_knots: tuple[float, ...]

@dataclass(frozen=True)
class ResolvedSigmaSchedule:
    baseline_sigmas: tuple[float, ...]
    normalized_positions: tuple[float, ...]
    resolved_sigmas: tuple[float, ...]
    mode: SigmaScheduleMode
    phase_shares: dict[str, float]

class PoseGateController:
    ...
```

Pure functions:

```python
effective_step_count(...)
soft_gate_strength(...)
gate_strengths(...)
phase_weighted_positions(...)
resample_advanced_positions(...)
resolve_sigma_schedule(...)
validate_sigma_schedule(...)
```

The module must not import the web layer.

ComfyUI-specific baseline-sigma retrieval may live in the worker runtime or a thin integration module; keep the math independently testable with synthetic baseline curves.

---

## 11. Runtime integration

Primary current target:

```text
src/k2_region_lab/worker/runtime.py
```

### 11.1 Replace current arguments

Remove active generation arguments for:

- `pose_conditioning_enabled`;
- `pose_controlnet_path`;
- `pose_control_strength`;
- `pose_control_start`;
- `pose_control_end`.

Add typed arguments or a single validated settings object for:

- `pose_gating_enabled`;
- `pose_hard_gate_steps`;
- `pose_soft_gate_steps`;
- `pose_soft_gate_schedule`;
- `pose_sigma_schedule_mode`;
- `pose_sigma_hard_share`;
- `pose_sigma_soft_share`;
- `pose_sigma_knots`.

Prefer passing a dataclass/Pydantic-derived object rather than adding an indefinitely growing list of primitives.

### 11.2 Generation flow

Required order:

1. validate project and generation settings;
2. compile current regional prompt/LoRA plan;
3. calculate `effective_steps`;
4. if gating is enabled:
   - require at least one pose-enabled subject when `H + S > 0`;
   - render volumetric mask bundle;
   - validate mask coverage;
   - prepare dynamic regional hard fields;
5. initialize the model and ordinary T2I empty latent/noise as currently;
6. obtain baseline sigmas from the selected ComfyUI sampler/scheduler for `effective_steps`;
7. resolve custom sigmas;
8. create and bind `PoseGateController`;
9. install dynamic denoise-mask callback;
10. call `comfy.sample.sample(...)` exactly once;
11. decode/save through the current output path;
12. attach resolved pose-gating metadata;
13. restore hooks/resources in `finally`.

### 11.3 Sampling pseudocode

Adapt exact signatures to the checked-out ComfyUI version:

```python
phases = PoseGatePhases(
    hard_steps=settings.pose_hard_gate_steps,
    soft_steps=settings.pose_soft_gate_steps,
    normal_steps=settings.steps,
)

effective_steps = phases.effective_steps

baseline_sigmas = build_comfy_baseline_sigmas(
    model=generation_model,
    steps=effective_steps,
    sampler_name=settings.sampler,
    scheduler=settings.scheduler,
    denoise=1.0,
)

resolved = resolve_sigma_schedule(
    baseline_sigmas=baseline_sigmas,
    phases=phases,
    request=settings.pose_sigma_schedule,
)

bundle = render_volumetric_masks(
    regions=regions,
    width=settings.width,
    height=settings.height,
    priority=compiled_plan.priority,
)

controller = PoseGateController(
    phases=phases,
    soft_schedule=settings.pose_soft_gate_schedule,
    resolved_sigmas=resolved,
)

support_tensor = mask_to_tensor(bundle.union_support)

def dynamic_denoise_mask(sigma, prepared_support_mask, extra_options):
    controller.observe_sigma(sigma)
    g = controller.gate_strength
    return 1.0 - g * (1.0 - prepared_support_mask)

previous_hook = generation_model.model_options.get("denoise_mask_function")
if previous_hook is not None:
    raise PoseGateRuntimeError(
        "A pre-existing denoise-mask callback is incompatible with pose gating."
    )

generation_model.model_options["denoise_mask_function"] = dynamic_denoise_mask
bind_regional_gate_controller(controller, bundle)

try:
    samples = comfy.sample.sample(
        generation_model,
        noise,
        effective_steps,
        1.0,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        noise_mask=support_tensor,
        callback=wrapped_callback,
        seed=seed,
        sigmas=resolved.tensor,
        ...
    )
finally:
    unbind_regional_gate_controller()
    restore_model_option_exactly(...)
```

The exact ComfyUI call must be based on the vendored/installed version in the workspace image.

### 11.4 Progress callback

Wrap the existing callback rather than replacing its memory/cancellation behavior.

For each completed transition, report:

- `current = step + 1`;
- `total = effective_steps`;
- phase;
- gate strength used;
- current and next sigma, rounded for event display;
- normalized trajectory progress.

Example message:

```text
Denoising 3/12 — soft gate, strength 0.65
```

Do not expose prompts or raw paths.

Update the controller for the next transition only after all model evaluations for the current transition are complete.

### 11.5 Gating-disabled parity

When `pose_gating_enabled` is false:

- do not render masks;
- do not install a mask function;
- do not alter regional masks;
- use `steps` exactly as before;
- use the prior sampler call behavior;
- ignore stored gating/sigma tuning fields;
- produce output equivalent to the existing path for the same seed/settings, subject to ordinary model nondeterminism.

Add a regression test around call arguments and compiled routing.

---

## 12. Project schema version 21

Primary backend source:

```text
src/k2_region_lab/project.py
```

Primary frontend source:

```text
web/client/src/studioProject.ts
```

Bump:

```text
PROJECT_VERSION = 21
```

Update any worker protocol, agent version, capability, and compatibility constants required by the changed job payload.

### 12.1 New generation fields

Add:

```json
{
  "pose_gating_enabled": false,
  "pose_hard_gate_steps": 2,
  "pose_soft_gate_steps": 2,
  "pose_soft_gate_schedule": "cosine",
  "pose_sigma_schedule_mode": "automatic",
  "pose_sigma_hard_share": 0.20,
  "pose_sigma_soft_share": 0.30,
  "pose_sigma_knots": []
}
```

The existing `steps` field remains the normal-step count.

Defaults:

- gating disabled;
- hard steps 2;
- soft steps 2;
- cosine;
- automatic;
- balanced phase-share values preserved but ignored until phase-weighted mode is selected;
- empty advanced knots.

Do not persist a preset name; presets merely set the underlying values.

### 12.2 Removed version-21 fields

Do not write these old pose-ControlNet settings in version 21:

- pose ControlNet model/path/file ID;
- pose ControlNet strength;
- pose conditioning start;
- pose conditioning end;
- legacy `pose_conditioning_enabled` as a runtime feature.

### 12.3 Validation

Backend and frontend validation must agree:

- normal `steps`: preserve current valid bounds;
- hard steps: integer `>= 0`;
- soft steps: integer `>= 0`;
- effective steps: at least 1 and at most the existing safe application maximum, recommended 100 unless current constraints dictate a different documented maximum;
- release schedule: known enum;
- sigma mode: known enum;
- shares finite;
- hard/soft shares each in `[0, 1]`;
- calculated normal share positive for nonzero normal steps;
- advanced knots exactly `effective_steps + 1`;
- advanced endpoints exactly `0` and `1`;
- advanced interior values finite and strictly increasing;
- pose format and all 13 joints valid;
- head radii positive and bounded;
- no NaN/Infinity in pose or sigma settings.

### 12.4 Version-20 migration

Support loading version 20 and migrate it to version 21.

#### Pose migration

Copy the 13 retained body joints.

Derive the head ellipse:

1. center from legacy nose when valid;
2. otherwise average valid eye/ear positions;
3. otherwise place it above the neck using standing-pose proportions;
4. estimate `rx` from horizontal face-point spread when available;
5. estimate `ry` from neck-to-head-center distance;
6. clamp both radii to safe defaults/ranges.

Do not retain face nodes in the migrated version-21 document.

#### Pose-control setting migration

The incompatible old ControlNet path must not be silently activated as volumetric gating.

On migration:

- set `pose_gating_enabled = false`;
- initialize the new tuning defaults;
- discard old pose ControlNet model/strength/start/end from the canonical version-21 output;
- produce a nonfatal migration notice:

```text
Legacy Qwen pose-ControlNet settings were removed. Volumetric pose gating is available but remains disabled until enabled.
```

Surface this notice in local import/open UI where practical and record it in backend logs.

#### Round trips

Add tests for:

- version-21 exact round trip;
- version-20 to version-21 migration;
- PNG-embedded project migration;
- cloud project migration;
- no reappearance of removed fields.

---

## 13. Agent, web API, and worker protocol

Relevant current areas include:

```text
src/k2_region_lab/agent/domain.py
src/k2_region_lab/agent/jobs.py
src/k2_region_lab/worker/entrypoint.py
web/client/src/api.ts
```

### 13.1 Job request

Remove `pose_controlnet_file_id` from the new job request and browser submit path.

Do not resolve or require a ControlNet file for pose-gated generation.

The canonical project document should carry the authored gating settings. The agent should parse/version-check it and send validated worker fields.

### 13.2 Worker command

Bump the worker protocol version because generation payload semantics changed.

The worker command must include:

- version-21 project;
- new pose-gating settings;
- no pose ControlNet path;
- no old strength/start/end values.

Old worker/new client and new worker/old client mismatches must fail with the existing explicit version/capability mechanism rather than producing an obscure runtime error.

### 13.3 Capabilities

Advertise, directly or through project/protocol version:

```json
{
  "volumetric_pose_gating": {
    "format": "k2-volumetric-pose-v1",
    "soft_schedules": ["cosine", "linear", "exponential", "stepped"],
    "sigma_modes": ["automatic", "phase_weighted", "advanced"]
  }
}
```

Use the project's existing capability compatibility conventions.

### 13.4 Canonical job snapshot

Freeze the submitted project and resolved generation settings at job start. UI changes after submission must not affect a running job.

Resolved sigmas are worker-derived and should be attached to job result metadata, not written back into the user's authored project automatically.

---

## 14. Error handling and diagnostics

Create typed exceptions, recommended:

```python
class PoseGatingError(RuntimeError): ...
class PoseMaskBuildError(PoseGatingError): ...
class PoseGateScheduleError(PoseGatingError): ...
class SigmaScheduleError(PoseGatingError): ...
class PoseGateRuntimeError(PoseGatingError): ...
```

Map them to stable browser error codes:

- `pose_mask_invalid`;
- `pose_gate_schedule_invalid`;
- `sigma_schedule_invalid`;
- `pose_gate_runtime_failed`;
- `pose_gate_hook_incompatible`.

Do not classify by searching exception text for `"controlnet"` or `"pose"`.

The generic sampler error must not claim that a LoRA is incompatible unless the actual typed LoRA compatibility inspection established that fact.

Recommended generic message:

```text
Generation failed during model sampling. Review the diagnostic ID and generation settings.
```

### 14.1 Preserve tracebacks

The worker already logs/prints tracebacks, while the current agent path can discard stderr. Replace silent discard with bounded diagnostic capture.

Requirements:

- retain worker stderr/traceback in private agent logs or a job-specific diagnostic log;
- cap total captured size, for example 4 MiB per command;
- redact according to current project rules;
- assign/use the command ID as a diagnostic identifier;
- expose only the stable error code, safe message, and diagnostic ID to the browser;
- never expose raw credentials, provider tokens, or unrestricted filesystem paths.

### 14.2 Events

Emit:

- `pose_gating_prepared` before sampling;
- normal progress events with phase/gate information;
- `pose_gating_completed` only after successful sampling.

Do not emit `"released"` or `"completed"` from a `finally` block.

Cleanup may be logged at debug level without a success implication.

---

## 15. Web GUI

Primary current targets include:

```text
web/client/src/components/RegionCanvas.tsx
web/client/src/components/Inspector.tsx
web/client/src/WorkspaceStudio.tsx
web/client/src/pose.ts
web/client/src/studioProject.ts
```

Create focused components rather than making `Inspector.tsx` unmaintainably large, recommended:

```text
web/client/src/components/PoseGatingControls.tsx
web/client/src/components/SigmaScheduleEditor.tsx
web/client/src/volumetricPose.ts
```

### 15.1 Region canvas

Replace ball-and-stick rendering with the volumetric SVG geometry described above.

Drag modes should include:

- joint;
- head move;
- head horizontal resize;
- head vertical resize;
- existing region move/resize modes.

Preserve:

- subject-region drawing;
- pose coordinates relative to subject box;
- out-of-box limbs;
- front/back ordering;
- standing/squatting/mirror actions;
- normal region behavior;
- region resize behavior.

The same visual mannequin should remain understandable at mobile and desktop canvas scales.

### 15.2 Global controls

Replace the old **Subject pose control** model section with **Volumetric pose gating**.

Controls:

- checkbox: `Constrain generation to subject mannequins`;
- integer: `Hard gate steps`;
- integer: `Soft gate steps`;
- selector: `Soft release`;
- read-only normal-step value sourced from existing `Steps`;
- read-only `Effective total`;
- selector: `Sigma schedule`;
- mode-specific sigma controls;
- concise experimental warning for nonstandard Turbo schedules.

Suggested display:

```text
Hard 2 + Soft 2 + Normal 8 = 12 total transitions
```

When enabling gating and both hard/soft values are zero, initialize them to the stored defaults or show an actionable warning.

Block submission when:

- gating is enabled;
- `H + S > 0`;
- no pose-enabled subject mannequin exists.

### 15.3 Phase-weighted controls

Show:

- hard percentage;
- soft percentage;
- calculated normal percentage;
- preset buttons: Balanced, Pose lock, Gentle;
- validation sum/state.

Percent UI should store exact normalized fractions.

### 15.4 Advanced editor

Use an accessible dependency-free SVG plus numeric table.

Graph:

- x-axis: transition boundary `0..N`;
- y-axis: normalized remaining trajectory/noise, visually `1 - p`;
- background bands: hard, soft, normal;
- draggable intermediate knots;
- locked endpoints;
- overlay: gate-strength sequence;
- tooltip/label for each knot.

Table:

- boundary index;
- phase immediately following it;
- normalized progress percentage;
- editable intermediate value;
- read-only endpoints.

The worker-resolved absolute sigma array is not known until the model/scheduler is loaded. Show it in completed job details/output metadata, not as a fabricated browser preview.

### 15.5 Project persistence

All controls and mannequin state must survive:

- New/Open;
- Save/Save As;
- browser refresh;
- cloud save/restore;
- PNG metadata import;
- version-20 import migration.

### 15.6 Remove old pose asset flow

Remove from the pose UI and generation submission:

- Choose pose model;
- ControlNet file requirement;
- old strength/start/end controls;
- InstantX installation guidance.

Do not remove generic asset infrastructure used elsewhere.

---

## 16. Output metadata and reproducibility

Extend the existing project/output metadata path.

For a gated generation, store a derived block similar to:

```json
{
  "pose_gating_runtime": {
    "format": "k2-volumetric-pose-v1",
    "enabled": true,
    "hard_steps": 2,
    "soft_steps": 2,
    "normal_steps": 8,
    "effective_steps": 12,
    "soft_schedule": "cosine",
    "sigma_mode": "phase_weighted",
    "phase_shares": {
      "hard": 0.25,
      "soft": 0.40,
      "normal": 0.35
    },
    "normalized_positions": [0.0, "...", 1.0],
    "resolved_sigmas": ["... exact finite values ...", 0.0],
    "gate_strengths": [1.0, 1.0, "...", 0.0],
    "mask_summary": {
      "subject_count": 2,
      "union_support_coverage": 0.31,
      "overlap_pixels": 1042,
      "priority_region_ids": ["..."]
    },
    "sampler": "euler",
    "scheduler": "...",
    "seed": 123
  }
}
```

Requirements:

- authored project stores normalized user settings;
- immutable job/output metadata stores exact resolved values;
- arrays are bounded by the application's effective-step limit;
- no raw mask image embedded by default;
- no secret paths or prompts added to public events;
- metadata remains sufficient to reproduce the scheduling/gating configuration.

For non-PNG outputs, retain equivalent job result metadata through the current durable result mechanism.

---

## 17. Documentation updates

Update:

```text
README.md
docs/subject_pose_control.md
docs/web_desktop_parity.md
```

Required documentation changes:

- remove InstantX/Qwen ControlNet setup instructions for Krea;
- explain volumetric mannequin editing;
- explain hard/soft/normal step accounting;
- explain all four release schedules;
- explain automatic, phase-weighted, and advanced sigma modes;
- warn that Krea Turbo custom schedules are experimental;
- explain that gating is hard only during configured gated phases and later normal denoising may alter the silhouette;
- explain that no pose model is downloaded or trained;
- include a same-seed gated/ungated test procedure;
- record project migration behavior.

Update README milestone language from version 20 to version 21 where applicable.

---

## 18. File-level implementation checklist

Codex must inspect for all references, but the expected changes include at least:

### Python core

- `src/k2_region_lab/pose.py`
  - new canonical pose;
  - legacy migration helpers;
  - presets/mirror.
- remove or repurpose `src/k2_region_lab/pose_control.py`
  - no active OpenPose/InstantX runtime.
- new `src/k2_region_lab/volumetric_pose.py` or `pose_masks.py`
  - geometry and mask bundle.
- new `src/k2_region_lab/pose_gating.py`
  - phase, release, sigma math, controller.
- `src/k2_region_lab/project.py`
  - version 21 and validation/migration.
- `src/k2_region_lab/regional_prompting.py`
  - optional hard volumetric fields/runtime provider.
- `src/k2_region_lab/spatial_attention.py`
  - dynamic regional fields.
- `src/k2_region_lab/regional_lora.py`
  - dynamic hard/normal image masks.
- `src/k2_region_lab/worker/runtime.py`
  - one-pass gate/sigma integration; remove ControlNet.
- `src/k2_region_lab/worker/entrypoint.py`
  - new payload.
- `src/k2_region_lab/worker/protocol.py`
  - typed errors.
- `src/k2_region_lab/agent/domain.py`
  - remove pose file ID; capabilities.
- `src/k2_region_lab/agent/jobs.py`
  - no pose asset resolution; new payload.
- version constants under `src/k2_region_lab/agent/__init__.py` or current equivalent.
- current PNG/result metadata writer.

### Browser

- `web/client/src/pose.ts`
  - new state and migration.
- recommended `web/client/src/volumetricPose.ts`
  - pure geometry helpers.
- `web/client/src/components/RegionCanvas.tsx`
  - filled mannequin and interactions.
- `web/client/src/components/Inspector.tsx`
  - replace old controls.
- recommended `PoseGatingControls.tsx`.
- recommended `SigmaScheduleEditor.tsx`.
- `web/client/src/studioProject.ts`
  - version 21/schema/migration/defaults.
- `web/client/src/api.ts`
  - new job contract.
- `web/client/src/WorkspaceStudio.tsx`
  - remove pose-model chooser and submit field.
- relevant CSS.
- browser contract scripts.

### Documentation/tests/build

- `README.md`;
- `docs/subject_pose_control.md`;
- `docs/web_desktop_parity.md`;
- Python tests;
- browser contract tests;
- workspace image/version metadata if required.

Use repository-wide search for:

```text
pose_control
pose_conditioning
pose_controlnet
InstantX
Qwen-Image-ControlNet
OPENPOSE
nose
left_eye
right_eye
left_ear
right_ear
PROJECT_VERSION
WORKER_PROTOCOL_VERSION
```

Every remaining occurrence must be intentional, such as migration code or historical documentation.

---

## 19. Automated tests

### 19.1 Pose state and geometry

Test:

- exactly 13 articulation joints;
- no facial joints in v1;
- required head ellipse;
- coordinate validation;
- standing preset;
- squatting preset;
- mirror involution;
- head move/resize;
- zero-length limb segment;
- limbs outside subject box remain rendered until canvas boundary;
- deterministic primitive bounds;
- frontend/backend geometry golden fixtures.

### 19.2 Mask generation

Test:

- core is binary;
- support stays in `[0, 1]`;
- support contains core;
- union is per-pixel max;
- support is larger than core;
- correct feather behavior;
- overlap priority;
- exclusive ownership;
- disabled subject excluded;
- empty subject set behavior;
- very small and large subject boxes;
- output dimensions;
- summary metrics.

Use semantic pixel assertions and compact fixtures rather than fragile full-image hashes alone.

### 19.3 Gate schedules

For every schedule:

- correct vector length;
- hard values exactly 1;
- soft values strictly between 0 and 1;
- normal values exactly 0;
- monotone nonincreasing soft release;
- edge cases `H=0`, `S=0`, `S=1`, `R=1`.

### 19.4 Dynamic denoise masks

With synthetic support masks:

- hard output equals support;
- normal output equals ones;
- soft output equals formula;
- core remains 1;
- outside increases as gate releases;
- hook restored after success;
- hook restored after exception;
- preexisting hook rejected with typed error;
- controller does not advance on repeated model evaluations within one transition.

### 19.5 Sigma schedules

Test with synthetic monotone baseline and, where importable, current ComfyUI baseline:

- automatic exact parity;
- `N + 1` length;
- phase boundary positions;
- weighted presets;
- zero-step phase behavior;
- advanced knot validation;
- advanced resampling after step-count change;
- interpolation endpoints;
- strict monotonicity;
- invalid NaN/Infinity;
- duplicates;
- invalid shares;
- final exact zero;
- serialization of normalized and resolved arrays.

### 19.6 Regional prompting/LoRA

Test:

- hard phase uses subject ownership;
- soft phase blends toward ordinary region;
- normal phase exactly matches prior regional mask;
- subject A does not own subject B's hard volume;
- overlap follows priority;
- global LoRA unchanged;
- text-only LoRA unchanged;
- combined/image route changes correctly;
- gating disabled reproduces old route;
- current regional relaxation remains functional.

### 19.7 Project and protocol

Test:

- version-21 exact round trip;
- v20 migration;
- old face points removed;
- head ellipse derived;
- old ControlNet settings removed and gating disabled;
- no `pose_controlnet_file_id` in new submit;
- agent capability/version mismatch;
- worker receives exact new settings;
- job snapshot immutable;
- output metadata exact.

### 19.8 Error classification

Test each typed error code.

Test that an unrelated sampling failure no longer returns the misleading LoRA compatibility message.

Test bounded stderr/diagnostic capture.

### 19.9 Browser

Run/add:

```bash
cd web/client
npm run test:project
npm run test:ui
npm run typecheck
npm run build
```

Browser contracts must cover:

- new defaults;
- v20 migration;
- v21 round trip;
- effective-step display;
- phase-share validation;
- advanced knots;
- no pose-model chooser;
- job payload without pose file ID;
- all 13 joints/head ellipse;
- no face handles.

### 19.10 Full Python validation

Run:

```bash
uv run pytest -q
uv run ruff check .
```

Do not limit final validation to only the newly added tests.

---

## 20. RunPod GUI acceptance procedure

The code milestone is reached only when the real workspace image can execute this flow.

### 20.1 Build/deploy readiness

Use the repository's existing versioned workspace-image workflow. Update image/agent version metadata as required. Codex must provide the exact commands or tag workflow based on the final branch.

The repository currently publishes through:

```text
.github/workflows/workspace-image.yml
```

and launches through:

```bash
./scripts/k2lab-runpod \
  --image 'ghcr.io/OWNER/k2lab-runpod-workspace@sha256:...'
```

Do not publish or provision billable infrastructure without the user's explicit execution outside Codex. Prepare the code and instructions.

### 20.2 Smoke test A: gating-disabled regression

- one subject region;
- ordinary regional prompt and existing character LoRA;
- pose gating disabled;
- same sampler/scheduler/steps previously known to work;
- confirm generation completes and output metadata remains valid.

### 20.3 Smoke test B: one mannequin

- one large subject region on the left half;
- volumetric mannequin with one arm raised and the other lowered;
- regional prompt describes the subject and pose semantics;
- one character LoRA;
- fixed seed;
- batch 1;
- normal steps 8;
- hard steps 2;
- soft steps 2;
- cosine release;
- automatic sigma schedule;
- Euler for the first diagnostic run.

Confirm:

- GUI shows 12 total transitions;
- job submission requires no ControlNet asset;
- progress identifies hard, soft, and normal phases;
- generation completes;
- metadata contains 12 transition gate values and 13 sigma endpoints;
- output includes resolved sigmas;
- no misleading LoRA/ControlNet diagnostic appears.

Then repeat same seed with gating disabled and compare placement.

### 20.4 Smoke test C: phase-weighted tuning

Repeat B with:

```text
Pose lock preset:
Hard 25%
Soft 40%
Normal 35%
```

Confirm exact normalized positions and sigmas are recorded.

### 20.5 Smoke test D: advanced sigma

- switch to advanced;
- move at least two intermediate knots while preserving monotonicity;
- generate;
- verify the worker receives and resolves the edited curve;
- verify exact output metadata;
- verify invalid crossing knots are blocked before submission.

### 20.6 Smoke test E: two subjects

- two separated subject regions;
- distinct mannequins and poses;
- distinct regional prompts;
- distinct character LoRAs;
- one subject standing, one crouching;
- hard 2, soft 2, normal 8.

Confirm:

- two ownership masks;
- union gate;
- no crash in regional prompt/LoRA routing;
- correct priority metadata;
- generated identities are at least routed to their intended volumes strongly enough to evaluate visually.

### 20.7 Behavioral evaluation, not code-completion fiction

Record screenshots/outputs for:

- ungated;
- automatic;
- Balanced;
- Pose lock;
- Gentle;
- at least one advanced curve.

The first empirical question is:

> Does strict early volumetric gating, followed by controlled release and ordinary regional denoising, preserve major head, torso, arm, and leg placement sufficiently without a trained pose adapter?

If it does not, do not conceal that result. The code should provide enough mask, phase, and sigma observability to diagnose whether the failure is:

- insufficient hard trajectory share;
- mask geometry/support width;
- release too fast;
- normal-phase drift;
- regional identity routing;
- lack of learned pose semantics.

---

## 21. Completion criteria

All must be true:

### Backend

- old Qwen/InstantX pose runtime is gone;
- volumetric masks are generated from the new canonical pose;
- hard/soft/normal phases run in one sampler call;
- outside-mask state uses scheduler-consistent masked sampling;
- all four release schedules work;
- all three sigma modes work;
- explicit sigmas reach ComfyUI;
- regional prompt and LoRA masks follow mannequin ownership while gated and return to existing behavior when normal;
- hooks restore after success/failure;
- errors are typed and tracebacks retained privately.

### Frontend

- no facial/head joints;
- draggable/resizable head ellipse;
- all 13 body articulation handles;
- filled torso/limbs;
- no visible ball-and-stick mannequin when unselected;
- hard/soft controls;
- effective total;
- release selector;
- sigma modes/presets/editor;
- persistence/migration;
- no pose-ControlNet chooser.

### Compatibility

- gating-disabled generation preserves existing behavior;
- image edit and face refinement remain unaffected;
- version-20 projects load through explicit migration;
- version-21 exact round trips;
- agent/worker capability versions prevent mismatches.

### Validation

- complete Python suite passes;
- Ruff passes;
- browser contracts/typecheck/build pass;
- workspace image can be built;
- RunPod web GUI reaches actual GPU generation without requiring a pose model.

---

## 22. Explicitly deferred work

Do not implement in this milestone:

- Krea-native OpenPose Control-LoRA;
- Krea-native volumetric pose Control-LoRA;
- depth conditioning;
- 3D mannequin/depth renderer;
- DensePose/IUV/normal-map conditioning;
- head rotation or facial landmarks;
- finger/hand rig;
- user-adjustable limb thickness;
- user-adjustable support dilation/feather;
- background-first generation;
- img2img;
- a second denoising pass;
- automatic pose-estimator scoring/retry;
- exact front/back limb-depth control.

Keep mask-bundle and runtime-controller APIs sufficiently clean that a future Krea-native adapter can consume `k2-volumetric-pose-v1` without replacing the editor or gating system.

Only pursue an adapter after the RunPod experiment establishes a repeatable failure that gating/sigma tuning cannot solve—for example, Krea fills the allowed body volume but repeatedly ignores the arrangement of individual limb volumes.

---

## 23. Design rationale

This approach tests the least expensive strong constraint first.

A pose Control-LoRA would teach the model the semantic meaning of a pose representation. The proposed system instead imposes a direct early optimization constraint:

```text
organized denoising updates may initially occur only inside the desired body volumes
```

The later normal phase restores Krea's ability to integrate anatomy, clothing, lighting, and scene coherence. User-controlled phase counts and sigma allocation make it possible to independently tune:

- how many times Krea evaluates while constrained;
- how much high-to-low-noise trajectory occurs while constrained;
- how abruptly the spatial restriction opens;
- how much unrestricted refinement remains.

The experiment is therefore useful even if the first mask constants are imperfect: it produces measurable controls rather than hiding all pose authority behind a single opaque strength slider.

---

## 24. Primary sources inspected

Current project:

- Branch:  
  `https://github.com/soomrenald/k2lab_runpod/tree/k2lab_pose`
- Current pose state:  
  `https://raw.githubusercontent.com/soomrenald/k2lab_runpod/k2lab_pose/src/k2_region_lab/pose.py`
- Current OpenPose renderer:  
  `https://raw.githubusercontent.com/soomrenald/k2lab_runpod/k2lab_pose/src/k2_region_lab/pose_control.py`
- Current worker runtime:  
  `https://raw.githubusercontent.com/soomrenald/k2lab_runpod/k2lab_pose/src/k2_region_lab/worker/runtime.py`
- Current project schema:  
  `https://raw.githubusercontent.com/soomrenald/k2lab_runpod/k2lab_pose/src/k2_region_lab/project.py`
- Current agent domain:  
  `https://raw.githubusercontent.com/soomrenald/k2lab_runpod/k2lab_pose/src/k2_region_lab/agent/domain.py`
- Current README/deployment flow:  
  `https://raw.githubusercontent.com/soomrenald/k2lab_runpod/k2lab_pose/README.md`

ComfyUI interfaces used by this design:

- `comfy/sample.py`:  
  `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/sample.py`
- `comfy/samplers.py`:  
  `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/samplers.py`

Krea 2 behavior:

- Official repository:  
  `https://github.com/krea-ai/krea-2`
- Official sampler:  
  `https://raw.githubusercontent.com/krea-ai/krea-2/main/sampling.py`

Because ComfyUI and the repository can change, Codex must verify exact installed/current signatures before committing integration code.

---

## 25. Final implementation report template

Codex should finish with:

```text
Starting branch/commit:
Ending branch/commit:

Implemented:
- ...

Removed:
- ...

Project/protocol migrations:
- ...

Tests:
- command → result

Browser build:
- command → result

GPU-dependent tests not run:
- ...

Workspace image build/publish instructions:
1. ...
2. ...

RunPod GUI test:
1. ...
2. ...

Known limitations:
- ...

Deviations from specification:
- ...
```

Do not substitute a new planning document for the requested implementation.
