# K2Lab Subject-Semantic Pose Conditioning

## End-to-end implementation specification for Codex 5.6 Sol High

**Repository:** `https://github.com/soomrenald/k2lab_runpod`  
**Starting branch:** `k2lab_pose`  
**Tested baseline commit:** `cb3159f4fde88a9e0ae2afc638230241900aafa5`  
**Existing project schema:** version 21  
**Target project schema:** version 22  
**Existing pose format:** `k2-volumetric-pose-v1`  
**Target milestone:** a complete implementation testable through the real RunPod web GUI

---

## 0. Directive to Codex

Implement this specification end to end. Do not stop after writing a plan, isolated backend prototype, UI mockup, or partial sampler experiment.

Before changing code:

1. Check out `k2lab_pose`.
2. Record the exact starting commit with:

   ```bash
   git rev-parse HEAD
   git log -1 --oneline
   ```

3. Compare the checked-out code with tested commit:

   ```text
   cb3159f4fde88a9e0ae2afc638230241900aafa5
   ```

4. Treat the checked-out branch as authoritative where later commits differ.
5. Run the existing Python tests, browser contract tests, typecheck, and build.
6. Inspect the exact ComfyUI commit pinned by `Dockerfile.workspace` before using any private or semi-private API. At the tested baseline it is:

   ```text
   285a98944c397a4a81f15ac63d69fa3dbc0a27b9
   ```

Implement through the first point at which the user can:

1. launch the RunPod web application;
2. open or create a version-22 project;
3. enter a scene prompt and shared visual context;
4. create two volumetric subject mannequins;
5. assign distinct subject prompts and distinct regional LoRAs;
6. select **Prediction composite** semantic routing;
7. configure the already-implemented hard, soft, normal, release, and sigma controls;
8. generate through the real RunPod web GUI;
9. compare **Spatial only**, **Attention isolation**, and **Prediction composite** using the same seed;
10. inspect runtime semantic-routing metadata and forward counts.

Do not:

- train or add a pose ControlNet or Control-LoRA;
- implement img2img, background-first generation, a second sampler pass, VAE round trips, or re-noising;
- remove the existing volumetric mannequin, hard/soft spatial gate, sigma scheduler, regional prompt system, or regional LoRA system;
- replace the current regional implementation with an unrelated parallel implementation;
- disguise multiple denoiser evaluations as multiple sampler passes;
- silently fall back from semantic routing to the current spatial-only behavior;
- swallow scope, conditioning, attention, LoRA, or sampler-hook errors;
- claim semantic obedience is successful until it has been tested on Krea 2 through RunPod.

At completion, report:

- starting and ending commits;
- files changed;
- schema/protocol migrations;
- tests and builds run;
- exact workspace-image build/publish steps;
- exact RunPod launch/test steps;
- GPU-dependent tests not run;
- measured forward counts and timing from the first live run;
- known limitations;
- deviations from this specification.

---

## 1. Problem established by the first implementation test

The current volumetric gate answers only:

> Where may a denoising update be accepted?

It does not answer:

> Which prompt must produce the accepted update?

At the tested commit, ordinary generation does the following:

1. compiles one unified text prompt containing:
   - the global scene;
   - all regional clauses;
   - all subject descriptions;
   - identity-trigger instructions;
   - relationship clauses;
2. encodes that unified prompt once;
3. asks Krea for one full-scene prediction;
4. uses the volumetric denoise mask to accept that prediction only within the currently open cells.

Consequently, a hard-gated mannequin volume can receive any part of the prediction for the unified scene:

- furniture;
- architecture;
- foliage;
- background texture;
- another subject;
- a local continuation of the scene.

The current attention partition does not solve this because:

- global/shared text tokens have owner `0` and remain accessible to subject-owned image queries;
- subject image queries block other subject-owned text but not global-scene text;
- image-to-image attention is explicitly unmodified;
- global scene structure can therefore enter a mannequin cell through both text-to-image and image-to-image pathways.

This is not primarily a mask-width, hard-step-count, soft-release, or sigma problem. Those controls only determine how strongly and for how long a spatially cropped **full-scene prediction** is accepted.

The required correction is to separate:

```text
Spatial authority:
Where can this update be accepted?

Semantic authority:
Which conditioning produced this update?
```

---

## 2. Product objective

During hard and soft pose-gated phases, every subject-owned latent cell must receive a denoising direction generated from that subject's own conditioning, not merely a cropped prediction generated from the entire scene prompt.

The system must support three semantic-routing modes:

1. **Spatial only**
   - preserves version-21 behavior;
   - one unified prediction;
   - volumetric spatial gate only;
   - retained strictly as a regression/diagnostic mode.

2. **Attention isolation**
   - one unified model prediction per model evaluation;
   - strict semantic-island attention rules during the hard phase;
   - gradually relaxed attention penalties during the soft phase;
   - faster but intrinsically less isolated because the unified text encoder has already contextualized all prompt tokens together.

3. **Prediction composite** — recommended
   - one full-scene prediction plus one separately encoded subject-only prediction per enabled mannequin during hard/soft phases;
   - exclusive mannequin masks fuse those predictions into one denoising direction;
   - subject-specific regional LoRAs apply in the corresponding subject-only forward;
   - normal phase returns to one ordinary unified prediction;
   - remains one latent trajectory and one sampler invocation.

The intended hard-phase behavior is:

```text
Subject A mannequin cells
    → Subject A prompt + Subject A LoRAs + shared visual context

Subject B mannequin cells
    → Subject B prompt + Subject B LoRAs + shared visual context

Outside subject ownership
    → full-scene prediction, although the existing hard denoise gate
      may prevent those cells from accepting an update
```

The normal phase must restore the existing unified regional pipeline exactly.

---

## 3. Non-negotiable generation invariants

All semantic modes must preserve:

- one initial seeded noise tensor;
- one latent canvas;
- one monotonically decreasing sigma array;
- one call to `comfy.sample.sample(...)`;
- the existing hard/soft/normal transition count;
- the existing user-selected soft-release schedule;
- the existing automatic, phase-weighted, or advanced sigma schedule;
- one final VAE decode;
- no sampler restart;
- no image source;
- no second denoising trajectory.

**Prediction composite uses multiple model evaluations inside one sampler evaluation.** It does not create multiple samples or multiple latent trajectories.

For one full prediction `P_full`, subject predictions `P_r`, exclusive ownership masks `O_r`, and current gate strength `g`:

```text
W_r    = g · O_r
W_sum  = clamp(Σ_r W_r, 0, 1)
W_full = 1 - W_sum

P_fused = W_full · P_full + Σ_r W_r · P_r
```

Equivalent residual form:

```text
P_fused = P_full + Σ_r W_r · (P_r - P_full)
```

Required phase behavior:

```text
hard:   g = 1
soft:   0 < g < 1
normal: g = 0
```

Therefore:

- hard subject cores use the subject-only prediction;
- feathered ownership edges blend with the full prediction;
- soft steps progressively restore full-scene influence;
- normal steps use only `P_full`;
- a subject prediction is not evaluated when `g == 0`.

This is a region-weighted fusion of denoising paths, conceptually aligned with region-based MultiDiffusion, but adapted to K2Lab's single full-canvas Krea latent and current ComfyUI sampler.

---

## 4. Keep the current spatial gate

Do not replace the current volumetric denoise gate.

The existing gate still performs an essential independent function:

```text
D_i = 1 - g_i · (1 - U)
```

Where:

- `U` is the union support mask;
- `g_i` is the hard/soft/normal gate strength.

Keep:

- `PoseGateController`;
- `PoseGatePhases`;
- all four release schedules;
- all sigma modes;
- scheduler-consistent outside-mask noise handling;
- `denoise_mask_function`;
- current mask summaries and ownership priority.

The semantic composite controls **which model prediction** is used.  
The denoise mask controls **where the sampler accepts that prediction**.

Both are required.

---

## 5. Prompt model: scene, shared context, and subjects

### 5.1 Separate scene content from shared visual context

Add a new project field:

```text
shared_visual_prompt
```

Keep the canonical backend field:

```text
global_prompt
```

but relabel it in the GUI as:

```text
Scene prompt
```

Definitions:

### Scene prompt

Content belonging to the overall scene:

- environment;
- architecture;
- furniture;
- landscape;
- weather;
- background objects;
- broad subject relationships;
- scene action not owned by one subject.

Example:

```text
A tropical greenhouse filled with broad green leaves,
glass walls, wet stone flooring, and distant potted plants.
```

### Shared visual context

Information that may safely be given to every subject-only forward:

- medium;
- photographic/rendering style;
- lens;
- camera treatment;
- lighting;
- palette;
- film stock;
- broad temporal/genre treatment.

Example:

```text
Realistic cinematic photography, soft overcast window light,
35 mm lens, restrained natural colors.
```

The application must not attempt unreliable NLP extraction from the scene prompt. The user explicitly authors shared visual context.

UI help text must say that scene objects do not belong in this field.

### 5.2 Full unified prompt

Extend `compile_regional_prompt_plan(...)` to accept:

```python
shared_visual_prompt: str
```

The full prompt should contain, in a deterministic documented order:

1. shared visual context;
2. scene prompt;
3. existing regional clauses in current priority order;
4. existing relationship clause.

The current prompt semantics, identity instructions, face-identity text, spatial clauses, and region order must otherwise remain intact.

Store character spans for:

- shared visual context;
- scene prompt;
- every region clause;
- relationship clause.

Bind them to token spans after Krea tokenization.

### 5.3 Subject-only prompt

Create a pure compiler:

```python
compile_subject_conditioning_prompt(...)
```

For every enabled subject with pose enabled, construct a separate prompt containing only:

1. shared visual context;
2. that subject's face-identity prompt;
3. that subject's regional prompt;
4. that subject's character-identity trigger instructions;
5. a concise generated instruction stating that this branch generates one coherent visible subject and no unrelated scene content;
6. coarse framing derived from the subject box/mannequin scale if the existing framing helper is useful.

Do not include:

- `global_prompt`;
- another subject's prompt;
- background/object region prompts;
- relationship clauses;
- left/right canvas-location prose;
- greenhouse, room, landscape, furniture, or other scene content unless the user explicitly placed it in the subject prompt;
- another subject's identity trigger.

Recommended deterministic template:

```text
{shared visual context, if nonempty}

Generate one coherent visible person described as:
{face identity, if nonempty}
{subject prompt}

Use the described action, posture, clothing, held items, and body orientation.
Do not generate unrelated scenery, architecture, furniture, landscape,
background objects, text, borders, guides, or additional people in this
subject-conditioning branch.

{character identity instructions, if any}
```

Do not insert an automatic claim about exact pose compliance. The mannequin and user prompt provide the intended pose.

Preserve legitimate subject-owned props:

- clothing;
- armor;
- weapons held by the subject;
- carried objects;
- a chair explicitly described as part of the subject's seated interaction.

The generic exclusion wording must not contradict explicit subject text.

### 5.4 Subject prompt binding

Create a bound subject-prompt plan containing:

- region ID/name;
- compiled prompt;
- total token count;
- shared-context token span;
- subject-description token span;
- face-identity token span;
- each character-trigger token span;
- subject-scoped prompt-emphasis spans;
- prompt SHA-256 for diagnostics.

The LoRA router must use these exact spans.

### 5.5 Prompt emphases

Existing behavior:

- global/scene emphasis remains part of the full unified forward only;
- region-scoped emphasis for a subject applies in:
  - the full unified forward through the existing regional route;
  - that subject's subject-only forward through its local bound span;
- an emphasis for another region does not enter a subject-only forward;
- shared visual context has no new emphasis UI in version 22.

### 5.6 Preview

Extend the current unified-prompt preview with a **Conditioning prompts** view:

- Full scene;
- Subject A;
- Subject B;
- Subject C, if present.

This preview must be generated from the same canonical Python compiler used by the worker, not re-created independently in TypeScript.

Do not expose prompts in public job events. Existing project/PNG persistence may contain the authored prompts as it already does.

---

## 6. New semantic-conditioning domain module

Create:

```text
src/k2_region_lab/semantic_conditioning.py
```

Recommended public API:

```python
from enum import StrEnum
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Mapping

class PoseSemanticMode(StrEnum):
    SPATIAL_ONLY = "spatial_only"
    ATTENTION_ISOLATION = "attention_isolation"
    PREDICTION_COMPOSITE = "prediction_composite"

class ConditioningScopeKind(StrEnum):
    FULL = "full"
    SUBJECT = "subject"

@dataclass(frozen=True, slots=True)
class ConditioningScope:
    kind: ConditioningScopeKind
    region_id: str | None = None

@dataclass(frozen=True, slots=True)
class BoundSubjectPrompt:
    region_id: str
    region_name: str
    prompt: str
    text_token_count: int
    shared_visual_span: tuple[int, int] | None
    subject_span: tuple[int, int]
    face_identity_span: tuple[int, int] | None
    character_trigger_spans: Mapping[str, tuple[tuple[int, int], ...]]
    emphasis_spans: tuple[...]
    prompt_sha256: str

@dataclass(frozen=True, slots=True)
class SubjectSemanticConditioning:
    scope: ConditioningScope
    bound_prompt: BoundSubjectPrompt
    conditioning: tuple[...]
    ownership_mask: object
    ownership_coverage: float

@dataclass(frozen=True, slots=True)
class PoseSemanticPlan:
    mode: PoseSemanticMode
    full_scope: ConditioningScope
    full_conditioning: tuple[...]
    subjects: tuple[SubjectSemanticConditioning, ...]
    shared_visual_prompt_sha256: str
    estimated_forwards_per_gated_evaluation: int

@dataclass(frozen=True, slots=True)
class ConditioningExecutionContext:
    scope: ConditioningScope
    text_token_count: int
```

Use a `ContextVar`:

```python
CURRENT_CONDITIONING_CONTEXT
```

Provide an exception-safe context manager:

```python
with conditioning_execution_scope(scope, text_token_count):
    ...
```

It must always restore the previous value after success or exception.

Do not store the active conditioning scope as a mutable process-global singleton.

---

## 7. Conditioning metadata

When constructing Comfy conditioning records, copy metadata and attach:

```text
k2_conditioning_scope = "full" | "subject"
k2_conditioning_region_id = null | region_id
k2_conditioning_prompt_sha256 = ...
k2_conditioning_text_token_count = ...
```

The tested ComfyUI `convert_cond(...)` copies arbitrary metadata and adds a UUID, so these fields should survive conditioning conversion.

Requirements:

- do not use Comfy's `mask` or `default` metadata for the prediction-composite implementation;
- do not rely on undocumented averaging among a mixed condition list;
- the custom sampler hook explicitly groups and evaluates scopes;
- preserve all ordinary Krea model-conditioning metadata generated by `encode_from_tokens_scheduled(...)`.

Support multiple conditioning records returned for one encoded prompt by grouping all records with the same scope. Do not assume exactly one record unless the installed Krea encoder contract proves that and tests enforce it.

Every positive conditioning record must have exactly one valid K2 scope.

Negative conditioning remains the existing empty prompt. Krea generation currently uses CFG 1.0; do not introduce a new CFG design in this milestone.

---

## 8. Recommended mode: prediction compositing

### 8.1 Install through ComfyUI's conditional-batch hook

The pinned ComfyUI sampler supports:

```text
model_options["sampler_calc_cond_batch_function"]
```

Install a custom callback only for `prediction_composite`.

Create an exception-safe context manager, recommended:

```python
installed_semantic_prediction_hook(
    model_options,
    semantic_runtime,
    comfy_samplers,
)
```

Before installing:

- snapshot any existing `sampler_calc_cond_batch_function`;
- if one exists, do not silently replace it;
- either compose it through a tested adapter or raise:
  - `semantic_sampler_hook_incompatible`.

For version 22, an explicit typed failure is preferred over an untested composition.

Restore the exact previous value in `finally`.

### 8.2 Wrapper contract

The installed function receives:

```python
{
    "conds": [positive_conds, negative_conds_or_none],
    "input": x_t,
    "sigma": sigma,
    "model": model,
    "model_options": model_options,
}
```

The callback must:

1. determine current phase and `g` from the existing `PoseGateController`;
2. group positive conditioning records by K2 scope;
3. validate exactly one full scope;
4. calculate the full prediction with the installed ComfyUI:

   ```python
   comfy.samplers.calc_cond_batch(...)
   ```

5. when `g > 0`, calculate one prediction per active subject scope;
6. prepare exclusive ownership masks at the current prediction resolution;
7. fuse predictions using the formula in section 3;
8. return a list matching Comfy's expected positive/negative outputs;
9. preserve CFG-1 behavior;
10. update semantic diagnostics without advancing the sampler transition.

Calling `comfy.samplers.calc_cond_batch(...)` from this wrapper is not recursive: recursion occurs only through `sampling_function`, while `calc_cond_batch` directly executes the conditional model batches.

### 8.3 Pseudocode

Adapt exact types/signatures to the pinned ComfyUI version:

```python
def semantic_calc_cond_batch(args):
    cond_sets = args["conds"]
    x = args["input"]
    sigma = args["sigma"]
    model = args["model"]
    model_options = args["model_options"]

    controller.assert_same_transition_for_model_evaluation(sigma)
    g = controller.gate_strength

    positive = cond_sets[0]
    negative = cond_sets[1] if len(cond_sets) > 1 else None

    groups = group_conditioning_by_scope(positive)

    full_records = groups.require_full()

    with conditioning_execution_scope(
        ConditioningScope(ConditioningScopeKind.FULL),
        groups.full_text_token_count,
    ):
        full_pred = comfy.samplers.calc_cond_batch(
            model,
            [full_records],
            x,
            sigma,
            model_options,
        )[0]

    if g <= 0.0:
        negative_pred = zero_or_original_negative_prediction(...)
        diagnostics.observe_full_only(...)
        return [full_pred, negative_pred]

    subject_predictions = {}
    for subject in semantic_plan.subjects:
        records = groups.require_subject(subject.region_id)
        with conditioning_execution_scope(
            subject.scope,
            subject.bound_prompt.text_token_count,
        ):
            subject_predictions[subject.region_id] = (
                comfy.samplers.calc_cond_batch(
                    model,
                    [records],
                    x,
                    sigma,
                    model_options,
                )[0]
            )

    masks = semantic_mask_cache.for_prediction(
        x,
        region_ids=subject_predictions.keys(),
    )

    fused = fuse_subject_predictions(
        full=full_pred,
        subjects=subject_predictions,
        ownership_masks=masks,
        gate_strength=g,
    )

    negative_pred = zero_or_original_negative_prediction(...)
    diagnostics.observe(...)
    return [fused, negative_pred]
```

### 8.4 Negative output at CFG 1

Current generation calls the sampler with:

```text
cfg = 1.0
```

Comfy may omit the unconditioned forward as an optimization.

The semantic callback must handle:

- `negative is None`;
- an empty negative list;
- a real negative conditioning list if future code disables the CFG-1 optimization.

For current CFG-1:

- returning a zero tensor for the unused unconditioned slot is acceptable only if verified against the pinned `cfg_function`;
- otherwise call the ordinary negative path once.

Add an integration test against the pinned sampler behavior. Do not guess silently.

### 8.5 Ownership-mask preparation

Use the exclusive per-subject ownership masks already generated by `VolumetricMaskBundle`.

Prepare each mask for the exact prediction tensor:

```text
[B, C, H, W] prediction
[1, 1, H, W] broadcastable ownership
```

Requirements:

- use bilinear resizing for soft ownership fields;
- clamp to `[0, 1]`;
- preserve front-to-back exclusivity;
- after resizing, calculate `sum_O`;
- where `sum_O > 1`, divide every ownership value by `sum_O`;
- cache by:
  - region ID;
  - device;
  - dtype;
  - spatial shape;
- do not repeat masks across channels in memory when broadcasting is sufficient;
- do not use subject region boxes in place of mannequin ownership.

Then:

```python
weights = {rid: g * ownership[rid] for rid in region_ids}
weight_sum = torch.clamp(sum(weights.values()), 0.0, 1.0)
fused = full * (1.0 - weight_sum)
for rid, prediction in subjects.items():
    fused = fused + prediction * weights[rid]
```

Validate all prediction shapes, dtypes, and devices before fusion.

### 8.6 Full prediction remains necessary

Do not optimize the full prediction away in the first implementation.

Even in the hard phase it supplies:

- feathered support-edge remainder;
- gaps between exclusive ownership masks;
- any denoisable area not fully owned;
- stable behavior when a support mask has partial alpha.

A later optimization may skip it only after proving that the accepted denoise area is completely covered and that output remains equivalent.

### 8.7 Number of forwards

For `S` enabled posed subjects:

```text
hard/soft model evaluation: 1 + S predictions
normal model evaluation:    1 prediction
```

For Euler, estimated total model forwards:

```text
normal_steps
+ (hard_steps + soft_steps) · (1 + S)
```

Example:

```text
2 subjects
2 hard + 2 soft + 8 normal

8 + 4 · 3 = 20 model forwards
```

This is an estimate only for one-evaluation-per-transition samplers. Keep sampler transition progress distinct from internal model-forward count.

### 8.8 Multi-GPU behavior

The first production implementation targets one RunPod GPU.

Because scope context and sequential conditional evaluation must be correct before optimization:

- detect Comfy's `multigpu_clones`;
- reject Prediction composite with a typed, explicit error if multi-GPU execution is active;
- do not silently produce scope-less LoRA routing across worker threads.

Attention isolation and Spatial only may keep existing supported behavior.

A future implementation may propagate scope metadata through the model-function wrapper in each GPU worker thread.

---

## 9. Scope-aware regional LoRA routing

Prediction compositing is incomplete unless a subject-only prediction receives:

- global LoRAs;
- LoRAs assigned to that subject;
- no LoRAs assigned only to another subject.

The current routed LoRA implementation compiles one text mask for the unified prompt and one image mask for the full canvas. Extend it to be conditioning-scope aware.

### 9.1 Route structure

Replace or extend `LoraDeltaRoute` with scope-specific masks.

Recommended structure:

```python
@dataclass(frozen=True, slots=True)
class ConditioningScopeMask:
    enabled: bool
    text_token_mask: tuple[float, ...]
    image_token_mask: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class LoraDeltaRoute:
    ...
    full_scope_mask: ConditioningScopeMask
    subject_scope_masks: Mapping[str, ConditioningScopeMask]
```

Preserve compatibility properties if they reduce unrelated churn.

### 9.2 Full scope

For the full unified forward, preserve the existing version-21 behavior:

- global LoRA: all text/image tokens;
- regional standard LoRA: its current unified regional text and image routing;
- character-identity LoRA:
  - identity trigger spans;
  - assigned regional image field;
- current dynamic normal/hard image-field blend;
- current unsupported-target checks.

### 9.3 Subject scope

For subject scope `r`:

#### Global LoRA

```text
text:  all subject-prompt tokens
image: all image tokens
```

#### Regional LoRA assigned to `r`

Standard regional mode:

```text
text:  the subject-description/identity portion of r's subject-only prompt;
       do not require it to modify the shared-visual prefix
image: all image tokens in this subject-only forward
```

Character-identity mode:

```text
text:  exact trigger token spans in r's subject-only prompt
image: all image tokens in this subject-only forward
```

Why all image tokens are allowed in the subject-only forward:

- this forward exists solely to predict Subject `r`;
- only the exclusive `O_r` portion is accepted into `P_fused`;
- allowing the LoRA across that forward gives identity/style deltas freedom to construct a coherent subject;
- output outside `O_r` is discarded by prediction fusion.

#### Regional LoRA not assigned to `r`

```text
text:  all zeros
image: all zeros
```

#### LoRA assigned to several regions including `r`

Treat it as assigned to `r`.

### 9.4 Runtime scope lookup

`RoutedCompositeAdapter._mask(...)` must read:

```python
CURRENT_CONDITIONING_CONTEXT.get()
```

Behavior:

- no active context:
  - preserve existing behavior for edit, face refinement, tests, and nonsemantic paths;
- full context:
  - use full-scope masks;
- subject context:
  - use that subject's scope mask;
- unknown/missing subject:
  - raise `conditioning_scope_mismatch`;
  - never fall back to global application.

### 9.5 Token-axis handling

Continue supporting all current route kinds:

- `text_layerwise`;
- `text_projector`;
- `text_refiner`;
- `combined`.

For every scope:

- use that scope's actual text-token count;
- validate the token axis exactly;
- for folded text batches, infer and validate the fold against the scope token count;
- cache masks by:
  - LoRA ID;
  - route kind;
  - scope;
  - tensor shape;
  - device;
  - dtype;
  - pose gate value where relevant.

A subject-only prompt may have a different token count from the unified prompt. No code may assume one global text-token count for every prediction scope.

### 9.6 Statistics

Extend LoRA diagnostics to report, per LoRA:

- full-scope forward calls;
- subject-scope forward calls by region ID;
- text delta RMS by scope;
- image delta RMS by scope;
- disabled-scope count;
- mismatched-scope errors.

Do not synchronize GPU tensors on every layer solely for UI logging. Aggregate as the existing statistics system does.

---

## 10. Attention isolation

Attention isolation has two uses:

1. an independent lower-cost semantic mode using the unified prompt;
2. extra protection inside each subject-only forward in Prediction composite.

### 10.1 Explicit token roles

The current owner value `0` conflates:

- scene text;
- shared visual context;
- special/padding tokens;
- relationship text;
- non-subject regional text.

Replace this ambiguity with explicit token roles.

Recommended:

```python
class PromptTokenRoleKind(StrEnum):
    SPECIAL = "special"
    SHARED_VISUAL = "shared_visual"
    SCENE = "scene"
    RELATIONSHIP = "relationship"
    SUBJECT = "subject"
    OTHER_REGION = "other_region"

@dataclass(frozen=True, slots=True)
class PromptTokenRole:
    kind: PromptTokenRoleKind
    region_id: str | None = None
```

`BoundRegionalPromptPlan` should expose one role per text token.

Special prefix/suffix/padding tokens not covered by a character span become `SPECIAL`.

### 10.2 Prevent shared tokens from becoming a bridge

During hard semantic isolation:

- shared visual tokens may be used as keys by all subjects;
- shared visual query tokens must not absorb scene or subject-specific content;
- special-token queries must not become a bridge between ownership domains.

Otherwise:

```text
scene → shared token → subject
```

would defeat isolation.

### 10.3 Attention-isolation mode using the unified prompt

During hard phase, enforce:

#### Subject image query for subject `r`

Allow keys from:

- subject `r` text;
- shared visual text;
- safe special tokens;
- subject `r` image tokens.

Block keys from:

- scene text;
- relationship text;
- other-region text;
- another subject's text;
- outside/unowned image tokens;
- another subject's image tokens.

#### Subject `r` text query

Allow:

- subject `r` text;
- shared visual keys;
- safe special keys;
- subject `r` image keys in the main stream.

Block:

- scene;
- relationships;
- other regions/subjects;
- outside image keys.

#### Scene/unowned image query

Allow:

- scene;
- relationships;
- shared visual;
- safe special tokens;
- unowned image tokens.

Block:

- subject text;
- subject-owned image keys.

#### Shared-visual query

Allow:

- shared visual;
- safe special tokens.

Do not allow it to absorb scene or subject content in the hard phase.

#### Text-refiner stage

Apply the same ownership partition before text enters the main single-stream transformer.

### 10.4 Prediction-composite subject scope

A subject-only prompt already excludes scene and other subjects. For that scope:

- all subject-prompt text may interact normally;
- image tokens inside the current subject island may attend to:
  - subject-prompt text;
  - image tokens inside the same island;
- inside image queries must not attend outside image keys during hard isolation;
- subject text queries in the main stream must not attend outside image keys during hard isolation;
- outside image output is later discarded, but it must not feed back into accepted subject tokens.

Use the subject's exclusive ownership field to define the image island.

Initial binary island threshold:

```text
ownership >= 0.50
```

Centralize this constant. Do not expose it in the version-22 GUI.

### 10.5 Hard and soft penalties

For hard phase (`g == 1`):

```text
blocked logits → -inf
```

For soft phase (`0 < g < 1`):

```text
blocked logits += -P · g
```

Use:

```text
P = 20.0
```

as a centralized first-test constant.

For normal phase:

```text
penalty = 0
```

Do not expose `P` as a user control in version 22. It is an implementation calibration constant.

The existing regional spatial biases still apply where appropriate. Semantic exclusion is evaluated before additive regional/emphasis bias so a positive regional bias cannot resurrect a blocked pair.

### 10.6 Batch awareness

The optimized attention override must read the active `ConditioningExecutionContext`.

For Prediction composite, conditional model calls are sequential per scope, so one scope should apply to the whole conditional batch.

Validate this assumption:

- if a model call contains mixed K2 scopes, raise a typed error;
- do not apply one scope's matrix to another scope.

### 10.7 Limitation of Attention isolation mode

Document clearly:

- unified-prompt attention isolation occurs after the external text encoder has already contextualized the full prompt;
- scene information may already be embedded in subject token vectors;
- therefore this mode is a faster experiment, not the strongest semantic guarantee;
- separately encoded subject prompts in Prediction composite are the recommended mode.

---

## 11. Semantic runtime controller

The existing `PoseGateController` remains the source of truth for:

- current transition;
- phase;
- gate strength;
- sigma diagnostics.

Add a `PoseSemanticRuntime` that references—not duplicates—the pose controller.

Recommended state:

```python
@dataclass
class PoseSemanticRuntime:
    mode: PoseSemanticMode
    pose_controller: PoseGateController
    plan: PoseSemanticPlan
    mask_cache: SemanticMaskCache
    diagnostics: SemanticPredictionDiagnostics
```

Do not create a second independent hard/soft schedule.

The semantic wrapper must use:

```python
g = pose_controller.gate_strength
```

The sampler callback remains the only code that advances transitions.

Repeated model evaluations within one sampler transition must all see the same:

- phase;
- `g`;
- sigma interval;
- ownership masks.

---

## 12. Runtime generation flow

Primary target:

```text
src/k2_region_lab/worker/runtime.py
```

Required order:

1. validate project/settings;
2. compile the full regional prompt plan, including shared visual spans;
3. determine enabled pose subjects;
4. build the existing volumetric mask bundle;
5. build the existing pose phase/sigma controller;
6. encode full unified conditioning;
7. when mode is Prediction composite:
   - compile each subject-only prompt;
   - encode each separately;
   - build `PoseSemanticPlan`;
8. bind full and subject prompt token spans;
9. compile scope-aware LoRA routes;
10. clone/patch the generation model once;
11. build the context-aware attention override;
12. install:
    - existing dynamic denoise-mask hook;
    - semantic conditional-batch hook for Prediction composite;
    - attention override;
13. call `comfy.sample.sample(...)` exactly once;
14. restore every hook and context in `finally`;
15. validate callback/forward counts;
16. decode/save through the existing path;
17. attach semantic runtime metadata.

### 12.1 Conceptual runtime pseudocode

```python
full_plan = compile_regional_prompt_plan(
    width=width,
    height=height,
    global_prompt=scene_prompt,
    shared_visual_prompt=shared_visual_prompt,
    regions=regions,
    ...
)

full_positive = encode_prompt(full_plan.prompt)
bound_full = bind_full_plan(full_plan, full_positive)

semantic_plan = None
if pose_gating_enabled and semantic_mode == PREDICTION_COMPOSITE:
    subject_conditionings = []
    for subject in enabled_pose_subjects:
        subject_prompt_plan = compile_subject_conditioning_prompt(
            shared_visual_prompt=shared_visual_prompt,
            region=subject,
            identity_triggers=...,
            emphases=...,
        )
        subject_positive = encode_prompt(subject_prompt_plan.prompt)
        bound_subject = bind_subject_prompt(
            subject_prompt_plan,
            subject_positive,
        )
        subject_conditionings.append(
            SubjectSemanticConditioning(
                scope=ConditioningScope.subject(subject.region_id),
                bound_prompt=bound_subject,
                conditioning=annotate_conditioning(
                    subject_positive,
                    scope=...,
                ),
                ownership_mask=mask_bundle.ownership(subject.region_id),
                ...
            )
        )

    positive = concatenate_conditioning_records(
        annotate_conditioning(full_positive, scope=FULL),
        *(subject.conditioning for subject in subject_conditionings),
    )
    semantic_plan = PoseSemanticPlan(...)
else:
    positive = full_positive
```

Then:

```python
generation_model, lora_reports, stats = apply_scope_aware_loras(...)

attention_override = KreaSpatialAttentionOverride(
    bound_full,
    pose_gate_binding=pose_gate_binding,
    semantic_mode=semantic_mode,
    semantic_plan=semantic_plan,
)

with installed_pose_gate_hook(...), \
     installed_semantic_prediction_hook_if_needed(...), \
     installed_attention_override(...):
    samples = comfy.sample.sample(
        generation_model,
        noise,
        effective_steps,
        1.0,
        sampler,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        noise_mask=pose_support_tensor,
        sigmas=resolved_sigmas,
        callback=callback,
        disable_pbar=True,
        seed=seed,
    )
```

### 12.2 Gating disabled

When pose gating is disabled:

- semantic mode has no effect;
- encode only the existing full prompt;
- install no semantic sampler hook;
- install no new semantic isolation beyond existing regional behavior;
- preserve version-21 output behavior.

### 12.3 Spatial-only mode

When pose gating is enabled and mode is Spatial only:

- use current version-21 behavior exactly;
- no subject-only prompt encoding;
- no semantic prediction hook;
- current spatial attention/LoRA behavior;
- retain for same-seed regression tests.

### 12.4 Attention-isolation mode

When pose gating is enabled and mode is Attention isolation:

- encode only the full prompt;
- use explicit token roles;
- install strict/soft semantic attention partition;
- use current single prediction;
- no per-subject prediction fusion.

### 12.5 Prediction-composite mode

When enabled:

- encode full plus each enabled posed subject;
- install prediction composite;
- use context-aware attention;
- use scope-aware LoRAs;
- skip subject predictions at `g == 0`.

---

## 13. Internal forward progress and timeout accounting

Prediction composite increases compute time but not sampler transition count.

### 13.1 Progress

Keep the primary progress bar in sampler transitions:

```text
Denoising 3/12
```

Add secondary status:

```text
Soft gate — semantic predictions 2/3
```

Internal events may report:

- full prediction started/completed;
- subject region ID prediction started/completed;
- internal forward index;
- current sampler transition;
- current semantic mode.

Do not emit one durable event for every transformer layer.

### 13.2 Timeout

The latest branch contains generation-timeout recovery. Update timeout/work estimates so Prediction composite is not falsely treated as hung.

For Euler:

```text
estimated_forward_equivalents =
    normal_steps
  + (hard_steps + soft_steps) · (1 + subject_count)
```

Use this workload estimate to scale any generation worker timeout or heartbeat expectation.

For samplers with more than one model evaluation per transition:

- do not assume the Euler formula is exact;
- continue heartbeat/status updates from the semantic wrapper;
- use a conservative multiplier based on sampler family or current observed forward rate.

Do not relax cancellation. Cancellation must be checked between scope predictions as well as between sampler transitions.

### 13.3 Memory

Subject predictions run sequentially for correctness.

Do not retain all large prediction tensors unnecessarily:

```python
fused = full * full_weight
for subject:
    prediction = calculate_subject(...)
    fused.add_(prediction * subject_weight)
    del prediction
```

Maintain whatever tensors are needed for diagnostics without retaining full copies.

---

## 14. Project schema version 22

Backend:

```text
src/k2_region_lab/project.py
```

Frontend:

```text
web/client/src/studioProject.ts
```

Set:

```text
PROJECT_VERSION = 22
```

### 14.1 New fields

Add to project state:

```json
{
  "shared_visual_prompt": "",
  "pose_semantic_mode": "prediction_composite"
}
```

No new user control is needed for the semantic attention penalty in version 22.

### 14.2 Defaults

For a brand-new version-22 project:

```text
pose_semantic_mode = prediction_composite
shared_visual_prompt = ""
```

Pose gating itself may retain its current enabled/disabled default.

### 14.3 Version-21 migration

Migration must preserve prior behavior.

For a version-21 project:

```text
shared_visual_prompt = ""
pose_semantic_mode = spatial_only
```

Add a nonfatal migration notice:

```text
This project was created before subject-semantic pose routing.
It was opened in Spatial only mode to preserve its previous behavior.
Select Prediction composite to bind gated mannequin cells to each
subject's own prompt and LoRAs.
```

Do not silently change an old project's output semantics.

### 14.4 Validation

Validate:

- shared visual prompt within the same safe text-size bounds as other prompts;
- semantic mode is a known enum;
- Prediction composite requires:
  - pose gating enabled;
  - at least one enabled posed subject;
  - every posed subject has nonempty subject or face-identity text;
- semantic subject region IDs are unique;
- subject conditioning exists for every ownership mask;
- no subject conditioning exists for a disabled/unposed region.

### 14.5 Serialization

Update:

- canonical JSON;
- cloud project copies;
- browser local/open/save;
- PNG-embedded project metadata;
- worker payload;
- version migration tests;
- exact version-22 round trips.

---

## 15. Agent, API, worker protocol, and capabilities

Update relevant versions and contracts.

Recommended capability:

```json
{
  "pose_semantic_routing": {
    "version": 1,
    "modes": [
      "spatial_only",
      "attention_isolation",
      "prediction_composite"
    ],
    "subject_prompt_encoding": true,
    "scope_aware_regional_lora": true,
    "single_sampler_trajectory": true,
    "multigpu_prediction_composite": false
  }
}
```

The job request must include:

- `shared_visual_prompt`;
- `pose_semantic_mode`;
- canonical version-22 project.

Do not send pre-encoded conditioning across process boundaries.

The GPU worker compiles and encodes prompts using the live model/tokenizer.

Version mismatches must fail through the existing explicit project/protocol/capability mechanism.

---

## 16. Web GUI

### 16.1 Prompt controls

Relabel:

```text
Global prompt → Scene prompt
```

Add below it:

```text
Shared visual context
```

Help text:

```text
Lighting, lens, medium, palette, and visual style shared by the
scene and every subject-only conditioning branch. Do not put
scene objects or background content here.
```

### 16.2 Pose semantic routing control

In `PoseGatingControls.tsx`, add:

```text
Subject semantic routing
```

Options:

#### Prediction composite — recommended

```text
Computes a clean subject-only prediction for every enabled
mannequin during hard and soft steps, then fuses it into that
subject's exclusive pose volume. Slower but strongest.
```

#### Attention isolation

```text
Uses one unified prediction but blocks scene/other-subject
attention into each mannequin during gated steps. Faster,
but the unified text encoding can still carry some leakage.
```

#### Spatial only — legacy diagnostic

```text
Restricts where denoising occurs without restricting what
the unified scene prompt generates there.
```

Show Spatial only with a warning icon when gating is enabled.

### 16.3 Work estimate

When Prediction composite is selected, show:

```text
Estimated Euler model forwards:
{normal} + ({hard} + {soft}) × (1 + {subjects}) = {total}
```

Also show:

```text
One sampler trajectory; subject predictions are evaluated sequentially.
```

Update live when subject count or step count changes.

Do not claim exact timing for non-Euler samplers.

### 16.4 Conditioning preview

Extend preview UI to show tabs/sections:

- Full scene;
- each subject by region name.

Show token counts when returned by the backend preview endpoint.

### 16.5 Validation before submit

Block submit when Prediction composite is selected and:

- no posed subject exists;
- a posed subject has no descriptive or identity prompt;
- project/backend capability lacks semantic routing;
- current worker protocol is too old.

Do not silently switch modes.

### 16.6 Mobile/responsive behavior

Keep controls usable on the existing mobile layout:

- selector does not overflow;
- prompt preview scrolls;
- forward estimate wraps cleanly;
- no graph changes are required.

---

## 17. Error handling

Create typed exceptions:

```python
class PoseSemanticError(RuntimeError): ...
class SubjectConditioningCompileError(PoseSemanticError): ...
class ConditioningScopeMismatchError(PoseSemanticError): ...
class SemanticSamplerHookError(PoseSemanticError): ...
class SemanticPredictionShapeError(PoseSemanticError): ...
class SemanticAttentionError(PoseSemanticError): ...
class SemanticLoraRoutingError(PoseSemanticError): ...
class SemanticMultiGpuUnsupportedError(PoseSemanticError): ...
```

Stable browser codes:

- `subject_conditioning_compile_failed`;
- `conditioning_scope_mismatch`;
- `semantic_sampler_hook_incompatible`;
- `semantic_prediction_shape_invalid`;
- `semantic_attention_failed`;
- `semantic_lora_routing_failed`;
- `semantic_prediction_composite_multigpu_unsupported`.

Requirements:

- retain full traceback in private job diagnostics;
- safe browser message includes diagnostic ID;
- no prompt text in public events;
- no LoRA compatibility claim unless actual LoRA inspection established it;
- hook/context cleanup occurs after every error;
- no fallback to Spatial only after a semantic error.

---

## 18. Runtime diagnostics and output metadata

Add:

```json
{
  "pose_semantic_runtime": {
    "version": 1,
    "mode": "prediction_composite",
    "shared_visual_prompt_sha256": "...",
    "full_prompt_sha256": "...",
    "subjects": [
      {
        "region_id": "...",
        "region_name": "...",
        "prompt_sha256": "...",
        "text_token_count": 123,
        "ownership_coverage": 0.12,
        "forward_calls": 4,
        "prediction_delta_rms_inside_ownership": 0.18
      }
    ],
    "full_forward_calls": 12,
    "subject_forward_calls": 8,
    "total_conditional_forward_calls": 20,
    "semantic_isolation": {
      "hard_blocks": 1234,
      "soft_penalty": 20.0,
      "image_to_image_isolation": true
    },
    "lora_scope_summary": {},
    "multigpu": false
  }
}
```

Also retain the existing pose-gating runtime block.

### 18.1 Prediction diagnostics

For each subject, calculate without storing full prediction tensors:

```text
RMS(P_subject - P_full) inside ownership
RMS(P_subject - P_full) outside ownership, optional diagnostic
mean ownership weight
maximum summed ownership weight
```

These values answer whether subject conditioning materially changed the denoising direction.

Synchronize only once per model evaluation or aggregate on GPU and synchronize at completion.

### 18.2 Events

Emit:

- `pose_semantic_conditioning_prepared`;
- internal progress/heartbeat events;
- `pose_semantic_conditioning_completed` only after successful sampling.

Do not emit completion from `finally`.

---

## 19. File-level implementation checklist

Codex must perform repository-wide search, but expected targets include:

### New Python module

- `src/k2_region_lab/semantic_conditioning.py`

### Existing Python

- `src/k2_region_lab/regional_prompting.py`
  - shared visual context;
  - full prompt token roles;
  - subject-only compiler/binder;
  - prompt preview summaries.
- `src/k2_region_lab/spatial_attention.py`
  - explicit token roles;
  - context-aware semantic matrices;
  - image-to-image isolation;
  - hard/soft behavior.
- `src/k2_region_lab/regional_lora.py`
  - full and subject scope masks.
- `src/k2_region_lab/worker/runtime.py`
  - separate encodes;
  - semantic plan;
  - sampler conditional-batch hook;
  - fusion;
  - diagnostics.
- `src/k2_region_lab/pose_gating.py`
  - only if needed to expose shared controller/binding cleanly.
- `src/k2_region_lab/project.py`
  - version 22/migration.
- `src/k2_region_lab/worker/protocol.py`
  - typed errors/version.
- `src/k2_region_lab/worker/entrypoint.py`
  - payload.
- `src/k2_region_lab/agent/domain.py`
  - capabilities/contracts.
- `src/k2_region_lab/agent/jobs.py`
  - job submission and timeout/work estimate.
- current metadata/output writer.

### Browser

- `web/client/src/studioProject.ts`
- `web/client/src/api.ts`
- `web/client/src/WorkspaceStudio.tsx`
- `web/client/src/components/PoseGatingControls.tsx`
- prompt editor component containing current global prompt
- prompt preview component
- CSS
- browser project/UI contract scripts.

### Documentation

- `README.md`
- `docs/subject_pose_control.md`
- `docs/web_desktop_parity.md`

Search all references to:

```text
PROJECT_VERSION
WORKER_PROTOCOL_VERSION
global_prompt
Unified spatial prompt
image_to_image_attention
text_region_ownership
PoseGateRegionBinding
sampler_calc_cond_batch_function
RoutedCompositeAdapter
LoraDeltaRoute
pose_semantic
```

Every remaining old behavior must be intentional.

---

## 20. Automated tests

### 20.1 Prompt compilation

Test:

- shared context included in full and each subject prompt;
- scene prompt excluded from subject-only prompts;
- another subject excluded;
- relationships excluded;
- face-identity prompt included only for its subject;
- identity triggers included only for their subject;
- subject-owned held object retained;
- deterministic output and hash;
- exact token span binding;
- subject emphasis mapping;
- empty shared context;
- punctuation/whitespace edge cases.

Use a hostile fixture:

```text
Scene:
A greenhouse crowded with enormous tropical leaves and red sofas.

Shared:
Cinematic realistic photography, 35 mm lens.

A:
A white astronaut raising the right arm.

B:
A medieval knight crouching in black plate armor.
```

Assert no greenhouse/leaves/sofas/knight text enters A's prompt.

### 20.2 Fusion math

Use fake tensors:

```text
P_full = 1
P_A = 10
P_B = 20
```

Test:

- hard core exactly selects subject;
- feather blends;
- soft `g`;
- normal equals full;
- exclusive masks;
- masks summing over one are normalized;
- dtype/device;
- batch broadcasting;
- shape mismatch errors.

### 20.3 Conditional-batch hook

With a fake `calc_cond_batch`:

- full evaluated once;
- every subject evaluated once when `g > 0`;
- no subject evaluated at `g == 0`;
- output order compatible with CFG function;
- no recursion;
- current transition not advanced;
- cancellation between subject forwards;
- prior hook restored;
- exception restores hook/context;
- existing hook rejected.

### 20.4 Scope context

Test:

- full context;
- subject context;
- nested context restoration;
- exception restoration;
- unknown subject failure;
- no stale context after a job.

### 20.5 LoRA routing

For global, Subject A, and Subject B LoRAs:

#### Full scope

Assert current version-21 masks remain unchanged.

#### Subject A scope

Assert:

- global LoRA active;
- A LoRA active;
- B LoRA zero;
- A identity trigger span exact;
- image mask all one for A's subject forward.

#### Subject B scope

Symmetric.

Test:

- varying subject/full text lengths;
- folded layerwise text path;
- projector;
- text refiner;
- combined sequence;
- cache keys include scope;
- statistics by scope;
- scope mismatch typed error.

### 20.6 Attention matrices

Use small synthetic token/image layouts.

Test Attention isolation hard phase:

- A image query cannot attend scene text;
- A image query cannot attend B text;
- A image query cannot attend outside image;
- A text cannot attend scene/outside image;
- shared query cannot bridge;
- scene query cannot attend A;
- permitted pairs remain finite.

Test soft:

- blocked pair receives `-20 · g`;
- penalty decreases monotonically;
- no `-inf` once `0 < g < 1`, unless another permanent regional rule requires it.

Test normal:

- existing regional attention behavior restored.

Test Prediction composite subject scope:

- subject text available;
- inside image isolated from outside image;
- subject scope does not use full token-role arrays.

### 20.7 Project/protocol

Test:

- version-22 exact round trip;
- version-21 migration to Spatial only;
- migration notice;
- new-project default Prediction composite;
- shared prompt persistence;
- PNG/cloud persistence;
- API payload;
- capability mismatch;
- no silent fallback.

### 20.8 Timeout/work estimate

Test:

- one/two/three subjects;
- zero hard/soft;
- Euler formula;
- non-Euler marked estimate;
- worker heartbeat during long subject sequence.

### 20.9 Worker integration with fake Comfy

Mock the pinned APIs and assert:

- one `comfy.sample.sample` call;
- multiple internal `calc_cond_batch` calls;
- explicit sigmas unchanged;
- denoise mask unchanged;
- context-aware LoRA/attention installed;
- all hooks restored;
- metadata counts exact.

### 20.10 Existing suite

Run:

```bash
uv run pytest -q
uv run ruff check .
```

Browser:

```bash
cd web/client
npm run test:project
npm run test:ui
npm run typecheck
npm run build
```

Do not report success based only on new tests.

---

## 21. RunPod GUI acceptance tests

Use Euler for first diagnostics because it gives one model evaluation per sampler transition and makes expected forward counts clear.

Use the same model, LoRAs, seed, resolution, mannequin state, hard/soft/normal steps, and sigma schedule across mode comparisons.

### 21.1 Test A: reproduce the defect

Mode:

```text
Spatial only
```

Scene:

```text
A tropical greenhouse crowded with enormous green leaves,
red sofas, glass walls, and potted plants.
```

Subject:

```text
A full-body astronaut in a clean white spacesuit, standing
upright with the right arm raised and left arm lowered.
```

Expected diagnostic purpose:

- establish that scene content can occupy the mannequin;
- record output, timing, metadata.

### 21.2 Test B: Attention isolation

Use identical seed/settings.

Confirm:

- one model prediction per sampler evaluation;
- attention diagnostics report text and image isolation;
- scene intrusion is lower than Spatial only;
- no crash in regional LoRA paths.

Do not treat this result as final if leakage remains.

### 21.3 Test C: Prediction composite

Use identical seed/settings.

Confirm:

- full and subject prompt previews are correct;
- subject prompt contains no greenhouse/leaves/sofas;
- expected Euler forward count is exact;
- subject prediction delta RMS is nonzero;
- astronaut appears inside the mannequin volume rather than scene content;
- normal phase integrates scene/background after release.

### 21.4 Test D: two-subject prompt swap

Scene:

```text
A tropical greenhouse with glass walls and wet stone flooring.
```

Shared:

```text
Realistic cinematic photography, soft window light, 35 mm lens.
```

Left:

```text
A white astronaut standing with the right arm raised.
```

Right:

```text
A medieval knight crouching in black plate armor.
```

Assign distinct regional character LoRAs.

Generate, then swap only:

- subject prompts;
- their regional LoRA assignments.

Keep:

- seed;
- mannequins;
- scene;
- sigma schedule.

Success criterion:

- astronaut and knight semantics/identities exchange mannequin ownership;
- greenhouse remains the scene;
- left/right pose volumes do not exchange.

### 21.5 Test E: adversarial scene object

Scene:

```text
A room dominated by a giant red sofa.
```

Subject:

```text
A standing person in a blue coat with both arms extended.
```

Hard-gate mannequin overlaps the visual center where a sofa would ordinarily form.

Success criterion for Prediction composite:

- accepted hard-phase subject volume is pulled toward a person, not a sofa;
- later normal steps may place/integrate a sofa outside or behind it.

### 21.6 Test F: LoRA isolation

Use two visibly distinct identity LoRAs.

Confirm metadata and output show:

- A LoRA has subject-scope calls only in A plus full-scope regional calls;
- B likewise;
- B LoRA has zero delta in A's subject-only forward;
- global style LoRA is active in full and both subject forwards.

### 21.7 Test G: soft release

Compare:

- 2 hard / 0 soft / 8 normal;
- 2 hard / 2 soft / 8 normal;
- 2 hard / 4 soft / 8 normal.

Confirm:

- subject semantics established in hard phase;
- increasing soft span changes integration, not subject identity ownership;
- no discontinuous boundary artifact at hard→soft or soft→normal.

### 21.8 Test H: gating disabled

Confirm existing ordinary generation remains functional and does not incur subject-only forward cost.

---

## 22. Acceptance criteria

### Functional

- Scene content and subject content are represented by separate prompt fields.
- Subject-only prompts are separately encoded in Prediction composite.
- Subject-owned cells receive subject-only predictions during hard gating.
- Soft phase blends subject and full predictions through the same gate strength.
- Normal phase uses the existing full prediction only.
- Regional LoRAs are scope-aware.
- Cross-boundary image-to-image attention is controlled during semantic isolation.
- One sampler call and one latent trajectory are preserved.
- All prior pose/sigma controls remain functional.

### Diagnostic

- Spatial only, Attention isolation, and Prediction composite can be selected.
- Same-seed comparison is possible.
- Runtime records exact full/subject forward counts.
- Runtime records prompt hashes/token counts and prediction-delta measurements.
- No silent semantic fallback.

### Compatibility

- Version-21 projects open in Spatial only mode.
- Version-22 projects round-trip exactly.
- Gating-disabled generation preserves prior behavior.
- Edit and face-refinement paths are unchanged unless explicitly sharing improved prompt/compiler primitives.
- RunPod worker timeout accounts for added forward work.

### Quality gate

The implementation is ready for empirical evaluation when:

- Test C completes from the RunPod GUI;
- Test D completes with two distinct regional LoRAs;
- logs prove correct scope execution;
- output proves the intended subject category is being generated in the mannequin cells more reliably than Spatial only.

Do not claim the pose itself is solved merely because subject semantics are solved.

---

## 23. Interpretation of test outcomes

### Correct subject, incorrect limb arrangement

Meaning:

- semantic routing works;
- current volumetric hard gate is insufficient as a learned pose representation.

Next possible work:

- tune hard/soft/sigma/mask geometry;
- then consider a Krea-native volumetric pose adapter.

A Control-LoRA is relevant only at this point.

### Scene content still appears inside subject in Prediction composite

Investigate, in order:

1. subject-only prompt accidentally contains scene text;
2. scope grouping is wrong;
3. fusion weights/ownership are wrong;
4. full prediction remains weighted in subject core;
5. subject LoRA inactive;
6. subject image queries still attend outside image keys;
7. `PoseGateController` advanced before all scope predictions completed.

Do not jump directly to pose-model training.

### Subject category correct but identity wrong

Investigate scope-aware character LoRA routing and trigger spans.

### Strong subject but poor scene integration

Adjust:

- soft-step count;
- soft release;
- sigma allocation;
- normal trajectory share.

Do not weaken semantic correctness by returning to Spatial only.

---

## 24. Explicitly deferred

Do not implement in this milestone:

- pose ControlNet/Control-LoRA training;
- img2img;
- depth maps;
- 3D mannequin rendering;
- DensePose/IUV/normals;
- background-first generation;
- second sampler pass;
- subject crop diffusion;
- batched subject predictions;
- multi-GPU Prediction composite;
- automatic semantic segmentation scoring;
- automatic retry/rerank;
- user-adjustable semantic attention penalty;
- user-authored per-subject shared-visual context;
- exact limb front/back ordering.

---

## 25. Primary implementation references

### K2Lab tested baseline

- Repository branch:  
  `https://github.com/soomrenald/k2lab_runpod/tree/k2lab_pose`
- Tested commit:  
  `cb3159f4fde88a9e0ae2afc638230241900aafa5`
- Worker runtime:  
  `src/k2_region_lab/worker/runtime.py`
- Pose gating:  
  `src/k2_region_lab/pose_gating.py`
- Spatial attention:  
  `src/k2_region_lab/spatial_attention.py`
- Regional prompting:  
  `src/k2_region_lab/regional_prompting.py`
- Regional LoRA:  
  `src/k2_region_lab/regional_lora.py`
- Project schema:  
  `src/k2_region_lab/project.py`
- Workspace ComfyUI pin:  
  `Dockerfile.workspace`

### Pinned ComfyUI interfaces

At the tested baseline:

```text
COMFYUI_REF=285a98944c397a4a81f15ac63d69fa3dbc0a27b9
```

Relevant files:

- `comfy/samplers.py`
  - `sampling_function`;
  - `sampler_calc_cond_batch_function`;
  - `calc_cond_batch`;
  - `KSamplerX0Inpaint`;
  - condition-mask preparation.
- `comfy/sampler_helpers.py`
  - `convert_cond`;
  - conditioning metadata copy/UUID.
- `comfy/model_patcher.py`
  - sampler conditional-batch hook setters.
- `comfy/sample.py`
  - one-call sampling API.

Verify the exact live pin before implementation.

### Methodological reference

MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation, ICML 2023.

The relevant principle is to obtain multiple prompt-specific denoising directions at the same diffusion state and reconcile them spatially with per-pixel weights, while preserving one global trajectory.

K2Lab's implementation differs by:

- using the same full Krea latent for every prediction;
- using exclusive volumetric mannequin ownership;
- blending only during hard/soft phases;
- retaining the existing regional prompt/LoRA system;
- using one full-scene default path plus subject-only paths.

---

## 26. Final Codex report template

```text
Starting branch/commit:
Ending branch/commit:
Pinned ComfyUI commit verified:

Implemented:
- ...

Semantic modes:
- Spatial only:
- Attention isolation:
- Prediction composite:

Prompt/schema changes:
- ...

Regional LoRA scope changes:
- ...

Attention changes:
- ...

Sampler hook/fusion:
- ...

Timeout/progress changes:
- ...

Migrations:
- ...

Tests:
- command → result

Browser:
- command → result

GPU-dependent tests not run:
- ...

Workspace image build/publish:
1. ...
2. ...

RunPod test procedure:
1. ...
2. ...

Measured first-run forward counts/timing:
- ...

Known limitations:
- ...

Deviations from specification:
- ...
```

Do not substitute a new planning document for the requested implementation.
