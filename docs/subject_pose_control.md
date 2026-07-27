# Subject-semantic pose conditioning

K2Lab project schema 22 separates two independent controls:

- volumetric pose gating decides where a denoising update may be accepted;
- subject-semantic routing decides which conditioning produces the update accepted by each
  mannequin.

The Generation inspector calls the ordinary global field **Scene prompt**. Put environments,
furniture, landscape, weather, relationships, and background content there. **Shared visual
context** is copied into the full-scene and subject-only prompts; use it only for medium,
lighting, lens, palette, film treatment, and other visual style shared safely by every branch.

## Routing modes

**Prediction composite** is the default for new schema-22 projects. During hard and soft
steps, K2 evaluates one full-scene prediction and one separately encoded prediction for each
enabled posed subject at the same latent and sigma. Exclusive mannequin ownership masks fuse
the predictions. Hard cores use the subject branch, soft steps blend back toward the scene,
and normal steps evaluate only the full scene. Regional LoRAs are scope-aware: a subject
branch receives global LoRAs plus LoRAs assigned to that subject, never another subject's
regional LoRA.

**Attention isolation** uses one unified prediction. During gated phases it prevents scene,
relationship, other-subject, and outside-image attention from entering each subject island.
It is faster, but the external text encoder has already contextualized the unified prompt, so
it is not as isolated as Prediction composite.

**Spatial only** preserves schema-21 behavior for regression comparisons. It crops acceptance
spatially but does not bind the accepted prediction to a subject prompt. Schema-21 projects
migrate into this mode intentionally.

All three modes retain one seeded noise tensor, one latent canvas, one decreasing sigma
trajectory, one sampler call, and one final VAE decode. Prediction composite adds internal
model evaluations; it does not start another sample.

For Euler with `S` posed subjects:

```text
estimated forwards = normal + (hard + soft) × (1 + S)
```

The GUI shows this estimate and keeps its main progress display in sampler transitions.
Output PNG metadata contains `pose_semantic_runtime` with prompt hashes, token counts,
ownership coverage, full/subject forward counts, and subject-vs-full prediction delta RMS.
Authored prompt text is not added to public progress events.

## Same-seed comparison

Keep the model, LoRAs, seed, canvas, mannequin geometry, phase counts, release schedule, and
sigma schedule unchanged. Generate in Spatial only, Attention isolation, and Prediction
composite. For the strongest diagnostic, use a scene containing a dominant object and a
subject prompt describing a visibly different category, then confirm the Prediction
composite metadata reports the expected Euler forward count and a nonzero subject delta.

Prediction composite currently targets one GPU. If ComfyUI multi-GPU clones are active, the
worker fails explicitly instead of falling back to Spatial only.
