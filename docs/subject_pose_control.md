# Volumetric subject pose gating

K2 has two intentionally different box types:

- **Region boxes** have no mannequin. Use them for props, non-person objects, scenery,
  architecture, or background bands. They continue to control prompts, spatial attention,
  and regional LoRA routing.
- **Subject boxes** own a filled 13-joint mannequin plus an editable head ellipse. They retain
  the same regional behavior and can define an early denoising volume and exclusive
  prompt/LoRA ownership.

No pose ControlNet is installed or loaded. Version-20 projects migrate their old 18-point
mannequins into the volumetric format, discard incompatible Qwen pose-ControlNet settings, and
leave global pose gating disabled until the user enables it.

## Pose a subject

1. Choose **Draw subject** and draw a box around the intended person.
2. Select the box. Use **Standing**, **Squatting**, or **Mirror** as a starting point.
3. Pose the mannequin on the canvas:
   - drag a circular joint handle for individual articulation;
   - drag a pink diamond to move an entire arm or leg without changing its internal shape;
   - drag the gold torso-center handle to move the torso and attached head together;
   - drag the cream rotation handle above the torso to rotate the entire figure;
   - drag the head’s center control to move it, or its right/bottom controls to change its
     horizontal and vertical radii.
4. A limb may extend outside the subject box for interactions with another subject or object.
5. Open **Advanced → Volumetric pose gating** and enable
   **Constrain generation to subject mannequins**.

Ordinary regions never add mannequin gating. Disable **Enable this mannequin** on an individual
subject to leave that person prompt-directed while other mannequins remain active.

## Gate phases

The existing **Steps** value is the number of normal unrestricted transitions. **Hard gate
steps** and **Soft gate steps** are added before them:

```text
Hard 2 + Soft 2 + Normal 8 = 12 total transitions
```

During hard steps, predicted denoising updates are accepted inside the union of the filled
mannequin support volumes. Each subject prompt and regional/character LoRA is routed through
that subject’s exclusive mannequin ownership. During soft steps, the gate opens according to
the selected cosine, linear, exponential, or stepped release. Normal steps restore the existing
regional fields exactly.

This is one continuous text-to-image sampler run with one initial noise tensor. There is no
background pass, decode/re-encode boundary, or re-noising restart.

## Sigma scheduling

- **Scheduler default** uses the selected ComfyUI sampler/scheduler curve for the effective
  transition count without warping. Start here.
- **Phase weighted** assigns normalized trajectory shares to hard, soft, and normal phases.
  Balanced, Pose lock, and Gentle presets are provided.
- **Advanced curve** exposes every normalized transition boundary in a graph and numeric table.
  It changes scheduler trajectory allocation, not gate strength.

Nondefault sigma allocation is experimental with Turbo models and can materially affect image
quality. The completed output records baseline/resolved sigmas, normalized positions, phase
shares, gate strengths, and mask coverage in `pose_gating` PNG metadata.

## Diagnostics

Job progress identifies the current hard, soft, or normal phase, gate strength, current/next
sigma, and normalized trajectory progress. Extremely small or large mannequin coverage produces
a warning. A pose-gated generation is blocked when no enabled subject mannequin exists.

Disabling global pose gating preserves the previous generation path: no dynamic denoise-mask
hook, no explicit pose sigma schedule, and no additional sampler transitions.
