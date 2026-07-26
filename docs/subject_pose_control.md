# Subject mannequin pose control

K2 has two intentionally different box types:

- **Region boxes** have no mannequin. Use them for props, non-person objects, scenery,
  architecture, or background bands. They continue to control prompt, spatial attention, and
  regional LoRA routing.
- **Subject boxes** own an articulated 18-joint mannequin. They retain the same regional prompt,
  attention, and LoRA behavior, and can additionally contribute pose conditioning.

Pose conditioning is generation-only. Existing version-19 projects migrate every old box to an
ordinary region, so loading an older project never silently turns on a new conditioning path.

## Install the pose model

Use **Transfers → Hugging Face**, enter
`InstantX/Qwen-Image-ControlNet-Union`, select
`diffusion_pytorch_model.safetensors`, and install it into **ControlNet models**. The upstream
Apache-2.0 model is approximately 3.54 GB:

<https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union>

The same file can be uploaded through **Assets → ControlNet models**. Model data lives in the
persistent workspace volume and survives a normal Pod image update.

## Pose a subject

1. Choose **Draw subject** and draw a box that covers the person's intended full-body extent.
2. Select the box. Use **Standing**, **Squatting**, or **Mirror** as a starting point.
3. Drag the visible joints on the canvas. A joint may extend outside its subject box, which is
   useful for a hand touching another subject or object.
4. In **Advanced → Subject pose control**, enable conditioning and select the installed model.
5. Start with strength `0.75`, start `0.0`, and end `0.75`.

All enabled subject mannequins are rendered into one full-canvas OpenPose map. K2 sends that one
map through one ControlNet pass. It does not run a separate pose generation inside each box and
does not clip limbs at box boundaries. Regional prompts and regional/character LoRAs continue to
use the existing soft spatial attention and delta-routing paths.

Practical control ranges:

- Lower strength (`0.35`–`0.6`) allows more natural deviation from the mannequin.
- The default (`0.75`) is a balanced starting point.
- Higher strength (`0.9`–`1.2`) follows joints more strictly but can reduce anatomical
  naturalness.
- Ending around `0.65`–`0.8` establishes composition early and lets later denoising restore
  texture and detail.
- Ending near `1.0` keeps enforcing pose into the final steps and can look rigid.

Disable a subject's **Enable this mannequin** checkbox when that person should remain
prompt-directed while other subject mannequins stay active. Disable the global advanced option
to run the exact non-ControlNet generation path.

## Memory behavior

The ControlNet is loaded only for a generation that has pose conditioning enabled and at least
one enabled subject mannequin. After sampling, K2 explicitly unloads that ControlNet and clears
its GPU allocation. The **Keep baseline model loaded between runs** option applies only to the
baseline model; it does not retain the pose model.

PNG metadata records the selected model, strength, denoising interval, subject count, joint
count, and connection count under `pose_conditioning`.
