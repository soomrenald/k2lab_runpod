# Depth calibration

Generate a preprocessing grid without loading Krea:

```bash
python -m k2lab.depth.calibrate \
  --depth reports/blender-reference/depth_16bit.png \
  --output reports/depth-calibration
```

The grid compares min/max and percentile normalization, both conventions, and
gamma 0.7, 1.0, and 1.4. Start with min/max, no inversion, gamma 1.0 for the
Blender exporter.

Inspect the exact checkpoint, input encoding, histogram, schedule, and token
layout:

```bash
python -m k2lab.depth.validate \
  --base-model /models/krea2_turbo_fp8_scaled.safetensors \
  --text-encoder /models/qwen3vl_4b_fp8_scaled.safetensors \
  --vae /models/qwen_image_vae.safetensors \
  --depth-checkpoint /models/depth-control-lora.safetensors \
  --depth-image depth_16bit.png \
  --output reports/depth-validation \
  --mode turbo --inspect-only
```

Remove `--inspect-only` on an accelerator worker to generate an image. For a
response matrix, keep prompt, seed, model, and dimensions fixed and compare:
correct depth, shuffled depth, inverted depth, horizontally shifted depth, and
a near-constant control. Evaluate output-depth rank correlation, edge
alignment, silhouette overlap, prompt similarity, visual quality, runtime, and
peak VRAM. Do not select a convention from correlation alone.

Raw uses 28 steps and CFG 3.5 by default; Turbo uses 8 steps and CFG 0. Record
both separately. Calibration reports should only recommend a preprocessing
choice after the correct control beats shuffled and neutral controls and visual
inspection confirms the orientation.
