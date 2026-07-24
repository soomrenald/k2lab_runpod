# Regional isolation parity evidence

The browser/RunPod port must preserve the regional-control method. UI parity alone is not
evidence of model-path parity.

## Frozen references

- Original LoRA-delta tracking and asymmetric attention implementation:
  `krea_reg_lora` commit `fca6adb4f68bcfbab8450667987001822a540a35`.
- Desktop hard text/image ownership implementation immediately before hard regional image
  boundaries were removed: `krea_region_project` commit
  `617d0aa27fad6d76a0fce3b7d8c60e33a4875db7`.
- Qt Quick migration boundary: `093ec16c79487f572be0931c07f2eb1d469317fd`.
  That commit changed presentation files only. Its parent is not used as the sole model-path
  oracle because the earlier regional regression was already present there.

## Required behavior

1. Pixel boxes rasterize to Krea image-token cells.
2. A regional LoRA's direct delta is zero outside its image-token gate.
3. Actual relative LoRA deltas mark modified image tokens with sticky retention.
4. Outside image queries receive the original asymmetric penalty when reading modified keys.
5. Cross-LoRA image attention receives its separate penalty.
6. Regional text and image keys are inaccessible to other owners, including through
   image-to-image attention.
7. Subject and image-edit boxes use the same ownership contract.
8. Generated PNG metadata reports the live attention implementation instead of a hard-coded
   summary.

## Reproducible tensor oracle

Run in the Torch-enabled ComfyUI environment:

```bash
PYTHONPATH=src /path/to/comfy-python scripts/verify_regional_reference.py \
  --reference-repo ../krea_reg_lora \
  --desktop-reference-repo ../krea_region_project
```

The comparison executes the original and restored implementations on identical tensors. It
requires exact equality for every modified-token flag, every entry in the asymmetric
attention-bias matrix, and the desktop hard ownership matrix. A tolerance-based or visual-only
match is not accepted.

The release is not approved for deployment until the same candidate image also completes a
fixed-seed GPU validation and preserves its PNGs, project JSON, runtime events, and comparison
report.
