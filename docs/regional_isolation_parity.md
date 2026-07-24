# Regional control parity evidence

The browser/RunPod port must preserve the regional-control method from the approved
PySide6 backend. UI similarity is not evidence of model-path parity.

## Frozen reference

The reference is the complete source tree at k2lab commit
`9170f1fadffa4d380601e3c88ac0e982c09e88d8`.

## Required behavior

1. All active region descriptions are compiled into one unified scene prompt.
2. Pixel boxes rasterize to Krea image-token cells.
3. Subject text is private to its assigned subject box and other subject clauses.
4. Global text remains available to every image cell.
5. Image-to-image attention is never partitioned by rectangular region ownership.
6. Image-edit clauses receive soft spatial guidance without taking hard subject ownership.
7. A regional LoRA remains unfused and its direct delta is zero outside its token gate.
8. Standard regional LoRAs omit broadcast-prone main-stream key/value targets.
9. Runtime metadata reports `subject_text_private_to_box` and unmodified image-to-image
   attention.

Hard image-to-image ownership is a regression: it disconnects the shared latent into
rectangular attention islands and produces seams, lighting discontinuities, and broken
composition.

## Reproducible tensor oracle

Run in the Torch-enabled ComfyUI environment:

```bash
PYTHONPATH=src /path/to/comfy-python scripts/verify_regional_reference.py \
  --reference-repo ../krea_region_project \
  --reference-revision 9170f1fadffa4d380601e3c88ac0e982c09e88d8
```

The comparison requires:

- byte-for-byte equality for the regional prompt and LoRA route compilers;
- exact equality with the reference text/image ownership vectors;
- exact equality with the reference main-stream partition matrix; and
- no changes to any image-to-image attention score.

A release candidate must additionally complete a fixed-seed GPU validation and retain its
PNG, project JSON, runtime events, and parity report as deployment evidence.
