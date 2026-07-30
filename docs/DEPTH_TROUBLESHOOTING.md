# Depth troubleshooting and rollback

- “Feature disabled”: enable only the required deployment flag and restart the
  worker. Flags default off.
- “Checkpoint incompatible”: use the documented public file with the exact
  SHA-256. Renamed or modified files are not trusted.
- “Single-channel grayscale required”: export PNG/TIFF as BW, without RGB or
  alpha.
- “Fully constant”: adjust Blender clip planes or scene visibility. K2Lab does
  not silently run an ineffective control.
- Reversed foreground/background: inspect the preview. Enable inversion only
  when near objects are dark.
- Weak structure: raise strength gradually, use a wider useful depth range, and
  reduce conflicting prompt or LoRA strength.
- Rectangular regional artifacts: increase feather pixels and avoid mutually
  inconsistent local camera geometry.
- Adapter conflict: disable volumetric pose Control-LoRA; both adapters cannot
  own the expanded projection simultaneously.
- Raw/Turbo error: use the documented mode defaults and a native ComfyUI Krea 2
  model with the Krea/Qwen image VAE.

Immediate rollback requires no data migration: set all four `K2_DEPTH_*` /
`K2_BLENDER_*` flags to false and restart the worker. Old v23 and earlier
projects load with depth disabled. Depth-disabled requests do not inspect or
load the checkpoint, do not encode a control image, do not install wrappers,
and do not add transformer forwards.

If a depth job fails, its cloned model patcher is discarded. Projection tokens
and spatial fields are cleared in `finally`/cleanup paths, so a following normal
job uses the untouched baseline model.
