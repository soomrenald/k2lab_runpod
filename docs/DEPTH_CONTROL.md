# Depth control

K2Lab accepts 8-bit grayscale PNG, 16-bit grayscale PNG, and 16-bit grayscale
TIFF depth maps. RGB and alpha-bearing images are rejected instead of silently
being reinterpreted.

Depth is opt-in at both deployment and request level. All deployment flags
default to false:

```text
K2_DEPTH_CONTROL_ENABLED
K2_DEPTH_REGIONS_ENABLED
K2_DEPTH_OVERRIDE_ENABLED
K2_BLENDER_BUNDLE_IMPORT_ENABLED
```

The control plane advertises those values to the browser. Global and regional
controls remain hidden unless their respective flags are enabled, and the agent
rechecks the flags when resolving every job.

The currently trusted adapter is
`Patil/Krea-2-depth-controlnet/depth-control-lora.safetensors`, SHA-256
`fb80547ed79b47c1e3fea7bb9d36297e3917b2115fab6700ca1501350f9f483c`.
It is subject to the Krea 2 Community License. K2Lab verifies the exact hash,
one expanded 128-channel input projection, rank 64, and all eight adapter
targets in all 28 Krea blocks before loading it.

The checkpoint expects inverse depth: near is white and far is black. The
default K2Lab UI uses min/max normalization with no inversion. Percentile
normalization is useful for isolated outliers. Convention changes are always
explicit and persisted.

The control image is normalized and resized once, repeated to grayscale RGB,
encoded once with the Krea/Qwen image VAE, and attached to the same integrated
denoising pass used by prompts and regional LoRAs. The native image projection
remains intact; the adapter contributes only its control-token half. Projection
state is installed on a cloned model patcher and restored on cleanup.

Raw defaults in the standalone harness are 28 steps and CFG 3.5. Turbo defaults
are 8 steps and CFG 0. The public model card reports support for both. K2Lab’s
normal Turbo path remains unchanged when depth is disabled.

Example request fragment:

```json
{
  "depth_control": {
    "enabled": true,
    "checkpoint": "depth-control-lora.safetensors",
    "depth_image": "depth_16bit.png",
    "global_strength": 1.0,
    "start_percent": 0.0,
    "end_percent": 1.0,
    "invert": false,
    "normalization": {"mode": "minmax", "gamma": 1.0},
    "feather_pixels": 32,
    "regions": []
  }
}
```

Depth and the experimental volumetric pose Control-LoRA cannot be enabled
together because both own Krea’s expanded input projection. K2Lab returns an
explicit conflict instead of choosing one silently. Ordinary global and
regional style/character LoRAs remain supported.

Depth controls coarse geometry, silhouette, scale, perspective, and occlusion.
It does not guarantee exact anatomy, hands, fingers, or joint angles. Strong or
contradictory prompts and LoRAs can fight the depth condition.
