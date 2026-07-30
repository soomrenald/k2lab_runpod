# Regional depth weighting

Regional depth settings modulate one global control image inside the one normal
Krea denoising trajectory. K2Lab never generates regions separately or
composites independent diffusion results.

Modes:

- `inherit`: use the global strength.
- `emphasize`: multiply local influence by 1–3.
- `relax`: multiply local influence by 0–1.
- `ignore`: reduce explicit depth influence to zero.
- `override`: blend a separate normalized depth image through the region mask;
  hidden unless `K2_DEPTH_OVERRIDE_ENABLED=true`.

Final explicit control strength is clamped to the configured safe range,
normally 0–3. Global and per-region schedules use inclusive normalized progress
from 0 to 1. A field is prepared once for each sampler transition; the callback
advances it only after the current transition completes.

Masks are rasterized in canvas pixels, feathered with continuous linear edges,
and block-averaged deterministically to Krea’s 16×16 image-token cells.
Lower-priority regions are blended first. Higher-priority regions, and earlier
regions at equal priority, win through the same source-over operation. This
preserves the existing K2Lab front-to-back region order and avoids hard
rectangular seams.

Regional strength affects the control-token contribution at each image token.
The adapter’s block weights remain a single globally loaded patch, so region
weighting does not add transformer forwards or encode the depth map per region.
Debug job metadata records every step’s minimum, maximum, mean, overlap policy,
and regional multipliers.
