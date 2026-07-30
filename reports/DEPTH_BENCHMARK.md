# Depth validation and benchmark status

Date: 2026-07-30

Completed locally:

- Public checkpoint SHA-256 matches
  `fb80547ed79b47c1e3fea7bb9d36297e3917b2115fab6700ca1501350f9f483c`.
- 450 tensors; one expanded `first.weight`; rank 64; 224/224 required
  block-target pairs.
- 16-bit Blender export preserved values (`uint16`, 256×256 reference,
  non-constant range).
- Repeated reference exports had zero differing depth pixels and zero differing
  object-ID pixels.
- Shared k2core suite: 225 passed, 2 environment skips, 14 subtests.
- K2Lab depth, project, worker, web-control, and ordinary regional regression
  selections: 131 passed, 6 expected Torch-environment skips, 6 subtests.
- The regional token-strength projection passed with Torch 2.10.0+rocm7.1
  using CPU tensors and isolated Comfy device helpers.
- Web TypeScript and production asset build passed.
- The fixed 11-scene/variant/metric matrix is recorded in
  `reports/depth-evaluation-suite.json`.

Live Raw/Turbo image response, runtime, and VRAM measurements are not recorded
here because the implementation host exposes no Torch accelerator
(`torch.cuda.is_available() == false`, device count 0). The standalone harness
fails closed at model loading in that environment. Gate B therefore remains an
external accelerator validation requirement; no production depth deployment
should enable the feature flags until correct depth beats shuffled and neutral
controls on both modes.

The unchanged depth-disabled regression suite and full integration suite are
run as part of merge readiness. The repository's complete workspace-agent test
module cannot finish on this host because its pre-existing chunked-upload test
blocks in the sandbox; focused agent capability, payload, pose-adapter, and
depth-adapter tests pass. Production rollout remains flag-gated.
