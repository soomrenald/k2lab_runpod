# K2Lab RunPod workspace image versions

This document maps the human-readable workspace version reported as
`K2LAB_IMAGE_VERSION` to the corresponding RunPod container image. It covers the
stable rollback image and the native K2 `0.4.0` release-candidate series.

Always configure a released image by its immutable `@sha256:` reference. Tags are
included for discovery and provenance, but a tag alone is not a deployment pin.
The local control plane uses `K2LAB_RUNPOD_IMAGE_VERSION`; it passes the same
value to the Pod as `K2LAB_IMAGE_VERSION` and verifies that the workspace reports
the expected value.

## Release matrix

| Workspace version | Release tag | Status | Summary |
| --- | --- | --- | --- |
| `0.3.0` | Not recorded in this branch | Stable rollback image | ComfyUI-backed K2 workspace used as the tested rollback baseline. |
| `0.4.0-rc.1` | `native-v0.4.0-rc.1` | Rejected; do not deploy | First native-only publication attempt. Its empty-workspace validation step was invalid, so the candidate was not signed or approved. |
| `0.4.0-rc.2` | `native-v0.4.0-rc.2` | Superseded, GPU accepted | First signed and fully validated native-only image; passed A40 native generation and rollback acceptance. |
| `0.4.0-rc.3` | `native-v0.4.0-rc.3` | Current recommended candidate | RC2 native runtime plus explicit Apache-2.0 source licensing, model-use policy, updated notices, and licensed k2core pin. |
| `0.4.0-rc.4` | Not published | Source-only candidate | Adds submission preflight for unsupported native model components and missing or mismatched tokenizer assets. No release image exists yet. |

## `0.3.0` — ComfyUI rollback baseline

```text
K2LAB_RUNPOD_IMAGE_VERSION=0.3.0
ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:19652733039379d1ef47cd3279e6b266b802c7a68a1c380173221a2d8ace6435
```

This is the retained ComfyUI-backed workspace image. Gate 12 demonstrated that
an existing workspace could be rolled back to this digest on the same Pod,
retain its hash-verified inventory, become healthy, and complete a generation.
Use it for rollback rather than native-backend acceptance.

## `0.4.0-rc.1` — rejected publication attempt

```text
Release tag: native-v0.4.0-rc.1
Source commit: ff569c4189acf14ba7da60128953ad9091f379d6
Deployment: prohibited
```

RC1 introduced publication of the clean native-only workspace candidate. The
release workflow failed because the empty-workspace health check contained
syntactically invalid embedded Python. It was not signed and has no approved
immutable deployment reference.

## `0.4.0-rc.2` — first accepted native-only image

```text
K2LAB_RUNPOD_IMAGE_VERSION=0.4.0-rc.2
Release tag: ghcr.io/soomrenald/k2lab-runpod-workspace:native-v0.4.0-rc.2
Immutable image: ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:7662f6440bd4e2a1f6059876c042df98a1e00284c89c35e6aaec3aa446be856f
Source commit: 4b091c54162fc689833b5115f78e47b1955525cb
```

RC2 comprises:

- a native K2 worker image with no ComfyUI installation;
- strict, content-addressed Krea 2 component loading;
- native generation, image-edit, standard LoRA, and regional LoRA paths;
- durable tokenizer assets stored in the workspace;
- isolated GPU workers, timeout classification, diagnostics, and recovery;
- locked Python dependencies, an SPDX SBOM, zero known high or critical
  vulnerabilities in the release scan, and a GitHub OIDC Cosign signature;
- successful fixed-candidate NVIDIA A40 generation and same-Pod rollback to the
  `0.3.0` ComfyUI image.

The retained acceptance record is
[`gate12_published_native_rc.json`](../tests/fixtures/parity/integration/gate12_published_native_rc.json).

## `0.4.0-rc.3` — licensed native release candidate

```text
K2LAB_RUNPOD_IMAGE_VERSION=0.4.0-rc.3
Release tag: ghcr.io/soomrenald/k2lab-runpod-workspace:native-v0.4.0-rc.3
Immutable image: ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:491067b900a0203d60df7134542d7dbbd41bca5534a1051dab2ea1b298080ede
Source commit: cc905bb584149e7e12f0b65121aa6747826197c8
k2core commit: 903166f756614b13c0add0196fb5705206370dc3
```

RC3 retains RC2's native inference behavior and adds the release licensing and
distribution boundary:

- Apache-2.0 licensing in the native image definition;
- `MODEL_USE_POLICY.md` and updated third-party notices;
- an exact licensed k2core source pin;
- no bundled model weights and no pre-populated model cache;
- locked dependencies, successful empty-workspace boot, SPDX SBOM, zero known
  high or critical release-scan findings, and a GitHub OIDC signature.

RC3 inherits the RC2 GPU acceptance because its native runtime implementation
did not change. Its publication record is
[`gate12_licensed_native_rc.json`](../tests/fixtures/parity/integration/gate12_licensed_native_rc.json).

The currently approved native component set is:

| Component | Workspace file | SHA-256 |
| --- | --- | --- |
| Transformer | `krea2_turbo_fp8_scaled.safetensors` | `eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1` |
| Text encoder | `qwen3vl_4b_fp8_scaled.safetensors` | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| VAE | `qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |
| Tokenizer directory | `models/tokenizers` | `9362730d7f1fe82e277f363f2294f30edb2bb81b5c67b0d1b83813a5ac21f34d` |

RC3 does not yet support face refinement, projector conditioning, post-upscale,
volumetric pose gating, or volumetric pose-control LoRA in the native worker.
Reference-projector conditioning is likewise unavailable for native image edits.

## Planned `0.4.0-rc.4` — not yet an image

Commit `c182ad9` adds fast native-asset validation at job submission. It rejects
an unapproved transformer, text encoder, or VAE and detects a missing or
mismatched tokenizer before allocating a GPU worker.

This change has been committed to `feature/native-k2-backend`, but it has not
been tagged, built, scanned, signed, or GPU accepted. Do not configure
`K2LAB_RUNPOD_IMAGE_VERSION=0.4.0-rc.4` until an immutable release image and its
acceptance record are published.

## Current deployment selection

Until RC4 is published and accepted, use RC3:

```bash
./scripts/k2lab-runpod \
  --image ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:491067b900a0203d60df7134542d7dbbd41bca5534a1051dab2ea1b298080ede \
  --image-version 0.4.0-rc.3
```
