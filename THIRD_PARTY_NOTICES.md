# Third-party notices

Last reviewed: 2026-07-29

This file records the direct third-party software boundaries used by the K2Lab RunPod
service and workspace image. It is an engineering inventory, not a legal opinion or a
replacement for upstream license texts. `pyproject.toml`, `uv.lock`, and
`Dockerfile.workspace` are the authoritative version and image records.

## Service and native-inference dependencies

| Component | Upstream | License recorded by upstream/package metadata |
| --- | --- | --- |
| NumPy | <https://github.com/numpy/numpy> | BSD-3-Clause plus separately identified bundled components; see the installed distribution |
| Pillow | <https://github.com/python-pillow/Pillow> | MIT-CMU |
| Pydantic | <https://github.com/pydantic/pydantic> | MIT |
| safetensors | <https://github.com/huggingface/safetensors> | Apache-2.0 |
| Diffusers | <https://github.com/huggingface/diffusers> | Apache-2.0 |
| PyTorch, TorchVision, TorchAudio | <https://github.com/pytorch/pytorch> | BSD-3-Clause; the distributions include additional third-party notices |
| Transformers | <https://github.com/huggingface/transformers> | Apache-2.0 |
| ONNX Runtime | <https://github.com/microsoft/onnxruntime> | MIT |

The separately pinned `k2core` repository is maintained with K2Lab but currently has
no declared project license. The K2Lab RunPod repository also has no declared
first-party project license. Both must be resolved by their copyright owner before a
publicly licensed distribution.

## Web-service dependencies

| Component | Upstream | License recorded by upstream/package metadata |
| --- | --- | --- |
| aiosqlite | <https://github.com/omnilib/aiosqlite> | MIT |
| asyncpg | <https://github.com/MagicStack/asyncpg> | Apache-2.0 |
| cryptography | <https://github.com/pyca/cryptography> | Apache-2.0 OR BSD-3-Clause |
| FastAPI | <https://github.com/fastapi/fastapi> | MIT |
| huggingface_hub | <https://github.com/huggingface/huggingface_hub> | Apache-2.0 |
| HTTPX | <https://github.com/encode/httpx> | BSD-3-Clause |
| SQLAlchemy | <https://github.com/sqlalchemy/sqlalchemy> | MIT |
| Uvicorn | <https://github.com/Kludex/uvicorn> | BSD-3-Clause |

## Current workspace-image compatibility layer

`Dockerfile.workspace` clones
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) at immutable revision
`285a98944c397a4a81f15ac63d69fa3dbc0a27b9`. ComfyUI declares GPL-3.0. Its
license file and the notices installed with its Python dependencies must remain in the
image. K2Lab does not copy ComfyUI source into its own Python packages.

The base image is supplied through the required `RUNPOD_BASE_IMAGE` build argument.
Release evidence must record an immutable image digest, its license terms, and the
CUDA/cuDNN and operating-system notices contained in that exact base image.

`Dockerfile.native-workspace` is a separate release-candidate definition that does not
clone or install ComfyUI. It retains the same Python, model, base-image, CUDA, and
operating-system notice obligations listed above.

## Models and user-supplied assets

The repository and workspace image do not grant rights to Krea, Qwen, LoRA,
face-detector, pose-adapter, or upscaler weights mounted into `/workspace`. File hashes
prove identity, not permission to use or redistribute an asset. A release that
downloads or bundles any model must preserve its license, model-card restrictions,
attribution, and redistribution terms separately.

The FantasyPortrait face detector and the pose-training external repositories retain
open provenance/license findings and must not be represented as redistributable until
those records are completed.

## Distribution checklist

Before publishing an image:

1. generate a complete bill of materials from the final lockfile and built image;
2. include license and notice files from every Python package and OS/CUDA component;
3. resolve the K2Lab and `k2core` first-party licenses;
4. record licenses for every downloaded or bundled model;
5. verify the ComfyUI license remains present while the compatibility image includes it;
6. repeat review whenever a dependency, base-image digest, source revision, or model
   hash changes.
