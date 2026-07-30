# K2Lab Krea-2 volumetric Control-LoRA

This toolchain trains the hidden `k2-volumetric-pose-control-v1` RGB rendering
against Krea-2-Raw, then writes a checkpoint contract accepted by K2Lab's native
ComfyUI Krea inference path. Train on Raw; validate inference on Raw and Turbo.
Turbo's initial inference setting is 8 steps with CFG disabled.

The trainer and native reference repositories had no license file at their
pinned revisions, so their source is not vendored here. `train.py` requires an
external checkout at exactly
`909682ae0bdd9eb87c8258894c0003224db00d0b`; see `upstream.json`.
The training container installs this directory's frozen `uv.lock`, including
the pinned Torch/torchvision pair and the trainer's Transformers 5.x,
Diffusers, Einops, Accelerate, SentencePiece, and Weights & Biases runtime.

## Data

Supply local COCO 2017 train/validation images, person-keypoint annotations, and
caption annotations. The scripts never download or redistribute COCO. Obtain it
yourself or explicitly consent to the official download terms outside this
tool.

```bash
python build_coco_pairs.py \
  --images /data/coco/train2017 \
  --keypoints /data/coco/annotations/person_keypoints_train2017.json \
  --captions /data/coco/annotations/captions_train2017.json \
  --output /data/k2-control --split train

python build_coco_pairs.py \
  --images /data/coco/val2017 \
  --keypoints /data/coco/annotations/person_keypoints_val2017.json \
  --captions /data/coco/annotations/captions_val2017.json \
  --output /data/k2-control --split validation
```

The train split defaults to 70% full images and 30% subject crops, with
15–40% crop context. Resize/crop/flip is applied to target and semantic joints
first; the canonical control is rendered once at the final ~1MP bucket. A flip
swaps left/right joint identities before rendering.

## Latent shards

```bash
python prepare_latent_shards.py \
  --dataset /data/k2-control/train \
  --output /data/k2-latents
```

Target and control must already have identical final dimensions. Each is
encoded independently through the Qwen-Image VAE and normalized by its declared
latent mean/std. `_DONE` enables shard-level resume.

## Training

Use BF16 on an H100/H200/A100-80GB-class GPU or larger. The defaults are rank
64, LR `1e-4`, AdamW betas `0.9/0.99`, weight decay 0, batch 8, accumulation 4,
warmup 200, caption dropout 0.10, checkpoints every 500, validation every 250,
and 6,000 maximum steps.

```bash
python train.py \
  --upstream-checkout /opt/Krea-2-controlnet \
  --data-dir /data/k2-latents \
  --raw-checkpoint /models/krea2_raw.safetensors \
  --checkpoint-dir /checkpoints \
  --dataset-manifest /data/k2-control/train/manifest.json \
  --training-commit "$(git rev-parse HEAD)"
```

Recommended sequence: 20-step synthetic smoke test, 500-step small-data test,
2,000-step capacity test, then a 6,000-step first serious run. Select by fixed
validation; 6,000 is not claimed optimal.

The pinned upstream `--resume` restores trainable weights and step only. It does
not restore optimizer, scheduler, RNG, data position, or accumulation state, so
`train.py` labels resume as approximate rather than exact.

`inspect_checkpoint.py` validates safetensors metadata, the doubled Krea input
projection, rank, all 28×8 block LoRA pairs, base family, renderer hash, and
format. Use `evaluate.py` to create the fixed Raw/Turbo, adapter-off, and
strength-sweep evaluation matrix.
