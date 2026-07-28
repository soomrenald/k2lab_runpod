from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .dataset import sha256_file
except ImportError:
    from dataset import sha256_file


def _tensor(image: Image.Image, torch):
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


class QwenImageVaeEncoder:
    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from diffusers import AutoencoderKLQwenImage

        self.torch = torch
        self.device = device
        self.vae = (
            AutoencoderKLQwenImage.from_pretrained(
                "Qwen/Qwen-Image",
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.mean = torch.tensor(
            self.vae.config.latents_mean,
            device=device,
        ).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(
            self.vae.config.latents_std,
            device=device,
        ).view(1, -1, 1, 1, 1)

    def encode(self, image: Image.Image) -> np.ndarray:
        torch = self.torch
        pixels = _tensor(image, torch).unsqueeze(0).unsqueeze(2).to(
            self.device,
            torch.bfloat16,
        )
        with torch.inference_mode():
            latent = self.vae.encode(pixels).latent_dist.sample()
            latent = (latent - self.mean) / self.std
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise RuntimeError(f"Qwen VAE returned unexpected latent shape {tuple(latent.shape)}")
        return latent[0, :, 0].float().cpu().numpy().astype(np.float16)


def prepare(
    dataset: Path,
    output: Path,
    *,
    samples_per_shard: int = 1000,
    encoder: QwenImageVaeEncoder | None = None,
) -> int:
    metadata_path = dataset / "metadata.jsonl"
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    encoder = encoder or QwenImageVaeEncoder()
    completed = 0
    for shard_number, start in enumerate(range(0, len(records), samples_per_shard)):
        shard_name = f"shard{shard_number:05d}"
        shard = output / shard_name
        done = shard / "_DONE"
        if done.is_file():
            completed += min(samples_per_shard, len(records) - start)
            continue
        shard.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, object]] = []
        for record in records[start : start + samples_per_shard]:
            target_path = dataset / "images" / record["file_name"]
            control_path = dataset / "controls" / record["control_file_name"]
            target = Image.open(target_path).convert("RGB")
            control = Image.open(control_path).convert("RGB")
            bucket = tuple(int(value) for value in record["bucket"])
            if target.size != control.size or target.size != bucket:
                raise ValueError(
                    f"aligned pair {record['file_name']!r} does not match bucket {bucket}"
                )
            if sha256_file(control_path) != record["control_sha256"]:
                raise ValueError(f"control hash changed for {record['control_file_name']!r}")
            latent = encoder.encode(target)
            control_latent = encoder.encode(control)
            if latent.shape != control_latent.shape:
                raise ValueError(
                    f"target/control latent mismatch for {record['file_name']!r}: "
                    f"{latent.shape} != {control_latent.shape}"
                )
            sample_name = Path(record["file_name"]).stem + ".npz"
            np.savez_compressed(
                shard / sample_name,
                latent=latent,
                control=control_latent,
                prompt=np.str_(record["text"]),
                size=np.asarray(bucket, dtype=np.int32),
                source_sha256=np.str_(sha256_file(target_path)),
                control_sha256=np.str_(record["control_sha256"]),
            )
            relative = f"{shard_name}/{sample_name}"
            index.append(
                {
                    "file": relative,
                    "bucket": list(bucket),
                    "source_sha256": sha256_file(target_path),
                    "control_sha256": record["control_sha256"],
                }
            )
            completed += 1
        (shard / "index.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in index),
            encoding="utf-8",
        )
        for item in index:
            path = output / str(item["file"])
            with np.load(path) as data:
                if data["latent"].shape != data["control"].shape:
                    raise ValueError(f"shard validation failed for {path}")
        done.write_text("complete\n", encoding="utf-8")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prepare(
        args.dataset,
        args.output,
        samples_per_shard=args.samples_per_shard,
        encoder=QwenImageVaeEncoder(device=args.device),
    )


if __name__ == "__main__":
    main()
