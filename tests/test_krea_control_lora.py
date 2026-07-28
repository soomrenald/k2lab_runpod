from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import k2_region_lab.krea_control_lora as control_lora
from k2_region_lab.krea_control_lora import (
    EXPECTED_BLOCKS,
    EXPECTED_TARGETS,
    KreaControlInputProjection,
    encode_control_bundle,
    inspect_krea_control_checkpoint,
    select_control_latent,
)
from k2_region_lab.pose import default_volumetric_subject_pose
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.semantic_conditioning import (
    ConditioningScope,
    conditioning_execution_scope,
)
from k2_region_lab.volumetric_control import (
    K2_VOLUMETRIC_CONTROL_FORMAT,
    K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
    render_krea_volumetric_control_bundle,
)


def _metadata() -> dict[str, str]:
    return {
        "k2lab_adapter_kind": "krea2_control_lora",
        "k2lab_adapter_version": "1",
        "k2lab_control_format": K2_VOLUMETRIC_CONTROL_FORMAT,
        "k2lab_control_format_sha256": K2_VOLUMETRIC_CONTROL_FORMAT_SHA256,
        "k2lab_renderer_version": "1",
        "k2lab_base_model": "krea/Krea-2-Raw",
        "k2lab_inference_targets": "krea/Krea-2-Raw,krea/Krea-2-Turbo",
        "k2lab_rank": "64",
        "k2lab_expanded_input_projection": "true",
        "k2lab_expected_transformer_blocks": "28",
        "k2lab_control_channel_mode": "rgb",
        "k2lab_control_normalize": "none",
        "k2lab_control_invert": "false",
        "k2lab_dataset_manifest_sha256": "d" * 64,
        "k2lab_trainer_repository": "https://github.com/Tanmaypatil123/Krea-2-controlnet",
        "k2lab_trainer_commit": "909682ae0bdd9eb87c8258894c0003224db00d0b",
        "k2lab_training_commit": "c" * 40,
        "k2lab_created_at": datetime.now(UTC).isoformat(),
    }


def _checkpoint(path: Path, *, metadata: dict[str, str] | None = None) -> None:
    tensors = {
        "first.weight": np.zeros((16, 128), dtype=np.float16),
        "first.bias": np.zeros((16,), dtype=np.float16),
    }
    for block in range(EXPECTED_BLOCKS):
        for target in EXPECTED_TARGETS:
            tensors[f"blocks.{block}.{target}.A"] = np.zeros((64, 1), dtype=np.float16)
            tensors[f"blocks.{block}.{target}.B"] = np.zeros((1, 64), dtype=np.float16)
    save_file(tensors, path, metadata=metadata or {})


def test_checkpoint_validator_accepts_complete_verified_rank64(tmp_path: Path) -> None:
    path = tmp_path / "control.safetensors"
    _checkpoint(path, metadata=_metadata())
    report = inspect_krea_control_checkpoint(path)
    assert report.compatible
    assert report.verified
    assert report.checkpoint is not None
    assert report.checkpoint.rank == 64
    assert report.checkpoint.compatible_block_pairs == 28 * 8
    assert report.checkpoint.expanded_projection_key == "first.weight"


def test_checkpoint_validator_rejects_missing_projection_and_wrong_format(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ordinary.safetensors"
    metadata = _metadata()
    metadata["k2lab_control_format_sha256"] = "0" * 64
    save_file(
        {
            "blocks.0.attn.wq.A": np.zeros((64, 1), dtype=np.float16),
            "blocks.0.attn.wq.B": np.zeros((1, 64), dtype=np.float16),
        },
        path,
        metadata=metadata,
    )
    report = inspect_krea_control_checkpoint(path)
    assert not report.compatible
    assert any("renderer/palette" in error for error in report.errors)
    assert any("projection" in error for error in report.errors)
    assert any("missing" in error and "LoRA" in error for error in report.errors)


def test_legacy_checkpoint_requires_explicit_override(tmp_path: Path) -> None:
    path = tmp_path / "legacy.safetensors"
    _checkpoint(path)
    blocked = inspect_krea_control_checkpoint(path)
    allowed = inspect_krea_control_checkpoint(path, allow_unverified_legacy=True)
    assert not blocked.compatible
    assert allowed.compatible
    assert not allowed.verified
    assert allowed.warnings == ("Unverified legacy Krea control checkpoint",)


def test_unsafe_or_non_safetensors_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "control.ckpt"
    path.write_bytes(b"pickle")
    report = inspect_krea_control_checkpoint(path)
    assert not report.compatible
    assert report.checkpoint is None


def test_checkpoint_validator_rejects_wrong_projection_width_and_missing_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-shape.safetensors"
    tensors = {
        "first.weight": np.zeros((16, 64), dtype=np.float16),
        "first.bias": np.zeros((16,), dtype=np.float16),
    }
    for block in range(EXPECTED_BLOCKS):
        for target in EXPECTED_TARGETS:
            if block == EXPECTED_BLOCKS - 1 and target == EXPECTED_TARGETS[-1]:
                continue
            tensors[f"blocks.{block}.{target}.A"] = np.zeros(
                (64, 1),
                dtype=np.float16,
            )
            tensors[f"blocks.{block}.{target}.B"] = np.zeros(
                (1, 64),
                dtype=np.float16,
            )
    save_file(tensors, path, metadata=_metadata())

    report = inspect_krea_control_checkpoint(path)

    assert not report.compatible
    assert any("doubled-width" in error for error in report.errors)
    assert any("missing 1 required" in error for error in report.errors)


def test_ordinary_lora_is_not_accepted_as_control_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "identity.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": np.zeros(
                (16, 64),
                dtype=np.float16,
            ),
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": np.zeros(
                (64, 16),
                dtype=np.float16,
            ),
        },
        path,
        metadata={"ss_network_module": "networks.lora"},
    )

    report = inspect_krea_control_checkpoint(
        path,
        allow_unverified_legacy=True,
    )

    assert not report.compatible
    assert any("projection" in error for error in report.errors)
    assert any("missing" in error and "LoRA" in error for error in report.errors)


def test_scope_selection_uses_full_and_exact_subject_controls() -> None:
    mapping = {"full": "full-control", "subjects": {"a": "subject-a-control"}}

    assert select_control_latent(mapping) == ("full", "full-control")
    with conditioning_execution_scope(ConditioningScope.full(), 8):
        assert select_control_latent(mapping) == ("full", "full-control")
    with conditioning_execution_scope(ConditioningScope.subject("a"), 8):
        assert select_control_latent(mapping) == (
            "subject:a",
            "subject-a-control",
        )
    with conditioning_execution_scope(ConditioningScope.subject("missing"), 8):
        with pytest.raises(Exception, match="subject scope"):
            select_control_latent(mapping)
    assert select_control_latent(mapping) == ("full", "full-control")


def test_expanded_projection_preserves_native_image_path_and_scales_only_control() -> None:
    torch = pytest.importorskip("torch")
    original = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        original.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    expanded = torch.zeros((3, 8), dtype=torch.float32)
    expanded[:, 4:] = 2
    image_tokens = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    control_tokens = torch.ones_like(image_tokens)
    projection = KreaControlInputProjection(
        expanded,
        image_features=4,
        original_first=original,
        strength=0.5,
    )
    projection.control_tokens = control_tokens

    output = projection(image_tokens)

    assert torch.equal(
        output,
        original(image_tokens) + torch.full((1, 2, 3), 4.0),
    )
    projection.strength = 0
    assert torch.equal(projection(image_tokens), original(image_tokens))
    with pytest.raises(Exception, match="latent shape"):
        projection(torch.ones((1, 2, 5)))
    projection.control_tokens = torch.ones((1, 3, 4))
    with pytest.raises(Exception, match="token count"):
        projection(image_tokens)


def test_control_wrapper_restores_projection_after_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original = object()

    class Projection:
        control_features = 4
        control_tokens = None

    projection = Projection()
    projection.original_first = original
    diffusion = type("Diffusion", (), {"first": original, "patch": 2})()
    monkeypatch.setattr(
        control_lora,
        "_control_tokens",
        lambda *_args, **_kwargs: torch.ones((1, 4, 4)),
    )

    class Executor:
        class_obj = diffusion

        def __init__(self) -> None:
            self.fail = False

        def __call__(self, *_args, **_kwargs):
            assert diffusion.first is projection
            assert projection.control_tokens is not None
            if self.fail:
                raise RuntimeError("synthetic denoiser failure")
            return "sample"

    executor = Executor()
    wrapper = control_lora._control_wrapper(projection, {})
    options = {
        control_lora.CONTROL_LATENT_MAPPING_KEY: {
            "full": torch.zeros((1, 4, 4, 4)),
            "subjects": {},
        }
    }
    assert wrapper(
        executor,
        torch.zeros((1, 4, 4, 4)),
        transformer_options=options,
    ) == "sample"
    assert diffusion.first is original
    assert projection.control_tokens is None

    executor.fail = True
    with pytest.raises(RuntimeError, match="synthetic denoiser failure"):
        wrapper(
            executor,
            torch.zeros((1, 4, 4, 4)),
            transformer_options=options,
        )
    assert diffusion.first is original
    assert projection.control_tokens is None


def test_control_images_are_encoded_independently_and_processed_as_model_latents() -> None:
    torch = pytest.importorskip("torch")
    controls = render_krea_volumetric_control_bundle(
        regions=(
            RegionDefinition(
                region_id="a",
                name="A",
                box=PixelBox(8, 4, 56, 62),
                prompt="subject",
                priority=1,
                spatial_role="subject",
                region_type="subject",
                pose=default_volumetric_subject_pose(),
            ),
        ),
        width=64,
        height=64,
        include_subjects=True,
    )

    class FakeVae:
        def __init__(self) -> None:
            self.calls = []

        def encode(self, pixels):
            self.calls.append(pixels.clone())
            return torch.zeros((1, 16, 8, 8), dtype=torch.float32)

    class FakeModel:
        @staticmethod
        def process_latent_in(latent):
            return latent + 1

    class FakePatcher:
        model = FakeModel()

        @staticmethod
        def get_model_object(name):
            assert name == "latent_format"
            return type(
                "LatentFormat",
                (),
                {"latent_channels": 16, "latent_dimensions": 2},
            )()

    vae = FakeVae()
    encoded = encode_control_bundle(vae, FakePatcher(), controls)

    assert len(vae.calls) == 2
    assert vae.calls[0].data_ptr() != vae.calls[1].data_ptr()
    assert tuple(encoded.full.shape) == (1, 16, 8, 8)
    assert torch.all(encoded.full == 1)
    assert tuple(encoded.subjects["a"].shape) == (1, 16, 8, 8)
