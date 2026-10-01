import asyncio
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from k2_region_lab.agent.domain import FileKind, JobKind, JobSubmitRequest
from k2_region_lab.agent.jobs import JobError, JobManager
from k2_region_lab.agent.storage import WorkspaceLayout
from k2_region_lab.agent.transfers import TransferManager
from k2_region_lab.image2image import Image2ImageSettings
from k2_region_lab.project import ProjectState, project_document, project_state
from k2_region_lab.worker.image2image import run_image2image


def test_project_round_trip_keeps_edit_and_image2image_independent():
    state = ProjectState(
        canvas_width=1024,
        canvas_height=1024,
        image2image=Image2ImageSettings(prompt="recolor the coat", cfg=3, steps=20),
    )
    document = project_document(state)
    restored = project_state(document)
    assert restored.image2image == state.image2image
    assert restored.image_edit == state.image_edit
    document.pop("image2image")
    assert project_state(document).image2image == Image2ImageSettings()


@pytest.mark.parametrize(
    "patch",
    [
        {"width": 1050},
        {"width": 2048, "height": 2048},
        {"ref_boost": float("nan")},
        {"sampler": "invalid"},
        {"fit_mode": "stretch"},
        {"cfg": 0},
    ],
)
def test_invalid_settings_fail_before_sampling(patch):
    with pytest.raises((ValueError, ValidationError)):
        Image2ImageSettings(**patch)


def test_agent_routes_new_job_without_edit_settings_or_loras(tmp_path):
    async def check():
        layout = WorkspaceLayout(tmp_path / "workspace")
        layout.initialize()
        transfers = TransferManager(layout)
        manager = JobManager(
            layout, transfers, worker_python=Path("/unused"), comfyui_root=Path("/unused/comfy")
        )
        source = layout.destination("inputs") / "source.png"
        Image.new("RGB", (256, 256)).save(source)
        source_record = await transfers.index_existing_file(FileKind.INPUTS, source)
        lora = layout.destination("loras") / "identity.safetensors"
        lora.write_bytes(b"test binding only")
        lora_record = await transfers.index_existing_file(FileKind.LORAS, lora)
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            keep_model_loaded=True,
            image2image=Image2ImageSettings(prompt="make the coat blue"),
        )
        state = replace(
            state,
            image_edit=replace(
                state.image_edit, global_prompt="legacy instruction must not be used"
            ),
        )
        document = project_document(state)
        request = JobSubmitRequest(
            command_id="i2i-test",
            kind=JobKind.IMAGE2IMAGE,
            project_id="project",
            project=document,
            input_file_id=source_record.id,
            identity_lora_file_id=lora_record.id,
        )
        validated, sanitized = manager._validate_request(request)
        payload = await manager._job_payload("job", request, validated, sanitized)
        assert payload["prompt"] == "make the coat blue"
        assert payload["identity_lora_path"] == str(lora)
        assert payload["image_path"] == str(source)
        assert payload["keep_model_loaded"] is False
        assert "regions" not in payload and "denoise" not in payload and "loras" not in payload
        assert [item["kind"] for item in manager._commands("job", request, payload)] == [
            "probe",
            def test_old_runtime_rejects_image2image_with_actionable_conflict():
    from k2_region_lab.web.domain import WorkspaceError
    from k2_region_lab.web.runpod_backend import RunPodPersistentPodBackend

    class OldAgent:
        async def capabilities(self):
            return SimpleNamespace(supported_job_kinds=["generate", "edit_image"])

        async def submit_job(self, request):
            pytest.fail("An unsupported job must not reach the old runtime")

    class Backend:
        async def _workspace_agent(self, workspace_id):
            return OldAgent()

    request = SimpleNamespace(kind=JobKind.IMAGE2IMAGE)
    with pytest.raises(WorkspaceError) as caught:
        asyncio.run(RunPodPersistentPodBackend.submit_job(Backend(), "workspace", request))
    assert caught.value.status_code == 409
    assert caught.value.code == "image2image_runtime_required"
    assert "Update the workspace image" in caught.value.message


"load_model",
            "image2image",
        ]
        with pytest.raises(JobError, match="Identity Edit"):
            manager._validate_request(request.model_copy(update={"identity_lora_file_id": None}))
        with pytest.raises(JobError, match="input file"):
            manager._validate_request(request.model_copy(update={"input_file_id": None}))

    asyncio.run(check())


class FakeTensor:
    def __init__(self, array):
        self.array = array

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def clamp(self, lower, upper):
        return FakeTensor(self.array.clip(lower, upper))

    def numpy(self):
        return self.array


@pytest.mark.parametrize("cfg", [1, 3])
def test_pipeline_grounds_instruction_uses_clean_reference_and_noise_target(
    tmp_path,
    monkeypatch,
    capsys,
    cfg,
):
    """Exercise orchestration without claiming to test actual GPU inference."""
    source = tmp_path / "source.png"
    Image.new("RGB", (256, 256)).save(source)
    encodings, patches, samples = [], [], []
    target_tensor = FakeTensor(np.zeros((1, 16, 32, 32), dtype=np.float32))
    source_latent = object()
    base_model = SimpleNamespace(clone=lambda: model)
    model = SimpleNamespace(add_patches=lambda weights, strength: ["matched"])

    class Encoder:
        def encode(self, clip, prompt, image, **kwargs):
            encodings.append((prompt, image, kwargs))
            return ([prompt],)

    class Patch:
        def patch(self, supplied_model, reference, **kwargs):
            assert supplied_model is model
            patches.append((reference, kwargs))
            print("upstream diagnostic")
            return (model,)

    def sample(*args, **kwargs):
        samples.append((args, kwargs))
        kwargs["callback"](0, None, None, 8)
        return object()

    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.sample = SimpleNamespace(sample=sample, prepare_noise=lambda tensor, seed: "noise")
    comfy.model_management = SimpleNamespace(
        intermediate_device=lambda: "cpu", intermediate_dtype=lambda: "float32"
    )
    comfy.samplers = SimpleNamespace(
        KSampler=SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
    )
    for name, module in {
        "comfy": comfy,
        "comfy.sample": comfy.sample,
        "comfy.samplers": comfy.samplers,
        "torch": SimpleNamespace(
            from_numpy=FakeTensor,
            float32="float32",
            no_grad=nullcontext,
            zeros=lambda *args, **kwargs: target_tensor,
        ),
        "comfy.model_management": comfy.model_management,
        "k2_region_lab.worker.krea2edit": SimpleNamespace(
            Krea2EditGroundedEncode=Encoder, Krea2EditModelPatch=Patch
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "k2_region_lab.sampling.register_bong_tangent_scheduler", lambda *args: None
    )
        import builtins

    original_import = builtins.__import__

    def noisy_import(name, *args, **kwargs):
        if name == "k2_region_lab.worker.krea2edit":
            print("[krea2edit] nodes loaded")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", noisy_import)
snapshot = {"gpu_free_bytes": 100, "critical_free_bytes": 1}
    runtime = SimpleNamespace(
        loaded=True,
        clip=object(),
        vae=object(),
        model=base_model,
        _ensure_memory=lambda *args: None,
        _load_lora_patches=lambda spec: ({"weights": object()}, None, {}),
        _encode_vae=lambda pixels: source_latent,
        _prepare_vae_handoff=lambda *args: None,
        _decode_vae=lambda tensor: FakeTensor(np.zeros((1, 1, 256, 256, 3))),
        memory_snapshot=lambda stage: snapshot,
    )
    options = Image2ImageSettings(prompt="make it blue", width=256, height=256, cfg=cfg)
    result = run_image2image(
        runtime,
        {
            **options.model_dump(),
            "image_path": str(source),
            "identity_lora_path": "identity",
            "output_directory": str(tmp_path),
            "project_json": {"image2image": options.model_dump()},
        },
        progress=lambda *args: print(json.dumps({"step": args[0]})),
    )
    assert encodings[0][0] == "make it blue"
    assert [item[0] for item in encodings] == (
        ["make it blue", ""] if cfg > 1 else ["make it blue"]
    )
    assert all(item[2]["grounding_px"] == 768 for item in encodings)
    assert patches[0][0]["samples"] is source_latent
    assert patches[0][1]["target_latent"]["samples"] is target_tensor
    assert patches[0][1]["source_image"] is encodings[0][1]
    assert samples[0][0][-1] is target_tensor
    assert samples[0][1]["denoise"] == 1
    assert "noise_mask" not in samples[0][1]
    assert runtime.model is base_model
    output = capsys.readouterr()
    assert '"step": 1' in output.out and "upstream diagnostic" not in output.out
    assert "upstream diagnostic" in output.err
    with Image.open(result["image_path"]) as image:
        assert image.info["k2lab_mode"] == "krea2_identity_image2image"
        assert json.loads(image.info["k2lab_project"])["image2image"]["prompt"] == "make it blue"
