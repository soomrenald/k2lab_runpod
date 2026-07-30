from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from k2_region_lab.pose import default_volumetric_subject_pose
from k2_region_lab.pose_gating import PoseGatingSettings
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.worker.runtime import ComfyBaselineRuntime


def test_control_adapter_preflight_does_not_require_pose_gating(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pose-control.safetensors"
    checkpoint.write_bytes(b"validated by the worker installation step")
    runtime = ComfyBaselineRuntime(Path("/unused"))
    runtime.model = object()
    runtime.clip = object()
    runtime.vae = object()
    runtime._generate_once = Mock(return_value={"image_path": "/tmp/result.png"})
    region = RegionDefinition(
        region_id="subject-a",
        name="Subject A",
        box=PixelBox(32, 24, 224, 240),
        prompt="a standing person",
        enabled=True,
        priority=1,
        spatial_role="subject",
        region_type="subject",
        pose=default_volumetric_subject_pose(),
    )

    result = runtime.generate(
        prompt="a simple portrait",
        width=256,
        height=256,
        steps=8,
        seed=17,
        output_directory=tmp_path,
        regions=(region,),
        regional_prompting=False,
        pose_gating=PoseGatingSettings(enabled=False),
        pose_control_lora_enabled=True,
        pose_control_lora_path=checkpoint,
    )

    assert result["image_path"] == "/tmp/result.png"
    call = runtime._generate_once.call_args
    assert call.kwargs["pose_mask_bundle"] is None
    assert call.kwargs["control_bundle"].full.coverage > 0
    assert call.kwargs["control_bundle"].subjects == {}
