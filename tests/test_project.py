from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from PIL import Image, PngImagePlugin

from k2core.depth import (
    DepthControlSettings,
    DepthNormalizationSettings,
    DepthRegionMode,
    DepthRegionSettings,
)
from k2_region_lab.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    STANDARD_LORA_ROUTING,
)
from k2_region_lab.project import (
    PROJECT_VERSION,
    ProjectState,
    SavedLora,
    load_project_image,
    project_document,
    project_state,
)
from k2_region_lab.semantic_conditioning import PoseSemanticMode
from k2_region_lab.regional_prompting import GLOBAL_EMPHASIS_SCOPE, PromptEmphasis
from k2_region_lab.regions import PixelBox, RegionDefinition


class ProjectStateTests(unittest.TestCase):
    def test_depth_control_round_trips_and_version_23_defaults_disabled(self) -> None:
        depth = DepthControlSettings(
            enabled=True,
            checkpoint=Path("depth-control-lora.safetensors"),
            depth_image=Path("blender-depth.png"),
            global_strength=1.25,
            normalization=DepthNormalizationSettings(invert=True, gamma=0.8),
            regions=(
                DepthRegionSettings("subject-a", DepthRegionMode.EMPHASIZE, 1.5),
            ),
        )
        document = project_document(
            ProjectState(
                canvas_width=1024,
                canvas_height=1024,
                depth_control=depth,
                regions=(
                    RegionDefinition(
                        region_id="subject-a",
                        name="Subject A",
                        box=PixelBox(32, 32, 512, 900),
                        prompt="a person",
                    ),
                ),
            )
        )

        restored = project_state(document)

        self.assertTrue(restored.depth_control.enabled)
        self.assertEqual(restored.depth_control.global_strength, 1.25)
        self.assertTrue(restored.depth_control.normalization.invert)
        self.assertEqual(
            restored.depth_control.regions[0].mode,
            DepthRegionMode.EMPHASIZE,
        )

        document["version"] = 23
        document["generation"].pop("depth_control")
        legacy = project_state(document)
        self.assertFalse(legacy.depth_control.enabled)

    def test_depth_control_rejects_a_missing_project_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth settings reference"):
            ProjectState(
                canvas_width=1024,
                canvas_height=1024,
                depth_control=DepthControlSettings(
                    regions=(
                        DepthRegionSettings(
                            "missing-subject",
                            DepthRegionMode.EMPHASIZE,
                            1.5,
                        ),
                    ),
                ),
            )

    def test_version_twenty_two_preserves_semantics_and_disables_new_adapter(self) -> None:
        document = project_document(
            ProjectState(
                canvas_width=1024,
                canvas_height=1024,
                pose_gating_enabled=False,
                pose_semantic_mode=PoseSemanticMode.ATTENTION_ISOLATION.value,
            )
        )
        document["version"] = 22
        document["generation"].pop("pose_control_lora_enabled")
        document["generation"].pop("pose_control_lora_model")
        document["generation"].pop("pose_control_lora_strength")
        document["generation"].pop("pose_control_format")

        restored = project_state(document)

        self.assertFalse(restored.pose_gating_enabled)
        self.assertEqual(
            restored.pose_semantic_mode,
            PoseSemanticMode.ATTENTION_ISOLATION.value,
        )
        self.assertFalse(restored.pose_control_lora_enabled)
        self.assertIsNone(restored.pose_control_lora_model)
        self.assertEqual(restored.pose_control_lora_strength, 1.0)

    def test_version_twenty_one_migrates_to_spatial_only(self) -> None:
        document = project_document(
            ProjectState(
                canvas_width=1024,
                canvas_height=1024,
                shared_visual_prompt="35 mm cinematic photography",
            )
        )
        document["version"] = 21
        document["generation"].pop("shared_visual_prompt")
        document["generation"].pop("pose_semantic_mode")

        restored = project_state(document)

        self.assertEqual(restored.shared_visual_prompt, "")
        self.assertEqual(
            restored.pose_semantic_mode,
            PoseSemanticMode.SPATIAL_ONLY.value,
        )
        self.assertTrue(
            any(
                "subject-semantic pose routing" in notice
                for notice in restored.runtime["migration_notices"]
            )
        )

    def test_project_image_metadata_loads_and_uses_imported_png_as_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.png"
            document = project_document(
                ProjectState(
                    canvas_width=768,
                    canvas_height=512,
                    global_prompt="a glass observatory",
                    background_image=Path("old-output.png"),
                )
            )
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("k2lab_project", json.dumps(document))
            Image.new("RGB", (768, 512), "navy").save(path, pnginfo=metadata)

            restored = load_project_image(path)

            self.assertEqual(restored.global_prompt, "a glass observatory")
            self.assertEqual(restored.canvas_width, 768)
            self.assertEqual(restored.background_image, path.resolve())

    def test_project_image_requires_embedded_k2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary.png"
            Image.new("RGB", (32, 32), "black").save(path)

            with self.assertRaisesRegex(ValueError, "does not contain"):
                load_project_image(path)

    def test_late_step_scale_round_trips(self) -> None:
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            shared_visual_prompt="cinematic 35 mm photography",
            pose_semantic_mode=PoseSemanticMode.ATTENTION_ISOLATION.value,
            sampler="dpmpp_2m",
            scheduler="karras",
            regional_late_step_scale=0.8,
            regional_lora_delta_adaptation=True,
            regional_lora_delta_adaptation_gain=0.6,
            face_detail_seed=123,
            face_detail_steps=10,
            face_detail_denoise=0.25,
            face_detail_crop_size=768,
            face_detail_padding=1.8,
            face_detail_feather=0.16,
            face_detail_blend=0.4,
            face_detail_lora_scale=1.2,
            face_detail_detector_threshold=0.35,
            face_detail_detector_provider="cpu",
            projector_identity_protection=0.65,
            vram_mode="high_vram",
            reserve_vram_gb=1.5,
            keep_model_loaded=True,
            system_ram_guard_enabled=False,
            pose_control_lora_model=Path("krea-pose-r64.safetensors"),
            pose_control_lora_strength=0.85,
            prompt_emphases=(
                PromptEmphasis(GLOBAL_EMPHASIS_SCOPE, "two distinct people", 0.5),
            ),
        )

        document = project_document(state)

        self.assertEqual(document["version"], PROJECT_VERSION)
        self.assertEqual(document["generation"]["regional_late_step_scale"], 0.8)
        self.assertEqual(document["generation"]["sampler"], "dpmpp_2m")
        self.assertEqual(document["generation"]["scheduler"], "karras")
        self.assertFalse(document["generation"]["pose_control_lora_enabled"])
        self.assertEqual(
            document["generation"]["pose_control_lora_model"],
            "krea-pose-r64.safetensors",
        )
        self.assertEqual(document["generation"]["pose_control_lora_strength"], 0.85)
        self.assertEqual(project_state(document).regional_late_step_scale, 0.8)
        self.assertTrue(project_state(document).regional_lora_delta_adaptation)
        self.assertEqual(
            project_state(document).regional_lora_delta_adaptation_gain, 0.6
        )
        self.assertEqual(project_state(document).prompt_emphases[0].phrase, "two distinct people")
        restored = project_state(document)
        self.assertEqual(
            restored.shared_visual_prompt,
            "cinematic 35 mm photography",
        )
        self.assertEqual(
            restored.pose_semantic_mode,
            PoseSemanticMode.ATTENTION_ISOLATION.value,
        )
        self.assertEqual(restored.sampler, "dpmpp_2m")
        self.assertEqual(restored.scheduler, "karras")
        self.assertEqual(restored.face_detail_seed, 123)
        self.assertEqual(restored.face_detail_steps, 10)
        self.assertEqual(restored.face_detail_denoise, 0.25)
        self.assertEqual(restored.face_detail_crop_size, 768)
        self.assertEqual(restored.face_detail_padding, 1.8)
        self.assertEqual(restored.face_detail_feather, 0.16)
        self.assertEqual(restored.face_detail_blend, 0.4)
        self.assertEqual(restored.face_detail_lora_scale, 1.2)
        self.assertEqual(restored.face_detail_detector_threshold, 0.35)
        self.assertEqual(restored.face_detail_detector_provider, "cpu")
        self.assertEqual(restored.projector_identity_protection, 0.65)
        self.assertEqual(restored.vram_mode, "high_vram")
        self.assertEqual(restored.reserve_vram_gb, 1.5)
        self.assertTrue(restored.keep_model_loaded)
        self.assertFalse(restored.system_ram_guard_enabled)

    def test_vram_controls_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "VRAM mode"):
            ProjectState(canvas_width=1024, canvas_height=1024, vram_mode="unbounded")
        with self.assertRaisesRegex(ValueError, "VRAM reserve"):
            ProjectState(canvas_width=1024, canvas_height=1024, reserve_vram_gb=0.25)

    def test_character_identity_lora_routing_round_trips(self) -> None:
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            regions=(
                RegionDefinition(
                    "person",
                    "Person",
                    PixelBox(0, 0, 512, 1024),
                    "lface, an adult woman",
                    face_identity_prompt="lface, a specific woman with an oval face",
                ),
            ),
            loras=(
                SavedLora(
                    Path("lface.safetensors"),
                    global_scope=False,
                    region_ids=("person",),
                    strength=1.5,
                    routing_mode=CHARACTER_IDENTITY_LORA_ROUTING,
                    trigger_phrase="lface",
                ),
            ),
        )

        document = project_document(state)
        restored = project_state(document)

        self.assertEqual(
            document["loras"][0]["routing_mode"],
            CHARACTER_IDENTITY_LORA_ROUTING,
        )
        self.assertEqual(document["loras"][0]["trigger_phrase"], "lface")
        self.assertEqual(
            restored.loras[0].routing_mode,
            CHARACTER_IDENTITY_LORA_ROUTING,
        )
        self.assertEqual(restored.loras[0].trigger_phrase, "lface")
        self.assertEqual(
            restored.regions[0].face_identity_prompt,
            "lface, a specific woman with an oval face",
        )

    def test_version_twelve_lora_uses_standard_routing_defaults(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 12
        document["loras"] = [{"path": "style.safetensors", "global": True}]

        restored = project_state(document)

        self.assertEqual(restored.loras[0].routing_mode, STANDARD_LORA_ROUTING)
        self.assertEqual(restored.loras[0].trigger_phrase, "")

    def test_legacy_project_uses_existing_relaxation_default(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 8
        document["generation"].pop("regional_late_step_scale")

        self.assertEqual(project_state(document).regional_late_step_scale, 0.35)

    def test_version_eleven_project_uses_safe_face_refinement_defaults(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 11
        for key in tuple(document["generation"]):
            if key.startswith("face_detail_"):
                document["generation"].pop(key)

        restored = project_state(document)

        self.assertEqual(restored.face_detail_seed, 0)
        self.assertEqual(restored.face_detail_denoise, 0.15)
        self.assertEqual(restored.face_detail_blend, 0.5)

    def test_legacy_negative_prompt_is_discarded(self) -> None:
        document = project_document(
            ProjectState(
                canvas_width=1024,
                canvas_height=1024,
                regions=(
                    RegionDefinition(
                        "person",
                        "Person",
                        PixelBox(0, 0, 512, 1024),
                        "a person",
                    ),
                ),
            )
        )
        document["version"] = 14
        document["regions"][0]["negative_prompt"] = "legacy unused text"

        restored = project_state(document)

        self.assertEqual(restored.regions[0].negative_prompt, "")
        self.assertNotIn("negative_prompt", project_document(restored)["regions"][0])

    def test_legacy_project_uses_euler_simple_defaults(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 14
        document["generation"].pop("sampler")
        document["generation"].pop("scheduler")

        restored = project_state(document)

        self.assertEqual(restored.sampler, "euler")
        self.assertEqual(restored.scheduler, "simple")


if __name__ == "__main__":
    unittest.main()
