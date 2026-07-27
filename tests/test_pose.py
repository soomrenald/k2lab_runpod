from __future__ import annotations

import unittest

from k2_region_lab.pose import (
    POSE_JOINT_NAMES,
    PoseJoint,
    SubjectPose,
    default_subject_pose,
    default_volumetric_subject_pose,
    subject_pose_document,
    subject_pose_from_document,
)
from k2_region_lab.project import (
    LEGACY_POSE_MIGRATION_NOTICE,
    ProjectState,
    project_document,
    project_state,
)
from k2_region_lab.regions import PixelBox, RegionDefinition


class SubjectPoseTests(unittest.TestCase):
    def test_default_pose_has_every_openpose_joint(self) -> None:
        pose = default_subject_pose()

        self.assertEqual(
            tuple(joint.name for joint in pose.joints),
            POSE_JOINT_NAMES,
        )
        self.assertTrue(pose.enabled)

    def test_pose_round_trip_preserves_out_of_box_interaction_joint(self) -> None:
        pose = default_subject_pose()
        joints = tuple(
            PoseJoint(joint.name, -0.2, joint.y, joint.enabled)
            if joint.name == "left_wrist"
            else joint
            for joint in pose.joints
        )

        restored = subject_pose_from_document(subject_pose_document(SubjectPose(joints=joints)))

        self.assertEqual(restored.joint("left_wrist").x, -0.2)

    def test_ordinary_regions_reject_mannequins(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordinary region"):
            RegionDefinition(
                "object",
                "Lamp",
                PixelBox(10, 10, 100, 100),
                pose=default_subject_pose(),
            )

    def test_subject_region_requires_subject_spatial_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "subject spatial role"):
            RegionDefinition(
                "person",
                "Person",
                PixelBox(10, 10, 100, 200),
                region_type="subject",
                pose=default_subject_pose(),
            )

    def test_project_round_trip_preserves_subject_and_regular_region(self) -> None:
        state = ProjectState(
            1024,
            1024,
            pose_gating_enabled=True,
            pose_hard_gate_steps=3,
            pose_soft_gate_steps=2,
            regions=(
                RegionDefinition(
                    "person",
                    "Person",
                    PixelBox(64, 64, 448, 960),
                    prompt="an adult standing",
                    spatial_role="subject",
                    region_type="subject",
                    pose=default_volumetric_subject_pose(),
                ),
                RegionDefinition(
                    "lamp",
                    "Lamp",
                    PixelBox(640, 200, 900, 700),
                    prompt="a floor lamp",
                    spatial_role="auto",
                ),
            ),
        )

        restored = project_state(project_document(state))

        self.assertTrue(restored.pose_gating_enabled)
        self.assertEqual(restored.pose_hard_gate_steps, 3)
        self.assertEqual(restored.pose_soft_gate_steps, 2)
        self.assertEqual(restored.regions[0].region_type, "subject")
        self.assertIsNotNone(restored.regions[0].pose)
        self.assertEqual(restored.regions[1].region_type, "region")
        self.assertIsNone(restored.regions[1].pose)

    def test_v19_projects_load_regions_without_mannequins(self) -> None:
        document = project_document(ProjectState(1024, 1024))
        document["version"] = 19
        document["regions"] = [
            {
                "id": "legacy",
                "name": "Legacy subject",
                "box": {"x0": 0, "y0": 0, "x1": 512, "y1": 1024},
                "prompt": "a person",
                "enabled": True,
                "priority": 1,
                "spatial_role": "subject",
            }
        ]

        restored = project_state(document)

        self.assertEqual(restored.regions[0].region_type, "region")
        self.assertIsNone(restored.regions[0].pose)

    def test_v20_pose_control_migrates_without_silently_enabling_gating(self) -> None:
        document = project_document(ProjectState(1024, 1024))
        document["version"] = 20
        document["generation"].update(
            {
                "pose_conditioning_enabled": True,
                "pose_controlnet_model": "qwen-controlnet.safetensors",
                "pose_conditioning_strength": 1.0,
            }
        )
        document["regions"] = [{
            "id": "legacy-subject",
            "name": "Legacy subject",
            "box": {"x0": 64, "y0": 64, "x1": 448, "y1": 960},
            "prompt": "a person",
            "enabled": True,
            "priority": 1,
            "spatial_role": "subject",
            "region_type": "subject",
            "pose": subject_pose_document(default_subject_pose()),
        }]

        restored = project_state(document)
        canonical = project_document(restored)

        self.assertFalse(restored.pose_gating_enabled)
        self.assertEqual(restored.regions[0].pose.format, "k2-volumetric-pose-v1")
        self.assertIn(
            LEGACY_POSE_MIGRATION_NOTICE,
            restored.runtime["migration_notices"],
        )
        self.assertNotIn("pose_controlnet_model", canonical["generation"])
        self.assertNotIn("pose_conditioning_enabled", canonical["generation"])

    def test_advanced_sigma_knots_resample_when_step_counts_change(self) -> None:
        document = project_document(ProjectState(1024, 1024))
        document["generation"].update(
            {
                "steps": 4,
                "pose_hard_gate_steps": 1,
                "pose_soft_gate_steps": 1,
                "pose_sigma_schedule_mode": "advanced",
                "pose_sigma_knots": [0.0, 0.2, 0.8, 1.0],
            }
        )

        restored = project_state(document)

        self.assertEqual(len(restored.pose_sigma_knots), 7)
        self.assertEqual(restored.pose_sigma_knots[0], 0.0)
        self.assertEqual(restored.pose_sigma_knots[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
