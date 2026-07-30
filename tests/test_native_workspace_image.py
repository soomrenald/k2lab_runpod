from __future__ import annotations

import json
import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class NativeWorkspaceImageTests(unittest.TestCase):
    def test_native_image_is_pinned_and_has_no_comfyui_install_step(self) -> None:
        dockerfile = (ROOT / "Dockerfile.native-workspace").read_text(
            encoding="utf-8"
        )

        self.assertIn("ARG RUNPOD_BASE_IMAGE", dockerfile)
        self.assertIn("K2LAB_INFERENCE_BACKEND=native", dockerfile)
        self.assertIn(
            "K2LAB_WORKER_PYTHON=/opt/k2lab-venv/bin/python", dockerfile
        )
        self.assertIn(
            "K2LAB_NATIVE_TOKENIZER_PATH=/workspace/k2lab/models/tokenizers",
            dockerfile,
        )
        for pin in (
            "K2CORE_REF=903166f756614b13c0add0196fb5705206370dc3",
            "TORCH_VERSION=2.9.1",
            "PIP_VERSION=26.2",
            "SETUPTOOLS_VERSION=83.0.0",
            "WHEEL_VERSION=0.47.0",
            '"diffusers==0.39.0"',
            '"safetensors==0.8.0"',
            '"transformers==5.14.1"',
        ):
            self.assertIn(pin, dockerfile)
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn("native-web-requirements.lock", dockerfile)
        self.assertIn("pip install --no-deps /opt/k2lab-runpod", dockerfile)
        self.assertIn("python -m pip check", dockerfile)
        self.assertIn("python -m pip uninstall -y pip setuptools wheel", dockerfile)
        self.assertIn("COPY src /opt/k2lab-runpod/src", dockerfile)
        self.assertIn("COPY LICENSE MODEL_USE_POLICY.md", dockerfile)
        self.assertNotIn("COPY . /opt/k2lab-runpod", dockerfile)
        self.assertNotIn("COMFYUI_REPOSITORY", dockerfile)
        self.assertNotIn("COMFYUI_REF", dockerfile)
        self.assertNotIn("/opt/ComfyUI", dockerfile)
        self.assertNotIn("git clone", dockerfile)

    def test_native_web_requirements_match_locked_versions_and_have_hashes(
        self,
    ) -> None:
        requirements = (
            ROOT / "deploy" / "workspace" / "native-web-requirements.lock"
        ).read_text(encoding="utf-8")
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        locked_versions = {
            package["name"]: package["version"]
            for package in lock["package"]
            if "version" in package
        }
        exported = dict(
            re.findall(
                r"^([a-z0-9][a-z0-9._-]*)==([^ ;\\]+)",
                requirements,
                re.MULTILINE,
            )
        )

        for dependency in (
            "aiosqlite",
            "asyncpg",
            "cryptography",
            "fastapi",
            "httpx",
            "huggingface-hub",
            "numpy",
            "pillow",
            "pydantic",
            "safetensors",
            "sqlalchemy",
            "uvicorn",
        ):
            self.assertEqual(exported[dependency], locked_versions[dependency])
        self.assertNotIn("k2core", exported)
        self.assertNotIn("k2lab-runpod", exported)
        self.assertGreaterEqual(requirements.count("--hash=sha256:"), len(exported))

    def test_native_image_workflow_builds_scans_and_checks_absence(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "native-workspace-image.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("file: Dockerfile.native-workspace", workflow)
        self.assertIn('tags:\n      - "native-v*"', workflow)
        self.assertIn(
            "IMAGE: ghcr.io/${{ github.repository_owner }}/k2lab-runpod-workspace",
            workflow,
        )
        self.assertIn("push: ${{ startsWith(github.ref, 'refs/tags/native-v') }}", workflow)
        self.assertIn("Log in to GHCR for release-candidate builds", workflow)
        self.assertIn("steps.target.outputs.image", workflow)
        self.assertIn("test ! -e /opt/ComfyUI", workflow)
        self.assertIn("Smoke native imports", workflow)
        self.assertIn("Check locked Python environment", workflow)
        self.assertIn("test ! -e /opt/k2lab-venv/bin/pip", workflow)
        self.assertIn("Boot native agent from an empty workspace", workflow)
        self.assertIn("K2LAB_INFERENCE_BACKEND", workflow)
        self.assertIn("/v1/health", workflow)
        self.assertIn("severity: HIGH,CRITICAL", workflow)
        self.assertIn("version: v0.70.0", workflow)
        self.assertIn("native-workspace-image.spdx.json", workflow)
        self.assertIn("syft-version: v1.42.3", workflow)
        self.assertIn("Sign release-candidate digest with GitHub OIDC", workflow)

        health_command = re.search(
            r"docker exec .*? -c \\\n\s+'(?P<code>[^\n]+)'",
            workflow,
        )
        self.assertIsNotNone(health_command)
        compile(health_command.group("code"), "<native-workspace-health-smoke>", "exec")

        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("**/node_modules", dockerignore)

    def test_native_entrypoint_allows_empty_workspace_to_start_for_uploads(
        self,
    ) -> None:
        entrypoint = (
            ROOT / "deploy" / "workspace" / "entrypoint.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("import diffusers, k2core, safetensors, torch, transformers", entrypoint)
        self.assertNotIn(
            "The configured native tokenizer directory is unavailable", entrypoint
        )

    def test_published_native_rc_evidence_is_signed_and_gpu_accepted(self) -> None:
        evidence = json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "parity"
                / "integration"
                / "gate12_published_native_rc.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(evidence["status"], "published_signed_gpu_accepted")
        self.assertRegex(
            evidence["image"]["immutable"],
            r"^ghcr\.io/soomrenald/k2lab-runpod-workspace@sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(evidence["workflow"]["checks"]["trivy_high"], 0)
        self.assertEqual(evidence["workflow"]["checks"]["trivy_critical"], 0)
        self.assertTrue(evidence["signature"]["claims_validated"])
        self.assertTrue(evidence["signature"]["transparency_log_verified"])
        self.assertFalse(evidence["superseded_candidate"]["signed"])
        deployment = evidence["runpod_gpu_deployment"]
        self.assertEqual(deployment["status"], "passed")
        self.assertRegex(deployment["pod_id_suffix"], r"^[a-z0-9]{6}$")
        self.assertEqual(
            deployment["failure_recovery"]["expected_error_code"],
            "native_feature_unsupported",
        )
        self.assertEqual(deployment["native_generation"]["backend"], "native")
        self.assertTrue(deployment["rollback"]["same_pod"])
        self.assertEqual(
            deployment["rollback"]["generation"]["backend"],
            "comfyui",
        )
        self.assertTrue(deployment["cleanup"]["pod_deleted"])
        self.assertTrue(deployment["cleanup"]["pod_volume_deleted"])

    def test_licensed_native_rc_evidence_is_published_scanned_and_signed(self) -> None:
        evidence = json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "parity"
                / "integration"
                / "gate12_licensed_native_rc.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(evidence["status"], "published_scanned_signed")
        self.assertEqual(
            evidence["source"]["commit"],
            "cc905bb584149e7e12f0b65121aa6747826197c8",
        )
        self.assertRegex(
            evidence["image"]["immutable"],
            r"^ghcr\.io/soomrenald/k2lab-runpod-workspace@sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(evidence["workflow"]["checks"]["trivy_high"], 0)
        self.assertEqual(evidence["workflow"]["checks"]["trivy_critical"], 0)
        self.assertTrue(evidence["signature"]["workflow_sign_step_passed"])
        scope = evidence["distribution_scope"]
        self.assertTrue(scope["apache_license_in_native_image_definition"])
        self.assertTrue(scope["model_use_policy_in_native_image_definition"])
        self.assertFalse(scope["model_weights_bundled"])
        self.assertTrue(scope["operator_requested_upstream_downloads_supported"])
        self.assertFalse(scope["public_shared_inference_approved"])
        self.assertTrue(evidence["gpu_evidence"]["inherited"])
