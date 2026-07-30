from __future__ import annotations

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
            "K2CORE_REF=237fd23dc4a578e9d1a095fac0587d4d6bdf88e4",
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
        self.assertIn("push: false", workflow)
        self.assertIn("test ! -e /opt/ComfyUI", workflow)
        self.assertIn("Smoke native imports", workflow)
        self.assertIn("severity: HIGH,CRITICAL", workflow)
        self.assertIn("native-workspace-image.spdx.json", workflow)

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
