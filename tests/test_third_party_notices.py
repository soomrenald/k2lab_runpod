from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ThirdPartyNoticeTests(unittest.TestCase):
    def test_notice_tracks_runtime_dependencies_and_comfy_image_revision(self) -> None:
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile.workspace").read_text(encoding="utf-8")

        for dependency in (
            "NumPy",
            "Pillow",
            "Pydantic",
            "safetensors",
            "Diffusers",
            "PyTorch",
            "Transformers",
            "aiosqlite",
            "asyncpg",
            "cryptography",
            "FastAPI",
            "huggingface_hub",
            "HTTPX",
            "SQLAlchemy",
            "Uvicorn",
        ):
            self.assertIn(dependency.casefold(), notice.casefold())
        comfy_ref = re.search(r"^ARG COMFYUI_REF=([0-9a-f]{40})$", dockerfile, re.MULTILINE)
        self.assertIsNotNone(comfy_ref)
        self.assertIn(comfy_ref.group(1), notice)
        self.assertIn("RUNPOD_BASE_IMAGE", notice)
        self.assertIn("k2core", manifest)
        self.assertIn("no declared project license", notice)
