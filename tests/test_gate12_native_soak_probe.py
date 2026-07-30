from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "gate12_native_soak_probe.py"
)
SPEC = importlib.util.spec_from_file_location("gate12_native_soak_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class Gate12NativeSoakProbeTests(unittest.TestCase):
    def test_growth_summary_uses_warm_stable_windows(self) -> None:
        values = [100, 110, 120, 130, 140, *([150] * 95)]

        result = PROBE.growth_summary(values, tolerance=4)

        self.assertEqual(result["samples"], 100)
        self.assertEqual(result["growth"], 0)
        self.assertTrue(result["passed"])

    def test_growth_summary_fails_persistent_growth(self) -> None:
        result = PROBE.growth_summary(
            [index * 10 for index in range(100)],
            tolerance=64,
        )

        self.assertGreater(result["growth"], 64)
        self.assertFalse(result["passed"])

    def test_performance_summary_reports_distribution(self) -> None:
        iterations = [
            {
                "duration_seconds": float(index),
                "timings": {
                    "text_encoding_seconds": 1.0,
                    "transformer_seconds": 2.0,
                    "vae_decode_seconds": 3.0,
                    "image_output_seconds": 4.0,
                },
            }
            for index in range(1, 6)
        ]

        result = PROBE.performance_summary(iterations)

        self.assertEqual(result["jobs"], 5)
        self.assertEqual(result["p50_generation_seconds"], 3.0)
        self.assertEqual(result["p95_generation_seconds"], 5.0)
        self.assertEqual(result["mean_transformer_seconds"], 2.0)

    def test_state_records_and_rejects_workload_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({"prompt": "synthetic"}), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "state": root / "state.json",
                    "count": 100,
                    "fixture": fixture,
                    "width": 512,
                    "height": 512,
                    "steps": 8,
                    "transformer_sha256": "1" * 64,
                    "text_encoder_sha256": "2" * 64,
                    "vae_sha256": "3" * 64,
                    "tokenizer_sha256": "4" * 64,
                },
            )()
            state = PROBE.load_state(args)
            PROBE.atomic_json(args.state, state)
            args.steps = 2

            with self.assertRaisesRegex(ValueError, "workload"):
                PROBE.load_state(args)
