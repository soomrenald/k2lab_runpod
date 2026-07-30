from pathlib import Path

import numpy as np
from PIL import Image

from k2lab.depth import evaluate as depth_evaluate


def test_parse_image_requires_label_and_path() -> None:
    assert depth_evaluate._parse_image("correct=/tmp/image.png") == (
        "correct",
        Path("/tmp/image.png"),
    )


def test_comparative_metrics_rank_matching_depth_higher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reference = np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000
    reference_path = tmp_path / "depth.png"
    Image.fromarray(reference).save(reference_path)

    matching = np.arange(64, dtype=np.float32).reshape(8, 8)
    shuffled = matching[:, ::-1].copy()
    monkeypatch.setattr(
        depth_evaluate,
        "_estimate_depths",
        lambda *args, **kwargs: {"correct": matching, "shuffled": shuffled},
    )

    result = depth_evaluate.evaluate(
        reference_path,
        [("correct", tmp_path / "correct.png"), ("shuffled", tmp_path / "shuffled.png")],
        estimator="test-estimator",
        device="cpu",
    )

    correct = result["metrics"]["correct"]
    shuffled_result = result["metrics"]["shuffled"]
    assert correct["depth_correlation"] > shuffled_result["depth_correlation"]
    assert correct["depth_rank_correlation"] > shuffled_result["depth_rank_correlation"]
    assert correct["edge_alignment"] >= shuffled_result["edge_alignment"]
