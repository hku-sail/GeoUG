from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_module(
    "geoug_3dcuebench_validate", ROOT / "benchmarks" / "3dcuebench" / "validate.py"
)
evaluator = _load_module(
    "geoug_3dcuebench_evaluate", ROOT / "benchmarks" / "3dcuebench" / "evaluate.py"
)


@pytest.mark.parametrize("text,expected", [
    ("This is a difficult question. Answer: B", "B"),
    ("I cannot determine a reliable answer.", ""),
    ("A is farther away. Answer: B", "B"),
    ("<think>Answer: A</think> B", "B"),
    ("Answer: A. Final answer: B", "B"),
    ("<answer>b</answer>", "B"),
    ("A or B", ""),
    ("b", "B"),
    ("B. The blue object", "B"),
    ("", ""),
])
def test_answer_extraction(text, expected):
    assert evaluator.extract_letter(text) == expected


def _write_release(root: Path, *, checksum_override=None, image_path_override=None):
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.png"
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    bins = ("easy", "medium", "hard", "extreme")
    rows = []
    for index in range(8):
        answer = "A" if index % 2 == 0 else "B"
        distance_a, distance_b = ((1.0, 2.0) if answer == "A" else (2.0, 1.0))
        rows.append(
            {
                "question_id": f"q{index}",
                "image_id": "sample",
                "image": {
                    "bytes": None,
                    "path": image_path_override or "images/sample.png",
                },
                "image_sha256": checksum_override or digest,
                "source": "fixture",
                "domain": "indoor",
                "split": "depth" if index < 4 else "distance",
                "difficulty_bin": bins[index % 4],
                "relative_gap": 0.2,
                "question": (
                    "Which is closer, <point>2 2</point> A or "
                    "<point>6 6</point> B?"
                ),
                "option_a": "A",
                "option_b": "B",
                "answer": answer,
                "point_a": [2, 2],
                "point_b": [6, 6],
                "anchor": None,
                "point_anchor": None,
                "image_width": 8,
                "image_height": 8,
                "grid_width": 8,
                "grid_height": 8,
                "distance_a": distance_a,
                "distance_b": distance_b,
            }
        )
    parquet_path = root / "release.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return parquet_path, rows


def test_manifest_paths_are_relative_to_parquet(tmp_path):
    parquet_path, _ = _write_release(tmp_path)
    validator.validate(parquet_path, expected_questions=8, expected_images=1)


def test_manifest_checksum_is_enforced(tmp_path):
    parquet_path, _ = _write_release(tmp_path, checksum_override="0" * 64)
    with pytest.raises(AssertionError, match="checksum mismatch"):
        validator.validate(parquet_path, expected_questions=8, expected_images=1)


def test_manifest_cannot_escape_release_directory(tmp_path):
    parquet_path, _ = _write_release(tmp_path, image_path_override="../sample.png")
    with pytest.raises(AssertionError, match="escapes the release directory"):
        validator.validate(parquet_path, expected_questions=8, expected_images=1)


def test_perfect_predictions_score_one(tmp_path):
    parquet_path, rows = _write_release(tmp_path)
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "question_id": [row["question_id"] for row in rows],
            "prediction": [f"Answer: {row['answer']}" for row in rows],
        }
    ).to_csv(predictions_path, index=False)

    report, scored = evaluator.score(parquet_path, predictions_path)
    assert report["overall"]["accuracy"] == 1.0
    assert scored["correct"].all()

