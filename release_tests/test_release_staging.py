from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training_release = _load_module(
    "geoug_prepare_training_release",
    ROOT / "scripts" / "prepare_training_release.py",
)
evaluation_release = _load_module(
    "geoug_prepare_evaluation_release",
    ROOT / "scripts" / "prepare_evaluation_release.py",
)


def test_training_staging_materializes_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target.npy"
    target.write_bytes(b"depth")
    (source / "question.npy").symlink_to(target)
    destination = tmp_path / "staged"

    outputs = training_release._materialize(source, destination)

    assert outputs == [destination / "question.npy"]
    assert outputs[0].is_file()
    assert not outputs[0].is_symlink()
    assert outputs[0].read_bytes() == b"depth"


def test_evaluation_manifest_contains_no_source_path(tmp_path, monkeypatch):
    source = tmp_path / "private" / "run"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text('{"accuracy": 1.0}', encoding="utf-8")
    output = tmp_path / "release"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_evaluation_release.py",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--label",
            "test",
        ],
    )

    evaluation_release.main()
    manifest_path = output / "test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert str(source) not in manifest_path.read_text(encoding="utf-8")
    assert manifest["files"][0]["path"] == "metrics.json"
