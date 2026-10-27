from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.depth import eval_depth
from inference import depth, spatial

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_portability", ROOT / "scripts/check_portability.py")
portability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portability)


@pytest.mark.parametrize("module,required", [
    (depth, ["--image", "image.jpg"]),
    (spatial, ["--image", "image.jpg", "--question", "Which is closer?"]),
    (eval_depth, ["--dataset-config", "datasets.json"]),
])
def test_custom_model_selects_its_own_transform(module, required, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["test", *required, "--model-path", "custom-model"])
    args = module.parse_args()
    assert Path(args.model_path) == tmp_path / "custom-model"
    assert Path(args.transform_config) == tmp_path / "custom-model/transform_config.yaml"

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--transform-config", "override.yaml"])
    assert Path(module.parse_args().transform_config) == tmp_path / "override.yaml"


def test_missing_model_reports_resources_before_gpu_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test", "--image", "image.jpg", "--model-path", str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="Missing checkpoint resources"):
        depth.main()


def test_source_has_no_external_links():
    portability.check(ROOT)


def test_portability_check_rejects_external_directory_links(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "hidden_dependency").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="hidden_dependency"):
        portability.check(source)


def test_rl_bootstrap_selects_local_packages():
    # Run in a fresh interpreter: the base and RL model packages intentionally
    # have the same legacy name and must not be mixed in one process.
    code = '''
from pathlib import Path
import importlib.util
from train.rl.bootstrap import prepare_engine_imports
prepare_engine_imports()
prepare_engine_imports()
import data, train
from data.constant import DECIDE
root = Path.cwd()
assert Path(data.__file__).parent == root / "data"
assert Path(train.__file__).parent == root / "train"
assert Path(importlib.util.find_spec("modeling").origin).parent == root / "train/rl/engine/Bagel/modeling"
assert DECIDE == "<DECIDE>"
'''
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


@pytest.mark.parametrize("script", ["infer_depth.sh", "infer_spatial.sh", "eval_depth.sh", "train_sft.sh", "train_rl.sh"])
def test_launchers_preserve_caller_relative_paths(script, tmp_path):
    probe = tmp_path / "python-probe"
    probe.write_text("#!/bin/sh\npwd\nprintf '%s\\n' \"$@\"\n")
    probe.chmod(0o755)
    env = os.environ.copy()
    env.update(PYTHON=str(probe), MODEL_PATH="local-model", OUTPUT_DIR=str(tmp_path / "out"),
               GEOUG_DATASET_MANIFEST="data.json", RL_DATASET="rl-data", DEPTH_CACHE_DIR="depth",
               EVAL_QUESTIONS="eval.parquet", EVAL_DEPTH_CACHE="eval-depth")
    result = subprocess.run(["bash", str(ROOT / "scripts" / script), "--help"],
                            cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines()[0] == str(tmp_path)
