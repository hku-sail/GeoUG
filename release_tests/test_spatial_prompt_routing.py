from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VLMEVAL_ROOT = ROOT / "third_party" / "VLMEvalKit"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompting = _load_module(
    "geoug_prompting", VLMEVAL_ROOT / "vlmeval" / "prompting.py"
)
textual_coords = _load_module(
    "geoug_textual_coords",
    VLMEVAL_ROOT
    / "vlmeval"
    / "dataset"
    / "utils"
    / "spatial_bench"
    / "textual_coords.py",
)
build_prompt = prompting.build_prompt


class _CustomPromptModel:
    def use_custom_prompt(self, dataset):
        return True

    def build_prompt(self, line, dataset):
        return [{"type": "text", "value": "generic model prompt"}]


@pytest.mark.parametrize(
    ("style", "point_a"),
    [
        ("qwen25", "(200, 200)"),
        ("vst", "(200, 200)"),
        ("qwen3", "(250, 500)"),
        ("internvl", "[250, 500]"),
        ("bagel", "<point>250 500</point>"),
    ],
)
def test_spatial_dataset_prompt_wins_and_renders_coordinates(monkeypatch, style, point_a):
    def fixed_grid(selected_style, width, height):
        if textual_coords.STYLES[selected_style]["scale"] is None:
            return width, height
        return 1000, 1000

    monkeypatch.setattr(textual_coords, "grid", fixed_grid)

    class SpatialDataset:
        USE_DATASET_PROMPT = True

        def build_prompt(self, line):
            a = textual_coords.render_point(style, 250, 500, (1000, 1000), 800, 400)
            b = textual_coords.render_point(style, 750, 500, (1000, 1000), 800, 400)
            return [
                {
                    "type": "text",
                    "value": (
                        f"Which is closer, {a} or {b}?\n"
                        f"{textual_coords.ANSWER_INSTRUCTION}"
                    ),
                }
            ]

    prompt = build_prompt(
        _CustomPromptModel(),
        SpatialDataset(),
        f"DA2KTextual_circular_{style}",
        {},
    )
    text = prompt[-1]["value"]

    assert "generic model prompt" not in text
    assert point_a in text
    assert textual_coords.ANSWER_INSTRUCTION in text


def test_regular_mcq_still_uses_model_custom_prompt():
    class Dataset:
        def build_prompt(self, line):
            return [{"type": "text", "value": "dataset prompt"}]

    prompt = build_prompt(_CustomPromptModel(), Dataset(), "RegularMCQ", {})
    assert prompt[-1]["value"] == "generic model prompt"


def test_all_coordinate_dataset_classes_force_dataset_prompt():
    source_path = (
        VLMEVAL_ROOT / "vlmeval" / "dataset" / "spatial_textual.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = {
        node.name: {
            target.id
            for statement in node.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert "USE_DATASET_PROMPT" in classes["CVBench3DTexutalCircular"]
    assert "USE_DATASET_PROMPT" in classes["PointTextualCircular"]


def test_3dcuebench_circular_uses_binary_option_columns():
    source = (
        VLMEVAL_ROOT / "vlmeval" / "dataset" / "spatialdepth3d.py"
    ).read_text(encoding="utf-8")
    assert "letters = OPTION_LETTERS[:2]" in source
    assert "texts = [row[c] for c in OPTION_LETTERS]" not in source


def test_local_tsv_is_not_sent_to_url_downloader():
    source = (
        VLMEVAL_ROOT / "vlmeval" / "dataset" / "image_base.py"
    ).read_text(encoding="utf-8")
    local_guard = source.index("if osp.isfile(url):")
    download = source.index("download_file(url, data_path)")
    assert local_guard < download


def test_cvbench_scoring_keeps_in_memory_fallback_result():
    source = (
        VLMEVAL_ROOT / "vlmeval" / "dataset" / "cvbench.py"
    ).read_text(encoding="utf-8")
    assert "data = load(get_intermediate_file_path(eval_file" not in source
