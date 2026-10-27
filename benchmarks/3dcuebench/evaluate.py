#!/usr/bin/env python3
"""Score 3DCueBench predictions with the official v1.0 accuracy breakdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


LETTER_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")
ANSWER_RE = re.compile(r"\b(?:final\s+)?answer\s*[:：]\s*([AB])\b", re.IGNORECASE)


def extract_letter(value):
    # Match the original reasoning evaluator's answer-region convention.
    # Lower-case 'a' in prose is an article, not an option selection.
    text = str(value).rsplit("</think>", 1)[-1]
    text = re.sub(r"<\|.*?\|>", " ", text)
    tagged = re.findall(r"<answer>\s*([AB])\s*</answer>", text, re.IGNORECASE)
    if tagged:
        return tagged[-1].upper()
    declared = ANSWER_RE.findall(text)
    if declared:
        return declared[-1].upper()
    bare = re.fullmatch(r"\s*\(?([AB])\)?[.!]?\s*", text, re.IGNORECASE)
    if bare:
        return bare.group(1).upper()
    letters = set(LETTER_RE.findall(text))
    return letters.pop() if len(letters) == 1 else ""


def read_predictions(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def score(dataset: Path, predictions: Path):
    truth = pq.read_table(
        dataset,
        columns=["question_id", "answer", "split", "difficulty_bin", "domain", "source"],
    ).to_pandas()
    pred = read_predictions(predictions)
    required = {"question_id", "prediction"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
    if pred["question_id"].duplicated().any():
        raise ValueError("prediction file has duplicate question_id values")

    merged = truth.merge(
        pred[["question_id", "prediction"]], on="question_id", how="left", validate="one_to_one")
    merged["parsed_answer"] = merged["prediction"].fillna("").map(extract_letter)
    merged["correct"] = merged["parsed_answer"] == merged["answer"]
    merged["parsed"] = merged["parsed_answer"] != ""

    def aggregate(columns):
        return (
            merged.groupby(columns, dropna=False)
            .agg(questions=("question_id", "size"),
                 accuracy=("correct", "mean"),
                 parse_rate=("parsed", "mean"))
            .reset_index()
        )

    reports = {
        "overall": {
            "questions": int(len(merged)),
            "accuracy": float(merged["correct"].mean()),
            "parse_rate": float(merged["parsed"].mean()),
        },
        "by_split": aggregate(["split"]).to_dict(orient="records"),
        "by_difficulty": aggregate(["split", "difficulty_bin"]).to_dict(orient="records"),
        "by_domain": aggregate(["split", "domain"]).to_dict(orient="records"),
        "by_source": aggregate(["split", "source"]).to_dict(orient="records"),
    }
    return reports, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports, rows = score(args.dataset, args.predictions)
    print(json.dumps(reports, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        rows.to_csv(args.output.with_suffix(".rows.csv"), index=False)


if __name__ == "__main__":
    main()

