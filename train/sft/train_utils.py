# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os


def create_logger(logging_dir, rank, filename="log"):
    """Create a rank-zero logger that writes to stdout and a log file."""
    if rank == 0 and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/{filename}.txt"),
            ],
        )
        return logging.getLogger(__name__)

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger


def append_loss_record(results_dir, record):
    """Append one JSON training-metrics record and flush it immediately."""
    path = os.path.join(results_dir, "loss.jsonl")
    try:
        os.makedirs(results_dir, exist_ok=True)
        with open(path, "a") as file:
            file.write(json.dumps(record) + "\n")
    except OSError as error:
        logging.getLogger(__name__).warning(
            "could not append step %s to %s: %s",
            record.get("step"),
            path,
            error,
        )


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [
        name
        for name in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, name))
    ]
    if not step_dirs:
        return None
    step_dirs = sorted(step_dirs, key=int)
    return os.path.join(checkpoint_dir, step_dirs[-1])


def dump_data_indexes(
    dump_dir,
    step,
    rank,
    total_norm,
    ce,
    batch_data_indexes,
    reason,
):
    """Append sample locations for a training batch to a per-rank JSONL file."""
    if not batch_data_indexes:
        return
    os.makedirs(dump_dir, exist_ok=True)
    samples = []
    for item in batch_data_indexes:
        data_idx = item.get("data_indexes")
        row_idx = (
            data_idx[1]
            if isinstance(data_idx, (list, tuple)) and len(data_idx) > 1
            else data_idx
        )
        samples.append(
            {
                "dataset_name": item.get("dataset_name"),
                "file_path": item.get("file_path"),
                "row_group_id": item.get("row_group_id"),
                "row_idx": row_idx,
                "data_indexes": data_idx,
                "worker_id": item.get("worker_id"),
            }
        )
    record = {
        "step": step,
        "rank": rank,
        "reason": reason,
        "total_norm": total_norm,
        "ce": ce,
        "num_samples": len(samples),
        "samples": samples,
    }
    path = os.path.join(dump_dir, f"data_indexes_rank{rank}.jsonl")
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")
