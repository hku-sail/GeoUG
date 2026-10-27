# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import os
import logging

import pyarrow.fs as pf
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
    num_data_dirs = len(data_dir_list)
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths):
        if "://" in data_dir:
            raise ValueError(
                "Public GeoUG supports local parquet paths only; "
                f"materialize the remote dataset locally first: {data_dir}"
            )
        else:
            # Recurse into sub-directories so a data_dir that groups its parquet
            # files under task sub-folders (e.g. vst_500k_bagel/{mi_*, si_*})
            # is fully picked up. Sorted for deterministic ordering across ranks.
            data_paths_per_dir = []
            for root, dirnames, names in os.walk(data_dir):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in names:
                    if name.endswith(".parquet"):
                        data_paths_per_dir.append(os.path.join(root, name))
            data_paths_per_dir.sort()
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        local_data_paths.extend(data_paths_per_dir[:num_data_path])

    if world_size > 1:
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks


def init_arrow_pf_fs(parquet_file_path):
    if "://" in parquet_file_path:
        raise ValueError(
            "Public GeoUG supports local parquet paths only; "
            f"materialize the remote file locally first: {parquet_file_path}"
        )
    return pf.LocalFileSystem()
