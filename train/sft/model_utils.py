import os

import torch
from loguru import logger
from tabulate import tabulate
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel


_DISTRIBUTED_WRAPPERS = (
    torch.nn.parallel.DistributedDataParallel,
    torch.nn.DataParallel,
    FullyShardedDataParallel,
)


def print_model_params(model):
    """Log trainable parameter shapes, counts, and their share of the model."""
    while isinstance(model, _DISTRIBUTED_WRAPPERS):
        model = model.module

    params_trainable = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        num_params = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
        total_params += num_params
        if param.requires_grad:
            params_trainable.append([name, tuple(param.shape), num_params])
            trainable_params += num_params

    params_trainable.sort(key=lambda row: row[2], reverse=True)
    for row in params_trainable:
        row[2] = f"{row[2]:,}"

    table = tabulate(
        params_trainable,
        ["Name", "Shape", "Parameters"],
        tablefmt="pretty",
        stralign="left",
    )
    trainable_percent = (
        trainable_params / total_params * 100 if total_params > 0 else 0
    )
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(
            "\n"
            f"\033[93mTrainable parameters: (~{trainable_params / 1e6:.2f}M) / "
            f"(~{total_params / 1e6:.2f}M)[{trainable_percent:.2f}%]\n"
            f"{table}\n\033[0m"
        )
    return table


def color_logger(message):
    logger.info(f"\n\033[93m{message}\n\033[0m")
