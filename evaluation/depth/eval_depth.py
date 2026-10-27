"""Distributed BAGEL metric-depth evaluation entry point."""

import argparse
import json
import math
import os
import zlib
from pathlib import Path

from inference.paths import resolve_model_paths, validate_model_paths

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = _ROOT / "ckpt" / "geoug-metric-depth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BAGEL depth maps and evaluate depth metrics."
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="JSON file describing one or more depth benchmarks.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Metric-depth checkpoint directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/depth_eval",
        help="Directory for metrics and optional predictions.",
    )
    parser.add_argument(
        "--transform-config",
        default=None,
        help="Image/depth transform YAML (default: <model-path>/transform_config.yaml).",
    )
    parser.add_argument(
        "--depth-type",
        choices=("affine_invariant", "metric"),
        default="affine_invariant",
        help=(
            "Interpret generated depth as affine-invariant or metric. Metric "
            "predictions also receive all compatible invariant metrics."
        ),
    )
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; each sample derives a stable seed from its filename.",
    )
    parser.add_argument(
        "--include-vit-incontext",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--is-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load ema.safetensors instead of model.safetensors.",
    )
    parser.add_argument(
        "--prompt",
        default="<GENERATE_DEPTH>Estimate the metric depth map.</GENERATE_DEPTH>",
    )
    parser.add_argument(
        "--fov-prompt",
        action="store_true",
        help="Append ground-truth horizontal FOV to the generation prompt.",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Save per-sample prediction and ground truth as compressed NPZ.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse predictions already saved below output-dir/depth.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per process.",
    )
    return resolve_model_paths(parser.parse_args())


def _prompt_with_fov(prompt: str, horizontal_fov: float) -> str:
    sentence = f"The horizontal field of view is {horizontal_fov:.1f} degrees."
    closing_tag = "</GENERATE_DEPTH>"
    if closing_tag in prompt:
        prefix, suffix = prompt.rsplit(closing_tag, 1)
        return f"{prefix.rstrip()} {sentence}{closing_tag}{suffix}"
    return f"{prompt.rstrip()} {sentence}"


def _single_sample(batch):
    return batch[0]


def main(args) -> None:
    validate_model_paths(args)
    import numpy as np
    import torch
    import torch.distributed as distributed
    from PIL import Image
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from .bagel_depth_runtime import BagelDepthModel
    from .eval_dataset import DepthEvalDataset
    from .metrics import average_metrics, compute_depth_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("Depth generation requires a CUDA-capable GPU")

    distributed_run = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed_run:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    output_root = Path(args.output_dir)
    metrics_root = output_root / "metrics"
    if rank == 0:
        metrics_root.mkdir(parents=True, exist_ok=True)
        (output_root / "args.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )
    if distributed_run:
        distributed.barrier()

    config_path = Path(args.dataset_config)
    benchmarks = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("dataset-config must be a non-empty JSON object")

    model = BagelDepthModel(
        model_path=args.model_path,
        transform_config=args.transform_config,
        device=device,
        is_ema=args.is_ema,
    )
    generation_options = {
        "include_vit_incontext": args.include_vit_incontext,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "num_timesteps": args.num_timesteps,
    }
    all_summaries = {}

    try:
        for benchmark_name, benchmark_config in benchmarks.items():
            if not isinstance(benchmark_config, dict):
                raise ValueError(
                    f"Config for benchmark {benchmark_name!r} must be an object"
                )
            dataset = DepthEvalDataset(
                benchmark_name=benchmark_name, **benchmark_config
            )
            indices = list(range(rank, len(dataset), world_size))
            local_dataset = Subset(dataset, indices)
            dataloader = DataLoader(
                local_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=_single_sample,
            )
            iterator = (
                tqdm(dataloader, desc=benchmark_name)
                if rank == 0
                else dataloader
            )
            local_metrics = []
            for sample in iterator:
                image = sample["image"]
                image_pil = Image.fromarray(
                    (
                        image.numpy().transpose(1, 2, 0).clip(0, 1) * 255
                    ).astype(np.uint8)
                )
                safe_filename = sample["filename"].replace("/", "_").replace(
                    "\\", "_"
                )
                prediction_path = (
                    output_root
                    / "depth"
                    / benchmark_name
                    / f"{safe_filename}.npz"
                )
                horizontal_fov = math.degrees(
                    2
                    * math.atan(
                        1.0 / (2.0 * float(sample["intrinsics"][0, 0]))
                    )
                )
                prompt = (
                    _prompt_with_fov(args.prompt, horizontal_fov)
                    if args.fov_prompt
                    else args.prompt
                )
                resumed = args.resume and prediction_path.exists()
                if resumed:
                    with np.load(prediction_path) as saved:
                        depth = torch.from_numpy(
                            saved["pred"].astype(np.float32)
                        )
                else:
                    sample_seed = args.seed + zlib.crc32(
                        sample["filename"].encode("utf-8")
                    )
                    torch.manual_seed(sample_seed)
                    torch.cuda.manual_seed_all(sample_seed)
                    depth = model.infer_depth(
                        [image_pil], prompt, **generation_options
                    )

                prediction = {f"depth_{args.depth_type}": depth.float()}
                metrics = compute_depth_metrics(prediction, sample)
                if args.depth_type == "metric" and sample["is_metric"]:
                    mask = sample["depth_mask"]
                    ratio = (
                        sample["depth"][mask]
                        / depth[mask].clamp_min(1e-6)
                    ).median()
                    metrics["scale"] = {
                        "log_gt_over_pred": ratio.log().item()
                    }
                local_metrics.append(metrics)

                if args.save_depth and not resumed:
                    prediction_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    np.savez_compressed(
                        prediction_path,
                        pred=depth.numpy().astype(np.float16),
                        gt=sample["depth"].numpy().astype(np.float16),
                        mask=sample["depth_mask"].numpy(),
                        intrinsics=sample["intrinsics"].numpy(),
                        is_metric=sample["is_metric"],
                        hfov=horizontal_fov,
                        prompt=prompt,
                    )

            if distributed_run:
                gathered = [None] * world_size
                distributed.all_gather_object(gathered, local_metrics)
                metrics = [
                    item for process_metrics in gathered for item in process_metrics
                ]
            else:
                metrics = local_metrics

            if rank == 0:
                summary = average_metrics(metrics)
                all_summaries[benchmark_name] = summary
                payload = json.dumps(
                    {benchmark_name: summary}, indent=2
                )
                print(
                    f"{benchmark_name}: {len(metrics)} samples\n{payload}"
                )
                (metrics_root / f"{benchmark_name}.json").write_text(
                    payload, encoding="utf-8"
                )
                (
                    metrics_root / f"{benchmark_name}_per_sample.json"
                ).write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            if distributed_run:
                distributed.barrier()

        if rank == 0:
            all_summaries["mean"] = average_metrics(
                all_summaries.values()
            )
            (metrics_root / "all.json").write_text(
                json.dumps(all_summaries, indent=2), encoding="utf-8"
            )
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
