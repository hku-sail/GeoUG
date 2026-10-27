"""Generate a metric depth map for one image with a GeoUG checkpoint."""

import argparse
from pathlib import Path

from .paths import resolve_model_paths, validate_model_paths

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = _ROOT / "ckpt" / "geoug-metric-depth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Metric-depth checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/inference/depth")
    parser.add_argument(
        "--transform-config",
        default=None,
        help="Transform YAML (default: <model-path>/transform_config.yaml).",
    )
    parser.add_argument(
        "--prompt",
        default="<GENERATE_DEPTH>Estimate the metric depth map.</GENERATE_DEPTH>",
    )
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--is-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return resolve_model_paths(parser.parse_args())


def _save_preview(depth, path):
    import numpy as np
    from PIL import Image

    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        near, far = np.quantile(depth[valid], [0.02, 0.98])
        scaled = (depth - near) / max(float(far - near), 1e-6)
        preview[valid] = ((1.0 - scaled[valid].clip(0, 1)) * 255).astype(
            np.uint8
        )
    Image.fromarray(preview, mode="L").save(path)


def main(args=None):
    args = args or parse_args()
    validate_model_paths(args)

    import numpy as np
    import torch
    from PIL import Image

    from evaluation.depth.bagel_depth_runtime import BagelDepthModel

    if not torch.cuda.is_available():
        raise RuntimeError("Depth generation requires a CUDA-capable GPU")
    image = Image.open(args.image).convert("RGB")
    model = BagelDepthModel(
        model_path=args.model_path,
        transform_config=args.transform_config,
        device=torch.device("cuda", 0),
        is_ema=args.is_ema,
    )
    # Keep sampling independent of checkpoint initialization/loading details.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    depth = model.infer_depth(
        [image],
        args.prompt,
        num_timesteps=args.num_timesteps,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
    ).numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    depth_path = output_dir / f"{stem}.depth.npy"
    preview_path = output_dir / f"{stem}.depth.png"
    np.save(depth_path, depth.astype(np.float32))
    _save_preview(depth, preview_path)
    print(f"depth: {depth_path}")
    print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()
