"""Run GeoUG adaptive spatial reasoning on one image and question."""

import argparse
import json
import os
from pathlib import Path
from types import MethodType

from .paths import resolve_model_paths, validate_model_paths

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = _ROOT / "ckpt" / "geoug-spatial-rl-lambda0.1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Spatial RL checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", default="outputs/inference/spatial")
    parser.add_argument(
        "--transform-config",
        default=None,
        help="Transform YAML (default: <model-path>/transform_config.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=("adaptive", "force_depth", "force_answer"),
        default="adaptive",
    )
    parser.add_argument(
        "--intrinsics",
        help="Optional JSON file containing a 3x3 matrix or an intrinsics field.",
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--is-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return resolve_model_paths(parser.parse_args())


def _load_intrinsics(path):
    if not path:
        return None
    import numpy as np

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value["intrinsics"]
    intrinsics = np.asarray(value, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError("--intrinsics must contain a 3x3 matrix")
    return intrinsics


def _prepare_engine_imports():
    # The RL engine carries the forward-dispatch hooks used by the adaptive
    # inferencer. Cache GeoUG's public data package first, then put the engine's
    # BAGEL implementation ahead of the base implementation.
    from train.rl.bootstrap import prepare_engine_imports

    prepare_engine_imports()


def main(args=None):
    args = args or parse_args()
    validate_model_paths(args)
    os.environ.setdefault("BAGEL_REASONING_TOKENS", "1")
    _prepare_engine_imports()

    import numpy as np
    import torch
    from PIL import Image

    from evaluation.depth.bagel_depth_runtime import BagelDepthModel
    from train.rl.prompt import UNIFIED_PROMPT
    from train.rl.spatial_inferencer import make_inferencer_cls

    if not torch.cuda.is_available():
        raise RuntimeError("Spatial inference requires a CUDA-capable GPU")
    image = Image.open(args.image).convert("RGB")
    bagel = BagelDepthModel(
        model_path=args.model_path,
        transform_config=args.transform_config,
        device=torch.device("cuda", 0),
        is_ema=args.is_ema,
    )
    # Keep sampling independent of checkpoint initialization/loading details.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    inferencer_type = make_inferencer_cls()
    inferencer = inferencer_type(
        model=bagel.gen_model,
        vae_model=bagel.vae_model,
        tokenizer=bagel.tokenizer,
        vae_transform=bagel.vae_transform,
        vit_transform=bagel.vit_transform,
        new_token_ids=bagel.new_token_ids,
    )
    inferencer.setup_spatial(args.transform_config, args.output_dir)

    generated = {}

    def generate_depth(_self, _image_key, _depth_dir=None):
        depth = bagel.infer_depth(
            [image],
            "<GENERATE_DEPTH>Estimate the metric depth map.</GENERATE_DEPTH>",
            num_timesteps=args.num_timesteps,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
        )
        generated["depth"] = depth
        return depth

    # Training reads a precomputed cache; public inference generates the same
    # observation lazily, only when the policy takes the depth branch.
    inferencer.load_cached_depth = MethodType(generate_depth, inferencer)

    force = {
        "adaptive": None,
        "force_depth": "depth",
        "force_answer": "answer",
    }[args.mode]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = inferencer.rollout_one(
            image_pil=image,
            question=args.question,
            image_key=Path(args.image).stem,
            system_prompt=UNIFIED_PROMPT,
            intrinsics=_load_intrinsics(args.intrinsics),
            image_wh=image.size,
            max_rounds=args.max_rounds,
            max_text_tokens=args.max_text_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            force=force,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    if "depth" in generated:
        np.save(
            output_dir / f"{stem}.depth.npy",
            generated["depth"].numpy().astype(np.float32),
        )
    record = {
        "image": str(Path(args.image)),
        "question": args.question,
        "mode": args.mode,
        "response": result["text"],
        "used_depth": bool(result["used_depth"]),
        "reached_decide": bool(result["reached_decide"]),
        "forced": bool(result["forced"]),
        "n_rounds": int(result["n_rounds"]),
    }
    output_path = output_dir / f"{stem}.json"
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result["text"])
    print(f"\nresult: {output_path}")


if __name__ == "__main__":
    main()
