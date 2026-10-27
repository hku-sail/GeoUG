"""Resolve checkpoint resources without depending on the checkout location."""

from pathlib import Path


def resolve_model_paths(args):
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    config = getattr(args, "transform_config", None)
    args.transform_config = str(
        Path(config).expanduser().resolve()
        if config else Path(args.model_path) / "transform_config.yaml"
    )
    return args


def validate_model_paths(args):
    """Report missing local artifacts before importing the GPU model stack."""
    resolve_model_paths(args)
    root = Path(args.model_path)
    required = [
        root / ("ema.safetensors" if args.is_ema else "model.safetensors"),
        root / "ae.safetensors",
        root / "llm_config.json",
        root / "vit_config.json",
        root / "tokenizer_config.json",
        Path(args.transform_config),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint resources:\n  " + "\n  ".join(missing)
            + "\nSee ckpt/README.md; pass --model-path to a complete local package "
            "and optionally --transform-config to override its transform."
        )
