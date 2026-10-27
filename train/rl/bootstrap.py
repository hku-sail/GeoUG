"""Select the bundled RL model while preserving GeoUG's full data package."""

import importlib
import sys
from pathlib import Path


def prepare_engine_imports():
    root = Path(__file__).resolve().parents[2]
    engine = root / "train" / "rl" / "engine"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # BAGEL also has packages called data and train. Cache our public packages
    # first so import order cannot select its incomplete training-data tree.
    for name in ("data", "train"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != root / name:
            raise RuntimeError(f"{name} was imported from {module.__file__}; run GeoUG in a fresh process")
    model = sys.modules.get("modeling")
    if model is not None and Path(model.__file__).resolve().parent != engine / "Bagel" / "modeling":
        raise RuntimeError("The base modeling package is already loaded; run spatial inference in a fresh process")
    for path in (engine, engine / "Bagel"):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
