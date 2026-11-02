"""Small helpers shared by the quantization modules."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import ProjectPaths


def _read(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if is_dataclass(cfg):
        return asdict(cfg).get(key, default)
    return getattr(cfg, key, default)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_existing_path(*filenames: str, search_dirs: list[Path] | None = None) -> Path:
    search_dirs = search_dirs or [
        Path.cwd(),
        Path("/mnt/data"),
        Path.cwd().parent,
    ]

    for filename in filenames:
        candidate = Path(filename)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        for base_dir in search_dirs:
            candidate = (base_dir / filename).resolve()
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Unable to find one of the files: {filenames}")


def build_default_paths(
    model_filename: str = "MLP_ideal_sigmoid.pt",
    dataset_filename: str = "MLP_dataset_ideal.npz",
    output_dirname: str = "quantized_integrated_artifacts_simplified",
) -> ProjectPaths:
    model_path = resolve_existing_path(model_filename)
    dataset_path = resolve_existing_path(dataset_filename)
    output_dir = Path.cwd() / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(model_path=model_path, dataset_path=dataset_path, output_dir=output_dir)

