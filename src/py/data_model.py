"""Dataset loading and PyTorch MLP model definition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from activations import activation_float, build_pytorch_activation, canonical_activation_name
from config import DatasetBundle, ModelBundle, QuantConfig


class FeedforwardNN(nn.Module):
    """Configurable fully connected network used by training and quantization."""

    def __init__(
        self,
        input_dim: int,
        layers: tuple[int, ...],
        activation: str,
        output_dim: int = 2,
        activation_negative_slope: float = 1 / 64,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive integers")
        if not layers or any(units <= 0 for units in layers):
            raise ValueError("layers must contain at least one positive integer")

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else layers[i - 1], units) for i, units in enumerate(layers)]
        )
        self.out = nn.Linear(layers[-1], output_dim)
        self.activation_name = canonical_activation_name(activation)
        self.activation = build_pytorch_activation(
            self.activation_name,
            negative_slope=activation_negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.out(x)


def build_model(
    config: dict[str, Any],
    *,
    activation_negative_slope: float = 1 / 64,
) -> FeedforwardNN:
    """Build the MLP from the shared configuration dictionary."""
    return FeedforwardNN(
        input_dim=int(config["input_dim"]),
        layers=tuple(int(units) for units in config["layers"]),
        activation=str(config["activation"]),
        output_dim=int(config.get("output_dim", 2)),
        activation_negative_slope=activation_negative_slope,
    )


def load_walter_dataset(
    dataset_path: str | Path,
    *,
    target_unit: str = "degrees",
) -> DatasetBundle:
    """Load Walter inputs and convert target angles to radians.

    The NPZ file must contain ``ph_val`` (model inputs) and ``pos`` (two target
    angles). ``target_unit`` describes the unit used by ``pos``. The returned
    ``DatasetBundle.y`` is always in radians, which is the model output unit.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    unit = str(target_unit).lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("target_unit must be either 'degrees' or 'radians'")

    with np.load(path, allow_pickle=False) as raw:
        missing = {"ph_val", "pos"}.difference(raw.files)
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}")
        x = np.asarray(raw["ph_val"], dtype=np.float32)
        pos = np.asarray(raw["pos"], dtype=np.float32)

    if x.ndim != 2 or pos.ndim != 2:
        raise ValueError("ph_val and pos must both be two-dimensional arrays")
    if x.shape[0] != pos.shape[0]:
        raise ValueError("ph_val and pos must contain the same number of samples")
    if not np.isfinite(x).all() or not np.isfinite(pos).all():
        raise ValueError("Dataset contains NaN or infinite values")

    if unit == "degrees":
        y_deg = pos
        y_rad = np.radians(pos).astype(np.float32)
    else:
        y_rad = pos
        y_deg = np.degrees(pos).astype(np.float32)

    return DatasetBundle(x=x, y_deg=y_deg, y_rad=y_rad, y=y_rad, task="regression")


def prepare_mnist_dataset(x: np.ndarray, y: np.ndarray, *, normalize: bool = True) -> DatasetBundle:
    """Prepare MNIST for a basic MLP: flatten images and keep labels as classes."""
    x = np.asarray(x)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    x = x.astype(np.float32)
    if normalize and x.size and float(x.max()) > 1.0:
        x = x / 255.0
    return DatasetBundle(x=x, y=np.asarray(y).astype(np.int64), task="classification")


def load_fashion_dataset(dataset_path: str | Path) -> DatasetBundle:
    """Load the Fashion-MNIST NPZ written by the training notebook."""
    with np.load(dataset_path, allow_pickle=False) as raw:
        missing = {"x_train", "y_train", "x_test", "y_test"}.difference(raw.files)
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}")
        return DatasetBundle(
            x=np.asarray(raw["x_train"], dtype=np.float32),
            y=np.asarray(raw["y_train"], dtype=np.int64),
            x_test=np.asarray(raw["x_test"], dtype=np.float32),
            y_test=np.asarray(raw["y_test"], dtype=np.int64),
            task="classification",
        )


def load_dataset(dataset_path: str | Path) -> DatasetBundle:
    """Select the loader from the arrays stored in the NPZ file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as raw:
        is_classification = "x_train" in raw.files
    return load_fashion_dataset(path) if is_classification else load_walter_dataset(path)


def build_activation_plan(config: dict[str, Any], n_hidden: int) -> list[dict[str, Any]]:
    activation_spec = config.get("activations_per_layer")
    if activation_spec is None:
        activation_spec = [config["activation"]] * n_hidden

    if len(activation_spec) != n_hidden:
        raise ValueError("activations_per_layer is inconsistent with the number of hidden layers")

    return [{"layer_index": i, "name": canonical_activation_name(name)} for i, name in enumerate(activation_spec)]


def load_model_bundle_from_dataset(
    model_path: Path,
    dataset: DatasetBundle,
    qcfg: QuantConfig,
    device: str = "cpu",
) -> ModelBundle:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = {**config, "input_dim": int(dataset.x.shape[1])}
    model = build_model(model_config, activation_negative_slope=qcfg.activation_negative_slope)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError("No state_dict found in the checkpoint.")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    activation_plan = build_activation_plan(config, len(model.hidden_layers))

    return ModelBundle(model=model, checkpoint=checkpoint, config=config, activation_plan=activation_plan)
