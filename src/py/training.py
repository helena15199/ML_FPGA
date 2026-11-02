"""Reusable training pipeline for the Walter angular-regression MLP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from activations import canonical_activation_name
from data_model import FeedforwardNN, build_model, load_walter_dataset


SUPPORTED_ACTIVATIONS = {"relu", "leaky_relu", "gelu", "sigmoid", "tanh"}
SUPPORTED_OPTIMIZERS = {"adam", "adamw", "rmsprop"}
SUPPORTED_INITIALIZATIONS = {"glorot_uniform", "glorot_normal", "he_uniform", "he_normal"}
REQUIRED_CONFIG_KEYS = {
    "dataset_name",
    "dataset_path",
    "input_dim",
    "output_dim",
    "layers",
    "activation",
    "target_unit",
    "optimizer",
    "init_weight",
    "learning_rate",
    "batch_size",
    "epochs",
    "patience",
    "nb_splits",
    "shuffle",
    "random_state",
    "save_dir",
    "name",
}


@dataclass(frozen=True)
class TrainingDataset:
    path: Path
    x: np.ndarray
    y_radians: np.ndarray
    y_degrees: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    config: dict[str, Any]
    best_state_dict: dict[str, torch.Tensor]
    best_fold: int
    best_val_deg: float
    fold_scores: list[dict[str, float | int]]
    histories: list[list[dict[str, float | int]]]
    device: str

    @property
    def history_rows(self) -> list[dict[str, float | int]]:
        return [row for fold_history in self.histories for row in fold_history]


def validate_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate once at the pipeline boundary and return a detached copy."""
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Missing training configuration keys: {missing}")

    validated = dict(config)
    validated["layers"] = tuple(int(units) for units in config["layers"])

    for key in ("input_dim", "output_dim", "batch_size", "epochs", "patience", "nb_splits"):
        if not isinstance(validated[key], (int, np.integer)) or int(validated[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
        validated[key] = int(validated[key])

    if not validated["layers"] or any(units <= 0 for units in validated["layers"]):
        raise ValueError("layers must contain at least one positive integer")
    if validated["output_dim"] != 2:
        raise ValueError("Walter angular regression requires output_dim=2")
    if validated["nb_splits"] < 2:
        raise ValueError("nb_splits must be at least 2")
    if float(validated["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")

    activation = canonical_activation_name(str(validated["activation"]))
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {validated['activation']}")
    validated["activation"] = activation

    unit = str(validated["target_unit"]).lower()
    if unit not in {"degrees", "radians"}:
        raise ValueError("target_unit must be either 'degrees' or 'radians'")
    validated["target_unit"] = unit

    if str(validated["optimizer"]).lower() not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {validated['optimizer']}")
    if str(validated["init_weight"]).lower() not in SUPPORTED_INITIALIZATIONS:
        raise ValueError(f"Unsupported weight initialization: {validated['init_weight']}")

    if Path(str(validated["dataset_path"])).name != str(validated["dataset_name"]):
        raise ValueError("dataset_name must match the filename in dataset_path")
    if not str(validated["name"]).endswith(".pt"):
        raise ValueError("name must use the .pt extension")

    validated["learning_rate"] = float(validated["learning_rate"])
    validated["shuffle"] = bool(validated["shuffle"])
    validated["random_state"] = int(validated["random_state"])
    return validated


def _resolve_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project_root).resolve() / path


def load_training_dataset(config: dict[str, Any], *, project_root: str | Path = ".") -> TrainingDataset:
    """Load and validate the database selected by the training configuration."""
    validated = validate_training_config(config)
    path = _resolve_path(project_root, validated["dataset_path"])
    dataset = load_walter_dataset(path, target_unit=validated["target_unit"])

    if dataset.x.shape[1] != validated["input_dim"]:
        raise ValueError(
            f"input_dim={validated['input_dim']} but ph_val has {dataset.x.shape[1]} columns"
        )
    if dataset.y.shape[1] != validated["output_dim"]:
        raise ValueError(
            f"output_dim={validated['output_dim']} but pos has {dataset.y.shape[1]} columns"
        )
    if dataset.x.shape[0] < validated["nb_splits"]:
        raise ValueError("The dataset must contain at least nb_splits samples")

    return TrainingDataset(
        path=path,
        x=dataset.x,
        y_radians=dataset.y_rad,
        y_degrees=dataset.y_deg,
    )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible folds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_linear_layers(model: nn.Module, method: str) -> None:
    """Initialize every linear layer using the configured strategy."""
    method = method.lower()
    if method not in SUPPORTED_INITIALIZATIONS:
        raise ValueError(f"Unsupported weight initialization: {method}")

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if method == "glorot_uniform":
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
        elif method == "glorot_normal":
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))
        elif method == "he_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        else:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def angular_error_deg(
    y_true_radians: torch.Tensor,
    y_pred_radians: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the stable 3D angular distance between angle pairs.

    ``atan2(||u x v||, u . v)`` avoids the artificial ~0.81 degree floor caused
    by clamping the cosine before ``acos``.
    """
    if y_true_radians.shape != y_pred_radians.shape:
        raise ValueError("Targets and predictions must have identical shapes")
    if y_true_radians.ndim != 2 or y_true_radians.shape[1] != 2:
        raise ValueError("Angular tensors must have shape (n_samples, 2)")

    def to_direction(angles: torch.Tensor) -> torch.Tensor:
        first, second = angles[:, 0], angles[:, 1]
        return torch.stack(
            (
                torch.cos(first) * torch.cos(second),
                torch.sin(first) * torch.cos(second),
                -torch.sin(second),
            ),
            dim=1,
        )

    true_direction = to_direction(y_true_radians)
    pred_direction = to_direction(y_pred_radians)
    cross_norm = torch.linalg.vector_norm(
        torch.cross(true_direction, pred_direction, dim=1),
        dim=1,
    )
    dot_product = torch.sum(true_direction * pred_direction, dim=1)
    error = torch.rad2deg(torch.atan2(cross_norm, dot_product))

    if reduction == "none":
        return error
    if reduction == "mean":
        return error.mean()
    if reduction == "sum":
        return error.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def build_optimizer(model: nn.Module, name: str, learning_rate: float) -> torch.optim.Optimizer:
    """Create an optimizer without leaking optimizer-specific code into notebooks."""
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    key = name.lower()
    if key not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}")
    return optimizers[key](model.parameters(), lr=learning_rate)


def _mean_loss(
    model: FeedforwardNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = angular_error_deg(y_batch, prediction)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(x_batch.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot compute a loss on an empty data loader")
    return total_loss / total_samples


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_cross_validated(
    config: dict[str, Any],
    *,
    project_root: str | Path = ".",
    device: str | torch.device | None = None,
    show_progress: bool = True,
    verbose: bool = True,
) -> TrainingResult:
    """Train one model per fold and retain the best validation checkpoint."""
    validated = validate_training_config(config)
    dataset = load_training_dataset(validated, project_root=project_root)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    splitter = KFold(
        n_splits=validated["nb_splits"],
        shuffle=validated["shuffle"],
        random_state=validated["random_state"] if validated["shuffle"] else None,
    )

    histories: list[list[dict[str, float | int]]] = []
    fold_scores: list[dict[str, float | int]] = []
    best_global_value = float("inf")
    best_global_state: dict[str, torch.Tensor] | None = None
    best_global_fold = -1

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(dataset.x), start=1):
        fold_seed = validated["random_state"] + fold
        set_random_seed(fold_seed)
        generator = torch.Generator().manual_seed(fold_seed)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(dataset.x[train_indices]),
                torch.from_numpy(dataset.y_radians[train_indices]),
            ),
            batch_size=validated["batch_size"],
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(dataset.x[validation_indices]),
                torch.from_numpy(dataset.y_radians[validation_indices]),
            ),
            batch_size=validated["batch_size"],
            shuffle=False,
        )

        model = build_model(validated).to(selected_device)
        initialize_linear_layers(model, validated["init_weight"])
        optimizer = build_optimizer(model, validated["optimizer"], validated["learning_rate"])
        best_fold_value = float("inf")
        best_fold_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        fold_history: list[dict[str, float | int]] = []

        epoch_iterator: Iterable[int] = range(1, validated["epochs"] + 1)
        if show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc=f"Fold {fold}/{validated['nb_splits']}")

        for epoch in epoch_iterator:
            train_loss = _mean_loss(model, train_loader, selected_device, optimizer)
            validation_loss = _mean_loss(model, validation_loader, selected_device)
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                raise RuntimeError(f"Non-finite loss detected in fold {fold}, epoch {epoch}")

            fold_history.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss_deg": train_loss,
                    "val_loss_deg": validation_loss,
                }
            )
            if validation_loss < best_fold_value:
                best_fold_value = validation_loss
                best_fold_state = _cpu_state_dict(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= validated["patience"]:
                if verbose:
                    print(
                        f"Fold {fold}: early stopping at epoch {epoch} "
                        f"(best={best_fold_value:.4f} deg)"
                    )
                break

        if best_fold_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a checkpoint")
        histories.append(fold_history)
        fold_scores.append({"fold": fold, "best_val_deg": best_fold_value})

        if best_fold_value < best_global_value:
            best_global_value = best_fold_value
            best_global_state = best_fold_state
            best_global_fold = fold

    if best_global_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    return TrainingResult(
        config=validated,
        best_state_dict=best_global_state,
        best_fold=best_global_fold,
        best_val_deg=best_global_value,
        fold_scores=fold_scores,
        histories=histories,
        device=str(selected_device),
    )


def checkpoint_path(config: dict[str, Any], *, project_root: str | Path = ".") -> Path:
    validated = validate_training_config(config)
    return _resolve_path(project_root, validated["save_dir"]) / validated["name"]


def save_training_result(result: TrainingResult, *, project_root: str | Path = ".") -> Path:
    """Save a backward-compatible checkpoint for quantization and export."""
    path = checkpoint_path(result.config, project_root=project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "training_schema_version": 1,
            "model_state_dict": result.best_state_dict,
            "config": result.config,
            "model_output_unit": "radians",
            "best_global_fold": result.best_fold,
            "best_global_val": result.best_val_deg,
            "best_global_val_deg": result.best_val_deg,
            "fold_scores": result.fold_scores,
        },
        path,
    )
    return path


def load_trained_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FeedforwardNN, dict[str, Any]]:
    """Reload a training checkpoint and reconstruct its model."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = validate_training_config(checkpoint["config"])
    model = build_model(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError("No model_state_dict found in checkpoint")
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint
