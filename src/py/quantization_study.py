"""Audit and compare quantization choices without hiding the error source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


def angular_error_numpy(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Angular distance in degrees, kept dependency-free for export audits."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("Les angles doivent avoir la même forme (n_samples, 2).")

    def direction(angles: np.ndarray) -> np.ndarray:
        first, second = angles[:, 0], angles[:, 1]
        return np.stack(
            (
                np.cos(first) * np.cos(second),
                np.sin(first) * np.cos(second),
                -np.sin(second),
            ),
            axis=1,
        )

    ref_vector = direction(reference)
    pred_vector = direction(prediction)
    return np.degrees(
        np.arctan2(
            np.linalg.norm(np.cross(ref_vector, pred_vector), axis=1),
            np.sum(ref_vector * pred_vector, axis=1),
        )
    )


@dataclass(frozen=True)
class QuantizationAudit:
    """Quality decomposition for float, Python fixed-point and real FPGA data."""

    summary: pd.DataFrame
    axis_report: pd.DataFrame
    sample_report: pd.DataFrame
    diagnosis: tuple[str, ...]

    def print_diagnosis(self) -> None:
        for message in self.diagnosis:
            print(f"- {message}")


def _resolve(root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else Path(root).resolve() / path


def _error_row(stage: str, reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = angular_error_numpy(reference, prediction)
    return {
        "comparison": stage,
        "samples": int(error.size),
        "mean_deg": float(np.mean(error)),
        "median_deg": float(np.median(error)),
        "p95_deg": float(np.percentile(error, 95)),
        "p99_deg": float(np.percentile(error, 99)),
        "max_deg": float(np.max(error)),
    }


def _first_key(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _load_board_predictions(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    frame = pd.read_csv(path)
    float_pairs = (
        ("fpga_float_0", "fpga_float_1"),
        ("pred_board_0", "pred_board_1"),
        ("board_pred_0", "board_pred_1"),
        ("pred_0", "pred_1"),
        ("angle_0_rad", "angle_1_rad"),
    )
    int_pairs = (
        ("fpga_quantized_0", "fpga_quantized_1"),
        ("pred_board_int_0", "pred_board_int_1"),
        ("board_pred_int_0", "board_pred_int_1"),
        ("pred_int_0", "pred_int_1"),
        ("raw_0", "raw_1"),
    )
    float_pair = next((pair for pair in float_pairs if set(pair) <= set(frame.columns)), None)
    int_pair = next((pair for pair in int_pairs if set(pair) <= set(frame.columns)), None)
    sample_key = _first_key(frame.columns, ("sample", "sample_index", "index"))
    indices = frame[sample_key].to_numpy(dtype=np.int64) if sample_key else None
    board_int = frame[list(int_pair)].to_numpy(dtype=np.int64) if int_pair else None
    if float_pair:
        board_float = frame[list(float_pair)].to_numpy(dtype=np.float64)
    elif board_int is not None:
        exponent_key = _first_key(
            frame.columns, ("output_exp", "output_scale_exponent", "pred_scale_exp")
        )
        if exponent_key is None:
            raise ValueError(
                "Le CSV contient des mots entiers mais aucune sortie flottante ni colonne output_exp."
            )
        board_float = board_int.astype(np.float64) * 2.0 ** (-int(frame[exponent_key].iloc[0]))
    else:
        raise ValueError(
            "Colonnes FPGA introuvables. Utilisez pred_board_0/1, board_pred_0/1 "
            "ou pred_int_0/1 avec output_exp."
        )
    return board_float, board_int, indices


def audit_quantization_export(
    quantized_dataset: str | Path,
    *,
    board_results: str | Path | None = None,
    project_root: str | Path = ".",
) -> QuantizationAudit:
    """Separate model, Python fixed-point and physical-board errors."""
    dataset_path = _resolve(project_root, quantized_dataset)
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {"pos", "pred_float", "pred_board_float", "pred_board_int"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Clés absentes du dataset quantifié : {missing}")
        target = np.asarray(data["pos"], dtype=np.float64)
        float_pred = np.asarray(data["pred_float"], dtype=np.float64)
        python_pred = np.asarray(data["pred_board_float"], dtype=np.float64)
        python_int = np.asarray(data["pred_board_int"], dtype=np.int64)
        output_exp = int(np.asarray(data["pred_scale_exp"]).item())

    rows = [
        _error_row("modèle flottant → cible", target, float_pred),
        _error_row("Python entier → modèle flottant", float_pred, python_pred),
        _error_row("Python entier → cible", target, python_pred),
    ]
    axis_rows: list[dict[str, Any]] = []
    sample_report = pd.DataFrame()
    diagnosis: list[str] = []

    quant_error = rows[1]["mean_deg"]
    model_error = rows[0]["mean_deg"]
    if quant_error <= max(0.02, model_error * 0.10):
        diagnosis.append(
            f"La quantification Python n'est pas la cause principale : {quant_error:.6f}° "
            f"contre {model_error:.6f}° pour le modèle flottant face à la cible."
        )
    else:
        diagnosis.append(
            f"La quantification est significative : {quant_error:.6f}° d'écart moyen au flottant."
        )

    if board_results is not None:
        board_float, board_int, indices = _load_board_predictions(
            _resolve(project_root, board_results)
        )

        # Conversion réelle utilisée pour l'audit :
        # entier FPGA -> radians
        if board_int is None:
            raise ValueError(
                "Le CSV carte doit contenir les sorties entières FPGA."
            )

        board_float = board_int.astype(np.float64) * 2.0 ** (-output_exp)

        if indices is None:
            indices = np.arange(board_float.shape[0], dtype=np.int64)

        if (
            indices.size != board_float.shape[0]
            or np.any(indices < 0)
            or np.any(indices >= target.shape[0])
        ):
            raise ValueError(
                "Les indices du CSV FPGA sont incompatibles avec le dataset quantifié."
            )

        target_b = target[indices]
        float_b = float_pred[indices]
        python_b = python_pred[indices]
        python_int_b = python_int[indices]
        rows.extend(
            [
                _error_row("carte → Python entier", python_b, board_float),
                _error_row("carte → modèle flottant", float_b, board_float),
                _error_row("carte → cible", target_b, board_float),
            ]
        )
        card_error = rows[-3]["mean_deg"]
        if card_error > max(0.02, quant_error * 5.0):
            diagnosis.append(
                f"L'écart carte/RTL domine l'erreur de quantification : {card_error:.6f}° "
                f"contre {quant_error:.6f}°. Vérifiez d'abord les échelles, signes et décalages RTL."
            )

        card_vs_python = angular_error_numpy(python_b, board_float)
        card_vs_float = angular_error_numpy(float_b, board_float)
        card_vs_target = angular_error_numpy(target_b, board_float)

        # Keep the aligned values used by the audit in the sample report.
        # This makes diagnostic plots reproducible without reopening the NPZ
        # dataset or the board CSV in the notebook.
        sample_columns: dict[str, np.ndarray] = {
            "sample": indices,
            "card_vs_python_deg": card_vs_python,
            "card_vs_float_deg": card_vs_float,
            "card_vs_target_deg": card_vs_target,
        }
        for axis in range(target_b.shape[1]):
            sample_columns[f"target_{axis}"] = target_b[:, axis]
            sample_columns[f"float_{axis}"] = float_b[:, axis]
            sample_columns[f"python_float_{axis}"] = python_b[:, axis]
            sample_columns[f"card_float_{axis}"] = board_float[:, axis]
            sample_columns[f"python_int_{axis}"] = python_int_b[:, axis]
            sample_columns[f"card_int_{axis}"] = board_int[:, axis]

        sample_report = pd.DataFrame(sample_columns).sort_values(
            "card_vs_python_deg", ascending=False, ignore_index=True
        )

        lsb_delta = board_int - python_int_b
        for axis in range(lsb_delta.shape[1]):
            delta = lsb_delta[:, axis]
            axis_rows.append(
                {
                    "output": axis,
                    "bias_lsb": float(np.mean(delta)),
                    "mae_lsb": float(np.mean(np.abs(delta))),
                    "p95_abs_lsb": float(np.percentile(np.abs(delta), 95)),
                    "min_lsb": int(np.min(delta)),
                    "max_lsb": int(np.max(delta)),
                    "exact_percent": float(100.0 * np.mean(delta == 0)),
                }
            )
        worst_axis = max(axis_rows, key=lambda row: abs(row["bias_lsb"]))
        diagnosis.append(
            f"La sortie {worst_axis['output']} présente le biais matériel le plus fort : "
            f"{worst_axis['bias_lsb']:+.2f} LSB en moyenne."
        )
    else:
        diagnosis.append("Aucun CSV carte fourni : la cohérence UART/RTL n'a pas été évaluée.")

    return QuantizationAudit(
        summary=pd.DataFrame(rows),
        axis_report=pd.DataFrame(axis_rows),
        sample_report=sample_report,
        diagnosis=tuple(diagnosis),
    )


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("Chaque essai doit avoir un label non vide.")
    return cleaned


def compare_quantization_candidates(
    base_config: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    *,
    project_root: str | Path = ".",
    runner: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Run controlled experiments and rank only successful, overflow-free variants."""
    if runner is None:
        from quantization import run_quantization

        runner = run_quantization

    base_name = str(base_config.get("name", "quantization"))
    rows: list[dict[str, Any]] = []
    labels: set[str] = set()
    for candidate in candidates:
        if "label" not in candidate:
            raise ValueError("Chaque essai doit contenir la clé 'label'.")
        label = str(candidate["label"])
        if label in labels:
            raise ValueError(f"Label d'essai dupliqué : {label!r}")
        labels.add(label)
        overrides = {key: value for key, value in candidate.items() if key != "label"}
        config = {
            **base_config,
            **overrides,
            "name": f"{base_name}__study_{_slug(label)}",
            "fail_on_overflow": False,
            "generate_report": False,
            "generate_rtl": False,
        }
        started = perf_counter()
        try:
            run = runner(config, project_root=project_root, show_summary=False)
            summary = run.simulation_summary
            sample_count = int(run.simulation.output_float.shape[0])
            float_pred = np.asarray(run.results["tensors"]["output"])[:sample_count]
            target = np.asarray(run.results["dataset"].y_rad)[:sample_count]
            quant_error = angular_error_numpy(float_pred, run.simulation.output_float)
            target_error = angular_error_numpy(target, run.simulation.output_float)
            rows.append(
                {
                    "label": label,
                    "status": "ok",
                    "dsp_bits": int(run.quant_config.dsp_bits),
                    "calibration_mode": run.quant_config.calibration_mode,
                    "calibration_percentile": float(run.quant_config.calibration_percentile),
                    "margin_weights": float(run.quant_config.margin_weights),
                    "margin_acts": float(run.quant_config.margin_acts),
                    "overflow_count": int(run.simulation.overflow_count),
                    "quant_mean_deg": float(np.mean(quant_error)),
                    "quant_median_deg": float(np.median(quant_error)),
                    "quant_p95_deg": float(np.percentile(quant_error, 95)),
                    "quant_max_deg": float(np.max(quant_error)),
                    "target_mean_deg": float(np.mean(target_error)),
                    "target_median_deg": float(np.median(target_error)),
                    "target_p95_deg": float(np.percentile(target_error, 95)),
                    "float_target_mean_deg": float(summary["mean_float_vs_target_deg"]),
                    "float_target_median_deg": float(summary["median_float_vs_target_deg"]),
                    "mean_abs_lsb": float(summary["mean_abs_lsb_vs_float_quantized"]),
                    "max_abs_lsb": int(summary["max_abs_lsb_vs_float_quantized"]),
                    "output_exp": int(run.simulation.output_exp),
                    "duration_s": perf_counter() - started,
                    "error": "",
                }
            )
        except Exception as exc:  # An invalid experiment must not discard the full study.
            rows.append(
                {
                    "label": label,
                    "status": "error",
                    "overflow_count": np.nan,
                    "duration_s": perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("La liste d'essais est vide.")
    for column in (
        "target_mean_deg",
        "target_median_deg",
        "quant_mean_deg",
        "quant_p95_deg",
    ):
        if column not in report:
            report[column] = np.nan
    report["safe"] = (report["status"] == "ok") & (report["overflow_count"] == 0)
    return report.sort_values(
        ["safe", "target_mean_deg", "target_median_deg", "quant_mean_deg"],
        ascending=[False, True, True, True],
        na_position="last",
        ignore_index=True,
    )
