"""Single configurable quantization flow for Ultra96, Lattice and Nexys targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

from analysis import run_print_distribution
from config import ProjectPaths, QuantConfig
from export import export_final_mem_artifacts, export_ph_val_pred_mem
from fixed_point import quantize_to_int
from fpga_sim import FPGASimulationResult, angular_error_numpy, simulate_fpga
from fpga_targets import TARGET_ALIASES, TARGET_PROFILES, TargetProfile
from lut import LUTSearchConfig, search_best_lut_size
from quantize import optimize_dsp_weight_scales, quantize_model


DEFAULT_QUANTIZATION_CONFIG = {
    "name": "walter_q16",
    "model_path": "data/walter/mlp_walter.pt",
    "dataset_path": "data/walter/dataset_walter_ajuste.npz",
    "output_root": "data/quantized_data",
    "target": "ultra96",
    "architecture": "sequential",
    "dsp_bits": None,
    "device": "cpu",
    "weight_bits": 16,
    "bias_bits": 16,
    "act_bits": 16,
    "margin_weights": 0.0,
    "margin_acts": 0.0,
    "calibration_mode": "max",
    "calibration_percentile": 99.9,
    "activation_negative_slope": 1 / 64,
    "dsp_acc_max_utilization": 0.90,
    "dsp_weight_search_max_steps": 24,
    "lut_addr_bits_min": 3,
    "lut_addr_bits_max_cap": 9,
    "lut_max_words": 1024,
    "lut_auto_window": 2,
    "activation_score_memory_weight": 0.03,
    "test_samples": 100,
    "simulation_samples": None,
    "fail_on_overflow": True,
    "generate_report": True,
    "generate_rtl": True,
}


@dataclass(frozen=True)
class QuantizationRun:
    config: dict[str, Any]
    target_profile: TargetProfile
    paths: ProjectPaths
    quant_config: QuantConfig
    results: dict[str, Any]
    simulation: FPGASimulationResult
    simulation_summary: dict[str, Any]
    simulation_report: pd.DataFrame
    dataset_path: Path
    manifest_path: Path
    rtl_project: dict[str, Any] | None


def _resolve_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else Path(project_root).resolve() / path


def validate_quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge safe defaults, validate values and normalize target/mode aliases."""
    unknown = sorted(set(config).difference(DEFAULT_QUANTIZATION_CONFIG))
    if unknown:
        raise ValueError(f"Unknown quantization configuration keys: {unknown}")
    validated = {**DEFAULT_QUANTIZATION_CONFIG, **config}

    target_key = str(validated["target"]).strip().lower().replace("-", "")
    if target_key not in TARGET_ALIASES:
        raise ValueError("target must be 'ultra96', 'lattice' or 'nexys'")
    validated["target"] = TARGET_ALIASES[target_key]

    architecture = str(validated["architecture"]).strip().lower()
    if architecture not in {"sequential", "pipeline"}:
        raise ValueError("architecture must be 'sequential' or 'pipeline'")
    validated["architecture"] = architecture

    dsp_bits = validated["dsp_bits"]
    if dsp_bits is None:
        dsp_bits = TARGET_PROFILES[validated["target"]].dsp_bits
    if not isinstance(dsp_bits, (int, np.integer)) or isinstance(dsp_bits, (bool, np.bool_)):
        raise ValueError("dsp_bits must be a positive integer or None")
    if int(dsp_bits) <= 0:
        raise ValueError("dsp_bits must be a positive integer or None")
    validated["dsp_bits"] = int(dsp_bits)

    if not str(validated["name"]).strip():
        raise ValueError("name cannot be empty")
    for key in ("model_path", "dataset_path", "output_root", "device"):
        if not str(validated[key]).strip():
            raise ValueError(f"{key} cannot be empty")
    for key in (
        "weight_bits",
        "bias_bits",
        "act_bits",
        "dsp_weight_search_max_steps",
        "lut_addr_bits_min",
        "lut_addr_bits_max_cap",
        "lut_max_words",
        "lut_auto_window",
        "test_samples",
    ):
        if not isinstance(validated[key], (int, np.integer)) or int(validated[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
        validated[key] = int(validated[key])

    if validated["act_bits"] > 16:
        raise ValueError("The current AXI/UART and testbench stream supports act_bits <= 16")
    if validated["lut_addr_bits_min"] > validated["lut_addr_bits_max_cap"]:
        raise ValueError("lut_addr_bits_min cannot exceed lut_addr_bits_max_cap")
    if validated["lut_addr_bits_max_cap"] > validated["act_bits"]:
        raise ValueError("lut_addr_bits_max_cap cannot exceed act_bits")
    if str(validated["calibration_mode"]).lower() not in {"max", "percentile"}:
        raise ValueError("calibration_mode must be 'max' or 'percentile'")
    validated["calibration_mode"] = str(validated["calibration_mode"]).lower()

    for key in (
        "margin_weights",
        "margin_acts",
        "calibration_percentile",
        "activation_negative_slope",
        "dsp_acc_max_utilization",
        "activation_score_memory_weight",
    ):
        validated[key] = float(validated[key])
    if not 0.0 < validated["dsp_acc_max_utilization"] <= 1.0:
        raise ValueError("dsp_acc_max_utilization must be in (0, 1]")
    if not 0.0 < validated["calibration_percentile"] <= 100.0:
        raise ValueError("calibration_percentile must be in (0, 100]")
    if validated["activation_negative_slope"] <= 0.0:
        raise ValueError("activation_negative_slope must be positive")
    leak_shift = int(round(-np.log2(validated["activation_negative_slope"])))
    if not np.isclose(validated["activation_negative_slope"], 2.0 ** (-leak_shift)):
        raise ValueError("activation_negative_slope must be of the form 2^-k")
    if validated["margin_weights"] < 0.0 or validated["margin_acts"] < 0.0:
        raise ValueError("margin_weights and margin_acts must be non-negative")
    if validated["activation_score_memory_weight"] < 0.0:
        raise ValueError("activation_score_memory_weight must be non-negative")

    simulation_samples = validated["simulation_samples"]
    if simulation_samples is not None:
        if not isinstance(simulation_samples, (int, np.integer)) or int(simulation_samples) <= 0:
            raise ValueError("simulation_samples must be a positive integer or None")
        validated["simulation_samples"] = int(simulation_samples)

    for key in ("fail_on_overflow", "generate_report", "generate_rtl"):
        if not isinstance(validated[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a boolean")
        validated[key] = bool(validated[key])
    validated["device"] = str(validated["device"])
    return validated


def build_quant_config(config: dict[str, Any]) -> tuple[QuantConfig, TargetProfile]:
    """Translate the user-facing dictionary into the numerical FPGA contract."""
    validated = validate_quantization_config(config)
    profile = TARGET_PROFILES[validated["target"]]
    qcfg = QuantConfig(
        weight_bits=validated["weight_bits"],
        bias_bits=validated["bias_bits"],
        act_bits=validated["act_bits"],
        dsp_bits=validated["dsp_bits"],
        margin_weights=validated["margin_weights"],
        margin_acts=validated["margin_acts"],
        calibration_mode=validated["calibration_mode"],
        calibration_percentile=validated["calibration_percentile"],
        activation_negative_slope=validated["activation_negative_slope"],
        dsp_acc_max_utilization=validated["dsp_acc_max_utilization"],
        dsp_weight_search_max_steps=validated["dsp_weight_search_max_steps"],
        fpga_mode=validated["architecture"],
        export_split_files=validated["architecture"] == "pipeline",
        last_segment_strategy="forbid_last",
        lut_addr_bits_min=validated["lut_addr_bits_min"],
        lut_addr_bits_max_cap=validated["lut_addr_bits_max_cap"],
        lut_max_words=validated["lut_max_words"],
        lut_auto_window=validated["lut_auto_window"],
        activation_score_memory_weight=validated["activation_score_memory_weight"],
        activation_artifact_subdir="activation_fpga",
        distribution_pdf_name="distribution_analysis_report.pdf",
    )
    return qcfg, profile


def _float_model_predictions(results: dict[str, Any], sample_count: int) -> np.ndarray:
    return np.asarray(results["tensors"]["output"], dtype=np.float32)[:sample_count]


def _build_simulation_summary(
    results: dict[str, Any],
    simulation: FPGASimulationResult,
) -> dict[str, Any]:
    sample_count = int(simulation.output_q.shape[0])
    float_prediction = _float_model_predictions(results, sample_count)
    target = np.asarray(results["dataset"].y_rad, dtype=np.float32)[:sample_count]
    float_prediction_q = quantize_to_int(
        float_prediction,
        simulation.output_exp,
        num_bits=int(results["qcfg"].act_bits),
        signed=True,
    ).astype(np.int64)
    lsb_delta = simulation.output_q.astype(np.int64) - float_prediction_q
    board_vs_float_deg = angular_error_numpy(float_prediction, simulation.output_float)
    board_vs_target_deg = angular_error_numpy(target, simulation.output_float)
    float_vs_target_deg = angular_error_numpy(target, float_prediction)

    return {
        "samples": sample_count,
        "target": simulation.target,
        "architecture": simulation.architecture,
        "transport": TARGET_PROFILES[simulation.target].transport,
        "dsp_bits": simulation.dsp_bits,
        "input_exp": simulation.input_exp,
        "output_exp": simulation.output_exp,
        "overflow_count": simulation.overflow_count,
        "exact_output_words_vs_float_quantized": int(np.count_nonzero(lsb_delta == 0)),
        "total_output_words": int(lsb_delta.size),
        "exact_samples_vs_float_quantized": int(np.count_nonzero(np.all(lsb_delta == 0, axis=1))),
        "mean_abs_lsb_vs_float_quantized": float(np.mean(np.abs(lsb_delta))),
        "max_abs_lsb_vs_float_quantized": int(np.max(np.abs(lsb_delta))) if lsb_delta.size else 0,
        "mean_board_vs_float_deg": float(np.mean(board_vs_float_deg)),
        "max_board_vs_float_deg": float(np.max(board_vs_float_deg)),
        "mean_board_vs_target_deg": float(np.mean(board_vs_target_deg)),
        "median_board_vs_target_deg": float(np.median(board_vs_target_deg)),
        "mean_float_vs_target_deg": float(np.mean(float_vs_target_deg)),
        "median_float_vs_target_deg": float(np.median(float_vs_target_deg)),
    }


def _export_simulated_dataset(
    results: dict[str, Any],
    simulation: FPGASimulationResult,
    output_dir: Path,
) -> Path:
    """Save board predictions; ``pred_int`` is now the bit-accurate reference."""
    sample_count = int(simulation.output_q.shape[0])
    dataset = results["dataset"]
    float_prediction = _float_model_predictions(results, sample_count)
    float_prediction_q = quantize_to_int(
        float_prediction,
        simulation.output_exp,
        num_bits=int(results["qcfg"].act_bits),
        signed=True,
    )
    target_radians = np.asarray(dataset.y_rad, dtype=np.float32)[:sample_count]
    target_q = quantize_to_int(
        target_radians,
        simulation.output_exp,
        num_bits=int(results["qcfg"].act_bits),
        signed=True,
    )
    path = output_dir / "dataset_walter_quantized.npz"
    np.savez_compressed(
        path,
        ph_val=np.asarray(dataset.x, dtype=np.float32)[:sample_count],
        ph_val_int=simulation.input_q,
        pos=target_radians,
        pos_int=target_q,
        pred_float=float_prediction,
        pred_float_int=float_prediction_q,
        pred_board_float=simulation.output_float,
        pred_board_int=simulation.output_q,
        pred_int=simulation.output_q,
        input_scale_exp=np.int32(simulation.input_exp),
        target_scale_exp=np.int32(simulation.output_exp),
        pred_scale_exp=np.int32(simulation.output_exp),
        target=np.asarray(simulation.target),
        architecture=np.asarray(simulation.architecture),
        dsp_bits=np.int32(simulation.dsp_bits),
    )
    return path


@torch.no_grad()
def _float_logits(results: dict[str, Any], x_float: np.ndarray) -> np.ndarray:
    model = results["model_bundle"].model
    return model(torch.from_numpy(np.asarray(x_float, dtype=np.float32))).cpu().numpy()


def _build_classification_summary(
    results: dict[str, Any],
    simulation: FPGASimulationResult,
) -> dict[str, Any]:
    """Compare the classes predicted by the board model and the float32 model."""
    sample_count = int(simulation.output_q.shape[0])
    dataset = results["dataset"]
    labels = np.asarray(dataset.y_test)[:sample_count]
    float_logits = _float_logits(results, dataset.x_test[:sample_count])
    float_logits_q = quantize_to_int(
        float_logits,
        simulation.output_exp,
        num_bits=int(results["qcfg"].act_bits),
        signed=True,
    ).astype(np.int64)
    lsb_delta = simulation.output_q.astype(np.int64) - float_logits_q
    board_class = np.argmax(simulation.output_q, axis=1)
    float_class = np.argmax(float_logits, axis=1)

    return {
        "samples": sample_count,
        "target": simulation.target,
        "architecture": simulation.architecture,
        "transport": TARGET_PROFILES[simulation.target].transport,
        "dsp_bits": simulation.dsp_bits,
        "input_exp": simulation.input_exp,
        "output_exp": simulation.output_exp,
        "overflow_count": simulation.overflow_count,
        "accuracy_float": float(np.mean(float_class == labels)),
        "accuracy_board": float(np.mean(board_class == labels)),
        "board_vs_float_agreement": float(np.mean(board_class == float_class)),
        "board_vs_float_disagreements": int(np.count_nonzero(board_class != float_class)),
        "mean_abs_lsb_vs_float_quantized": float(np.mean(np.abs(lsb_delta))),
        "max_abs_lsb_vs_float_quantized": int(np.max(np.abs(lsb_delta))) if lsb_delta.size else 0,
    }


def _export_classification_dataset(
    results: dict[str, Any],
    simulation: FPGASimulationResult,
    output_dir: Path,
) -> Path:
    """Save the test images and predictions used as the board reference."""
    sample_count = int(simulation.output_q.shape[0])
    dataset = results["dataset"]
    float_logits = _float_logits(results, dataset.x_test[:sample_count])
    path = output_dir / "dataset_fashion_quantized.npz"
    np.savez_compressed(
        path,
        x=np.asarray(dataset.x_test, dtype=np.float32)[:sample_count],
        x_int=simulation.input_q,
        y=np.asarray(dataset.y_test)[:sample_count],
        pred_float=float_logits,
        pred_board_float=simulation.output_float,
        pred_board_int=simulation.output_q,
        class_float=np.argmax(float_logits, axis=1),
        class_board=np.argmax(simulation.output_q, axis=1),
        input_scale_exp=np.int32(simulation.input_exp),
        pred_scale_exp=np.int32(simulation.output_exp),
        target=np.asarray(simulation.target),
        architecture=np.asarray(simulation.architecture),
        dsp_bits=np.int32(simulation.dsp_bits),
    )
    return path


def _load_python_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def generate_rtl_project(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    layout_path: Path,
) -> dict[str, Any]:
    """Select and render the existing editable RTL panel for one target/mode."""
    validated = validate_quantization_config(config)
    root = Path(project_root).resolve()
    target = validated["target"]
    mode = validated["architecture"]
    profile = TARGET_PROFILES[target]

    if target == "nexys":
        if mode != "sequential":
            raise NotImplementedError("The Nexys RTL only exists in sequential mode for now")
        generator = _load_python_file(
            root / "src" / "tcl" / "remodel_vivado_project.py",
            "recipe_remodel_vivado",
        )
        return generator.generate_nexys_sequential(
            root,
            layout_path,
            dsp_pw=validated["dsp_bits"],
        )

    if target == "ultra96":
        generator = _load_python_file(
            root / "src" / "tcl" / "remodel_vivado_project.py",
            "recipe_remodel_vivado",
        )
        return generator.generate_project(
            root,
            layout_path,
            fpga_mode=mode,
            template_rel=f"src/rtl/src_ultra_{mode}",
            generated_rel=f"src/rtl/generated/src_ultra_{mode}",
            project_name="SOC_MLP_ultra96",
            part_name=profile.device,
            dsp_pw=validated["dsp_bits"],
        )

    generator = _load_python_file(
        root / "src" / "tcl" / "remodel_lattice_project.py",
        "recipe_remodel_lattice",
    )
    return generator.generate_project(
        root,
        layout_path,
        fpga_mode=mode,
        template_rel=f"src/rtl/src_lattice_{mode}",
        project_name="SOC_MLP_lattice",
        device=profile.device,
        dsp_pw=validated["dsp_bits"],
    )


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_quantization(
    config: dict[str, Any],
    *,
    project_root: str | Path = ".",
    show_summary: bool = True,
) -> QuantizationRun:
    """Execute calibration, export, RTL simulation and optional RTL generation."""
    validated = validate_quantization_config(config)
    root = Path(project_root).resolve()
    qcfg, profile = build_quant_config(validated)
    output_dir = (
        _resolve_path(root, validated["output_root"])
        / str(validated["name"])
        / profile.name
        / validated["architecture"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        model_path=_resolve_path(root, validated["model_path"]),
        dataset_path=_resolve_path(root, validated["dataset_path"]),
        output_dir=output_dir,
    )
    if not paths.model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {paths.model_path}")
    if not paths.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {paths.dataset_path}")

    quantized = quantize_model(
        paths,
        qcfg,
        device=validated["device"],
        show_summary=show_summary,
    )
    dsp_optimized = optimize_dsp_weight_scales(
        quantized,
        show_summary=show_summary,
        save_report=True,
        report_filename="dsp_accumulator_report.csv",
    )
    activation_results = search_best_lut_size(
        dsp_optimized,
        cfg=LUTSearchConfig(
            save_artifacts=False,
            show_tables=False,
            show_best_tables=show_summary,
            print_debug=False,
        ),
    )
    export_result = export_final_mem_artifacts(
        activation_results,
        export_dir=output_dir,
        show_summary=show_summary,
    )

    dataset = activation_results["dataset"]
    is_classification = dataset.task == "classification"
    simulation = simulate_fpga(
        activation_results,
        target=profile.name,
        architecture=validated["architecture"],
        x_float=dataset.x_test if is_classification else None,
        max_samples=validated["simulation_samples"],
        strict=validated["fail_on_overflow"],
    )
    simulation_report = pd.DataFrame(simulation.layer_stats)
    simulation_report.to_csv(output_dir / "fpga_simulation_layers.csv", index=False)
    if is_classification:
        simulation_summary = _build_classification_summary(activation_results, simulation)
        dataset_path = _export_classification_dataset(activation_results, simulation, output_dir)
    else:
        simulation_summary = _build_simulation_summary(activation_results, simulation)
        dataset_path = _export_simulated_dataset(activation_results, simulation, output_dir)
    testbench_inputs, testbench_outputs = export_ph_val_pred_mem(
        npz_path=dataset_path,
        out_dir=output_dir,
        n_samples=validated["test_samples"],
        input_key="x_int" if is_classification else "ph_val_int",
        expected_key="pred_board_int",
        show_summary=show_summary,
    )

    if validated["generate_report"]:
        run_print_distribution(
            activation_results,
            save_pdf=True,
            show_table_dataframe=show_summary,
        )

    layout_path = output_dir / "mem_layout.json"
    rtl_project = None
    if validated["generate_rtl"]:
        rtl_project = generate_rtl_project(
            validated,
            project_root=root,
            layout_path=layout_path,
        )

    manifest = {
        "schema_version": 1,
        "config": validated,
        "target_profile": asdict(profile),
        "quant_config": asdict(qcfg),
        "paths": {
            "model": str(paths.model_path),
            "dataset": str(paths.dataset_path),
            "output": str(paths.output_dir),
            "quantized_dataset": str(dataset_path),
            "testbench_inputs": str(testbench_inputs),
            "testbench_outputs": str(testbench_outputs),
            "mem_layout": str(layout_path),
        },
        "simulation": simulation_summary,
        "rtl_project": rtl_project,
        "export": export_result,
    }
    manifest_path = output_dir / "quantization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if show_summary:
        print("\n" + "=" * 88)
        print("QUANTIZATION COMPLETE")
        print("=" * 88)
        print(f"Target / mode : {profile.name} / {validated['architecture']}")
        print(f"DSP / transport: {validated['dsp_bits']} bits / {profile.transport}")
        print(f"Output          : {output_dir}")
        print(f"FPGA overflows  : {simulation.overflow_count}")
        if is_classification:
            print(f"Accuracy float  : {simulation_summary['accuracy_float']:.2%}")
            print(f"Accuracy board  : {simulation_summary['accuracy_board']:.2%}")
            print(f"Board vs float  : {simulation_summary['board_vs_float_agreement']:.2%} agreement")
        else:
            print(f"Board vs float  : {simulation_summary['mean_board_vs_float_deg']:.6f} deg mean")
        print("=" * 88)

    return QuantizationRun(
        config=validated,
        target_profile=profile,
        paths=paths,
        quant_config=qcfg,
        results=activation_results,
        simulation=simulation,
        simulation_summary=simulation_summary,
        simulation_report=simulation_report,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        rtl_project=rtl_project,
    )
