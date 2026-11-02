"""FPGA export helpers: .mem files, scales.vh and quantized datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from activations import canonical_activation_name
from config import QuantConfig
from fixed_point import dtype_for_bits, int_to_hex_twos_complement, quantize_to_int
from utils import _cfg_value, _read


def format_layer_file_tag(layer_name: str) -> str:
    idx = int(str(layer_name).split("_")[1])
    return f"layer{idx:02d}"


def save_mem_file(values, bits: int, path: Path):
    lines = [int_to_hex_twos_complement(v, bits) for v in np.asarray(values).reshape(-1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_activation_lut_artifacts(
    *,
    candidate_dir: Path,
    module,
    qcfg: QuantConfig,
    layer_name: str,
    layer_tag: str,
    act_name: str,
    addr_bits: int,
    sx: int,
    sy: int,
    memory_words: int,
    memory_bits: int,
    abs_err: np.ndarray,
    save_debug_csv: bool = False,
) -> dict[str, str]:
    candidate_dir.mkdir(parents=True, exist_ok=True)

    y_q_hdl_order = reorder_lut_signed_order_to_hdl_raw_order(module.y_knots_q)
    mem_path = candidate_dir / "act.mem"
    save_mem_file(y_q_hdl_order, qcfg.act_bits, mem_path)

    metadata = {
        "layer_name": layer_name,
        "layer_tag": layer_tag,
        "activation_name": act_name,
        "implementation": "paper_fit",
        "act_bits": qcfg.act_bits,
        "addr_bits": addr_bits,
        "input_exp": sx,
        "output_exp": sy,
        "last_segment_strategy": qcfg.last_segment_strategy,
        "memory_words": memory_words,
        "memory_bits": memory_bits,
        "mae": float(abs_err.mean()),
        "max_abs_err": float(abs_err.max()),
        "mem_file": "act.mem",
        "mem_order": "hdl_raw_order",
    }

    json_path = candidate_dir / "metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    paths = {
        "dir": str(candidate_dir),
        "mem": str(mem_path),
        "json": str(json_path),
    }

    if save_debug_csv:
        knots_df = module.knots_df.copy()
        knots_df["y_float"] = module.y_knots_float
        knots_df["y_q_signed_order"] = module.y_knots_q
        knots_df["y_q_hdl_order"] = y_q_hdl_order
        knots_df["y_q_real"] = module.y_knots_quantized_float

        knots_path = candidate_dir / "lut_knots.csv"
        segments_path = candidate_dir / "segments.csv"

        knots_df.to_csv(knots_path, index=False)
        module.segments_df.to_csv(segments_path, index=False)

        paths["knots_csv"] = str(knots_path)
        paths["segments_csv"] = str(segments_path)

    return paths


def export_weight_bias_mem_files(
    activation_results: dict[str, Any],
    *,
    export_dir: Path,
) -> list[dict[str, Any]]:
    """
    Export quantized weight and bias .mem files.
    One weight file and one bias file per layer, including the output layer.
    """
    quantized_layers = activation_results["quantized_layers"]
    qcfg = activation_results["qcfg"]

    export_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for layer in quantized_layers:
        layer_name = _read(layer, "name")
        layer_idx = int(str(layer_name).split("_")[1])

        w_q = np.asarray(_read(layer, "w_q")).reshape(-1)
        b_q = np.asarray(_read(layer, "b_q")).reshape(-1)

        weight_path = export_dir / f"weight_layer{layer_idx:d}.mem"
        bias_path = export_dir / f"bias_layer{layer_idx:d}.mem"

        save_mem_file(w_q, qcfg.weight_bits, weight_path)
        save_mem_file(b_q, qcfg.bias_bits, bias_path)

        exported.append(
            {
                "layer_name": layer_name,
                "weight_mem": str(weight_path),
                "bias_mem": str(bias_path),
                "n_weights": int(w_q.size),
                "n_bias": int(b_q.size),
            }
        )

    return exported


def export_best_activation_mem_files(
    activation_results: dict[str, Any],
    *,
    export_dir: Path,
) -> list[dict[str, Any]]:
    """
    Export only the .mem files for the best activation LUTs.
    Pure integer analytical activations do not generate .mem files.

    IMPORTANT :
    module.y_knots_q is generated in natural signed order :
        [-32768 ... -1, 0 ... 32767]

    But the current rom_act.sv HDL addresses the ROM with :
        addr = x[ACT_W-1:FRAC_W]

    Therefore it expects raw two's-complement order :
        [0 ... 32767, -32768 ... -1]

    The LUT is therefore reordered before .mem export.
    """
    activation_best_df = activation_results.get("activation_best_df")
    activation_modules = activation_results.get("activation_modules", {})
    qcfg = activation_results["qcfg"]

    export_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    if activation_best_df is None or len(activation_best_df) == 0:
        return exported

    for _, row in activation_best_df.iterrows():
        implementation = row.get("implementation", "")
        layer_name = row["layer_name"]

        if implementation != "paper_fit":
            continue

        layer_idx = int(str(layer_name).split("_")[1])
        module = activation_modules.get(layer_idx)

        if module is None:
            continue

        # LUT in natural signed order
        y_q_signed_order = np.asarray(module.y_knots_q, dtype=np.int64)

        # LUT reordered for the current HDL
        y_q_hdl_order = reorder_lut_signed_order_to_hdl_raw_order(y_q_signed_order)

        act_mem_path = export_dir / f"act_layer{layer_idx:d}.mem"

        # Export the reordered version
        save_mem_file(y_q_hdl_order, qcfg.act_bits, act_mem_path)

        exported.append(
            {
                "layer_name": layer_name,
                "activation_name": row["activation_name"],
                "addr_bits": int(row["addr_bits"]),
                "implementation": implementation,
                "act_mem": str(act_mem_path),
                "n_words": int(len(y_q_hdl_order)),
                "reordered_for_hdl_raw_addressing": True,
            }
        )

    return exported


def export_pipeline_layout_json(
    activation_results: dict[str, Any],
    *,
    export_dir: Path,
    weight_bias_exports: list[dict[str, Any]],
    activation_exports: list[dict[str, Any]],
    scales_path: Path | None = None,
    layout_filename: str = "mem_layout.json",
) -> dict[str, Any]:
    """
    Write a JSON layout for the pipeline export.

    This gives the pipeline mode the same kind of human-readable JSON file as
    the sequential mode. In pipeline mode files are split per layer, so the JSON
    stores filenames and sizes instead of offsets in concatenated memories.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    qcfg = activation_results["qcfg"]
    quantized_layers = sorted(
        activation_results["quantized_layers"],
        key=lambda layer: int(str(_read(layer, "name")).split("_")[1]),
    )
    best_map = _build_activation_best_map(activation_results)

    wb_by_layer = {str(item["layer_name"]): item for item in weight_bias_exports}
    act_by_layer = {str(item["layer_name"]): item for item in activation_exports}

    weight_layout: list[dict[str, Any]] = []
    bias_layout: list[dict[str, Any]] = []
    act_layout: list[dict[str, Any]] = []

    total_weight_words = 0
    total_bias_words = 0
    total_activation_words = 0

    for layer in quantized_layers:
        layer_name = str(_read(layer, "name"))
        layer_idx = int(layer_name.split("_")[1])
        is_output = bool(_read(layer, "is_output"))

        w_q = np.asarray(_read(layer, "w_q"), dtype=np.int64)
        b_q = np.asarray(_read(layer, "b_q"), dtype=np.int64)
        wb_item = wb_by_layer.get(layer_name, {})

        weight_words = int(w_q.size)
        bias_words = int(b_q.size)
        total_weight_words += weight_words
        total_bias_words += bias_words

        weight_layout.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "file": wb_item.get("weight_mem", str(export_dir / f"weight_layer{layer_idx:d}.mem")),
                "length": weight_words,
                "shape": list(w_q.shape),
            }
        )
        bias_layout.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "file": wb_item.get("bias_mem", str(export_dir / f"bias_layer{layer_idx:d}.mem")),
                "length": bias_words,
                "shape": list(b_q.shape),
            }
        )

        best_meta = best_map.get(layer_name, {})
        act_item = act_by_layer.get(layer_name)

        if act_item is not None:
            act_words = int(act_item.get("n_words", 0))
            act_file = act_item.get("act_mem")
            addr_bits = act_item.get("addr_bits")
            reordered = bool(act_item.get("reordered_for_hdl_raw_addressing", False))
            implementation = act_item.get("implementation", best_meta.get("implementation", "paper_fit"))
            activation_name = act_item.get("activation_name", best_meta.get("activation_name", _read(layer, "activation_name", "-")))
        else:
            act_words = 0
            act_file = None
            addr_bits = None
            reordered = False
            implementation = "none" if is_output else best_meta.get("implementation", "analytic_or_none")
            activation_name = best_meta.get("activation_name", _read(layer, "activation_name", "-"))

        total_activation_words += act_words
        act_layout.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "file": act_file,
                "length": act_words,
                "activation_name": activation_name,
                "implementation": implementation,
                "addr_bits": None if addr_bits is None else int(addr_bits),
                "reordered_for_hdl_raw_addressing": reordered,
            }
        )

    layout = {
        "format": "pipeline_mem_v1",
        "mode": "pipeline",
        "description": "One weight file and one bias file per layer; one activation .mem file per LUT layer.",
        "scales_vh": None if scales_path is None else str(scales_path),
        "bits": {
            "act_bits": int(qcfg.act_bits),
            "weight_bits": int(qcfg.weight_bits),
            "bias_bits": int(qcfg.bias_bits),
            "dsp_bits": int(qcfg.dsp_bits),
        },
        "weights": weight_layout,
        "biases": bias_layout,
        "activations": act_layout,
        "total_weight_words": int(total_weight_words),
        "total_bias_words": int(total_bias_words),
        "total_activation_words": int(total_activation_words),
    }

    layout_path = export_dir / layout_filename
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    return {
        "mem_layout_json": str(layout_path),
        "mem_layout": layout,
        "total_weight_words": int(total_weight_words),
        "total_bias_words": int(total_bias_words),
        "total_activation_words": int(total_activation_words),
    }


def get_verilog_act_code(activation_name: str, implementation: str) -> int:
    """
    Activation codes for the Verilog header :
      0 = identity
      1 = relu
      2 = leaky_relu_pow2
      3 = lut
    """
    name = canonical_activation_name(activation_name)

    if implementation == "paper_fit":
        return 3
    if name == "relu":
        return 1
    if name == "leaky_relu":
        return 2
    return 0


def get_global_leak_shift(qcfg: QuantConfig) -> int:
    """
    Return k such that negative_slope = 2^-k.
    Used for the global LEAK_SHIFT in the header.
    """
    alpha = float(qcfg.activation_negative_slope)
    if alpha <= 0:
        raise ValueError("activation_negative_slope must be > 0.")

    k = int(round(-np.log2(alpha)))
    if not np.isclose(alpha, 2 ** (-k)):
        raise ValueError(
            f"activation_negative_slope={alpha} is not of the form 2^-k."
        )
    return k


def build_scales_vh_text(activation_results: dict[str, Any]) -> str:
    """
    Generate the scales.vh content with a compact convention :
    - ACTi : type d'activation
    - LEAK_SHIFT: global shift if leaky_relu_pow2
    - LUT_Ni : LUT N per layer if ACTi = 3
    - SPi : pre-activation scale of layer i
    - SXi : boundary scale (network input, puis layer outputs)
    """
    qcfg = activation_results["qcfg"]
    shift_exp = activation_results["shift_exp"]
    quantized_layers = activation_results["quantized_layers"]
    activation_best_df = activation_results.get("activation_best_df")
    model_bundle = activation_results["model_bundle"]
    dataset = activation_results["dataset"]

    config = model_bundle.config
    layers = tuple(config["layers"])
    output_dim = int(config.get("output_dim", 2))
    topology = [dataset.x.shape[1], *layers, output_dim]

    best_activation_map = {}
    if activation_best_df is not None and len(activation_best_df):
        for _, row in activation_best_df.iterrows():
            best_activation_map[row["layer_name"]] = row.to_dict()

    def layer_idx_of(layer_obj) -> int:
        return int(str(_read(layer_obj, "name")).split("_")[1])

    def sx_input_of_layer(layer_obj) -> int:
        idx = layer_idx_of(layer_obj)
        if idx == 0:
            return int(shift_exp["input"])
        prev_layer = quantized_layers[idx - 1]
        if bool(_read(prev_layer, "is_output")):
            return int(_read(prev_layer, "output_exp"))
        return int(_read(prev_layer, "act_exp"))

    def sx_output_of_network() -> int:
        last_layer = quantized_layers[-1]
        if bool(_read(last_layer, "is_output")):
            return int(_read(last_layer, "output_exp"))
        return int(_read(last_layer, "act_exp"))

    lines = []
    lines.append("// -----------------------------------------------------------------------------")
    lines.append("// Macro configuration (auto-generated)")
    lines.append("// -----------------------------------------------------------------------------")
    lines.append("`ifndef SCALES")
    lines.append("`define SCALES")
    lines.append("")
    lines.append("// V2 numerical widths")
    lines.append(f"`define ACT_W       {int(qcfg.act_bits)}")
    lines.append(f"`define WEIGHT_W    {int(qcfg.weight_bits)}")
    lines.append(f"`define BIAS_W      {int(qcfg.bias_bits)}")
    lines.append(f"`define DSP_PW      {int(qcfg.dsp_bits)}")
    lines.append("")
    lines.append("// --- Topologie ---")
    lines.append(f"`define NUM_LINEAR  {len(quantized_layers)}")
    for idx, width in enumerate(topology):
        lines.append(f"`define N{idx}  {int(width)}")
    lines.append("")
    lines.append("// Activation codes:")
    lines.append("// 0 = identity")
    lines.append("// 1 = relu")
    lines.append("// 2 = leaky_relu_pow2")
    lines.append("// 3 = lut")
    lines.append("")
    lines.append("// -----------------------------------------------------------------------------")
    lines.append("// Model scale factors")
    lines.append("//")
    lines.append("// SXi : tensor scale at the input of layer i")
    lines.append("//       and SX{NUM_LINEAR} = network output scale")
    lines.append("// SWi : weight scale of layer i")
    lines.append("// SBi : bias scale of layer i")
    lines.append("// SPi : pre-activation scale of layer i")
    lines.append("//       (present only for hidden layers)")
    lines.append("// -----------------------------------------------------------------------------")

    for layer in quantized_layers:
        idx = layer_idx_of(layer)
        is_output = bool(_read(layer, "is_output"))

        lines.append(f"// Couche {idx}")
        lines.append(f"`define SX{idx} {sx_input_of_layer(layer)}")
        lines.append(f"`define SW{idx} {int(_read(layer, 'weight_exp'))}")
        lines.append(f"`define SB{idx} {int(_read(layer, 'bias_exp'))}")

        if not is_output:
            lines.append(f"`define SP{idx} {int(_read(layer, 'preact_exp'))}")

        lines.append("")

    lines.append("// Network output")
    lines.append(f"`define SX{len(quantized_layers)} {sx_output_of_network()}")
    lines.append("")

    lines.append("// -----------------------------------------------------------------------------")
    lines.append("// Activation type per linear layer")
    lines.append("// -----------------------------------------------------------------------------")
    for layer in quantized_layers:
        layer_name = _read(layer, "name")
        idx = layer_idx_of(layer)

        if bool(_read(layer, "is_output")):
            act_code = 0
        else:
            best_meta = best_activation_map.get(layer_name, {})
            implementation = best_meta.get("implementation", "identity")
            activation_name = _read(layer, "activation_name", "identity")
            act_code = get_verilog_act_code(activation_name, implementation)

        lines.append(f"`define ACT{idx} {act_code}")
    lines.append("")

    lines.append("// Use only if ACTi = 2")
    lines.append(f"`define LEAK_SHIFT {int(get_global_leak_shift(qcfg))}")
    lines.append("")

    lines.append("// Use only if ACTi = 3")
    for layer in quantized_layers:
        layer_name = _read(layer, "name")
        idx = layer_idx_of(layer)

        lut_n = 0
        if not bool(_read(layer, "is_output")):
            best_meta = best_activation_map.get(layer_name, {})
            if best_meta.get("implementation") == "paper_fit" and pd.notna(best_meta.get("addr_bits", np.nan)):
                lut_n = int(best_meta["addr_bits"])

        lines.append(f"`define LUT_N{idx} {lut_n}")

    lines.append("")
    lines.append("`endif // SCALES")
    return "\n".join(lines) + "\n"


def export_scales_vh(
    activation_results: dict[str, Any],
    *,
    export_dir: Path,
    filename: str = "scales.vh",
) -> Path:
    """
    Write the scales.vh file.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    scales_path = export_dir / filename
    scales_text = build_scales_vh_text(activation_results)
    scales_path.write_text(scales_text, encoding="utf-8")
    return scales_path


def reorder_lut_signed_order_to_hdl_raw_order(y_q: np.ndarray) -> np.ndarray:
    """
    Reorder a LUT generated in natural signed order for the current HDL.

    Currently generated order :
        index 0       -> x = -32768
        middle index  -> x = 0
        last index -> x close to +32767

    Expected order by the current rom_act.sv :
        addr 0        -> x = 0
        middle address   -> x = -32768
        last addr   -> x close to -1

    So the two halves are swapped :
        [negative, positive] -> [positive, negative]
    """
    y_q = np.asarray(y_q, dtype=np.int64).reshape(-1)

    if len(y_q) == 0:
        return y_q

    if len(y_q) & (len(y_q) - 1):
        raise ValueError(f"LUT size must be a power of 2, got {len(y_q)}")

    half = len(y_q) // 2

    return np.concatenate([
        y_q[half:],   # partie x >= 0
        y_q[:half],   # partie x < 0
    ])


def export_pipeline_mem_artifacts(
    activation_results: dict[str, Any],
    *,
    export_dir: Path | None = None,
    show_summary: bool = True,
) -> dict[str, Any]:
    """
    Export final useful artifacts :
    - weight_layerXX.mem
    - bias_layerXX.mem
    - act_layerXX.mem (only for the best LUTs)
    - scales.vh
    """
    paths = activation_results["paths"]
    qcfg = activation_results["qcfg"]
    export_dir = Path(export_dir or (paths.output_dir / "final_mem"))
    export_dir.mkdir(parents=True, exist_ok=True)

    weight_bias_exports = export_weight_bias_mem_files(
        activation_results,
        export_dir=export_dir,
    )

    activation_exports = export_best_activation_mem_files(
        activation_results,
        export_dir=export_dir,
    )

    scales_path = export_scales_vh(
        activation_results,
        export_dir=export_dir,
        filename="scales.vh",
    )

    layout_exports = export_pipeline_layout_json(
        activation_results,
        export_dir=export_dir,
        weight_bias_exports=weight_bias_exports,
        activation_exports=activation_exports,
        scales_path=scales_path,
        layout_filename="mem_layout.json",
    )

    summary = {
        "export_dir": str(export_dir),
        "weight_bias_exports": weight_bias_exports,
        "activation_exports": activation_exports,
        "scales_vh": str(scales_path),
        **layout_exports,
        "act_width": int(qcfg.act_bits),
        "weight_width": int(qcfg.weight_bits),
        "bias_width": int(qcfg.bias_bits),
        "dsp_width": int(qcfg.dsp_bits),
    }

    if show_summary:
        print("\n" + "=" * 100)
        print("FINAL EXPORT OF .MEM ARTIFACTS")
        print("=" * 100)
        print(f"Export directory      : {export_dir}")
        print(f"ACT_W              : {qcfg.act_bits}")
        print(f"WEIGHT_W           : {qcfg.weight_bits}")
        print(f"BIAS_W             : {qcfg.bias_bits}")
        print(f"DSP_PW             : {qcfg.dsp_bits}")
        print(f"Exported weight/bias files: {len(weight_bias_exports)} layers")
        print(f"LUT activations     : {len(activation_exports)} layers")
        print(f"Verilog header      : {scales_path}")
        print(f"JSON layout         : {layout_exports['mem_layout_json']}")

        if len(weight_bias_exports):
            print("\nWeights / biases :")
            for item in weight_bias_exports:
                print(
                    f"- {item['layer_name']} | "
                    f"weights={item['weight_mem']} | "
                    f"bias={item['bias_mem']}"
                )

        if len(activation_exports):
            print("\nSelected LUT activations :")
            for item in activation_exports:
                print(
                    f"- {item['layer_name']} | "
                    f"act={item['act_mem']} | "
                    f"N={item['addr_bits']} | "
                    f"words={item['n_words']}"
                )

        print("=" * 100)

    return summary


def _build_activation_best_map(activation_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    activation_best_df = activation_results.get("activation_best_df")
    best_map: dict[str, dict[str, Any]] = {}
    if activation_best_df is not None and len(activation_best_df):
        for _, row in activation_best_df.iterrows():
            best_map[str(row["layer_name"])] = row.to_dict()
    return best_map


def export_combined_mem_files(
    activation_results: dict[str, Any],
    *,
    export_dir: str | Path,
    weight_filename: str = "weight_all.mem",
    bias_filename: str = "bias_all.mem",
    act_filename: str = "act_all.mem",
    layout_filename: str = "mem_layout.json",
) -> dict[str, Any]:
    """
    Export only three concatenated .mem files :

    - weight_all.mem : all weights, layer by layer
    - bias_all.mem   : all biases, layer by layer
    - act_all.mem    : all selected LUTs, layer by layer

    Sizes and offsets are deduced from scales.vh when reloading.
    A JSON layout file is also written for human debugging, but reload
    does not depend on it.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    qcfg = activation_results["qcfg"]
    quantized_layers = sorted(
        activation_results["quantized_layers"],
        key=lambda layer: int(str(_read(layer, "name")).split("_")[1]),
    )
    activation_modules = activation_results.get("activation_modules", {})
    best_map = _build_activation_best_map(activation_results)

    all_weights: list[np.ndarray] = []
    all_biases: list[np.ndarray] = []
    all_acts: list[np.ndarray] = []

    weight_layout = []
    bias_layout = []
    act_layout = []

    w_offset = 0
    b_offset = 0
    a_offset = 0

    for layer in quantized_layers:
        layer_name = _read(layer, "name")
        layer_idx = int(str(layer_name).split("_")[1])
        is_output = bool(_read(layer, "is_output"))

        w_q = np.asarray(_read(layer, "w_q"), dtype=np.int64).reshape(-1)
        b_q = np.asarray(_read(layer, "b_q"), dtype=np.int64).reshape(-1)

        all_weights.append(w_q)
        all_biases.append(b_q)

        weight_layout.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "offset": w_offset,
                "length": int(w_q.size),
                "shape": list(np.asarray(_read(layer, "w_q")).shape),
            }
        )
        bias_layout.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "offset": b_offset,
                "length": int(b_q.size),
                "shape": list(np.asarray(_read(layer, "b_q")).shape),
            }
        )

        w_offset += int(w_q.size)
        b_offset += int(b_q.size)

        act_words = np.asarray([], dtype=np.int64)
        act_meta = best_map.get(layer_name, {})
        if (not is_output) and act_meta.get("implementation") == "paper_fit":
            module = activation_modules.get(layer_idx)
            if module is not None and hasattr(module, "y_knots_q"):
                # Stored in the order expected by the current HDL: raw two's-complement address.
                act_words = reorder_lut_signed_order_to_hdl_raw_order(
                    np.asarray(module.y_knots_q, dtype=np.int64)
                )

        all_acts.append(act_words)
        act_layout.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "offset": a_offset,
                "length": int(act_words.size),
                "activation_name": act_meta.get("activation_name", _read(layer, "activation_name", "-")),
                "implementation": act_meta.get("implementation", "none" if is_output else "analytic_or_none"),
                "addr_bits": None if int(act_words.size) == 0 else int(round(np.log2(int(act_words.size)))),
                "reordered_for_hdl_raw_addressing": bool(int(act_words.size) > 0),
            }
        )
        a_offset += int(act_words.size)

    weight_all = np.concatenate(all_weights) if len(all_weights) else np.asarray([], dtype=np.int64)
    bias_all = np.concatenate(all_biases) if len(all_biases) else np.asarray([], dtype=np.int64)
    act_all = np.concatenate(all_acts) if len(all_acts) else np.asarray([], dtype=np.int64)

    weight_path = export_dir / weight_filename
    bias_path = export_dir / bias_filename
    act_path = export_dir / act_filename

    save_mem_file(weight_all, qcfg.weight_bits, weight_path)
    save_mem_file(bias_all, qcfg.bias_bits, bias_path)
    save_mem_file(act_all, qcfg.act_bits, act_path)

    layout = {
        "format": "concat_mem_v1",
        "description": "V2 sequential layout. ACT_W is shared by network input, activations, LUT words and final output.",
        "bits": {
            "act_bits": int(qcfg.act_bits),
            "weight_bits": int(qcfg.weight_bits),
            "bias_bits": int(qcfg.bias_bits),
            "dsp_bits": int(qcfg.dsp_bits),
        },
        "weight_all_mem": str(weight_path),
        "bias_all_mem": str(bias_path),
        "act_all_mem": str(act_path),
        "weights": weight_layout,
        "biases": bias_layout,
        "activations": act_layout,
        "total_weight_words": int(weight_all.size),
        "total_bias_words": int(bias_all.size),
        "total_activation_words": int(act_all.size),
    }

    layout_path = export_dir / layout_filename
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    return {
        "weight_all_mem": str(weight_path),
        "bias_all_mem": str(bias_path),
        "act_all_mem": str(act_path),
        "mem_layout_json": str(layout_path),
        "mem_layout": layout,
        "total_weight_words": int(weight_all.size),
        "total_bias_words": int(bias_all.size),
        "total_activation_words": int(act_all.size),
    }


def write_mem_signed(values, bits: int, path: str | Path) -> Path:
    """Write a signed two's-complement .mem file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mem_file(np.asarray(values).reshape(-1), int(bits), path)
    return path


def export_sequential_mem_artifacts(
    activation_results: dict[str, Any],
    *,
    export_dir: Path | None = None,
    show_summary: bool = True,
    export_split_files: bool = False,
) -> dict[str, Any]:
    """
    Export final artifacts.

    By default, only three .mem files are generated :
    - weight_all.mem
    - bias_all.mem
    - act_all.mem

    The legacy per-layer files can be re-enabled with export_split_files=True.
    """
    paths = activation_results["paths"]
    qcfg = activation_results["qcfg"]
    export_dir = Path(export_dir or (paths.output_dir / "final_mem"))
    export_dir.mkdir(parents=True, exist_ok=True)

    scales_path = export_scales_vh(
        activation_results,
        export_dir=export_dir,
        filename="scales.vh",
    )

    combined_exports = export_combined_mem_files(
        activation_results,
        export_dir=export_dir,
        weight_filename="weight_all.mem",
        bias_filename="bias_all.mem",
        act_filename="act_all.mem",
    )

    split_exports: dict[str, Any] | None = None
    if export_split_files:
        split_exports = {
            "weight_bias_exports": export_weight_bias_mem_files(activation_results, export_dir=export_dir),
            "activation_exports": export_best_activation_mem_files(activation_results, export_dir=export_dir),
        }

    summary = {
        "export_dir": str(export_dir),
        "scales_vh": str(scales_path),
        **combined_exports,
        "split_exports": split_exports,
        "act_width": int(qcfg.act_bits),
        "weight_width": int(qcfg.weight_bits),
        "bias_width": int(qcfg.bias_bits),
        "dsp_width": int(qcfg.dsp_bits),
    }

    if show_summary:
        layout = combined_exports["mem_layout"]
        print("\n" + "=" * 100)
        print("FINAL EXPORT OF CONCATENATED .MEM ARTIFACTS")
        print("=" * 100)
        print(f"Export directory      : {export_dir}")
        print(f"Verilog header      : {scales_path}")
        print(".mem files       :")
        print(f"- weights             : {combined_exports['weight_all_mem']} ({layout['total_weight_words']} words)")
        print(f"- biases             : {combined_exports['bias_all_mem']} ({layout['total_bias_words']} words)")
        print(f"- LUT activations   : {combined_exports['act_all_mem']} ({layout['total_activation_words']} words)")
        print(f"Layout debug        : {combined_exports['mem_layout_json']}")

        print("\nWeight/bias layout :")
        for w_item, b_item in zip(layout["weights"], layout["biases"]):
            print(
                f"- layer_{w_item['layer_idx']} | "
                f"W offset={w_item['offset']} len={w_item['length']} shape={tuple(w_item['shape'])} | "
                f"B offset={b_item['offset']} len={b_item['length']}"
            )

        print("\nActivation LUT layout :")
        any_lut = False
        for a_item in layout["activations"]:
            if int(a_item["length"]) > 0:
                any_lut = True
                print(
                    f"- layer_{a_item['layer_idx']} | "
                    f"offset={a_item['offset']} len={a_item['length']} "
                    f"N={a_item['addr_bits']} act={a_item['activation_name']}"
                )
        if not any_lut:
            print("- no LUT exported in act_all.mem")

        if export_split_files:
            print("\nLegacy per-layer files also exported (export_split_files=True).")
        print("=" * 100)

    return summary


def export_final_mem_artifacts(
    activation_results: dict[str, Any],
    *,
    export_dir: Path | None = None,
    show_summary: bool = True,
    export_split_files: bool | None = None,
) -> dict[str, Any]:
    """
    Export the final FPGA memory artifacts using qcfg.fpga_mode.

    Modes:
    - qcfg.fpga_mode == "pipeline": export one weight/bias file per layer and one
      activation LUT file per LUT layer.
    - qcfg.fpga_mode == "sequential": export three concatenated files:
      weight_all.mem, bias_all.mem and act_all.mem.

    The optional export_split_files argument only affects sequential mode. If it is
    None, qcfg.export_split_files is used.
    """
    qcfg = activation_results["qcfg"]
    mode = str(_cfg_value(qcfg, "fpga_mode", "pipeline")).lower().strip()

    if mode in {"pipeline", "pipelined"}:
        return export_pipeline_mem_artifacts(
            activation_results,
            export_dir=export_dir,
            show_summary=show_summary,
        )

    if mode in {"sequential", "seq"}:
        split = bool(_cfg_value(qcfg, "export_split_files", False)) if export_split_files is None else bool(export_split_files)
        return export_sequential_mem_artifacts(
            activation_results,
            export_dir=export_dir,
            show_summary=show_summary,
            export_split_files=split,
        )

    raise ValueError(
        f"Unknown fpga_mode={mode!r}. Expected 'pipeline' or 'sequential'."
    )


def export_quantized_dataset(
    *,
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    input_exp: int,
    export_dir: str | Path,
    qcfg: QuantConfig,
    filename: str = "quantized_dataset.npz",
    target_exp: int | None = None,
    input_name: str = "x_q",
    target_name: str = "y",
    show_summary: bool = True,
) -> dict[str, Any]:
    """Export a quantized dataset for regression or classification.

    - Regression: quantize both x and y.
    - Classification: quantize x and keep labels as integer classes.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    task = str(task).lower().strip()

    x_q = quantize_to_int(x, int(input_exp), num_bits=qcfg.act_bits, signed=True)

    payload: dict[str, Any] = {
        input_name: x_q,
        "input_exp": np.int32(input_exp),
        "task": np.asarray(task),
    }

    if task == "regression":
        if target_exp is None:
            raise ValueError("target_exp is required for a regression dataset.")
        y_q = quantize_to_int(y, int(target_exp), num_bits=qcfg.act_bits, signed=True)
        payload[target_name] = y_q
        payload["target_exp"] = np.int32(target_exp)
    elif task == "classification":
        y_q = np.asarray(y).astype(np.int64)
        payload[target_name] = y_q
    else:
        raise ValueError("task must be 'regression' or 'classification'.")

    metadata = {
        "input_name": input_name,
        "target_name": target_name,
        "task": task,
        "input_exp": int(input_exp),
        "target_exp": None if target_exp is None else int(target_exp),
        "input_bits": int(qcfg.act_bits),
        "target_bits": None if task == "classification" else int(qcfg.act_bits),
        "input_shape": tuple(np.asarray(x_q).shape),
        "target_shape": tuple(np.asarray(y_q).shape),
        "classification_rule": "argmax_logits" if task == "classification" else None,
    }
    payload["metadata_json"] = json.dumps(metadata, indent=2)

    out_path = export_dir / filename
    np.savez(out_path, **payload)

    if show_summary:
        print("\n" + "=" * 100)
        print("QUANTIZED DATASET EXPORT")
        print("=" * 100)
        print(f"File         : {out_path}")
        print(f"Task         : {task}")
        print(f"Input name   : {input_name}")
        print(f"Target name  : {target_name}")
        print(f"Input exp    : {input_exp}")
        print(f"Target exp   : {target_exp}")
        print(f"Input shape  : {np.asarray(x_q).shape}")
        print(f"Target shape : {np.asarray(y_q).shape}")
        print("=" * 100)

    return {"dataset_path": str(out_path), "metadata": metadata}


def export_quantized_dataset_npz(
    results,
    export_dir,
    filename="dataset_walter_quantized.npz",
    show_summary=True,
):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    dataset = results["dataset"]
    model_bundle = results["model_bundle"]
    summary_df = results["summary_df"]
    qcfg = results["qcfg"]
    model = model_bundle.model
    model.eval()

    x_float = dataset.x.astype(np.float32)
    y_float = dataset.y_rad.astype(np.float32)

    input_exp = int(summary_df["input_exp"].iloc[0])
    target_exp = int(summary_df["output_exp"].dropna().iloc[-1])

    x_int = quantize_to_int(x_float, input_exp, num_bits=qcfg.act_bits, signed=True)
    y_int = quantize_to_int(y_float, target_exp, num_bits=qcfg.act_bits, signed=True)

    device = next(model.parameters()).device

    with torch.no_grad():
        x_t = torch.tensor(x_float, dtype=torch.float32, device=device)
        pred_float = model(x_t).cpu().numpy().astype(np.float32)

    pred_exp = target_exp

    # This remains a quantized FLOAT-model reference, not yet the future
    # bit-exact sequential RTL reference.  No saturation is applied.
    pred_int = quantize_to_int(pred_float, pred_exp, num_bits=qcfg.act_bits, signed=True)

    output_path = export_dir / filename

    np.savez(
        output_path,

        ph_val=x_float,
        ph_val_int=x_int,

        pos=y_float,
        pos_int=y_int,

        pred_float=pred_float,
        pred_int=pred_int,

        input_scale_exp=np.array(input_exp, dtype=np.int32),
        target_scale_exp=np.array(target_exp, dtype=np.int32),
        pred_scale_exp=np.array(pred_exp, dtype=np.int32),
    )

    if show_summary:
        print("Quantized dataset exported:")
        print(" ", output_path)
        print("ph_val      :", x_float.shape, x_float.dtype)
        print("ph_val_int  :", x_int.shape, x_int.dtype)
        print("pos         :", y_float.shape, y_float.dtype)
        print("pos_int     :", y_int.shape, y_int.dtype)
        print("pred_float  :", pred_float.shape, pred_float.dtype)
        print("pred_int    :", pred_int.shape, pred_int.dtype)
        print("input_exp   :", input_exp)
        print("target_exp  :", target_exp)
        print("pred_exp    :", pred_exp)

    return output_path

def int16_to_hex_word(v: int) -> str:
    """Convertit un int16 signé en mot hexadécimal 16 bits complément à 2."""
    return f"{(int(v) & 0xFFFF):04X}"


def write_int16_mem(path: Path, array: np.ndarray) -> None:
    """Écrit un tableau int16 aplati, un mot hex 16 bits par ligne."""
    arr = np.asarray(array, dtype=np.int16).reshape(-1)

    with path.open("w", encoding="utf-8") as f:
        for v in arr:
            f.write(int16_to_hex_word(v) + "\n")

def export_ph_val_pred_mem(
    npz_path,
    out_dir="tb_vectors_soc_mlp",
    n_samples=100,
    input_key="ph_val_int",
    expected_key="pred_int",
    input_filename="tb_inputs_stream.mem",
    expected_filename="tb_expected_outputs.mem",
    show_summary=True,
):
    """
    Extrait ph_val_int et pred_int depuis dataset_walter_quantized.npz
    et génère les fichiers .mem pour le testbench FPGA.
    """
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = np.load(npz_path, allow_pickle=True)

    # The current UART/testbench stream is explicitly 16-bit.  ACT_W may become
    # configurable later, but silently narrowing another width here would violate
    # the V2 no-saturation/no-hidden-truncation contract.
    if input_key in ds.files and np.asarray(ds[input_key]).dtype.itemsize > 2:
        vals = np.asarray(ds[input_key], dtype=np.int64)
        if vals.size and (vals.min() < -32768 or vals.max() > 32767):
            raise OverflowError("export_ph_val_pred_mem currently requires ACT_W <= 16.")

    if input_key not in ds.files:
        raise KeyError(f"Clé absente: {input_key}. Clés disponibles: {ds.files}")

    if expected_key not in ds.files:
        raise KeyError(f"Clé absente: {expected_key}. Clés disponibles: {ds.files}")

    x_all = np.asarray(ds[input_key], dtype=np.int16)
    y_all = np.asarray(ds[expected_key], dtype=np.int16)

    if x_all.ndim != 2:
        raise ValueError(f"{input_key} doit être 2D, shape reçue: {x_all.shape}")

    if y_all.ndim != 2:
        raise ValueError(f"{expected_key} doit être 2D, shape reçue: {y_all.shape}")

    n = min(int(n_samples), x_all.shape[0], y_all.shape[0])

    x = x_all[:n]
    y = y_all[:n]

    input_path = out_dir / input_filename
    expected_path = out_dir / expected_filename

    write_int16_mem(input_path, x)
    write_int16_mem(expected_path, y)

    if show_summary:
        print("Fichiers .mem générés:")
        print(" input   :", input_path)
        print(" expected:", expected_path)
        print(" n       :", n)
        print(" x shape :", x.shape)
        print(" y shape :", y.shape)

    return input_path, expected_path
