"""Core model calibration and quantization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

from activations import activation_float
from config import LayerCalibration, ProjectPaths, QuantConfig, QuantizedLayer
from data_model import load_dataset, load_model_bundle_from_dataset
from fixed_point import choose_shift_exponent, int_bounds, quantize_to_int
from utils import _read


@torch.no_grad()
def collect_model_tensors(model, x_np: np.ndarray, activation_plan: list[dict[str, Any]], qcfg: QuantConfig) -> dict[str, np.ndarray]:
    """Run the float model once and collect input, pre-activations, activations and output."""
    x = torch.tensor(x_np, dtype=torch.float32)
    tensors: dict[str, np.ndarray] = {"input": x.detach().cpu().numpy().copy()}

    current = x
    for index, layer in enumerate(model.hidden_layers):
        preact = layer(current)
        preact_np = preact.detach().cpu().numpy().copy()
        act_name = activation_plan[index]["name"]
        act_np = activation_float(preact_np, act_name, negative_slope=qcfg.activation_negative_slope).astype(np.float32)

        tensors[f"layer_{index}_preact"] = preact_np
        tensors[f"layer_{index}_act"] = act_np
        current = torch.tensor(act_np, dtype=torch.float32)

    tensors["output"] = model.out(current).detach().cpu().numpy().copy()
    return tensors


# Backward-compatible name from the original all-in-one file.
collect_model_tensors_per_layer = collect_model_tensors


def _select_exponent(values, *, bits: int, margin: float, qcfg: QuantConfig) -> int:
    return choose_shift_exponent(
        values,
        max_int=int_bounds(bits)[1],
        margin=margin,
        mode=qcfg.calibration_mode,
        percentile=qcfg.calibration_percentile,
    )


def calibrate_and_quantize_layers(model, tensors: dict[str, np.ndarray], activation_plan: list[dict[str, Any]], qcfg: QuantConfig):
    """Calibrate fixed-point exponents and quantize all linear layers.

    V2 uses ACT_W for input, hidden activation/LUT words, pre-activation LUT
    representation and final output.  The physical MAC width is DSP_PW and its
    scale-aware overflow optimization is intentionally handled in a later step.
    """
    linear_layers = list(model.hidden_layers) + [model.out]
    shift_exp: dict[str, int] = {}
    calibrated_layers: list[LayerCalibration] = []
    quantized_layers: list[QuantizedLayer] = []

    input_exp = _select_exponent(tensors["input"], bits=qcfg.act_bits, margin=qcfg.margin_acts, qcfg=qcfg)
    shift_exp["input"] = input_exp
    prev_act_exp = input_exp

    for layer_index, layer in enumerate(linear_layers):
        w_float = layer.weight.detach().cpu().numpy()
        b_float = layer.bias.detach().cpu().numpy()
        is_output = layer_index == len(linear_layers) - 1

        weight_exp = _select_exponent(w_float, bits=qcfg.weight_bits, margin=qcfg.margin_weights, qcfg=qcfg)
        bias_exp = _select_exponent(b_float, bits=qcfg.bias_bits, margin=qcfg.margin_weights, qcfg=qcfg)

        preact_exp = None
        act_exp = None
        output_exp = None
        activation_name = None

        if is_output:
            output_exp = _select_exponent(tensors["output"], bits=qcfg.act_bits, margin=qcfg.margin_acts, qcfg=qcfg)
            shift_exp[f"layer_{layer_index}_output"] = output_exp
        else:
            preact_exp = _select_exponent(tensors[f"layer_{layer_index}_preact"], bits=qcfg.act_bits, margin=qcfg.margin_acts, qcfg=qcfg)
            act_exp = _select_exponent(tensors[f"layer_{layer_index}_act"], bits=qcfg.act_bits, margin=qcfg.margin_acts, qcfg=qcfg)
            activation_name = activation_plan[layer_index]["name"]
            shift_exp[f"layer_{layer_index}_preact"] = preact_exp
            shift_exp[f"layer_{layer_index}_act"] = act_exp
            next_prev_act_exp = act_exp

        calibration = LayerCalibration(
            name=f"layer_{layer_index}",
            is_output=is_output,
            w_float=w_float,
            b_float=b_float,
            input_exp=prev_act_exp,
            weight_exp=weight_exp,
            bias_exp=bias_exp,
            preact_exp=preact_exp,
            act_exp=act_exp,
            output_exp=output_exp,
            activation_name=activation_name,
        )
        calibrated_layers.append(calibration)

        quantized_layers.append(
            QuantizedLayer(
                name=calibration.name,
                is_output=calibration.is_output,
                w_float=w_float,
                b_float=b_float,
                w_q=quantize_to_int(w_float, weight_exp, num_bits=qcfg.weight_bits),
                b_q=quantize_to_int(b_float, bias_exp, num_bits=qcfg.bias_bits),
                input_exp=prev_act_exp,
                weight_exp=weight_exp,
                bias_exp=bias_exp,
                preact_exp=preact_exp,
                act_exp=act_exp,
                output_exp=output_exp,
                activation_name=activation_name,
            )
        )

        if not is_output:
            prev_act_exp = next_prev_act_exp

    return calibrated_layers, quantized_layers, shift_exp


def summarize_quantized_layers(quantized_layers: list[QuantizedLayer]) -> pd.DataFrame:
    rows = []
    for layer in quantized_layers:
        rows.append(
            {
                "layer": layer.name,
                "is_output": layer.is_output,
                "w_shape": tuple(layer.w_q.shape),
                "input_exp": layer.input_exp,
                "weight_exp": layer.weight_exp,
                "bias_exp": layer.bias_exp,
                "preact_exp": layer.preact_exp,
                "act_exp": layer.act_exp,
                "output_exp": layer.output_exp,
                "activation": layer.activation_name or "-",
            }
        )
    return pd.DataFrame(rows)


def quantize_model(paths: ProjectPaths, qcfg: QuantConfig, *, device: str = "cpu", show_summary: bool = True) -> dict[str, Any]:
    """Load the model and dataset, then calibrate and quantize all linear layers."""
    dataset = load_dataset(paths.dataset_path)
    model_bundle = load_model_bundle_from_dataset(paths.model_path, dataset=dataset, qcfg=qcfg, device=device)

    tensors = collect_model_tensors(
        model=model_bundle.model,
        x_np=dataset.x,
        activation_plan=model_bundle.activation_plan,
        qcfg=qcfg,
    )

    layer_export, quantized_layers, shift_exp = calibrate_and_quantize_layers(
        model=model_bundle.model,
        tensors=tensors,
        activation_plan=model_bundle.activation_plan,
        qcfg=qcfg,
    )
    summary_df = summarize_quantized_layers(quantized_layers)

    results = {
        "paths": paths,
        "qcfg": qcfg,
        "dataset": dataset,
        "model_bundle": model_bundle,
        "tensors": tensors,
        "layer_export": layer_export,
        "shift_exp": shift_exp,
        "quantized_layers": quantized_layers,
        "summary_df": summary_df,
    }

    if show_summary:
        print("\n" + "=" * 100)
        print("BASIC QUANTIZATION")
        print("=" * 100)
        print(f"Model path       : {paths.model_path}")
        print(f"Dataset path     : {paths.dataset_path}")
        print(f"Output directory : {paths.output_dir}")
        print(f"FPGA export mode : {qcfg.fpga_mode}")
        print(f"ACT_W / DSP_PW   : {qcfg.act_bits} / {qcfg.dsp_bits}")
        print(f"Number of layers : {len(quantized_layers)}")
        print("=" * 100)
        display(summary_df)

    return results


# Backward-compatible name from the previous file.
run_basic_quant_pipeline = quantize_model

# =============================================================================
# DSP-aware weight-scale optimization
# =============================================================================
# =============================================================================
# DSP-aware weight-scale optimization -- FLOAT calibration
# =============================================================================


def _validate_dsp_utilization(max_utilization: float) -> float:
    """Validate and return the configured DSP accumulator utilization."""
    util = float(max_utilization)
    if not (0.0 < util <= 1.0):
        raise ValueError(
            f"dsp_acc_max_utilization must be in (0, 1], got {max_utilization}."
        )
    return util


def _dsp_real_bounds(
    *,
    dsp_bits: int,
    scale_exp: int,
    max_utilization: float,
) -> dict[str, float | int]:
    """Return physical/safe DSP ranges expressed in the real (float) domain.

    A DSP_PW signed integer P with scale ``Sacc`` represents::

        p_real = P / 2**Sacc

    The optimizer therefore compares FLOAT partial accumulations directly to
    these FLOAT limits.  No integer activation/weight/bias is used to choose SW.
    """
    util = _validate_dsp_utilization(max_utilization)
    qmin, qmax = int_bounds(int(dsp_bits), signed=True)

    safe_qmin = int(np.ceil(qmin * util))
    safe_qmax = int(np.floor(qmax * util))

    # np.ldexp(v, -S) computes v * 2**(-S) without constructing 2**S first.
    physical_real_min = float(np.ldexp(float(qmin), -int(scale_exp)))
    physical_real_max = float(np.ldexp(float(qmax), -int(scale_exp)))
    safe_real_min = float(np.ldexp(float(safe_qmin), -int(scale_exp)))
    safe_real_max = float(np.ldexp(float(safe_qmax), -int(scale_exp)))

    return {
        "physical_qmin": int(qmin),
        "physical_qmax": int(qmax),
        "safe_qmin": int(safe_qmin),
        "safe_qmax": int(safe_qmax),
        "physical_real_min": physical_real_min,
        "physical_real_max": physical_real_max,
        "safe_real_min": safe_real_min,
        "safe_real_max": safe_real_max,
    }


def _float_partial_accumulator_stats(
    x_float: np.ndarray,
    w_float: np.ndarray,
    b_float: np.ndarray,
    *,
    chunk_samples: int = 8192,
) -> dict[str, Any]:
    """Measure every partial MAC accumulation in FLOAT for one layer.

    For every sample s and neuron j the same accumulation order as the FPGA is
    inspected, but entirely in floating point::

        P_bias = b[j]
        P_0    = b[j] + x[s, 0] * w[j, 0]
        P_1    = P_0  + x[s, 1] * w[j, 1]
        ...

    This function is independent of SW.  Its output is therefore computed once
    per layer and reused while testing that layer's SW candidates.
    """
    x = np.asarray(x_float, dtype=np.float64)
    w = np.asarray(w_float, dtype=np.float64)
    b = np.asarray(b_float, dtype=np.float64).reshape(-1)

    if x.ndim != 2:
        raise ValueError(f"x_float must be 2D [samples, inputs], got {x.shape}.")
    if w.ndim != 2:
        raise ValueError(f"w_float must be 2D [neurons, inputs], got {w.shape}.")
    if x.shape[1] != w.shape[1]:
        raise ValueError(
            f"Input/weight mismatch: x_float has {x.shape[1]} inputs, "
            f"weights expect {w.shape[1]}."
        )
    if b.size != w.shape[0]:
        raise ValueError(
            f"Bias/weight mismatch: {b.size} biases for {w.shape[0]} neurons."
        )
    if x.shape[0] == 0:
        raise ValueError("x_float contains no calibration samples.")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be > 0.")

    n_samples = int(x.shape[0])
    n_neurons = int(w.shape[0])
    n_inputs = int(w.shape[1])

    bias_min_idx = int(np.argmin(b)) if b.size else 0
    bias_max_idx = int(np.argmax(b)) if b.size else 0
    partial_min = float(b[bias_min_idx]) if b.size else 0.0
    partial_max = float(b[bias_max_idx]) if b.size else 0.0
    min_loc = (0, bias_min_idx, -1)
    max_loc = (0, bias_max_idx, -1)

    final_min = np.inf
    final_max = -np.inf

    for start in range(0, n_samples, int(chunk_samples)):
        stop = min(start + int(chunk_samples), n_samples)
        x_chunk = x[start:stop]
        acc = np.broadcast_to(b.reshape(1, -1), (stop - start, n_neurons)).copy()

        for mac_idx in range(n_inputs):
            acc += x_chunk[:, mac_idx:mac_idx + 1] * w[:, mac_idx].reshape(1, -1)

            local_min_flat = int(np.argmin(acc))
            local_max_flat = int(np.argmax(acc))
            local_min_idx = np.unravel_index(local_min_flat, acc.shape)
            local_max_idx = np.unravel_index(local_max_flat, acc.shape)
            local_min = float(acc[local_min_idx])
            local_max = float(acc[local_max_idx])

            if local_min < partial_min:
                partial_min = local_min
                min_loc = (start + int(local_min_idx[0]), int(local_min_idx[1]), int(mac_idx))
            if local_max > partial_max:
                partial_max = local_max
                max_loc = (start + int(local_max_idx[0]), int(local_max_idx[1]), int(mac_idx))

        if n_inputs == 0:
            chunk_final_min = float(np.min(acc))
            chunk_final_max = float(np.max(acc))
        else:
            chunk_final_min = float(np.min(acc))
            chunk_final_max = float(np.max(acc))
        final_min = min(final_min, chunk_final_min)
        final_max = max(final_max, chunk_final_max)

    if abs(partial_min) >= abs(partial_max):
        worst_value = partial_min
        worst_loc = min_loc
        worst_sign = "negative" if partial_min < 0 else "zero"
    else:
        worst_value = partial_max
        worst_loc = max_loc
        worst_sign = "positive" if partial_max > 0 else "zero"

    return {
        "bias_min_float": float(np.min(b)) if b.size else 0.0,
        "bias_max_float": float(np.max(b)) if b.size else 0.0,
        "partial_min_float": float(partial_min),
        "partial_max_float": float(partial_max),
        "max_abs_partial_float": float(max(abs(partial_min), abs(partial_max))),
        "final_min_float": float(final_min),
        "final_max_float": float(final_max),
        "min_sample": int(min_loc[0]),
        "min_neuron": int(min_loc[1]),
        "min_mac_index": int(min_loc[2]),
        "max_sample": int(max_loc[0]),
        "max_neuron": int(max_loc[1]),
        "max_mac_index": int(max_loc[2]),
        "worst_value_float": float(worst_value),
        "worst_sample": int(worst_loc[0]),
        "worst_neuron": int(worst_loc[1]),
        "worst_mac_index": int(worst_loc[2]),
        "worst_sign": worst_sign,
        "n_samples": n_samples,
        "n_neurons": n_neurons,
        "n_inputs": n_inputs,
    }


def _evaluate_float_sw_candidate(
    float_stats: dict[str, Any],
    *,
    sx: int,
    sw: int,
    dsp_bits: int,
    max_utilization: float,
) -> dict[str, Any]:
    """Check one layer-local SW against FLOAT partial accumulations."""
    s_acc = int(sx) + int(sw)
    bounds = _dsp_real_bounds(
        dsp_bits=int(dsp_bits),
        scale_exp=s_acc,
        max_utilization=float(max_utilization),
    )

    partial_min = float(float_stats["partial_min_float"])
    partial_max = float(float_stats["partial_max_float"])

    safe = (
        partial_min >= float(bounds["safe_real_min"])
        and partial_max <= float(bounds["safe_real_max"])
    )

    physical_util_neg = (
        abs(partial_min) / abs(float(bounds["physical_real_min"]))
        if partial_min < 0.0 and bounds["physical_real_min"] != 0.0
        else 0.0
    )
    physical_util_pos = (
        partial_max / float(bounds["physical_real_max"])
        if partial_max > 0.0 and bounds["physical_real_max"] != 0.0
        else 0.0
    )
    safe_util_neg = (
        abs(partial_min) / abs(float(bounds["safe_real_min"]))
        if partial_min < 0.0 and bounds["safe_real_min"] != 0.0
        else 0.0
    )
    safe_util_pos = (
        partial_max / float(bounds["safe_real_max"])
        if partial_max > 0.0 and bounds["safe_real_max"] != 0.0
        else 0.0
    )

    return {
        **bounds,
        "safe": bool(safe),
        "s_acc": int(s_acc),
        "utilization_physical": float(max(physical_util_neg, physical_util_pos)),
        "utilization_safe_window": float(max(safe_util_neg, safe_util_pos)),
    }


def _float_input_for_layer(
    tensors: dict[str, np.ndarray],
    layer_index: int,
) -> np.ndarray:
    """Return the ORIGINAL FLOAT input tensor of one linear layer."""
    if int(layer_index) == 0:
        return np.asarray(tensors["input"], dtype=np.float32)
    return np.asarray(tensors[f"layer_{int(layer_index) - 1}_act"], dtype=np.float32)


def optimize_dsp_weight_scales(
    quant_results: dict[str, Any],
    *,
    show_summary: bool = True,
    save_report: bool = True,
    report_filename: str = "dsp_accumulator_report.csv",
) -> dict[str, Any]:
    """Optimize each layer's SW independently using only FLOAT model tensors.

    Numerical contract of this optimizer
    ------------------------------------
    1. SX, SP, activation SX, SB and the initial SW are calibrated from the
       floating-point model exactly as in ``quantize_model``.
    2. For each layer independently, all partial accumulations are recomputed
       from that layer's ORIGINAL float input, float weights and float bias.
    3. The layer-local SW is decreased only if those FLOAT partials do not fit
       inside ``dsp_acc_max_utilization`` of DSP_PW at Sacc = SX + SW.
    4. Changing SW of layer i NEVER changes the float tensor used to optimize
       layer i+1.  There is no integer propagation in the scale search.
    5. Once SW is selected, the stored weights are RE-QUANTIZED DIRECTLY FROM
       ``w_float`` with the selected SW.  Existing Wq values are not shifted.

    This function chooses scales from the float model.  A separate bit-exact
    integer/RTL emulation can be used afterwards as validation, but it must not
    feed back into the scale selection performed here.
    """
    qcfg = quant_results["qcfg"]
    tensors = quant_results["tensors"]
    old_layers = list(quant_results["quantized_layers"])

    if not old_layers:
        raise ValueError("No quantized layers are available for DSP optimization.")

    max_steps = int(qcfg.dsp_weight_search_max_steps)
    if max_steps < 0:
        raise ValueError("dsp_weight_search_max_steps must be >= 0.")
    _validate_dsp_utilization(qcfg.dsp_acc_max_utilization)

    optimized_layers: list[QuantizedLayer] = []
    rows: list[dict[str, Any]] = []
    layer_float_stats: dict[str, dict[str, Any]] = {}

    for layer_idx, old_layer in enumerate(old_layers):
        # IMPORTANT: each layer uses the original FLOAT forward tensors.
        # SW changes in previous layers do not modify x_float here.
        x_float = _float_input_for_layer(tensors, layer_idx)
        float_stats = _float_partial_accumulator_stats(
            x_float,
            old_layer.w_float,
            old_layer.b_float,
        )
        layer_float_stats[old_layer.name] = float_stats.copy()

        initial_sw = int(old_layer.weight_exp)
        selected_sw = None
        selected_eval = None
        attempts: list[dict[str, Any]] = []

        for step in range(max_steps + 1):
            candidate_sw = initial_sw - step
            candidate_eval = _evaluate_float_sw_candidate(
                float_stats,
                sx=int(old_layer.input_exp),
                sw=int(candidate_sw),
                dsp_bits=int(qcfg.dsp_bits),
                max_utilization=float(qcfg.dsp_acc_max_utilization),
            )

            attempts.append(
                {
                    "sw": int(candidate_sw),
                    "s_acc": int(candidate_eval["s_acc"]),
                    "safe": bool(candidate_eval["safe"]),
                    "safe_real_min": float(candidate_eval["safe_real_min"]),
                    "safe_real_max": float(candidate_eval["safe_real_max"]),
                    "utilization_physical": float(candidate_eval["utilization_physical"]),
                }
            )

            if candidate_eval["safe"]:
                selected_sw = int(candidate_sw)
                selected_eval = candidate_eval
                break

        if selected_sw is None or selected_eval is None:
            last = attempts[-1] if attempts else None
            raise OverflowError(
                f"{old_layer.name}: no FLOAT-safe SW found after {max_steps} decrements "
                f"from SW={initial_sw}. Last attempt={last}."
            )

        # CRITICAL: regenerate the ROM weights from the ORIGINAL float weights
        # at the selected exponent.  Never right-shift the previous Wq array.
        selected_w_q = quantize_to_int(
            old_layer.w_float,
            selected_sw,
            num_bits=qcfg.weight_bits,
            signed=True,
        )

        selected_layer = replace(
            old_layer,
            w_q=selected_w_q.astype(old_layer.w_q.dtype, copy=False),
            weight_exp=selected_sw,
        )
        optimized_layers.append(selected_layer)

        rows.append(
            {
                "layer": selected_layer.name,
                "is_output": bool(selected_layer.is_output),
                "calibration_domain": "float",
                "sx": int(selected_layer.input_exp),
                "sb": int(selected_layer.bias_exp),
                "sw_initial": int(initial_sw),
                "sw_selected": int(selected_sw),
                "sw_bits_lost": int(initial_sw - selected_sw),
                "s_acc_selected": int(selected_eval["s_acc"]),
                "dsp_bits": int(qcfg.dsp_bits),
                "dsp_max_utilization_target_%": 100.0 * float(qcfg.dsp_acc_max_utilization),
                "bias_min_float": float(float_stats["bias_min_float"]),
                "bias_max_float": float(float_stats["bias_max_float"]),
                "partial_min_float": float(float_stats["partial_min_float"]),
                "partial_max_float": float(float_stats["partial_max_float"]),
                "max_abs_partial_float": float(float_stats["max_abs_partial_float"]),
                "final_min_float": float(float_stats["final_min_float"]),
                "final_max_float": float(float_stats["final_max_float"]),
                "dsp_physical_real_min": float(selected_eval["physical_real_min"]),
                "dsp_physical_real_max": float(selected_eval["physical_real_max"]),
                "dsp_safe_real_min": float(selected_eval["safe_real_min"]),
                "dsp_safe_real_max": float(selected_eval["safe_real_max"]),
                "dsp_utilization_%": 100.0 * float(selected_eval["utilization_physical"]),
                "safe_window_utilization_%": 100.0 * float(selected_eval["utilization_safe_window"]),
                "worst_value_float": float(float_stats["worst_value_float"]),
                "worst_sample": int(float_stats["worst_sample"]),
                "worst_neuron": int(float_stats["worst_neuron"]),
                "worst_mac_index": int(float_stats["worst_mac_index"]),
                "worst_sign": str(float_stats["worst_sign"]),
                "candidate_attempts": int(len(attempts)),
            }
        )

    report_df = pd.DataFrame(rows)
    summary_df = summarize_quantized_layers(optimized_layers)

    updated = {
        **quant_results,
        "quantized_layers": optimized_layers,
        "summary_df": summary_df,
        "dsp_accumulator_report_df": report_df,
        "dsp_float_layer_stats": layer_float_stats,
        "dsp_weight_scale_optimized": True,
        "dsp_weight_scale_calibration_domain": "float",
    }

    report_path = None
    if save_report:
        report_path = Path(quant_results["paths"].output_dir) / report_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)
        updated["dsp_accumulator_report_csv"] = str(report_path)

    if show_summary:
        print("\n" + "=" * 124)
        print("DSP-AWARE WEIGHT SCALE OPTIMIZATION -- FLOAT CALIBRATION")
        print("=" * 124)
        print(f"DSP_PW                  : {qcfg.dsp_bits} bits")
        print(f"Allowed utilization     : {100.0 * qcfg.dsp_acc_max_utilization:.1f}%")
        print("Calibration source      : original FLOAT tensors, independently per layer")
        print("Scale modified          : SW only, independently for each layer")
        print("Stored weights          : re-quantized from W_float at each SW_selected")
        if report_path is not None:
            print(f"Report CSV              : {report_path}")
        print("=" * 124)
        display(report_df)

    return updated
