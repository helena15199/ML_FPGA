"""Bit-accurate NumPy reference model for the FPGA MLP datapath."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from activations import (
    canonical_activation_name,
    signed_ordered_addr_from_q,
    signed_ordered_frac_from_q,
)
from fixed_point import (
    arithmetic_shift_right,
    dequantize_from_int,
    int_bounds,
    quantize_to_int,
    shift_scale_int,
    wrap_signed,
)
from fpga_targets import TARGET_PROFILES


@dataclass(frozen=True)
class FPGASimulationResult:
    """Integer predictions and diagnostics produced by the RTL reference model."""

    input_q: np.ndarray
    output_q: np.ndarray
    output_float: np.ndarray
    input_exp: int
    output_exp: int
    target: str
    architecture: str
    dsp_bits: int
    layer_outputs: tuple[np.ndarray, ...]
    layer_stats: tuple[dict[str, Any], ...]

    @property
    def overflow_count(self) -> int:
        return sum(int(row["overflow_count"]) for row in self.layer_stats)


def _wrap_and_count(values: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.int64)
    qmin, qmax = int_bounds(bits, signed=True)
    count = int(np.count_nonzero((values < qmin) | (values > qmax)))
    return wrap_signed(values, bits).astype(np.int64, copy=False), count


def _shift_and_wrap(values: np.ndarray, shift: int, bits: int) -> tuple[np.ndarray, int]:
    shifted = shift_scale_int(np.asarray(values, dtype=np.int64), int(shift))
    return _wrap_and_count(shifted, bits)


def _simulate_mac(
    x_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    *,
    input_exp: int,
    weight_exp: int,
    bias_exp: int,
    dsp_bits: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reproduce the sequential MAC order and two's-complement DSP wrapping."""
    x_q = np.asarray(x_q, dtype=np.int64)
    weight_q = np.asarray(weight_q, dtype=np.int64)
    bias_q = np.asarray(bias_q, dtype=np.int64).reshape(-1)
    if x_q.ndim != 2 or weight_q.ndim != 2:
        raise ValueError("x_q and weight_q must be two-dimensional")
    if x_q.shape[1] != weight_q.shape[1]:
        raise ValueError("Input and weight dimensions are inconsistent")
    if bias_q.size != weight_q.shape[0]:
        raise ValueError("Bias and weight dimensions are inconsistent")

    accumulator_exp = int(input_exp) + int(weight_exp)
    aligned_bias, bias_wraps = _shift_and_wrap(
        bias_q,
        accumulator_exp - int(bias_exp),
        dsp_bits,
    )
    accumulator = np.broadcast_to(
        aligned_bias.reshape(1, -1),
        (x_q.shape[0], weight_q.shape[0]),
    ).copy()

    product_wraps = 0
    accumulator_wraps = 0
    for input_index in range(weight_q.shape[1]):
        raw_product = (
            x_q[:, input_index : input_index + 1]
            * weight_q[:, input_index].reshape(1, -1)
        )
        product, count = _wrap_and_count(raw_product, dsp_bits)
        product_wraps += count
        accumulator, count = _wrap_and_count(accumulator + product, dsp_bits)
        accumulator_wraps += count

    return accumulator, {
        "bias_wrap_count": bias_wraps,
        "product_wrap_count": product_wraps,
        "accumulator_wrap_count": accumulator_wraps,
    }


def _simulate_lut_activation(
    preactivation: np.ndarray,
    module: Any,
    *,
    act_bits: int,
    dsp_bits: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reproduce raw-address LUT lookup and DSP linear interpolation."""
    narrowed, input_wraps = _wrap_and_count(preactivation, act_bits)
    addr_bits = int(module.addr_bits)
    frac_bits = int(act_bits) - addr_bits
    address = signed_ordered_addr_from_q(narrowed, act_bits, addr_bits)
    fraction = signed_ordered_frac_from_q(narrowed, act_bits, addr_bits)
    address_mask = (1 << addr_bits) - 1
    next_address = (address + 1) & address_mask

    ya = np.asarray(module.y_knots_q, dtype=np.int64)[address]
    yb = np.asarray(module.y_knots_q, dtype=np.int64)[next_address]
    product, product_wraps = _wrap_and_count((yb - ya) * fraction, dsp_bits)
    c_term, c_wraps = _shift_and_wrap(ya, frac_bits, dsp_bits)
    interpolation, accumulator_wraps = _wrap_and_count(product + c_term, dsp_bits)
    if frac_bits:
        interpolation = arithmetic_shift_right(interpolation, frac_bits)
    output, output_wraps = _wrap_and_count(interpolation, act_bits)

    return output, {
        "preactivation_narrowing_count": input_wraps,
        "lut_product_wrap_count": product_wraps,
        "lut_c_wrap_count": c_wraps,
        "lut_accumulator_wrap_count": accumulator_wraps,
        "output_narrowing_count": output_wraps,
    }


def _simulate_direct_activation(
    preactivation: np.ndarray,
    activation_name: str,
    *,
    source_exp: int,
    output_exp: int,
    act_bits: int,
    dsp_bits: int,
    negative_slope: float,
) -> tuple[np.ndarray, dict[str, int]]:
    name = canonical_activation_name(activation_name)
    if name == "identity":
        activated = preactivation
    elif name == "relu":
        activated = np.maximum(preactivation, 0)
    elif name == "leaky_relu":
        leak_shift = int(round(-np.log2(float(negative_slope))))
        if not np.isclose(float(negative_slope), 2.0 ** (-leak_shift)):
            raise ValueError("The RTL requires activation_negative_slope to be 2^-k")
        activated = np.where(
            preactivation < 0,
            arithmetic_shift_right(preactivation, leak_shift),
            preactivation,
        )
    else:
        raise ValueError(f"Activation {name!r} requires a LUT module")

    shifted, dsp_shift_wraps = _shift_and_wrap(
        activated,
        int(output_exp) - int(source_exp),
        dsp_bits,
    )
    output, output_wraps = _wrap_and_count(shifted, act_bits)
    return output, {
        "activation_shift_wrap_count": dsp_shift_wraps,
        "output_narrowing_count": output_wraps,
    }


def simulate_fpga(
    results: dict[str, Any],
    *,
    target: str,
    architecture: str,
    x_float: np.ndarray | None = None,
    max_samples: int | None = None,
    strict: bool = True,
) -> FPGASimulationResult:
    """Run the quantized MLP exactly like the common RTL arithmetic datapath.

    Sequential and pipeline modes share the same arithmetic contract; they only
    differ in scheduling and resource duplication. The DSP width comes directly
    from the explicit ``QuantConfig.dsp_bits`` value.
    """
    qcfg = results["qcfg"]
    layers = list(results["quantized_layers"])
    activation_modules = dict(results.get("activation_modules", {}))
    if not layers:
        raise ValueError("No quantized layers are available for simulation")
    if architecture not in {"sequential", "pipeline"}:
        raise ValueError("architecture must be 'sequential' or 'pipeline'")
    if target not in TARGET_PROFILES:
        raise ValueError("target must be 'ultra96', 'lattice' or 'nexys'")
    if int(qcfg.dsp_bits) <= 0:
        raise ValueError("dsp_bits must be positive")

    if x_float is None:
        x_float = results["dataset"].x
    x_float = np.asarray(x_float, dtype=np.float32)
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive or None")
        x_float = x_float[: int(max_samples)]

    input_exp = int(layers[0].input_exp)
    current = quantize_to_int(
        x_float,
        input_exp,
        num_bits=int(qcfg.act_bits),
        signed=True,
    ).astype(np.int64, copy=False)
    input_q = current.copy()
    layer_outputs: list[np.ndarray] = []
    layer_stats: list[dict[str, Any]] = []

    for layer_index, layer in enumerate(layers):
        accumulator_exp = int(layer.input_exp) + int(layer.weight_exp)
        accumulator, mac_stats = _simulate_mac(
            current,
            layer.w_q,
            layer.b_q,
            input_exp=int(layer.input_exp),
            weight_exp=int(layer.weight_exp),
            bias_exp=int(layer.bias_exp),
            dsp_bits=int(qcfg.dsp_bits),
        )

        stage_stats: dict[str, Any] = {
            "layer": layer.name,
            "layer_index": layer_index,
            "input_exp": int(layer.input_exp),
            "weight_exp": int(layer.weight_exp),
            "bias_exp": int(layer.bias_exp),
            "accumulator_exp": accumulator_exp,
            "accumulator_min": int(accumulator.min()) if accumulator.size else 0,
            "accumulator_max": int(accumulator.max()) if accumulator.size else 0,
            **mac_stats,
        }

        if layer.is_output:
            output_exp = int(layer.output_exp)
            current, output_stats = _simulate_direct_activation(
                accumulator,
                "identity",
                source_exp=accumulator_exp,
                output_exp=output_exp,
                act_bits=int(qcfg.act_bits),
                dsp_bits=int(qcfg.dsp_bits),
                negative_slope=float(qcfg.activation_negative_slope),
            )
        else:
            preactivation_exp = int(layer.preact_exp)
            preactivation, pre_wraps = _shift_and_wrap(
                accumulator,
                preactivation_exp - accumulator_exp,
                int(qcfg.dsp_bits),
            )
            stage_stats["preactivation_exp"] = preactivation_exp
            stage_stats["preactivation_shift_wrap_count"] = pre_wraps

            activation_module = activation_modules.get(layer_index)
            if activation_module is not None and hasattr(activation_module, "y_knots_q"):
                if int(activation_module.input_exp) != preactivation_exp:
                    raise ValueError(
                        f"{layer.name}: LUT input_exp does not match preact_exp"
                    )
                if int(activation_module.output_exp) != int(layer.act_exp):
                    raise ValueError(
                        f"{layer.name}: LUT output_exp does not match act_exp"
                    )
                current, output_stats = _simulate_lut_activation(
                    preactivation,
                    activation_module,
                    act_bits=int(qcfg.act_bits),
                    dsp_bits=int(qcfg.dsp_bits),
                )
            else:
                current, output_stats = _simulate_direct_activation(
                    preactivation,
                    str(layer.activation_name),
                    source_exp=preactivation_exp,
                    output_exp=int(layer.act_exp),
                    act_bits=int(qcfg.act_bits),
                    dsp_bits=int(qcfg.dsp_bits),
                    negative_slope=float(qcfg.activation_negative_slope),
                )
            output_exp = int(layer.act_exp)

        stage_stats.update(output_stats)
        stage_stats["output_exp"] = output_exp
        stage_stats["output_min"] = int(current.min()) if current.size else 0
        stage_stats["output_max"] = int(current.max()) if current.size else 0
        overflow_keys = [key for key in stage_stats if key.endswith("_count")]
        stage_stats["overflow_count"] = sum(int(stage_stats[key]) for key in overflow_keys)
        layer_stats.append(stage_stats)
        layer_outputs.append(current.copy())

    simulation = FPGASimulationResult(
        input_q=input_q,
        output_q=current,
        output_float=dequantize_from_int(current, output_exp).astype(np.float32),
        input_exp=input_exp,
        output_exp=output_exp,
        target=str(target),
        architecture=str(architecture),
        dsp_bits=int(qcfg.dsp_bits),
        layer_outputs=tuple(layer_outputs),
        layer_stats=tuple(layer_stats),
    )
    expected_output_float = (
        simulation.output_q.astype(np.float64) * 2.0 ** (-simulation.output_exp)
    )
    if not np.array_equal(
        simulation.output_float,
        expected_output_float.astype(np.float32),
    ):
        raise AssertionError(
            "Integer simulation output and dequantized output do not share the same scale"
        )
    if strict and simulation.overflow_count:
        failing = [row["layer"] for row in simulation.layer_stats if row["overflow_count"]]
        raise OverflowError(
            f"The FPGA simulation detected {simulation.overflow_count} wrapped values "
            f"in layers {failing}. Adjust the quantization configuration."
        )
    return simulation


def angular_error_numpy(y_true_radians: np.ndarray, y_pred_radians: np.ndarray) -> np.ndarray:
    """Stable angular distance in degrees for NumPy reports."""
    y_true = np.asarray(y_true_radians, dtype=np.float64)
    y_pred = np.asarray(y_pred_radians, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError("Angular arrays must have the same shape (n_samples, 2)")

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

    true_direction = direction(y_true)
    pred_direction = direction(y_pred)
    cross_norm = np.linalg.norm(np.cross(true_direction, pred_direction), axis=1)
    dot_product = np.sum(true_direction * pred_direction, axis=1)
    return np.degrees(np.arctan2(cross_norm, dot_product))
