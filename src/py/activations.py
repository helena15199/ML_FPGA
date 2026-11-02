"""Activation functions and integer/LUT activation approximations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from config import QuantConfig
from fixed_point import (
    dequantize_from_int,
    assert_int_range,
    dtype_for_bits,
    int_bounds,
    quantize_to_int,
    requantize_int,
    signed_to_twos,
    arithmetic_shift_right,
)
from utils import _to_numpy


def canonical_activation_name(name: str) -> str:
    key = str(name).lower().replace("-", "_").replace(" ", "").replace("__", "_")
    aliases = {
        "relu": "relu",
        "leakyrelu": "leaky_relu",
        "leaky_relu": "leaky_relu",
        "gelu": "gelu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "identity": "identity",
        "linear": "identity",
        "none": "identity",
    }
    return aliases.get(key, key)


def build_pytorch_activation(name: str, *, negative_slope: float = 1 / 64) -> nn.Module:
    name = canonical_activation_name(name)
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def activation_float(x, activation_name: str, *, negative_slope: float = 1 / 64) -> np.ndarray:
    x_np = _to_numpy(x).astype(np.float32)
    name = canonical_activation_name(activation_name)

    if name == "relu":
        return np.maximum(x_np, 0.0)
    if name == "leaky_relu":
        return np.where(x_np >= 0.0, x_np, negative_slope * x_np)
    if name == "gelu":
        #return x_np * 0.5 * (1 + np.tanh(x_np / np.sqrt(2)))
        return 0.5 * x_np * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_np + 0.044715 * x_np**3)))
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x_np))
    if name == "tanh":
        return np.tanh(x_np)
    if name == "identity":
        return x_np

    raise ValueError(f"Unsupported activation: {activation_name}")


def activation_needs_lut(name: str) -> bool:
    return canonical_activation_name(name) not in {"relu", "leaky_relu", "identity"}


class ActivationBase:
    def __init__(self, name, act_bits, input_exp, output_exp, signed_input=True):
        self.name = canonical_activation_name(name)
        self.act_bits = int(act_bits)
        self.input_exp = int(input_exp)
        self.output_exp = int(output_exp)
        self.signed_input = bool(signed_input)

    def forward_int(self, x_q):
        raise NotImplementedError


class IdentityActivation(ActivationBase):
    def forward_int(self, x_q):
        y_q = requantize_int(np.asarray(x_q, dtype=np.int64), self.input_exp, self.output_exp)
        assert_int_range(y_q, self.act_bits, signed=True, label=f"{self.name} output")
        return y_q.astype(dtype_for_bits(self.act_bits))


class ReLUActivation(ActivationBase):
    def forward_int(self, x_q):
        x_q = np.asarray(x_q, dtype=np.int64)
        nonneg = np.maximum(x_q, 0)
        y_q = requantize_int(nonneg, self.input_exp, self.output_exp)
        assert_int_range(y_q, self.act_bits, signed=True, label=f"{self.name} output")
        return y_q.astype(dtype_for_bits(self.act_bits))


class LeakyReLUActivation(ActivationBase):
    def __init__(self, *args, negative_slope=1 / 64, **kwargs):
        super().__init__(*args, **kwargs)
        alpha = float(negative_slope)
        self.k = int(round(-np.log2(alpha)))
        if not np.isclose(alpha, 2 ** (-self.k)):
            raise ValueError("negative_slope must be of the form 2^-k for a pure integer datapath.")

    def forward_int(self, x_q):
        x_q = np.asarray(x_q, dtype=np.int64)
        out = x_q.copy()
        mask = out < 0
        # Match SystemVerilog signed ``>>>`` exactly for negative values.
        out[mask] = arithmetic_shift_right(out[mask], self.k)
        y_q = requantize_int(out, self.input_exp, self.output_exp)
        assert_int_range(y_q, self.act_bits, signed=True, label=f"{self.name} output")
        return y_q.astype(dtype_for_bits(self.act_bits))


def signed_ordered_addr_from_q(x_q, input_bits: int, addr_bits: int) -> np.ndarray:
    x_q = np.asarray(x_q, dtype=np.int64)
    step_shift = input_bits - addr_bits
    qmin, _ = int_bounds(input_bits, signed=True)
    assert_int_range(x_q, input_bits, signed=True, label="LUT input")
    return ((x_q - qmin) >> step_shift).astype(np.int64)


def signed_ordered_frac_from_q(x_q, input_bits: int, addr_bits: int) -> np.ndarray:
    x_q = np.asarray(x_q, dtype=np.int64)
    step_shift = input_bits - addr_bits
    qmin, _ = int_bounds(input_bits, signed=True)
    assert_int_range(x_q, input_bits, signed=True, label="LUT input")
    return ((x_q - qmin) & ((1 << step_shift) - 1)).astype(np.int64)


def generate_lut_knots_from_msb(input_bits: int, addr_bits: int, input_exp: int, signed_input: bool = True) -> pd.DataFrame:
    step_shift = input_bits - addr_bits
    rows = []

    if signed_input:
        qmin, _ = int_bounds(input_bits, signed=True)
        for addr_ord in range(2 ** addr_bits):
            q = qmin + (addr_ord << step_shift)
            raw_word = signed_to_twos(q, input_bits)
            raw_addr = raw_word >> step_shift
            x = q / (2 ** input_exp)
            rows.append({"addr": addr_ord, "raw_addr": raw_addr, "word": raw_word, "x_q": q, "x": x})
    else:
        for addr in range(2 ** addr_bits):
            word = addr << step_shift
            q = word
            x = q / (2 ** input_exp)
            rows.append({"addr": addr, "raw_addr": addr, "word": word, "x_q": q, "x": x})

    return pd.DataFrame(rows)


def build_segment_code_ranges(x_knots_q: np.ndarray) -> list[tuple[int, int]]:
    return [(int(x_knots_q[i]), int(x_knots_q[i + 1])) for i in range(len(x_knots_q) - 1)]


def fit_line_on_codes(q_start: int, q_end: int, sx: int, func_name: str, negative_slope: float = 1 / 64):
    q_codes = np.arange(q_start, q_end + 1, dtype=np.int64)
    x = q_codes / (2 ** sx)
    y = activation_float(x, func_name, negative_slope=negative_slope)
    if len(q_codes) < 2:
        m = 0.0
        b = float(y[0])
    else:
        m, b = np.polyfit(x, y, 1)
    y0 = m * (q_start / (2 ** sx)) + b
    y1 = m * (q_end / (2 ** sx)) + b
    return m, b, float(y0), float(y1)




def quantize_lut_table_words(values, exponent: int, *, bits: int) -> np.ndarray:
    """Encode the complete LUT ROM image in a signed ACT_W word.

    The ROM spans the full signed code space, including addresses that calibration
    proves unreachable.  Sigmoid/tanh asymptotes can round one LSB beyond the
    signed positive endpoint in those unused addresses (e.g. 32768 in Q15).
    Those table-only endpoints are capped to the representable word so the ROM
    image is valid.  Runtime tensors and datapath conversions remain strict and
    are never saturated by this helper.
    """
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (2 ** int(exponent))).astype(np.int64)
    qmin, qmax = int_bounds(bits, signed=True)
    return np.clip(scaled, qmin, qmax).astype(np.int64)

def build_paper_fit_knots(
    func_name: str,
    act_bits: int,
    addr_bits: int,
    input_exp: int,
    output_exp: int,
    *,
    signed_input: bool = True,
    negative_slope: float = 1 / 64,
):
    knots = generate_lut_knots_from_msb(act_bits, addr_bits, input_exp, signed_input=signed_input)
    x_knots_q = knots["x_q"].to_numpy(dtype=np.int64)
    x_knots = knots["x"].to_numpy(dtype=np.float64)
    seg_ranges = build_segment_code_ranges(x_knots_q)

    left_vals = [[] for _ in range(len(x_knots))]
    right_vals = [[] for _ in range(len(x_knots))]
    segments = []

    for segment_index, (q0, q1) in enumerate(seg_ranges):
        m, b, y0, y1 = fit_line_on_codes(q0, q1, input_exp, func_name, negative_slope=negative_slope)
        segments.append(
            {"segment": segment_index, "q_start": q0, "q_end": q1, "m": m, "b": b, "y_start": y0, "y_end": y1}
        )
        right_vals[segment_index].append(y0)
        left_vals[segment_index + 1].append(y1)

    y_knots = np.zeros(len(x_knots), dtype=np.float64)
    for knot_index in range(len(x_knots)):
        vals = []
        vals.extend(left_vals[knot_index])
        vals.extend(right_vals[knot_index])
        if len(vals) == 0:
            y_knots[knot_index] = activation_float(
                np.array([x_knots[knot_index]], dtype=np.float64),
                func_name,
                negative_slope=negative_slope,
            )[0]
        else:
            y_knots[knot_index] = float(np.mean(vals))

    y_q = quantize_lut_table_words(y_knots, output_exp, bits=act_bits)
    y_q_real = dequantize_from_int(y_q, output_exp)
    return knots, pd.DataFrame(segments), y_knots, y_q_real, y_q


def valid_eval_mask_exclude_last_segment(x_q, input_bits: int, addr_bits: int) -> np.ndarray:
    x_q = np.asarray(x_q, dtype=np.int64)
    step_shift = input_bits - addr_bits
    qmin, _ = int_bounds(input_bits, signed=True)

    guard_start_q = qmin + (((1 << addr_bits) - 1) << step_shift)
    return x_q < guard_start_q


def get_last_segment_guard_bounds(input_bits: int, addr_bits: int) -> dict[str, int]:
    """
    Return integer bounds for the forbidden zone (last segment / last LUT address).

    For a LUT addressed by the N most significant bits:
    - the last zone starts at guard_start_q
    - values q >= guard_start_q map to addr == max_addr
    """
    step_shift = input_bits - addr_bits
    qmin, qmax = int_bounds(input_bits, signed=True)

    guard_start_q = qmin + (((1 << addr_bits) - 1) << step_shift)
    max_safe_q = guard_start_q - 1

    return {
        "step_shift": int(step_shift),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "guard_start_q": int(guard_start_q),
        "max_safe_q": int(max_safe_q),
        "reserved_codes": int(1 << step_shift),
    }


def analyze_last_segment_coverage_from_q(
    x_q: np.ndarray,
    *,
    input_bits: int,
    addr_bits: int,
) -> dict[str, Any]:
    """
    Analyze whether quantized inputs x_q hit the forbidden last LUT segment.
    """
    x_q = np.asarray(x_q, dtype=np.int64).reshape(-1)
    bounds = get_last_segment_guard_bounds(input_bits=input_bits, addr_bits=addr_bits)

    mask_guard = x_q >= bounds["guard_start_q"]
    touch_count = int(mask_guard.sum())
    total_count = int(x_q.size)
    touch_ratio = float(touch_count / total_count) if total_count > 0 else 0.0

    if total_count > 0:
        max_pre_q = int(x_q.max())
        min_pre_q = int(x_q.min())
    else:
        max_pre_q = 0
        min_pre_q = 0

    headroom_q = int(bounds["max_safe_q"] - max_pre_q)

    return {
        "touch_last_segment": bool(touch_count > 0),
        "touch_count": touch_count,
        "total_count": total_count,
        "touch_ratio": touch_ratio,
        "min_pre_q": min_pre_q,
        "max_pre_q": max_pre_q,
        "guard_start_q": bounds["guard_start_q"],
        "max_safe_q": bounds["max_safe_q"],
        "headroom_q": headroom_q,
        "reserved_codes": bounds["reserved_codes"],
        "step_shift": bounds["step_shift"],
    }


def default_guard_info_for_analytic(x_q: np.ndarray) -> dict[str, Any]:
    """
    Default values for analytical activations (no LUT).
    Keeps a homogeneous table without KeyError.
    """
    x_q = np.asarray(x_q, dtype=np.int64).reshape(-1)

    if x_q.size > 0:
        min_pre_q = int(x_q.min())
        max_pre_q = int(x_q.max())
    else:
        min_pre_q = 0
        max_pre_q = 0

    return {
        "touch_last_segment": False,
        "touch_count": 0,
        "total_count": int(x_q.size),
        "touch_ratio": 0.0,
        "min_pre_q": min_pre_q,
        "max_pre_q": max_pre_q,
        "guard_start_q": np.nan,
        "max_safe_q": np.nan,
        "headroom_q": np.nan,
        "reserved_codes": np.nan,
        "step_shift": np.nan,
    }


def print_last_segment_guard_debug(
    *,
    layer_name: str,
    activation_name: str,
    addr_bits: int,
    guard_info: dict[str, Any],
):
    print(
        f"[Zone interdite] {layer_name} | act={activation_name} | N={addr_bits} | "
        f"guard_start_q={guard_info['guard_start_q']} | "
        f"max_safe_q={guard_info['max_safe_q']} | "
        f"max_pre_q={guard_info['max_pre_q']} | "
        f"headroom_q={guard_info['headroom_q']} | "
        f"touch_count={guard_info['touch_count']}/{guard_info['total_count']} "
        f"({100.0 * guard_info['touch_ratio']:.4f}%)"
    )


class LUTActivationPaperFit(ActivationBase):
    def __init__(
        self,
        *args,
        addr_bits: int,
        last_segment_strategy: str = "forbid_last",
        negative_slope: float = 1 / 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.addr_bits = int(addr_bits)
        self.step_shift = self.act_bits - self.addr_bits
        self.last_segment_strategy = last_segment_strategy

        (
            self.knots_df,
            self.segments_df,
            self.y_knots_float,
            self.y_knots_quantized_float,
            self.y_knots_q,
        ) = build_paper_fit_knots(
            func_name=self.name,
            act_bits=self.act_bits,
            addr_bits=self.addr_bits,
            input_exp=self.input_exp,
            output_exp=self.output_exp,
            signed_input=self.signed_input,
            negative_slope=negative_slope,
        )

    def forward_int(self, x_q):
        x_q = np.asarray(x_q, dtype=np.int64)

        if self.signed_input:
            addr = signed_ordered_addr_from_q(x_q, self.act_bits, self.addr_bits)
            frac = signed_ordered_frac_from_q(x_q, self.act_bits, self.addr_bits)
        else:
            word = x_q.astype(np.int64)
            addr = word >> self.step_shift
            frac = word & ((1 << self.step_shift) - 1)

        max_addr = (1 << self.addr_bits) - 1
        if np.any((addr < 0) | (addr > max_addr)):
            raise OverflowError("Computed LUT address is outside the configured address width.")

        # Match the sequential RTL exactly: the next address wraps modulo 2**N.
        # The final positive segment is intentionally forbidden by calibration, so
        # correct configurations never rely on this wrap-around interpolation.
        next_addr = (addr + 1) & max_addr

        ya = self.y_knots_q[addr].astype(np.int64)
        yb = self.y_knots_q[next_addr].astype(np.int64)
        interp = ya + arithmetic_shift_right((yb - ya) * frac, self.step_shift)
        y_out = interp

        assert_int_range(y_out, self.act_bits, signed=True, label=f"{self.name} LUT output")
        return y_out.astype(dtype_for_bits(self.act_bits))


def build_activation_module(
    name: str,
    act_bits: int,
    input_exp: int,
    output_exp: int,
    qcfg: QuantConfig,
    *,
    addr_bits: int | None = None,
):
    name = canonical_activation_name(name)
    if name == "relu":
        return ReLUActivation(name, act_bits, input_exp, output_exp, signed_input=True)
    if name == "leaky_relu":
        return LeakyReLUActivation(
            name,
            act_bits,
            input_exp,
            output_exp,
            signed_input=True,
            negative_slope=qcfg.activation_negative_slope,
        )
    if name == "identity":
        return IdentityActivation(name, act_bits, input_exp, output_exp, signed_input=True)
    if addr_bits is None:
        raise ValueError("addr_bits is required for a LUT activation.")
    return LUTActivationPaperFit(
        name,
        act_bits,
        input_exp,
        output_exp,
        signed_input=True,
        addr_bits=addr_bits,
        last_segment_strategy=qcfg.last_segment_strategy,
        negative_slope=qcfg.activation_negative_slope,
    )

