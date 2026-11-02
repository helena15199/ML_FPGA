"""Fixed-point numerical helpers aligned with SystemVerilog signed arithmetic."""

from __future__ import annotations

import math
import numpy as np
import torch


def int_bounds(bits: int, signed: bool = True) -> tuple[int, int]:
    if signed:
        return -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    return 0, (2**bits) - 1


def dtype_for_bits(bits: int):
    if bits <= 8:
        return np.int8
    if bits <= 16:
        return np.int16
    if bits <= 32:
        return np.int32
    return np.int64


def assert_int_range(values, bits: int, *, signed: bool = True, label: str = "value") -> np.ndarray:
    """Validate a value before narrowing it to a hardware word.

    V2 deliberately does not use saturation as a numerical correction.  A range
    violation means that the scale/dimensioning is invalid and must be fixed.
    """
    arr = np.asarray(values, dtype=np.int64)
    qmin, qmax = int_bounds(bits, signed=signed)
    if arr.size:
        amin = int(arr.min())
        amax = int(arr.max())
        if amin < qmin or amax > qmax:
            raise OverflowError(
                f"{label} does not fit in {'signed' if signed else 'unsigned'} {bits} bits: "
                f"range=[{amin}, {amax}], allowed=[{qmin}, {qmax}]"
            )
    return arr


def quantize_to_int(arr, exponent: int, *, num_bits: int, signed: bool = True) -> np.ndarray:
    """Round to the requested fixed-point scale, then validate without clipping."""
    arr = np.asarray(arr, dtype=np.float64)
    scaled = np.rint(arr * (2 ** int(exponent)))
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Cannot quantize NaN or infinite values.")
    scaled_i = scaled.astype(np.int64)
    assert_int_range(scaled_i, num_bits, signed=signed, label="quantized value")
    return scaled_i.astype(dtype_for_bits(num_bits))


def dequantize_from_int(arr_q, exponent: int) -> np.ndarray:
    return np.asarray(arr_q, dtype=np.float64) / (2 ** int(exponent))


def arithmetic_shift_right(x, shift: int) -> np.ndarray:
    """Signed arithmetic right shift: Python equivalent of SystemVerilog ``>>>``."""
    x = np.asarray(x, dtype=np.int64)
    shift = int(shift)
    if shift < 0:
        return x << (-shift)
    if shift == 0:
        return x
    return x >> shift


def shift_scale_int(x, shift: int) -> np.ndarray:
    """Match RTL ``shift_scale_p``: positive shift => ``<<<``, negative => ``>>>``."""
    x = np.asarray(x, dtype=np.int64)
    shift = int(shift)
    if shift == 0:
        return x
    if shift > 0:
        return x << shift
    return x >> (-shift)


def requantize_int(x_int, from_exp: int, to_exp: int) -> np.ndarray:
    """Change binary scale using the same arithmetic shift rule as the RTL."""
    # Real value is x_int / 2**from_exp.  Integer at to_exp therefore shifts by
    # (to_exp - from_exp), exactly like the RTL helper.
    return shift_scale_int(np.asarray(x_int, dtype=np.int64), int(to_exp) - int(from_exp))


def wrap_signed(values, bits: int) -> np.ndarray:
    """Model the natural two's-complement wrap of a signed RTL word.

    This is intended for a future bit-exact DSP reference model.  Configuration
    validation should ensure that normal inference never relies on this wrap.
    """
    values = np.asarray(values, dtype=np.int64)
    mask = (1 << int(bits)) - 1
    raw = values & mask
    sign = 1 << (int(bits) - 1)
    return np.where(raw & sign, raw - (1 << int(bits)), raw).astype(np.int64)


def signed_to_twos(v, bits: int) -> int:
    return int(v) & ((1 << bits) - 1)


def int_to_hex_twos_complement(v, bits: int) -> str:
    width = (bits + 3) // 4
    return f"{signed_to_twos(int(v), bits):0{width}X}"


def calibration_view(values, mode: str = "max", percentile: float = 99.9) -> np.ndarray:
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    flat = np.abs(arr).reshape(-1)

    if flat.size == 0:
        return arr

    if mode == "percentile":
        limit = np.percentile(flat, percentile)
        return np.clip(arr, -limit, limit)

    return arr


def choose_shift_exponent(
    values,
    *,
    max_int: int,
    margin: float = 0.0,
    mode: str = "max",
    percentile: float = 99.9,
) -> int:
    arr = calibration_view(values, mode=mode, percentile=percentile)
    max_abs = float(np.max(np.abs(arr))) if np.size(arr) else 0.0
    max_abs *= 1.0 + margin

    if max_abs == 0.0 or not np.isfinite(max_abs):
        return 0

    return int(math.floor(math.log2(max_int / max_abs)))

