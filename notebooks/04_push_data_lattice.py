#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a quantized SOC_MLP dataset on a Lattice FPGA over UART.

The script sends one quantized input frame at a time, reads the FPGA response,
aligns the output frames according to the selected architecture, and writes a
CSV report comparing the FPGA output against the dataset targets and optional
software predictions.

UART frame format:
    PC -> FPGA : INPUT_SIZE signed int16 words, little-endian, no SOF/EOF
    FPGA -> PC : OUTPUT_SIZE signed int16 words, little-endian, no SOF/EOF

Supported FPGA modes:
    sequential:
        One transmitted input frame produces one received output frame.
        pipeline_delay = 0

    pipeline:
        Output frames are delayed by NUM_LINEAR - 1 frames.
        NUM_LINEAR is read from scales.vh when available.

Main output:
    A CSV table with sample alignment, input/target/prediction values,
    FPGA results, quantized errors, hexadecimal TX/RX frames, and run metadata.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import serial
    import serial.tools.list_ports
except ImportError as exc:
    raise SystemExit("Missing pyserial module. Install it with: pip install pyserial") from exc


DEFAULT_INPUT_WORDS = 6
DEFAULT_OUTPUT_WORDS = 2
DEFAULT_NUM_LINEAR = 1

# ============================================================
# USER CONFIGURATION
# ============================================================

# Set PROJECT_ROOT_OVERRIDE to an absolute Path if automatic
# project-root detection is not suitable on your machine.
PROJECT_ROOT_OVERRIDE: Optional[Path] = None

# Dataset selection.
# To switch activation folders, normally only change DATASET_PROFILE.
DATASET_BASE_DIR = Path("data/quantized_data")
DATASET_PROFILE = "leaky_relu/lattice/sequential"
DATASET_FILENAME = "dataset_walter_quantized.npz"

# FPGA / generated RTL selection.
FPGA_TARGET = "lattice"
DEFAULT_FPGA_MODE = "sequential"
RTL_GENERATED_DIR = Path("src/rtl/generated")
SCALES_FILENAME = "scales.vh"

# Results.
RESULT_DIR = Path("Result")


# ============================================================
# Project paths
# ============================================================

def script_path() -> Path:
    """Return the absolute path of this script."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return Path.cwd().resolve() / "04_push_data_lattice.py"


def candidate_roots() -> list[Path]:
    """Possible project roots, used only when PROJECT_ROOT_OVERRIDE is None."""
    sp = script_path()
    cwd = Path.cwd().resolve()

    roots = [
        cwd,
        cwd.parent,
        sp.parent,
        sp.parent.parent,
        sp.parent.parent.parent,
    ]

    unique: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        root = root.resolve()
        if root not in seen:
            unique.append(root)
            seen.add(root)

    return unique


def resolve_project_root(explicit: Optional[Path] = None) -> Path:
    """Resolve the repository/project root in a deterministic order.

    Priority:
      1. --project-root
      2. PROJECT_ROOT_OVERRIDE
      3. automatic detection of a directory containing both src/ and data/
    """
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"--project-root is not a directory: {root}")
        return root

    if PROJECT_ROOT_OVERRIDE is not None:
        root = Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(
                f"PROJECT_ROOT_OVERRIDE is not a directory: {root}"
            )
        return root

    for root in candidate_roots():
        if (root / "src").is_dir() and (root / "data").is_dir():
            return root

    searched = "\n".join(f"  - {p}" for p in candidate_roots())
    raise FileNotFoundError(
        "Unable to determine the project root automatically.\n"
        "Expected a directory containing both src/ and data/.\n"
        f"Searched:\n{searched}\n\n"
        "Either set PROJECT_ROOT_OVERRIDE at the top of the script "
        "or use --project-root."
    )


def configured_dataset_path(project_root: Path) -> Path:
    """Dataset selected by the USER CONFIGURATION block."""
    return (
        project_root
        / DATASET_BASE_DIR
        / DATASET_PROFILE
        / DATASET_FILENAME
    ).resolve()


def resolve_dataset_path(
    explicit_dataset: Optional[Path],
    dataset_dir: Optional[Path],
    project_root: Path,
) -> Path:
    """Resolve the dataset path without ambiguous recursive searching.

    Priority:
      1. --dataset FILE
      2. --dataset-dir DIR
      3. USER CONFIGURATION:
           DATASET_BASE_DIR / DATASET_PROFILE / DATASET_FILENAME

    With --dataset-dir, both of these are accepted:
      DIR/dataset_walter_quantized.npz
      DIR/sigmoid/dataset_walter_quantized.npz
    """
    if explicit_dataset is not None:
        dataset = explicit_dataset.expanduser().resolve()

        if not dataset.is_file():
            raise FileNotFoundError(
                f"--dataset does not point to an existing file:\n  {dataset}"
            )

        return dataset

    if dataset_dir is not None:
        directory = dataset_dir.expanduser().resolve()

        if not directory.is_dir():
            raise NotADirectoryError(
                f"--dataset-dir does not point to a directory:\n  {directory}"
            )

        candidates = [
            directory / DATASET_FILENAME,
            directory / DATASET_PROFILE / DATASET_FILENAME,
        ]

        for dataset in candidates:
            if dataset.is_file():
                return dataset.resolve()

        expected = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            "--dataset-dir was provided, but the configured dataset "
            "was not found.\n"
            f"Expected one of:\n{expected}\n"
            "Edit DATASET_PROFILE / DATASET_FILENAME at the top of the script "
            "or pass --dataset explicitly."
        )

    dataset = configured_dataset_path(project_root)

    if not dataset.is_file():
        raise FileNotFoundError(
            "Configured dataset not found.\n"
            f"Expected:\n  {dataset}\n\n"
            "Current configuration:\n"
            f"  DATASET_BASE_DIR = {DATASET_BASE_DIR}\n"
            f"  DATASET_PROFILE  = {DATASET_PROFILE!r}\n"
            f"  DATASET_FILENAME = {DATASET_FILENAME!r}\n\n"
            "Change these values at the top of the script, "
            "or use --dataset."
        )

    return dataset


def configured_scales_path(project_root: Path, fpga_mode: str) -> Path:
    """Return the scales.vh path implied by target + FPGA mode."""
    generated_subdir = f"src_{FPGA_TARGET}_{fpga_mode}"
    return (
        project_root
        / RTL_GENERATED_DIR
        / generated_subdir
        / SCALES_FILENAME
    ).resolve()


def resolve_scales_vh(
    explicit: Optional[Path],
    project_root: Path,
    fpga_mode: str,
) -> Optional[Path]:
    """Resolve scales.vh.

    --scales has priority. Otherwise use the deterministic generated-RTL path.
    Missing scales.vh is not fatal because the script can infer dimensions
    from the dataset, but the exact expected path is printed.
    """
    if explicit is not None:
        scales = explicit.expanduser().resolve()

        if not scales.is_file():
            raise FileNotFoundError(
                f"--scales does not point to an existing file:\n  {scales}"
            )

        return scales

    scales = configured_scales_path(project_root, fpga_mode)

    if scales.is_file():
        return scales

    print(
        "WARNING: configured scales.vh was not found:\n"
        f"  {scales}\n"
        "Dataset dimensions will be used where possible."
    )
    return None


def resolve_out_dir(explicit: Optional[Path], project_root: Path) -> Path:
    """Resolve output directory."""
    if explicit is not None:
        return explicit.expanduser().resolve()

    return (project_root / RESULT_DIR).resolve()


# ============================================================
# SystemVerilog define helpers
# ============================================================

def parse_sv_define_int(path: Optional[Path], name: str) -> Optional[int]:
    if path is None or not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(rf"`define\s+{re.escape(name)}\s+(-?\d+)", text)
    return int(m.group(1)) if m else None


def read_num_linear(scales_path: Optional[Path], default: int = DEFAULT_NUM_LINEAR) -> int:
    value = parse_sv_define_int(scales_path, "NUM_LINEAR")
    return int(default) if value is None else int(value)


def infer_words_from_scales(scales_path: Optional[Path]) -> Tuple[int, int]:
    n0 = parse_sv_define_int(scales_path, "N0")
    num_linear = read_num_linear(scales_path, DEFAULT_NUM_LINEAR)

    nout = None
    if num_linear is not None:
        nout = parse_sv_define_int(scales_path, f"N{num_linear}")

    # Fallback for fixed six-layer exports that define N6 directly.
    if nout is None:
        nout = parse_sv_define_int(scales_path, "N6")

    return (
        DEFAULT_INPUT_WORDS if n0 is None else int(n0),
        DEFAULT_OUTPUT_WORDS if nout is None else int(nout),
    )


# ============================================================
# Dataset helpers
# ============================================================

def pick_npz_array(ds, preferred_names, ndim=None, required=True):
    for name in preferred_names:
        if name in ds.files:
            arr = ds[name]
            if ndim is None or arr.ndim == ndim:
                return name, arr
    if required:
        raise KeyError(
            f"Missing NPZ key. Tried: {preferred_names}. "
            f"Available keys: {list(ds.files)}"
        )
    return None, None


def get_npz_scalar(ds, names, default=None):
    for name in names:
        if name in ds.files:
            return int(np.asarray(ds[name]).reshape(-1)[0])
    return default


def q_to_float(x_int: np.ndarray, scale_exp: int) -> np.ndarray:
    return np.asarray(x_int, dtype=np.float64) / float(1 << int(scale_exp))


def load_dataset(dataset_npz: Path) -> dict:
    ds = np.load(dataset_npz, allow_pickle=True)
    print("Available NPZ keys:", list(ds.files))

    x_float_key, x_float = pick_npz_array(
        ds, ["ph_val", "x_float", "X_float", "x"], ndim=2, required=False
    )
    x_int_key, x = pick_npz_array(
        ds, ["ph_val_int", "X_int", "x_int", "inputs_int", "input_int"], ndim=2, required=False
    )

    y_float_key, y_float = pick_npz_array(
        ds, ["pos", "y_float", "Y_float", "target_float"], ndim=2, required=False
    )
    y_int_key, y_dataset = pick_npz_array(
        ds, ["pos_int", "Y_int", "y_int", "targets_int", "output_int"], ndim=2, required=False
    )

    pred_float_key, y_pred_float_ref = pick_npz_array(
        ds, ["pred_float", "float", "prediction_float"], ndim=2, required=False
    )
    pred_int_key, y_pred_int_ref = pick_npz_array(
        ds, ["pred_int", "float_int", "prediction_int"], ndim=2, required=False
    )

    # Format fallback: some NPZ files store integer data directly under ph_val/pos.
    if x is None:
        x_int_key, x = pick_npz_array(ds, ["ph_val", "X", "x", "inputs", "input", "features"], ndim=2)

    if y_dataset is None:
        y_int_key, y_dataset = pick_npz_array(ds, ["pos", "Y", "y", "outputs", "output", "labels", "targets"], ndim=2)

    x = np.asarray(x, dtype=np.int16)
    y_dataset = np.asarray(y_dataset, dtype=np.int16)

    if y_pred_int_ref is not None:
        y_pred_int_ref = np.asarray(y_pred_int_ref, dtype=np.int16)

    input_scale_exp = get_npz_scalar(ds, ["input_scale_exp", "input_exp", "x_scale_exp"])
    target_scale_exp = get_npz_scalar(ds, ["target_scale_exp", "target_exp", "output_scale_exp", "y_scale_exp"])
    pred_scale_exp = get_npz_scalar(
        ds,
        ["pred_scale_exp", "prediction_scale_exp", "float_scale_exp"],
        default=target_scale_exp,
    )

    n_samples, input_size = x.shape
    n_y, output_size = y_dataset.shape
    if n_y != n_samples:
        raise ValueError("Inputs and targets do not have the same number of samples.")

    print("X key              =", x_int_key, x.shape, x.dtype)
    print("Y target key       =", y_int_key, y_dataset.shape, y_dataset.dtype)
    print("X float key        =", x_float_key)
    print("Y float key        =", y_float_key)
    print("Pred float key     =", pred_float_key)
    print("Pred int key       =", pred_int_key)
    print("input_scale_exp    =", input_scale_exp)
    print("target_scale_exp   =", target_scale_exp)
    print("pred_scale_exp     =", pred_scale_exp)
    print("N_SAMPLES          =", n_samples)
    print("INPUT_SIZE         =", input_size)
    print("OUTPUT_SIZE        =", output_size)

    return {
        "X": x,
        "Y_dataset": y_dataset,
        "X_float": x_float,
        "Y_float": y_float,
        "Y_pred_float_ref": y_pred_float_ref,
        "Y_pred_int_ref": y_pred_int_ref,
        "input_scale_exp": input_scale_exp,
        "target_scale_exp": target_scale_exp,
        "pred_scale_exp": pred_scale_exp,
    }


# ============================================================
# UART helpers
# ============================================================

def ensure_int16_values(values: Iterable[int], expected_len: Optional[int] = None) -> list[int]:
    vals = [int(v) for v in values]
    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} int16 values, got {len(vals)}: {vals}")
    bad = [v for v in vals if v < -32768 or v > 32767]
    if bad:
        raise ValueError(f"Values outside int16 range [-32768, 32767]: {bad}")
    return vals


def pack_int16_words(values: Iterable[int], expected_len: int) -> bytes:
    vals = ensure_int16_values(values, expected_len=expected_len)
    return struct.pack("<" + f"{expected_len}h", *vals)


def unpack_int16_words(payload: bytes, expected_len: int) -> list[int]:
    expected_bytes = 2 * int(expected_len)
    if len(payload) != expected_bytes:
        raise ValueError(f"Invalid payload: got {len(payload)} bytes instead of {expected_bytes}")
    return list(struct.unpack("<" + f"{expected_len}h", payload))


def int16_words_hex(values: Iterable[int]) -> str:
    """Return int16 words as 4-hex-digit unsigned words, e.g. -3243 -> f355."""
    return " ".join(f"{(int(v) & 0xFFFF):04x}" for v in values)


def list_serial_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial port detected.")
        return
    print("Detected serial ports:")
    for p in ports:
        print(f"  {p.device:16s} {p.description}")


def drain_uart_input(ser: serial.Serial, quiet_time_s: float = 0.2) -> bytes:
    junk = bytearray()
    deadline = time.time() + float(quiet_time_s)
    while time.time() < deadline:
        chunk = ser.read(1024)
        if chunk:
            junk.extend(chunk)
            deadline = time.time() + float(quiet_time_s)
    return bytes(junk)


def open_uart(port: str, baud: int, timeout_s: float, startup_delay_s: float) -> serial.Serial:
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
            write_timeout=timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except serial.SerialException as exc:
        print(f"Cannot open {port}: {exc}", file=sys.stderr)
        if "Permission denied" in str(exc):
            print("\nPossible Linux fix: sudo usermod -a -G dialout $USER then log out/in.", file=sys.stderr)
        print("\nAvailable ports:", file=sys.stderr)
        list_serial_ports()
        raise

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    if startup_delay_s > 0:
        time.sleep(float(startup_delay_s))

    junk = drain_uart_input(ser, quiet_time_s=0.1)
    if junk:
        print("Ignored bytes after open:", junk.hex(" "))

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("UART port opened :", ser.port)
    print("Baudrate         :", ser.baudrate)
    print("UART format      : 8N1")
    return ser


def read_exact(ser: serial.Serial, n_bytes: int, timeout_s: float) -> bytes:
    deadline = time.time() + float(timeout_s)
    buf = bytearray()

    while len(buf) < n_bytes and time.time() < deadline:
        chunk = ser.read(n_bytes - len(buf))
        if chunk:
            buf.extend(chunk)

    if len(buf) != n_bytes:
        raise TimeoutError(
            f"RX timeout: got {len(buf)} bytes, expected {n_bytes}. "
            f"Received bytes: {bytes(buf).hex(' ')}"
        )

    return bytes(buf)


def transaction(
    ser: serial.Serial,
    values: Iterable[int],
    input_words: int,
    output_words: int,
    timeout_s: float,
    reset_input_buffer: bool = True,
) -> tuple[list[int], bytes, bytes]:
    tx_values = ensure_int16_values(values, expected_len=input_words)
    tx_bytes = pack_int16_words(tx_values, input_words)
    rx_nbytes = 2 * int(output_words)

    # This script keeps the known-good transaction style:
    # one TX frame, then one RX frame.
    # Since the UART protocol has no SOF/EOF, clearing old RX bytes before
    # sending a new frame avoids mixing a stale response with the next one.
    # Do not clear the input buffer between write() and read_exact().
    if reset_input_buffer:
        n_waiting = ser.in_waiting
        # if n_waiting:
        #     print(f"WARNING: RX buffer had {n_waiting} byte(s) before TX; clearing them.")
        ser.reset_input_buffer()

    ser.write(tx_bytes)
    ser.flush()

    rx_bytes = read_exact(ser, rx_nbytes, timeout_s)
    rx_values = unpack_int16_words(rx_bytes, output_words)

    return rx_values, tx_bytes, rx_bytes


def warmup_transactions(
    ser: serial.Serial,
    n: int,
    input_words: int,
    output_words: int,
    timeout_s: float,
) -> None:
    if n <= 0:
        return

    dummy = [0] * input_words
    print(f"UART warmup: {n} ignored dummy transaction(s)")

    for k in range(n):
        try:
            rx_values, _, rx_bytes = transaction(
                ser,
                dummy,
                input_words=input_words,
                output_words=output_words,
                timeout_s=timeout_s,
                reset_input_buffer=True,
            )
            print(f"  warmup {k + 1}/{n}: ignored RX bytes = {rx_bytes.hex(' ')} | RX int16 = {rx_values}")
        except TimeoutError as exc:
            print(f"  warmup {k + 1}/{n}: ignored timeout ({exc})")
            ser.reset_input_buffer()

    ser.reset_input_buffer()


# ============================================================
# CSV helpers
# ============================================================

def add_float_columns(cols: dict[str, object], prefix: str, arr, n_rows: int, width: int) -> None:
    if arr is None:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.full(n_rows, np.nan, dtype=np.float32)
    else:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.asarray(arr[:n_rows, j], dtype=np.float32)


def add_int_columns(cols: dict[str, object], prefix: str, arr, n_rows: int, width: int) -> None:
    if arr is None:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.full(n_rows, np.nan, dtype=np.float32)
    else:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.asarray(arr[:n_rows, j], dtype=np.int32)


# ============================================================
# Dataset execution
# ============================================================

def run_dataset(args: argparse.Namespace) -> int:
    fpga_mode = args.fpga_mode.lower().strip()
    if fpga_mode not in {"sequential", "pipeline"}:
        raise ValueError("--fpga-mode must be 'sequential' or 'pipeline'.")

    project_root = resolve_project_root(args.project_root)
    dataset_npz = resolve_dataset_path(
        args.dataset,
        args.dataset_dir,
        project_root,
    )
    out_dir = resolve_out_dir(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_path = resolve_scales_vh(
        args.scales,
        project_root,
        fpga_mode,
    )
    num_linear = read_num_linear(scales_path, args.num_linear_fallback)
    pipeline_delay = max(0, num_linear - 1) if fpga_mode == "pipeline" else 0

    inferred_input_words, inferred_output_words = infer_words_from_scales(scales_path)

    data = load_dataset(dataset_npz)
    x = data["X"]
    y_dataset = data["Y_dataset"]
    y_pred_int_ref = data["Y_pred_int_ref"]
    y_pred_float_ref = data["Y_pred_float_ref"]
    x_float = data["X_float"]
    y_float = data["Y_float"]

    n_samples, input_size = x.shape
    _, output_size = y_dataset.shape

    input_words = input_size if args.input_words is None else int(args.input_words)
    output_words = output_size if args.output_words is None else int(args.output_words)

    if input_words != input_size:
        raise ValueError(
            f"Mismatch: input_words={input_words}, but dataset INPUT_SIZE={input_size}. "
            "RTL and dataset must use the same input size."
        )

    if output_words != output_size:
        raise ValueError(
            f"Mismatch: output_words={output_words}, but dataset OUTPUT_SIZE={output_size}. "
            "RTL and dataset must use the same output size."
        )

    if input_words != inferred_input_words:
        print(f"WARNING: input_words={input_words}, but scales.vh suggests N0={inferred_input_words}.")
    if output_words != inferred_output_words:
        print(f"WARNING: output_words={output_words}, but scales.vh suggests output size={inferred_output_words}.")

    n_test = n_samples if args.max_samples is None or args.max_samples < 0 else min(int(args.max_samples), n_samples)

    # Pipeline mode needs extra dummy frames to flush the delayed real outputs.
    if fpga_mode == "pipeline":
        flush = np.full((pipeline_delay, input_size), int(args.flush_value), dtype=np.int16)
        x_stream = np.vstack([x[:n_test], flush]).astype(np.int16)
    else:
        x_stream = x[:n_test].astype(np.int16)

    n_stream = int(x_stream.shape[0])

    y_raw = np.zeros((n_stream, output_words), dtype=np.int16)
    tx_words_hex_raw: list[str] = []
    rx_words_hex_raw: list[str] = []

    print("\nUART dataset run configuration")
    print("  project_root   =", project_root)
    print("  fpga_mode      =", fpga_mode)
    print("  dataset        =", dataset_npz)
    print("  out_dir        =", out_dir)
    print("  scales         =", scales_path if scales_path else "not found")
    print("  num_linear     =", num_linear)
    print("  pipeline_delay =", pipeline_delay)
    print("  stream_frames  =", n_stream)
    print("  port           =", args.port)
    print("  baud           =", args.baud)
    print("  timeout        =", args.timeout)
    print("  startup_delay  =", args.startup_delay)
    print("  warmup         =", args.warmup_transactions)
    print("  N_TEST         =", n_test)
    print("  TX/frame       =", input_words, "int16 words =", 2 * input_words, "bytes")
    print("  RX/frame       =", output_words, "int16 words =", 2 * output_words, "bytes")
    print("  byte order     = lsb-first / little-endian")
    print("  SOF/EOF        = none")

    t0 = time.time()

    with open_uart(args.port, args.baud, args.timeout, args.startup_delay) as ser:
        warmup_transactions(
            ser,
            n=args.warmup_transactions,
            input_words=input_words,
            output_words=output_words,
            timeout_s=args.timeout,
        )

        for i in range(n_stream):
            rx_values, tx_bytes, rx_bytes = transaction(
                ser,
                x_stream[i],
                input_words=input_words,
                output_words=output_words,
                timeout_s=args.timeout,
                reset_input_buffer=args.reset_input_buffer_before_tx,
            )

            y_raw[i, :] = np.asarray(rx_values, dtype=np.int16)
            tx_words_hex_raw.append(int16_words_hex(x_stream[i]))
            rx_words_hex_raw.append(int16_words_hex(rx_values))

            if i < args.preview:
                tag = "flush" if i >= n_test else "data"
                print(
                    f"raw_frame {i:04d} <{tag}> | "
                    f"TX={x_stream[i].astype(np.int32).tolist()} "
                    f"| RX={y_raw[i].astype(np.int32).tolist()} "
                    f"| TX words={tx_words_hex_raw[-1]} "
                    f"| RX words={rx_words_hex_raw[-1]}"
                )

            if (i + 1) % args.progress_every == 0 or (i + 1) == n_stream:
                print(f"  {i + 1}/{n_stream} stream frame(s) processed")

    t1 = time.time()

    if fpga_mode == "pipeline":
        y_board = y_raw[pipeline_delay:pipeline_delay + n_test].copy()
        raw_output_frame = np.arange(pipeline_delay, pipeline_delay + n_test, dtype=np.int32)
        rx_words_hex = rx_words_hex_raw[pipeline_delay:pipeline_delay + n_test]
    else:
        y_board = y_raw[:n_test].copy()
        raw_output_frame = np.arange(n_test, dtype=np.int32)
        rx_words_hex = rx_words_hex_raw[:n_test]

    # TX words describe the original input sample, not the delayed raw output frame.
    tx_words_hex = [int16_words_hex(x[i]) for i in range(n_test)]

    y_ref_target = y_dataset[:n_test]
    err_vs_target = y_board.astype(np.int32) - y_ref_target.astype(np.int32)

    if y_pred_int_ref is not None:
        y_ref_pred = y_pred_int_ref[:n_test]
        err_vs_pred = y_board.astype(np.int32) - y_ref_pred.astype(np.int32)
    else:
        y_ref_pred = None
        err_vs_pred = None

    output_scale_exp = data["pred_scale_exp"]
    if output_scale_exp is None:
        output_scale_exp = data["target_scale_exp"]

    print("\nExecution completed.")
    print(f"Total time: {t1 - t0:.3f} s")
    print(f"Average time: {(t1 - t0) / max(n_test, 1) * 1000:.3f} ms/real sample")

    print("\nComparison vs target dataset")
    print("Max abs error  =", int(np.max(np.abs(err_vs_target))))
    print("Mean abs error =", float(np.mean(np.abs(err_vs_target))))
    print("Exact samples  =", int(np.sum(np.all(err_vs_target == 0, axis=1))), "/", n_test)

    if err_vs_pred is not None:
        print("\nComparison vs pred_int dataset")
        print("Max abs error  =", int(np.max(np.abs(err_vs_pred))))
        print("Mean abs error =", float(np.mean(np.abs(err_vs_pred))))
        print("Exact samples  =", int(np.sum(np.all(err_vs_pred == 0, axis=1))), "/", n_test)
    else:
        print("\nNo pred_int reference found in the dataset.")

    print("\nAligned preview:")
    for i in range(min(args.preview, n_test)):
        line = (
            f"sample {i:04d} | raw_output_frame={int(raw_output_frame[i]):04d} "
            f"| fpga={y_board[i].astype(np.int32).tolist()} "
            f"| target={y_ref_target[i].astype(np.int32).tolist()} "
            f"| err_target={err_vs_target[i].astype(np.int32).tolist()}"
        )
        if y_ref_pred is not None:
            line += (
                f" | pred_int={y_ref_pred[i].astype(np.int32).tolist()} "
                f"| err_pred={err_vs_pred[i].astype(np.int32).tolist()}"
            )
        print(line)

    if pd is None:
        print("\nPandas is missing: CSV not generated. Install it with: pip install pandas")
        return 0

    # Build CSV columns in a stable order shared by the project comparison scripts.
    cols: dict[str, object] = {}

    cols["sample"] = np.arange(n_test, dtype=np.int32)
    cols["raw_output_frame"] = raw_output_frame

    add_float_columns(cols, "input_float", x_float, n_test, input_size)
    add_int_columns(cols, "input_quantized", x, n_test, input_size)

    add_float_columns(cols, "target_float", y_float, n_test, output_size)
    add_int_columns(cols, "target_quantized", y_ref_target, n_test, output_size)

    add_float_columns(cols, "pred_float", y_pred_float_ref, n_test, output_size)
    add_int_columns(cols, "pred_quantized", y_ref_pred, n_test, output_size)

    add_int_columns(cols, "fpga_quantized", y_board, n_test, output_size)

    if output_scale_exp is None:
        fpga_float = np.full((n_test, output_size), np.nan, dtype=np.float32)
    else:
        fpga_float = q_to_float(y_board, output_scale_exp).astype(np.float32)
    add_float_columns(cols, "fpga_float", fpga_float, n_test, output_size)

    add_int_columns(cols, "error_fpga_target_quantized", err_vs_target, n_test, output_size)
    add_int_columns(cols, "error_fpga_pred_quantized", err_vs_pred, n_test, output_size)

    cols["tx_words_hex"] = np.asarray(tx_words_hex, dtype=object)
    cols["rx_words_hex"] = np.asarray(rx_words_hex, dtype=object)
    cols["transport"] = np.full(n_test, args.transport_label, dtype=object)
    cols["fpga_mode"] = np.full(n_test, fpga_mode, dtype=object)
    cols["num_linear"] = np.full(n_test, num_linear, dtype=np.int32)
    cols["pipeline_delay"] = np.full(n_test, pipeline_delay, dtype=np.int32)
    cols["input_scale_exponent"] = np.full(
        n_test,
        -1 if data["input_scale_exp"] is None else int(data["input_scale_exp"]),
        dtype=np.int32,
    )
    cols["output_scale_exponent"] = np.full(
        n_test,
        -1 if output_scale_exp is None else int(output_scale_exp),
        dtype=np.int32,
    )

    df = pd.DataFrame(cols)

    csv_path = (
        args.out_csv.expanduser().resolve()
        if args.out_csv
        else out_dir / f"soc_mlp_uart_{fpga_mode}_results.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    if args.save_raw_csv:
        raw_cols: dict[str, object] = {
            "raw_frame": np.arange(n_stream, dtype=np.int32),
            "tx_words_hex": np.asarray(tx_words_hex_raw, dtype=object),
            "rx_words_hex": np.asarray(rx_words_hex_raw, dtype=object),
        }
        for j in range(input_size):
            raw_cols[f"tx_quantized_{j}"] = x_stream[:, j].astype(np.int32)
        for j in range(output_size):
            raw_cols[f"rx_quantized_{j}"] = y_raw[:, j].astype(np.int32)

        raw_csv = csv_path.with_name(csv_path.stem + "_raw.csv")
        pd.DataFrame(raw_cols).to_csv(raw_csv, index=False)
        print("Raw CSV saved:", raw_csv)

    print("\nCSV saved:", csv_path)
    print("CSV preview:")
    print(df.head(min(10, len(df))).to_string(index=False))

    return 0


# ============================================================
# Main
# ============================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SOC_MLP through UART in sequential or pipeline mode."
    )

    # Project / generated files.
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Repository root. Default: PROJECT_ROOT_OVERRIDE or automatic "
            "detection of a directory containing src/ and data/."
        ),
    )
    parser.add_argument(
        "--fpga-mode",
        choices=["sequential", "pipeline"],
        default=DEFAULT_FPGA_MODE,
    )
    parser.add_argument(
        "--num-linear-fallback",
        type=int,
        default=DEFAULT_NUM_LINEAR,
    )
    parser.add_argument(
        "--scales",
        type=Path,
        default=None,
        help=(
            "Explicit scales.vh. Default is computed from "
            "FPGA_TARGET + --fpga-mode."
        ),
    )

    # UART.
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="UART baudrate")
    parser.add_argument("--timeout", type=float, default=10.0, help="RX timeout in seconds")
    parser.add_argument("--startup-delay", type=float, default=0.5, help="Delay after opening the port")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports then exit")

    # Dataset.
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Explicit quantized NPZ file. Overrides DATASET_PROFILE "
            "and DATASET_FILENAME."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Dataset directory. The script looks only for the configured "
            "DATASET_FILENAME, optionally under DATASET_PROFILE."
        ),
    )

    # Outputs.
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for CSV")
    parser.add_argument("--out-csv", type=Path, default=None, help="Complete output CSV path")

    # Frame dimensions.
    parser.add_argument(
        "--input-words",
        type=int,
        default=None,
        help="Default: dataset input width.",
    )
    parser.add_argument(
        "--output-words",
        type=int,
        default=None,
        help="Default: dataset output width.",
    )

    # Run control.
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of real samples. -1 = full dataset")
    parser.add_argument("--preview", type=int, default=10, help="Number of first aligned samples to print")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N stream frames")

    parser.add_argument("--warmup-transactions", type=int, default=1, help="Ignored dummy transactions before dataset")
    parser.add_argument("--flush-value", type=int, default=0, help="Dummy value used for pipeline flush frames")
    parser.add_argument("--save-raw-csv", action="store_true", help="Also save raw stream TX/RX frames")

    parser.add_argument(
        "--reset-input-buffer-before-tx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear stale RX bytes before each new TX frame.",
    )

    parser.add_argument(
        "--transport-label",
        default="uart_lsb_first_int16_no_sof_send_then_read",
        help="String stored in the CSV transport column.",
    )

    args = parser.parse_args(argv)

    if args.list_ports:
        list_serial_ports()
        return 0

    # Print the user-facing configuration before any UART access.
    print("Configured defaults:")
    print("  DATASET_BASE_DIR =", DATASET_BASE_DIR)
    print("  DATASET_PROFILE  =", DATASET_PROFILE)
    print("  DATASET_FILENAME =", DATASET_FILENAME)
    print("  FPGA_TARGET      =", FPGA_TARGET)
    print("  DEFAULT_FPGA_MODE=", DEFAULT_FPGA_MODE)
    print("  RESULT_DIR       =", RESULT_DIR)

    print("Linux user groups:", os.getgroups())
    return run_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
