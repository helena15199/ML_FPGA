"""Configuration objects and shared data containers for MLP quantization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch.nn as nn


@dataclass(frozen=True)
class QuantConfig:
    """Main fixed-point and FPGA configuration for the V2 numerical contract.

    V2 uses one common signed activation width for:
    - network input,
    - inter-layer activation RAMs,
    - LUT input/output words,
    - final network output.

    The MAC/pre-activation stays physically on ``dsp_bits``.  ``preact_exp``
    remains a scale exponent per hidden layer; there is deliberately no
    independent ``preact_bits`` width anymore.
    """

    weight_bits: int = 16
    bias_bits: int = 16
    act_bits: int = 16
    dsp_bits: int = 32

    margin_weights: float = 0.1
    margin_acts: float = 0.0
    calibration_mode: str = "max"
    calibration_percentile: float = 99.9

    activation_negative_slope: float = 1 / 64

    # DSP accumulator protection used during FLOAT calibration.
    # For each layer independently, the optimizer measures every floating-point
    # partial accumulation (bias, then bias + x0*w0, + x1*w1, ...), using the
    # original float activations/weights/biases.  It then reduces only that
    # layer's SW until the FLOAT accumulator range fits inside this fraction of
    # the real range representable by DSP_PW at Sacc = SX + SW.
    # Example: 0.90 keeps about 10% accumulator headroom.
    dsp_acc_max_utilization: float = 0.90
    dsp_weight_search_max_steps: int = 24

    # FPGA export mode.
    # - "pipeline": split .mem files per layer.
    # - "sequential": concatenated weight_all.mem, bias_all.mem and act_all.mem files.
    fpga_mode: Literal["pipeline", "sequential"] = "sequential"
    export_split_files: bool = False

    # Automatic search of the best LUT address width N.
    # The RTL deliberately keeps its raw wrap from the positive last segment to
    # the negative half.  Python therefore rejects candidates/data that touch
    # the forbidden last positive segment instead of relying on saturation/clamp.
    last_segment_strategy: str = "forbid_last"
    lut_addr_bits_min: int = 4
    lut_addr_bits_max_cap: int = 10
    lut_max_words: int = 1024
    lut_auto_window: int = 2
    activation_score_memory_weight: float = 0.35
    activation_artifact_subdir: str = "activation_fpga"

    distribution_pdf_name: str = "distribution_analysis_report.pdf"


@dataclass(frozen=True)
class ProjectPaths:
    model_path: Path
    dataset_path: Path
    output_dir: Path


@dataclass(frozen=True)
class DatasetBundle:
    x: np.ndarray
    y_deg: np.ndarray | None = None
    y_rad: np.ndarray | None = None
    y: np.ndarray | None = None
    task: Literal["regression", "classification"] = "regression"
    # Classification only: ``x``/``y`` calibrate the scales, the test split is
    # kept for the FPGA simulation.
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    checkpoint: dict[str, Any]
    config: dict[str, Any]
    activation_plan: list[dict[str, Any]]


@dataclass(frozen=True)
class LayerCalibration:
    name: str
    is_output: bool
    w_float: np.ndarray
    b_float: np.ndarray
    input_exp: int
    weight_exp: int
    bias_exp: int
    # Scale of the pre-activation presented to a hidden activation.  Its
    # physical datapath width is DSP_PW, while a LUT receives its ACT_W LSBs.
    preact_exp: int | None = None
    act_exp: int | None = None
    # Scale of the final network output.  Width is ACT_W in V2.
    output_exp: int | None = None
    activation_name: str | None = None


@dataclass(frozen=True)
class QuantizedLayer:
    name: str
    is_output: bool
    w_float: np.ndarray
    b_float: np.ndarray
    w_q: np.ndarray
    b_q: np.ndarray
    input_exp: int
    weight_exp: int
    bias_exp: int
    preact_exp: int | None = None
    act_exp: int | None = None
    output_exp: int | None = None
    activation_name: str | None = None


@dataclass(frozen=True)
class ActivationExport:
    layer_name: str
    activation_name: str
    implementation: str
    act_bits: int
    addr_bits: int | None
    signed_input: bool
    input_exp: int
    output_exp: int
    x_min: float
    x_max: float
    memory_words: int
    memory_bits: int
    mae: float
    max_abs_err: float
    mae_lsb_out: float
    max_abs_err_lsb_out: float
    last_segment_strategy: str
    paths: dict[str, Any]

