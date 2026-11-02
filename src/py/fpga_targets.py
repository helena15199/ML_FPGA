"""Shared FPGA target and architecture definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProfile:
    """Hardware contract used by quantization, RTL generation and build."""

    name: str
    transport: str
    dsp_bits: int
    toolchain: str
    device: str
    executables: tuple[str, ...]


TARGET_PROFILES = {
    "ultra96": TargetProfile(
        name="ultra96",
        transport="axi_stream",
        dsp_bits=48,
        toolchain="vivado",
        device="xczu3eg-sbva484-1-e",
        executables=("vivado",),
    ),
    "lattice": TargetProfile(
        name="lattice",
        transport="uart",
        dsp_bits=32,
        toolchain="radiant",
        device="iCE40UP5K-SG48I",
        executables=("radiantc", "pnmainc", "radiant"),
    ),
    "nexys": TargetProfile(
        name="nexys",
        transport="uart",
        dsp_bits=48,
        toolchain="vivado",
        device="xc7a100tcsg324-1",
        executables=("vivado",),
    ),
}

TARGET_ALIASES = {
    "ultra": "ultra96",
    "ultra96": "ultra96",
    "xilinx": "ultra96",
    "lattice": "lattice",
    "ice40": "lattice",
    "nexys": "nexys",
    "nexysa7": "nexys",
    "artix7": "nexys",
}

ARCHITECTURE_ALIASES = {
    "sequential": "sequential",
    "seq": "sequential",
    "pipeline": "pipeline",
    "pipelined": "pipeline",
}


def normalize_target(value: object) -> str:
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    try:
        return TARGET_ALIASES[key]
    except KeyError as exc:
        raise ValueError("target must be 'ultra96', 'lattice' or 'nexys'") from exc


def normalize_architecture(value: object) -> str:
    key = str(value).strip().lower().replace("-", "_")
    try:
        return ARCHITECTURE_ALIASES[key]
    except KeyError as exc:
        raise ValueError("architecture must be 'sequential' or 'pipeline'") from exc
