"""Generate the FPGA Tcl file and execute it with the selected vendor tool."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from fpga_targets import (
    TARGET_PROFILES,
    TargetProfile,
    normalize_architecture,
    normalize_target,
)


DEFAULT_FPGA_PROJECT_CONFIG = {
    "name": "walter_q16",
    "output_root": "data/quantized_data",
    "target": "ultra96",
    "architecture": "sequential",
    "project_name": None,
    "tool_executable": None,
    "jobs": 8,
    "clean_generated": True,
    "create_project": False,
    "run_flow": False,
}


@dataclass(frozen=True)
class FPGAProjectResult:
    """Paths and process status returned to the notebook."""

    config: dict[str, Any]
    profile: TargetProfile
    layout_path: Path
    dsp_bits: int
    generated_rtl_dir: Path
    tcl_path: Path
    project_dir: Path
    command: tuple[str, ...]
    log_path: Path | None
    return_code: int | None
    duration_seconds: float | None
    tool_error: str | None

    @property
    def succeeded(self) -> bool:
        return self.tool_error is None and (
            self.return_code == 0 or not self.config["create_project"]
        )

    def summary(self) -> dict[str, Any]:
        if self.tool_error:
            status = "Outil FPGA introuvable"
        elif self.return_code == 0:
            status = "Flow terminé (bitstream)" if self.config["run_flow"] else "Projet créé"
        elif self.return_code is None:
            status = "RTL et TCL générés"
        else:
            status = f"Échec outil (code {self.return_code})"
        return {
            "status": status,
            "target": self.profile.name,
            "architecture": self.config["architecture"],
            "transport": self.profile.transport,
            "DSP_PW": self.dsp_bits,
            "layout": str(self.layout_path),
            "RTL généré": str(self.generated_rtl_dir),
            "TCL": str(self.tcl_path),
            "projet": str(self.project_dir),
            "commande": shlex.join(self.command) if self.command else "non exécutée",
            "journal": str(self.log_path) if self.log_path else "non créé",
            "outil": self.tool_error or "disponible",
        }


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_fpga_project_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge defaults and normalize the two user-facing selectors."""
    unknown = sorted(set(config).difference(DEFAULT_FPGA_PROJECT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown FPGA project configuration keys: {unknown}")
    validated = {**DEFAULT_FPGA_PROJECT_CONFIG, **config}
    validated["target"] = normalize_target(validated["target"])
    validated["architecture"] = normalize_architecture(validated["architecture"])

    name = str(validated["name"]).strip()
    if not name:
        raise ValueError("name cannot be empty")
    validated["name"] = name
    if not str(validated["output_root"]).strip():
        raise ValueError("output_root cannot be empty")

    project_name = validated["project_name"]
    if project_name is None:
        project_name = f"SOC_MLP_{validated['target']}_{validated['architecture']}"
    project_name = str(project_name).strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", project_name):
        raise ValueError("project_name must contain only letters, digits and underscores")
    validated["project_name"] = project_name

    if not isinstance(validated["clean_generated"], bool):
        raise ValueError("clean_generated must be a boolean")
    for option in ("create_project", "run_flow"):
        if not isinstance(validated[option], bool):
            raise ValueError(f"{option} must be a boolean")
    if validated["run_flow"] and not validated["create_project"]:
        raise ValueError("run_flow=True requires create_project=True")
    if not isinstance(validated["jobs"], int) or isinstance(validated["jobs"], bool):
        raise ValueError("jobs must be a positive integer")
    if validated["jobs"] <= 0:
        raise ValueError("jobs must be a positive integer")

    executable = validated["tool_executable"]
    if executable is not None and not str(executable).strip():
        raise ValueError("tool_executable must be a path/name or None")
    validated["tool_executable"] = None if executable is None else str(executable).strip()
    return validated


def quantized_layout_path(config: dict[str, Any], *, project_root: str | Path) -> Path:
    """Return the export produced by notebook 02 for the same selectors."""
    validated = validate_fpga_project_config(config)
    root = Path(project_root).resolve()
    output_root = _resolve_path(root, validated["output_root"])
    return output_root / validated["name"] / validated["target"] / validated["architecture"] / "mem_layout.json"


def _read_export_contract(
    layout_path: Path,
    *,
    expected_target: str,
    expected_architecture: str,
) -> dict[str, Any]:
    """Read and validate the numerical contract produced by quantization."""
    with layout_path.open("r", encoding="utf-8") as stream:
        layout = json.load(stream)
    try:
        dsp_bits = int(layout["bits"]["dsp_bits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing bits.dsp_bits in {layout_path}") from exc
    if dsp_bits <= 0:
        raise ValueError(f"bits.dsp_bits must be positive in {layout_path}")

    manifest_path = layout_path.with_name("quantization_manifest.json")
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        config = manifest.get("config", {})
        manifest_target = config.get("target")
        manifest_architecture = config.get("architecture")
        if manifest_target != expected_target or manifest_architecture != expected_architecture:
            raise ValueError(
                "Quantized export does not match the requested FPGA project: "
                f"export={manifest_target}/{manifest_architecture}, "
                f"requested={expected_target}/{expected_architecture}"
            )
        manifest_dsp_bits = manifest.get("quant_config", {}).get("dsp_bits")
        if manifest_dsp_bits is not None and int(manifest_dsp_bits) != dsp_bits:
            raise ValueError(
                "DSP width mismatch between quantization_manifest.json "
                f"({manifest_dsp_bits}) and mem_layout.json ({dsp_bits})"
            )
    return {"layout": layout, "dsp_bits": dsp_bits}


def _load_generator(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"FPGA generator not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FPGA generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def generate_fpga_project(
    config: dict[str, Any],
    *,
    project_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the selected editable RTL panel and its project Tcl file."""
    validated = validate_fpga_project_config(config)
    root = Path(project_root).resolve()
    layout_path = quantized_layout_path(validated, project_root=root)
    if not layout_path.is_file():
        raise FileNotFoundError(
            f"Quantized layout not found: {layout_path}\n"
            "Run notebooks/02_quantize_MLP.ipynb with the same name, target and architecture."
        )

    contract = _read_export_contract(
        layout_path,
        expected_target=validated["target"],
        expected_architecture=validated["architecture"],
    )

    target = validated["target"]
    mode = validated["architecture"]
    profile = TARGET_PROFILES[target]
    if target == "nexys":
        raise NotImplementedError(
            "The Nexys project is created by hand in Vivado from "
            "src/rtl/generated/src_nexys_sequential (see notebook 02b)"
        )
    if target == "ultra96":
        generator = _load_generator(
            root / "src" / "tcl" / "remodel_vivado_project.py",
            "recipe_fpga_project_vivado",
        )
        generated = generator.generate_project(
            root,
            layout_path,
            fpga_mode=mode,
            template_rel=f"src/rtl/src_ultra_{mode}",
            generated_rel=f"src/rtl/generated/src_ultra_{mode}",
            project_name=validated["project_name"],
            part_name=profile.device,
            jobs=validated["jobs"],
            dsp_pw=contract["dsp_bits"],
            clean_generated=validated["clean_generated"],
        )
    else:
        generator = _load_generator(
            root / "src" / "tcl" / "remodel_lattice_project.py",
            "recipe_fpga_project_lattice",
        )
        generated = generator.generate_project(
            root,
            layout_path,
            fpga_mode=mode,
            template_rel=f"src/rtl/src_lattice_{mode}",
            project_name=validated["project_name"],
            device=profile.device,
            dsp_pw=contract["dsp_bits"],
            clean_generated=validated["clean_generated"],
            run_flow=validated["run_flow"],
        )
    return validated, generated


def _installation_candidates(profile: TargetProfile) -> list[Path]:
    """Return common tool locations without recursively scanning the machine."""
    home = Path.home()
    paths: list[Path] = []
    if profile.toolchain == "vivado":
        env_roots = [os.environ.get("XILINX_VIVADO")]
        roots = [Path(value).expanduser() for value in env_roots if value]
        roots += [Path("/tools/Xilinx/Vivado"), Path("/opt/Xilinx/Vivado")]
        for root in roots:
            paths.append(root / "bin" / "vivado")
            paths.extend(sorted(root.glob("*/bin/vivado"), reverse=True))
        return paths

    env_roots = [
        os.environ.get("RADIANT_HOME"),
        os.environ.get("LATTICE_RADIANT"),
        os.environ.get("LSC_RADIANT_HOME"),
    ]
    roots = [Path(value).expanduser() for value in env_roots if value]
    roots += [home / "lscc" / "radiant", Path("/usr/local/lscc/radiant"), Path("/opt/lscc/radiant")]
    relative_executables = (
        "bin/lin64/radiantc",
        "bin/lin64/pnmainc",
        "bin/lin64/radiant",
        "bin/nt64/radiantc.exe",
        "bin/nt64/pnmainc.exe",
        "bin/nt64/radiant.exe",
    )
    for root in roots:
        for relative in relative_executables:
            paths.append(root / relative)
            paths.extend(sorted(root.glob(f"*/{relative}"), reverse=True))
    return paths


def _find_executable(
    config: dict[str, Any],
    profile: TargetProfile,
    *,
    project_root: Path,
) -> str:
    requested = config["tool_executable"]
    candidates = (requested,) if requested else profile.executables
    for candidate in candidates:
        found = shutil.which(str(candidate))
        if found:
            return found
        path = Path(str(candidate)).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if path.is_file():
            return str(path.resolve())
    if not requested:
        for path in _installation_candidates(profile):
            if path.is_file():
                return str(path.resolve())
    names = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"{profile.toolchain} executable not found ({names}). "
        "Set tool_executable to the installed executable path."
    )


def _tool_command(
    executable: str,
    *,
    profile: TargetProfile,
    tcl_path: Path,
    run_flow: bool = False,
) -> tuple[str, ...]:
    if profile.toolchain == "vivado":
        command = (executable, "-mode", "batch", "-source", str(tcl_path))
        return command + ("-tclargs", "--run") if run_flow else command
    command = (executable, str(tcl_path))
    return command


def build_fpga_project(
    config: dict[str, Any],
    *,
    project_root: str | Path,
) -> FPGAProjectResult:
    """Generate RTL/Tcl and optionally create the FPGA project and bitstream."""
    root = Path(project_root).resolve()
    validated, generated = generate_fpga_project(config, project_root=root)
    profile = TARGET_PROFILES[validated["target"]]
    tcl_path = Path(generated["generated_tcl"]).resolve()
    generated_rtl_dir = Path(generated["generated_rtl_dir"]).resolve()
    layout_path = quantized_layout_path(validated, project_root=root)
    contract = _read_export_contract(
        layout_path,
        expected_target=validated["target"],
        expected_architecture=validated["architecture"],
    )
    tool_workspace = "vivado" if profile.toolchain == "vivado" else "lattice"
    project_dir = root / tool_workspace / "build" / validated["project_name"]

    if not validated["create_project"]:
        return FPGAProjectResult(
            config=validated,
            profile=profile,
            layout_path=layout_path,
            dsp_bits=contract["dsp_bits"],
            generated_rtl_dir=generated_rtl_dir,
            tcl_path=tcl_path,
            project_dir=project_dir,
            command=(),
            log_path=None,
            return_code=None,
            duration_seconds=None,
            tool_error=None,
        )

    try:
        executable = _find_executable(validated, profile, project_root=root)
    except FileNotFoundError as exc:
        suggested = _tool_command(
            profile.executables[0],
            profile=profile,
            tcl_path=tcl_path,
            run_flow=validated["run_flow"],
        )
        return FPGAProjectResult(
            config=validated,
            profile=profile,
            layout_path=layout_path,
            dsp_bits=contract["dsp_bits"],
            generated_rtl_dir=generated_rtl_dir,
            tcl_path=tcl_path,
            project_dir=project_dir,
            command=suggested,
            log_path=None,
            return_code=None,
            duration_seconds=None,
            tool_error=str(exc),
        )
    command = _tool_command(
        executable,
        profile=profile,
        tcl_path=tcl_path,
        run_flow=validated["run_flow"],
    )
    log_dir = root / tool_workspace / "build_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{validated['project_name']}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.monotonic() - started
    return FPGAProjectResult(
        config=validated,
        profile=profile,
        layout_path=layout_path,
        dsp_bits=contract["dsp_bits"],
        generated_rtl_dir=generated_rtl_dir,
        tcl_path=tcl_path,
        project_dir=project_dir,
        command=command,
        log_path=log_path,
        return_code=completed.returncode,
        duration_seconds=duration,
        tool_error=None,
    )
