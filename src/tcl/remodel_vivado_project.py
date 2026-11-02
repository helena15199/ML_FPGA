"""Generate Ultra96/Vivado RTL and TCL files from mem_layout.json.

This helper renders concrete SystemVerilog files from editable RTL templates and
creates the Vivado Tcl script used to package the SOC_MLP IP and build the block
design.

Supported Ultra96 RTL modes:
- sequential: AXI-Stream sequential MLP with concatenated ROM memories.
- pipeline: AXI-Stream pipeline MLP with generated layer modules and per-layer
  ROM memories.

Only explicit placeholders such as {memory_weight}, {module_layer_name},
{layer_instances}, etc. are replaced. Normal SystemVerilog braces are left
untouched.

Expected repository layout:
  <repo>/src/rtl/src_ultra_sequential/*.sv
  <repo>/src/rtl/src_ultra_pipeline/*.sv
  <repo>/data/quantized_data/mem_layout.json
  <repo>/data/quantized_data/scales.vh
  <repo>/data/quantized_data/*.mem
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ULTRA96_SEQUENTIAL_RTL_FILES = [
    "dsp.sv",
    "mem_a.sv",
    "mem_b.sv",
    "rom_weight_all.sv",
    "rom_bias_all.sv",
    "rom_act_all.sv",
    "mlp_sequential_ctrl.sv",
    "mlp_sequential_datapath.sv",
    "mlp_sequential.sv",
    "axi_input.sv",
    "axi_output.sv",
    "soc_mlp.sv",
]

ULTRA96_SEQUENTIAL_TCL_ONLY_FILES = [
    "scales.vh",
]

KNOWN_PLACEHOLDERS = {
    "memory_weight",
    "memory_bias",
    "memory_act_function",
    "layer_size_case",
    "weight_base_case",
    "bias_base_case",
    "act_depth_localparams",
    "act_depth_sum_expr",
    "sx_case",
    "sw_case",
    "sb_case",
    "sp_case",
    "act_case",
    "lutn_case",
    "act_base_case",
    "weight_depth_expr",
    "bias_depth_expr",
    "dsp_pw_expr",
    "output_size_expr",
    "max_neurons_expr",
}


@dataclass(frozen=True)
class LayoutInfo:
    layout_path: Path
    mem_dir: Path
    mode: str
    num_linear: int
    topology: list[int]
    weight_file: str
    bias_file: str
    act_file: str
    total_weight_words: int
    total_bias_words: int
    total_activation_words: int


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _shape_of_weight(item: dict[str, Any]) -> tuple[int, int] | None:
    shape = item.get("shape")
    if shape and len(shape) == 2:
        # PyTorch Linear weight shape = [out_features, in_features]
        return int(shape[0]), int(shape[1])
    return None


def _file_basename(path_like: Any) -> str | None:
    if path_like is None:
        return None
    s = str(path_like).strip()
    if not s or s.lower() == "none":
        return None
    return Path(s).name


def parse_layout(layout_path: str | Path, mode_override: str | None = None) -> LayoutInfo:
    """Read mem_layout.json and infer the sequential MLP topology.

    mode_override is accepted by the shared notebook interface. This section
    renders the sequential Ultra96 backend.
    """
    layout_path = Path(layout_path).resolve()
    layout = _read_json(layout_path)
    mem_dir = layout_path.parent

    weights = sorted(layout.get("weights", []), key=lambda x: int(x.get("layer_idx", 0)))
    biases = sorted(layout.get("biases", []), key=lambda x: int(x.get("layer_idx", 0)))
    activations = sorted(layout.get("activations", []), key=lambda x: int(x.get("layer_idx", 0)))

    if not weights:
        raise ValueError("mem_layout.json does not contain a 'weights' section.")

    shapes = [_shape_of_weight(w) for w in weights]
    if any(s is None for s in shapes):
        raise ValueError("Each weights[] entry must contain shape=[N_OUT, N_IN].")

    first_out, first_in = shapes[0]  # type: ignore[index]
    _ = first_out
    topology = [first_in]
    for layer_idx, shape in enumerate(shapes):
        n_out, n_in = shape  # type: ignore[misc]
        if topology[-1] != n_in:
            raise ValueError(
                f"Inconsistent topology at layer {layer_idx}: "
                f"expected N_IN={topology[-1]}, read shape={shape}."
            )
        topology.append(n_out)

    return LayoutInfo(
        layout_path=layout_path,
        mem_dir=mem_dir,
        mode="sequential",
        num_linear=len(weights),
        topology=topology,
        weight_file=_file_basename(layout.get("weight_all_mem")) or "weight_all.mem",
        bias_file=_file_basename(layout.get("bias_all_mem")) or "bias_all.mem",
        act_file=_file_basename(layout.get("act_all_mem")) or "act_all.mem",
        total_weight_words=int(layout.get("total_weight_words", sum(int(w.get("length", 0)) for w in weights))),
        total_bias_words=int(layout.get("total_bias_words", sum(int(b.get("length", 0)) for b in biases))),
        total_activation_words=int(layout.get("total_activation_words", sum(int(a.get("length", 0)) for a in activations))),
    )


def _sv_int(value: str | int | None, *, default_base: int = 10) -> int | None:
    """Parse a simple SystemVerilog integer literal used in scales.vh/.mem."""
    if value is None:
        return None
    if isinstance(value, int):
        return value

    s = str(value).strip().rstrip(",;")
    s = re.sub(r"//.*$", "", s).strip()
    s = s.replace("_", "")
    if not s:
        return None

    m = re.fullmatch(r"(?:(\d+))?'([sS])?([hHdDbBoO])([0-9a-fA-FxXzZ]+)", s)
    if m:
        base_ch = m.group(3).lower()
        digits = m.group(4).lower().replace("x", "0").replace("z", "0")
        base = {"h": 16, "d": 10, "b": 2, "o": 8}[base_ch]
        return int(digits, base)

    if re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
        return int(s, 16)
    if re.fullmatch(r"0[bB][01]+", s):
        return int(s, 2)
    if re.fullmatch(r"0[oO][0-7]+", s):
        return int(s, 8)
    return int(s, default_base)


def parse_scales_vh(path: str | Path) -> dict[str, int]:
    path = Path(path)
    macros: dict[str, int] = {}
    define_re = re.compile(r"^\s*`define\s+(\w+)\s+([^/\s]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = define_re.match(line)
        if not m:
            continue
        value = _sv_int(m.group(2))
        if value is not None:
            macros[m.group(1)] = value
    return macros


def _expect_macro(macros: dict[str, int], name: str, scales_path: Path) -> int:
    if name not in macros:
        raise ValueError(f"Macro `{name}` absente de {scales_path}")
    return int(macros[name])


def _numeric_widths(scales: dict[str, int], dsp_pw: int) -> tuple[int, int, int]:
    """Validate and return the four-macro V2 numerical contract."""
    try:
        act_width = int(scales["ACT_W"])
        weight_width = int(scales["WEIGHT_W"])
        bias_width = int(scales["BIAS_W"])
        header_dsp_width = int(scales["DSP_PW"])
    except KeyError as exc:
        raise ValueError(f"Required V2 width macro is missing: {exc.args[0]}") from exc

    if min(act_width, weight_width, bias_width, header_dsp_width) <= 0:
        raise ValueError("ACT_W, WEIGHT_W, BIAS_W and DSP_PW must be positive")
    if act_width > 16:
        raise ValueError("The Ultra96 AXI-Stream transport requires ACT_W <= 16")
    if max(act_width, weight_width, bias_width) > header_dsp_width:
        raise ValueError("DSP_PW must cover ACT_W, WEIGHT_W and BIAS_W")
    if header_dsp_width != int(dsp_pw):
        raise ValueError(
            f"DSP_PW mismatch: scales.vh gives {header_dsp_width}, "
            f"but the Ultra96 profile requires {dsp_pw}."
        )
    return act_width, weight_width, bias_width


def validate_scales_against_layout(scales_path: str | Path, info: LayoutInfo) -> dict[str, int]:
    scales_path = Path(scales_path)
    macros = parse_scales_vh(scales_path)

    num_linear = _expect_macro(macros, "NUM_LINEAR", scales_path)
    if num_linear != info.num_linear:
        raise ValueError(
            f"NUM_LINEAR mismatch: {scales_path} gives {num_linear}, "
            f"but {info.layout_path.name} gives {info.num_linear}."
        )

    for idx, expected in enumerate(info.topology):
        got = _expect_macro(macros, f"N{idx}", scales_path)
        if got != expected:
            raise ValueError(
                f"Topology mismatch: `N{idx}={got}` in {scales_path}, "
                f"but mem_layout.json gives {expected}."
            )
    return macros


def _mem_tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    tokens: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"//.*$", "", line)
        line = re.sub(r"#.*$", "", line)
        tokens.extend(tok for tok in line.replace(",", " ").split() if tok)
    return tokens


def read_mem_words(path: str | Path, width: int) -> list[int]:
    """Lire un fichier readmemh : tous les mots sont hexadécimaux."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f".mem file not found: {path}")

    mask = (1 << width) - 1
    words_by_addr: dict[int, int] = {}
    addr = 0
    max_addr = -1

    for tok in _mem_tokens(path):
        if tok.startswith("@"):
            # Les adresses readmemh sont également hexadécimales.
            addr = int(tok[1:].replace("_", ""), 16)
            max_addr = max(max_addr, addr - 1)
            continue

        try:
            # Important : ne jamais utiliser int(tok, 0).
            # Dans un readmemh, 0B11 est l'hexadécimal 0x0B11,
            # et non le binaire 0b11.
            value = int(tok.replace("_", ""), 16)
        except ValueError as exc:
            raise ValueError(
                f"Mot hexadécimal invalide dans {path}: {tok!r}"
            ) from exc

        words_by_addr[addr] = value & mask
        max_addr = max(max_addr, addr)
        addr += 1

    if max_addr < 0:
        return []

    return [
        words_by_addr.get(index, 0)
        for index in range(max_addr + 1)
    ]


def _check_word_count(kind: str, mem_path: Path, words: list[int], expected: int, *, allow_empty_as_zero: bool = False) -> None:
    actual = len(words)
    if allow_empty_as_zero and actual == 0 and expected == 0:
        return
    if actual != expected:
        raise ValueError(
            f"Inconsistent {kind} size: {mem_path.name} contains {actual} words, "
            f"but mem_layout.json expects {expected}."
        )


def _format_memory_init(words: list[int], width: int) -> str:
    """Return unindented SystemVerilog initialization lines."""
    init_words = words if words else [0]
    hex_digits = max(1, (width + 3) // 4)
    return "\n".join(
        f"mem[{idx}] = {width}'h{word:0{hex_digits}X};"
        for idx, word in enumerate(init_words)
    )


def _macro(i: int, prefix: str = "N") -> str:
    return f"`{prefix}{i}"


def _sum_expr(terms: Iterable[str]) -> str:
    terms = [t for t in terms if t]
    return " + ".join(terms) if terms else "0"


def _prod_expr(a: int, b: int) -> str:
    return f"({_macro(a)}*{_macro(b)})"


def _case_lines(function_name: str, cases: list[tuple[int, str]]) -> str:
    return "\n".join(f"{idx}: {function_name} = {expr};" for idx, expr in cases)


def _act_depth_lines(n: int) -> str:
    return "\n".join(
        f"localparam int ACT_DEPTH{i} = ((`ACT{i} == 3) && (`LUT_N{i} > 0)) ? (1 << `LUT_N{i}) : 0;"
        for i in range(n)
    )


def _act_depth_sum_expr(n: int) -> str:
    return _sum_expr([f"ACT_DEPTH{i}" for i in range(n)])


def _act_base_cases(n: int) -> list[tuple[int, str]]:
    return [(i, _sum_expr([f"ACT_DEPTH{j}" for j in range(i)])) for i in range(n)]


def build_sv_variables(info: LayoutInfo, scales: dict[str, int], *, dsp_pw: int = 48) -> dict[str, dict[str, str]]:
    """Build per-file placeholder dictionaries for the SystemVerilog templates."""
    n = info.num_linear
    last = n
    max_neurons = max(info.topology)

    act_width, w_width, b_width = _numeric_widths(scales, dsp_pw)

    weight_path = info.mem_dir / info.weight_file
    bias_path = info.mem_dir / info.bias_file
    act_path = info.mem_dir / info.act_file

    weight_words = read_mem_words(weight_path, w_width)
    bias_words = read_mem_words(bias_path, b_width)
    act_words = read_mem_words(act_path, act_width)

    _check_word_count("weight", weight_path, weight_words, info.total_weight_words)
    _check_word_count("bias", bias_path, bias_words, info.total_bias_words)
    _check_word_count("activation", act_path, act_words, info.total_activation_words, allow_empty_as_zero=True)

    weight_depth_expr = _sum_expr([_prod_expr(i, i + 1) for i in range(n)])
    bias_depth_expr = _sum_expr([_macro(i) for i in range(1, n + 1)])
    act_depth_localparams = _act_depth_lines(n)
    act_depth_sum_expr = _act_depth_sum_expr(n)

    ctrl_vars = {
        "layer_size_case": _case_lines("n_at", [(i, _macro(i)) for i in range(n + 1)]),
        "weight_base_case": _case_lines(
            "weight_base_at",
            [(0, "0")] + [(i, _sum_expr([_prod_expr(j, j + 1) for j in range(i)])) for i in range(1, n)],
        ),
        "bias_base_case": _case_lines(
            "bias_base_at",
            [(0, "0")] + [(i, _sum_expr([_macro(j) for j in range(1, i + 1)])) for i in range(1, n)],
        ),
    }

    datapath_vars = {
        "act_depth_localparams": "\n".join(
            [
                act_depth_localparams,
                f"localparam int ACT_DEPTH_RAW = {act_depth_sum_expr};",
                "localparam int ACT_DEPTH = (ACT_DEPTH_RAW > 0) ? ACT_DEPTH_RAW : 1;",
            ]
        ),
        "sx_case": _case_lines("sx_at", [(i, _macro(i, "SX")) for i in range(n + 1)]),
        "sw_case": _case_lines("sw_at", [(i, _macro(i, "SW")) for i in range(n)]),
        "sb_case": _case_lines("sb_at", [(i, _macro(i, "SB")) for i in range(n)]),
        "sp_case": _case_lines("sp_at", [(i, _macro(i, "SP")) for i in range(max(0, n - 1))]),
        "act_case": _case_lines("act_at", [(i, _macro(i, "ACT")) for i in range(n)]),
        "lutn_case": _case_lines("lutn_at", [(i, _macro(i, "LUT_N")) for i in range(n)]),
        "act_base_case": _case_lines("act_base_at", _act_base_cases(n)),
    }

    common: dict[str, str] = {
        "dsp_pw_expr": str(int(dsp_pw)),
        "output_size_expr": _macro(last),
        "max_neurons_expr": str(int(max_neurons)),
    }

    return {
        "common": common,
        "rom_weight_all.sv": {"memory_weight": _format_memory_init(weight_words, w_width)},
        "rom_bias_all.sv": {"memory_bias": _format_memory_init(bias_words, b_width)},
        "rom_act_all.sv": {"memory_act_function": _format_memory_init(act_words, act_width)},
        "mlp_sequential_ctrl.sv": ctrl_vars,
        "mlp_sequential_datapath.sv": datapath_vars,
        "mlp_sequential.sv": {
            "weight_depth_expr": weight_depth_expr,
            "bias_depth_expr": bias_depth_expr,
            "act_depth_localparams": act_depth_localparams,
            "act_depth_sum_expr": act_depth_sum_expr,
        },
        "soc_mlp.sv": common,
    }


def render_sv_placeholders(text: str, variables: dict[str, str], *, filename: str) -> str:
    """Replace only explicit {placeholder} tokens.

    This avoids Python str.format so SystemVerilog concatenations/replications
    such as {a, b} or {DATA_W{1'b1}} remain untouched.
    """
    result = text

    # First replace placeholders that occupy a full line, preserving indentation
    # for every generated line.
    for name, value in variables.items():
        full_line_re = re.compile(rf"^(?P<indent>[ \t]*)\{{{re.escape(name)}\}}[ \t]*$", re.M)

        def repl(match: re.Match[str]) -> str:
            indent = match.group("indent")
            if value == "":
                return ""
            return "\n".join(indent + line if line else "" for line in value.splitlines())

        result = full_line_re.sub(repl, result)

    # Then replace inline placeholders.
    for name, value in variables.items():
        result = result.replace("{" + name + "}", value)

    unresolved = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", result)) & KNOWN_PLACEHOLDERS)
    if unresolved:
        raise RuntimeError(f"Unresolved placeholders in {filename}: {unresolved}")

    return result


def sync_scales_vh(template_dir: Path, out_dir: Path, mem_dir: Path) -> Path:
    src = mem_dir / "scales.vh"
    dst = out_dir / "scales.vh"
    if src.exists():
        shutil.copy2(src, dst)
        return dst

    fallback = template_dir / "scales.vh"
    if fallback.exists():
        shutil.copy2(fallback, dst)
        return dst

    raise FileNotFoundError(
        f"scales.vh not found in {mem_dir} or in {template_dir}."
    )


def copy_and_render_ultra96_sequential_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str = "src/rtl/src_ultra_sequential",
    generated_rel: str = "src/rtl/generated/src_ultra_sequential",
    clean_generated: bool = True,
    dsp_pw: int = 48,
) -> Path:
    repo_root = Path(repo_root).resolve()
    template_dir = repo_root / template_rel
    if not template_dir.exists():
        raise FileNotFoundError(f"Ultra96 sequential RTL template directory not found: {template_dir}")

    missing = [f for f in ULTRA96_SEQUENTIAL_RTL_FILES if not (template_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Ultra96 sequential RTL files in {template_dir}: {missing}")

    out_dir = repo_root / generated_rel
    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_path = sync_scales_vh(template_dir, out_dir, info.mem_dir)
    scales = validate_scales_against_layout(scales_path, info)
    per_file_vars = build_sv_variables(info, scales, dsp_pw=dsp_pw)
    common = per_file_vars.get("common", {})

    for name in ULTRA96_SEQUENTIAL_RTL_FILES:
        src = template_dir / name
        text = src.read_text(encoding="utf-8")
        variables = dict(common)
        variables.update(per_file_vars.get(name, {}))
        rendered = render_sv_placeholders(text, variables, filename=name)
        (out_dir / name).write_text(rendered, encoding="utf-8")

    # Copy/render additional SystemVerilog helper files if present.
    for src in sorted(template_dir.glob("*.sv")):
        if src.name in ULTRA96_SEQUENTIAL_RTL_FILES:
            continue
        variables = dict(common)
        variables.update(per_file_vars.get(src.name, {}))
        rendered = render_sv_placeholders(src.read_text(encoding="utf-8"), variables, filename=src.name)
        (out_dir / src.name).write_text(rendered, encoding="utf-8")

    return out_dir


def copy_and_patch_ultra96_sequential_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str = "src/rtl/src_ultra_sequential",
    generated_rel: str = "src/rtl/generated/src_ultra_sequential",
    clean_generated: bool = True,
    dsp_pw: int = 48,
) -> Path:
    """Alias for the Ultra96 sequential RTL renderer."""
    return copy_and_render_ultra96_sequential_rtl(
        repo_root,
        info,
        template_rel=template_rel,
        generated_rel=generated_rel,
        clean_generated=clean_generated,
        dsp_pw=dsp_pw,
    )

def _tcl_list(items: list[str]) -> str:
    """Return a Tcl list continuation body."""
    sep = " " + "\\" + "\n    "
    return sep.join(items)


def make_ultra96_sequential_tcl(
    repo_root: str | Path,
    info: LayoutInfo,
    rtl_dir: Path,
    *,
    tcl_path: str | Path | None = None,
    project_name: str = "SOC_MLP_ultra96",
    top_name: str = "SOC_MLP",
    part_name: str = "xczu3eg-sbva484-1-e",
    board_part: str | None = None,
    jobs: int = 8,
) -> Path:
    """Write a Vivado Tcl script for the Ultra96 sequential AXI design.

    The generated Tcl now creates the full Vivado system:
      - RTL project from src/rtl/generated/src_ultra_sequential
      - SOC_MLP packaged as a local AXI-Stream IP
      - block design with Zynq UltraScale+ PS, AXI DMA, SmartConnect, reset, IRQ
      - HDL wrapper set as synthesis top

    The RTL top uses ports clk/rstn, not ap_clk/axi_resetn.
    """
    repo_root = Path(repo_root).resolve()
    rtl_dir = Path(rtl_dir).resolve()
    tcl_path = Path(tcl_path or (repo_root / "vivado" / "create_soc_mlp_project_ultra96_sequential.tcl"))
    tcl_path.parent.mkdir(parents=True, exist_ok=True)

    tcl_source_files: list[str] = []
    seen: set[str] = set()
    for f in ULTRA96_SEQUENTIAL_TCL_ONLY_FILES + ULTRA96_SEQUENTIAL_RTL_FILES:
        if f not in seen:
            seen.add(f)
            tcl_source_files.append(f)

    rtl_tcl = [f'[file join $rtl_dir "{f}"]' for f in tcl_source_files]
    rtl_list = _tcl_list(rtl_tcl)
    topology = " -> ".join(map(str, info.topology))
    board_part_value = board_part or ""

    template = r"""# -----------------------------------------------------------------------------
# Auto-generated Vivado project script for Ultra96 sequential AXI design.
# Layout    : __LAYOUT_NAME__
# Topology  : __TOPOLOGY__
# Mode      : sequential
# -----------------------------------------------------------------------------

proc print_help {} {
    puts "\nUsage: source [info script] -tclargs ?options?"
    puts "Options:"
    puts "  --run                  Run synthesis and implementation up to bitstream."
    puts "  --project-name <name>  Vivado project name."
    puts "  --project-dir <path>   Vivado project directory."
    puts "  --part <part>          FPGA part."
    puts "  --board-part <board>   Optional board_part. Empty disables it."
    puts "  --top <module>         RTL top for IP packaging. Default: SOC_MLP."
    puts "  --bd-name <name>       Block design name. Default: design_1."
    puts "  --ip-repo-dir <path>   Generated IP repository. Default: <repo>/vivado/ip_repo."
    puts "  --skip-ip              Do not package SOC_MLP as local IP."
    puts "  --skip-bd              Do not create the block design/wrapper."
    puts "  --jobs <N>             Vivado run parallelism."
    puts "  --ip-clk-freq-hz <Hz>  Packaged SOC_MLP AXI clock frequency. Default: 99999001."
    puts "  --ignore-freq-hz       Mark FREQ_HZ compatibility as ignored when supported."
    puts "  --strict-freq-hz       Do not request Ignore FREQ_HZ compatibility."
    puts "  --help                 Print this help.\n"
}

proc add_existing_files {fileset files file_type required} {
    foreach f $files {
        set nf [file normalize $f]
        if {[file exists $nf]} {
            add_files -norecurse -fileset $fileset $nf
            set obj [get_files -of_objects [get_filesets $fileset] [list $nf]]
            if {$obj ne ""} { set_property file_type $file_type $obj }
            puts "INFO: added $nf"
        } else {
            if {$required} { error "Required file not found: $nf" }
            puts "WARNING: optional file not found: $nf"
        }
    }
}

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]

set project_name "__PROJECT_NAME__"
set top_name     "__TOP_NAME__"
set part_name    "__PART_NAME__"
set board_part   "__BOARD_PART__"
set jobs         __JOBS__
set ip_clk_freq_hz 99999001
set ignore_freq_hz_compat 1
set run_flow     0
set create_ip    1
set create_bd    1
set project_dir  ""
set bd_name      "design_1"
set ip_repo_dir  ""
set ip_vendor    "user.org"
set ip_library   "user"
set ip_name      "SOC_MLP"
set ip_version   "1.0"

for {set i 0} {$i < [llength $::argv]} {incr i} {
    set arg [lindex $::argv $i]
    switch -- $arg {
        "--run"          { set run_flow 1 }
        "--skip-ip"      { set create_ip 0 }
        "--skip-bd"      { set create_bd 0 }
        "--project-name" { incr i; set project_name [lindex $::argv $i] }
        "--project-dir"  { incr i; set project_dir [file normalize [lindex $::argv $i]] }
        "--part"         { incr i; set part_name [lindex $::argv $i] }
        "--board-part"   { incr i; set board_part [lindex $::argv $i] }
        "--top"          { incr i; set top_name [lindex $::argv $i] }
        "--bd-name"      { incr i; set bd_name [lindex $::argv $i] }
        "--ip-repo-dir"  { incr i; set ip_repo_dir [file normalize [lindex $::argv $i]] }
        "--jobs"         { incr i; set jobs [lindex $::argv $i] }
        "--ip-clk-freq-hz" { incr i; set ip_clk_freq_hz [lindex $::argv $i] }
        "--ignore-freq-hz" { set ignore_freq_hz_compat 1 }
        "--strict-freq-hz" { set ignore_freq_hz_compat 0 }
        "--help"         { print_help; return }
        default           { puts "WARNING: unknown argument ignored: $arg" }
    }
}

if {$project_dir eq ""} { set project_dir [file normalize [file join $repo_root "vivado" "build" $project_name]] }
if {$ip_repo_dir eq ""} { set ip_repo_dir [file normalize [file join $repo_root "vivado" "ip_repo"]] }

set rtl_dir [file normalize [file join $repo_root "src" "rtl" "generated" "src_ultra_sequential"]]

puts "INFO: repo_root    = $repo_root"
puts "INFO: project_name = $project_name"
puts "INFO: project_dir  = $project_dir"
puts "INFO: part_name    = $part_name"
puts "INFO: board_part   = $board_part"
puts "INFO: top          = $top_name"
puts "INFO: rtl_dir      = $rtl_dir"
puts "INFO: ip_repo_dir  = $ip_repo_dir"
puts "INFO: bd_name      = $bd_name"
puts "INFO: ip_clk_freq = $ip_clk_freq_hz"
puts "INFO: ignore_freq_hz_compat = $ignore_freq_hz_compat"

if {[file exists $project_dir]} {
    file delete -force $project_dir
}
file mkdir $project_dir

create_project -force $project_name $project_dir -part $part_name
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property default_lib xil_defaultlib [current_project]
set_property source_mgmt_mode All [current_project]

if {$board_part ne ""} {
    if {[catch {set_property board_part $board_part [current_project]} msg]} {
        puts "WARNING: board_part could not be set: $msg"
    }
}

if {[get_filesets -quiet sources_1] eq ""} { create_fileset -srcset sources_1 }
if {[get_filesets -quiet sim_1] eq ""}     { create_fileset -simset sim_1 }
if {[get_filesets -quiet constrs_1] eq ""} { create_fileset -constrset constrs_1 }

set_property include_dirs [list $rtl_dir] [get_filesets sources_1]
set_property include_dirs [list $rtl_dir] [get_filesets sim_1]

set rtl_files [list \
    __RTL_LIST__ \
]
add_existing_files sources_1 $rtl_files "SystemVerilog" 1

set vh_path [file normalize [file join $rtl_dir "scales.vh"]]
if {[llength [get_files -quiet $vh_path]] > 0} {
    set_property file_type {Verilog Header} [get_files $vh_path]
}

foreach f [get_files -of_objects [get_filesets sources_1]] {
    set fname [get_property NAME $f]
    if {[string match "*.sv" $fname]} {
        set_property file_type {SystemVerilog} $f
    }
}

set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

# -----------------------------------------------------------------------------
# IP packaging and block-design generation
# -----------------------------------------------------------------------------
proc ipx_add_axis_interface {core if_name if_mode prefix} {
    if {[llength [ipx::get_bus_interfaces $if_name -of_objects $core -quiet]] == 0} {
        ipx::add_bus_interface $if_name $core
    }
    set bus_if [ipx::get_bus_interfaces $if_name -of_objects $core]
    set_property bus_type_vlnv xilinx.com:interface:axis:1.0 $bus_if
    set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 $bus_if
    set_property interface_mode $if_mode $bus_if

    foreach {logical physical} [list \
        TDATA  ${prefix}_tdata \
        TKEEP  ${prefix}_tkeep \
        TLAST  ${prefix}_tlast \
        TVALID ${prefix}_tvalid \
        TREADY ${prefix}_tready \
    ] {
        if {[llength [ipx::get_ports $physical -of_objects $core -quiet]] == 0} {
            puts "WARNING: IP port $physical not found while mapping $if_name/$logical"
            continue
        }
        if {[llength [ipx::get_port_maps $logical -of_objects $bus_if -quiet]] == 0} {
            ipx::add_port_map $logical $bus_if
        }
        set_property physical_name $physical [ipx::get_port_maps $logical -of_objects $bus_if]
    }
}

proc ipx_set_bus_parameter_value {bus_if param_name param_value} {
    if {$bus_if eq ""} { return }
    set params [ipx::get_bus_parameters $param_name -of_objects $bus_if -quiet]
    if {[llength $params] == 0} {
        catch {ipx::add_bus_parameter $param_name $bus_if}
        set params [ipx::get_bus_parameters $param_name -of_objects $bus_if -quiet]
    }
    if {[llength $params]} {
        catch {set_property value $param_value $params}
    }
}


proc try_set_property_if_exists {obj prop value} {
    if {$obj eq ""} { return 0 }
    if {[catch {set props [list_property $obj]}]} { return 0 }
    if {[lsearch -exact $props $prop] >= 0} {
        if {[catch {set_property $prop $value $obj} msg]} {
            puts "INFO: cannot set $prop on $obj to $value: $msg"
            return 0
        }
        puts "INFO: set $prop=$value on $obj"
        return 1
    }
    return 0
}

proc ipx_try_ignore_freq_hz_compat {bus_if} {
    if {$bus_if eq ""} { return }

    # Vivado versions expose the IP-packager "Ignore FREQ_HZ" option with
    # different internal property names. Try the known spellings and keep this
    # non-fatal so the script remains portable across Vivado releases.
    foreach p [ipx::get_bus_parameters FREQ_HZ -of_objects $bus_if -quiet] {
        foreach {prop val} [list \
            IS_IGNORED true \
            IS_IGNORE true \
            IGNORE true \
            IGNORE_PARAM true \
            IGNORE_IN_BD true \
            COMPATIBILITY_IGNORE true \
            VALUE_RESOLVE_TYPE user \
            value_resolve_type user \
        ] {
            try_set_property_if_exists $p $prop $val
        }
    }

    foreach {prop val} [list \
        CONFIG.FREQ_HZ.IGNORE true \
        CONFIG.FREQ_HZ_IGNORE true \
        FREQ_HZ_IGNORE true \
        IGNORE_FREQ_HZ true \
        COMPATIBILITY.FREQ_HZ.IGNORE true \
        COMPATIBILITY_IGNORE_FREQ_HZ true \
    ] {
        try_set_property_if_exists $bus_if $prop $val
    }
}

proc bd_try_ignore_freq_hz_compat {obj_name} {
    foreach obj [concat [get_bd_pins -quiet $obj_name] [get_bd_intf_pins -quiet $obj_name]] {
        foreach {prop val} [list \
            CONFIG.FREQ_HZ.IGNORE true \
            CONFIG.FREQ_HZ_IGNORE true \
            FREQ_HZ_IGNORE true \
            IGNORE_FREQ_HZ true \
            COMPATIBILITY.FREQ_HZ.IGNORE true \
            COMPATIBILITY_IGNORE_FREQ_HZ true \
        ] {
            try_set_property_if_exists $obj $prop $val
        }
    }
}

proc set_bd_property_if_exists {obj prop value} {
    if {$obj eq ""} { return }
    set props [list_property $obj]
    if {[lsearch -exact $props $prop] >= 0} {
        set_property $prop $value $obj
    } else {
        puts "INFO: property $prop not present on $obj; skipping"
    }
}

proc package_soc_mlp_ip {ip_repo_dir ip_vendor ip_library ip_name ip_version top_name ip_clk_freq_hz ignore_freq_hz_compat} {
    set ip_core_dir [file normalize [file join $ip_repo_dir $ip_name]]
    file mkdir $ip_repo_dir
    if {[file exists $ip_core_dir]} {
        puts "INFO: removing previous packaged IP: $ip_core_dir"
        file delete -force $ip_core_dir
    }

    set_property top $top_name [get_filesets sources_1]
    update_compile_order -fileset sources_1

    puts "INFO: RTL top for IP packaging = [get_property top [get_filesets sources_1]]"
    puts "INFO: packaging $top_name as IP: $ip_vendor:$ip_library:$ip_name:$ip_version"

    ipx::package_project \
        -root_dir $ip_core_dir \
        -vendor $ip_vendor \
        -library $ip_library \
        -taxonomy /UserIP \
        -import_files \
        -force

    set core [ipx::current_core]
    set_property vendor $ip_vendor $core
    set_property library $ip_library $core
    set_property name $ip_name $core
    set_property version $ip_version $core
    set_property display_name $ip_name $core
    set_property description "Auto-packaged SOC_MLP Ultra96 sequential AXI IP generated from mem_layout.json." $core
    set_property supported_families {zynquplus Production} $core

    # AXI-Stream interfaces first, then associate both of them to clk.
    # If clk is configured before the stream interfaces exist, Vivado may keep
    # only one ASSOCIATED_BUSIF value during auto-inference.
    ipx_add_axis_interface $core s_axis_s2mm slave  s_axis_s2mm
    ipx_add_axis_interface $core m_axis_mm2s master m_axis_mm2s

    # The generated Ultra96 top exposes clk/rstn.
    catch {ipx::infer_bus_interface clk xilinx.com:signal:clock_rtl:1.0 $core}
    catch {ipx::infer_bus_interface rstn xilinx.com:signal:reset_rtl:1.0 $core}

    set clk_if [ipx::get_bus_interfaces clk -of_objects $core -quiet]
    if {$clk_if ne ""} {
        ipx_set_bus_parameter_value $clk_if ASSOCIATED_BUSIF "s_axis_s2mm:m_axis_mm2s"
        ipx_set_bus_parameter_value $clk_if ASSOCIATED_RESET "rstn"
        # Match the PS/DMA clock used by Ultra96/Vivado. On this setup Vivado
        # reports pl_clk0 as 99999001 Hz, not exactly 100000000 Hz. If the IP is
        # packaged with 100000000, IP Integrator treats it as fixed metadata and
        # validate_bd_design fails on FREQ_HZ mismatches.
        ipx_set_bus_parameter_value $clk_if FREQ_HZ $ip_clk_freq_hz
    }

    set rst_if [ipx::get_bus_interfaces rstn -of_objects $core -quiet]
    if {$rst_if ne ""} {
        ipx_set_bus_parameter_value $rst_if POLARITY ACTIVE_LOW
    }

    foreach if_name [list s_axis_s2mm m_axis_mm2s] {
        set axis_if [ipx::get_bus_interfaces $if_name -of_objects $core -quiet]
        if {$axis_if ne ""} {
            ipx_set_bus_parameter_value $axis_if FREQ_HZ $ip_clk_freq_hz
        }
    }

    if {$ignore_freq_hz_compat} {
        foreach if_name [list clk s_axis_s2mm m_axis_mm2s] {
            set if_obj [ipx::get_bus_interfaces $if_name -of_objects $core -quiet]
            if {$if_obj ne ""} { ipx_try_ignore_freq_hz_compat $if_obj }
        }
    }

    ipx::check_integrity $core
    ipx::save_core $core
    ipx::unload_core $core

    set component_xml [file join $ip_core_dir component.xml]
    if {![file exists $component_xml]} {
        error "IP packaging failed: component.xml was not generated at $component_xml"
    }

    set_property ip_repo_paths [list $ip_repo_dir] [current_project]
    update_ip_catalog -rebuild

    puts "INFO: packaged IP available in: $ip_core_dir/component.xml"
    return [file join $ip_core_dir component.xml]
}

proc maybe_connect_bd_intf {from_pin to_pin net_name} {
    if {[llength [get_bd_intf_pins -quiet $from_pin]] && [llength [get_bd_intf_pins -quiet $to_pin]]} {
        connect_bd_intf_net -intf_net $net_name [get_bd_intf_pins $from_pin] [get_bd_intf_pins $to_pin]
    } else {
        puts "WARNING: cannot connect interface $from_pin -> $to_pin"
    }
}

proc maybe_connect_bd_net {net_name pin_list} {
    set pins [list]
    foreach p $pin_list {
        if {[llength [get_bd_pins -quiet $p]]} {
            lappend pins [get_bd_pins $p]
        } else {
            puts "WARNING: missing BD pin $p"
        }
    }
    if {[llength $pins] >= 2} {
        connect_bd_net -net $net_name {*}$pins
    }
}

proc get_bd_pin_freq_hz {pin_name fallback_hz} {
    set pin [get_bd_pins -quiet $pin_name]
    if {[llength $pin]} {
        foreach prop {CONFIG.FREQ_HZ FREQ_HZ} {
            set value ""
            catch {set value [get_property $prop $pin]}
            if {$value ne ""} { return $value }
        }
    }
    return $fallback_hz
}

proc set_bd_obj_freq_hz {obj_name freq_hz} {
    set pins [get_bd_pins -quiet $obj_name]
    if {[llength $pins]} {
        catch {set_property CONFIG.FREQ_HZ $freq_hz $pins}
        catch {set_property FREQ_HZ $freq_hz $pins}
    }

    set intfs [get_bd_intf_pins -quiet $obj_name]
    if {[llength $intfs]} {
        catch {set_property CONFIG.FREQ_HZ $freq_hz $intfs}
        catch {set_property FREQ_HZ $freq_hz $intfs}
    }
}

proc align_soc_mlp_axis_metadata {freq_hz ignore_freq_hz_compat} {
    puts "INFO: aligning SOC_MLP/DMA AXI-Stream metadata to FREQ_HZ=$freq_hz"

    foreach p [list \
        SOC_MLP_0/clk \
        SOC_MLP_0/s_axis_s2mm \
        SOC_MLP_0/m_axis_mm2s \
        axi_dma_0/M_AXIS_MM2S \
        axi_dma_0/S_AXIS_S2MM \
    ] {
        set_bd_obj_freq_hz $p $freq_hz
    }

    # Keep the IP and DMA AXI-Stream metadata coherent with the 16-bit stream.
    # Some packaged SystemVerilog IPs do not accept cell parameter propagation,
    # so set the interface-pin properties directly and ignore unsupported forms.
    foreach intf [list \
        SOC_MLP_0/s_axis_s2mm \
        SOC_MLP_0/m_axis_mm2s \
        axi_dma_0/M_AXIS_MM2S \
        axi_dma_0/S_AXIS_S2MM \
    ] {
        set ip [get_bd_intf_pins -quiet $intf]
        if {[llength $ip]} {
            catch {set_property CONFIG.TDATA_NUM_BYTES 2 $ip}
            catch {set_property TDATA_NUM_BYTES 2 $ip}
        }
    }

    if {$ignore_freq_hz_compat} {
        foreach p [list             SOC_MLP_0/clk             SOC_MLP_0/s_axis_s2mm             SOC_MLP_0/m_axis_mm2s             axi_dma_0/M_AXIS_MM2S             axi_dma_0/S_AXIS_S2MM         ] {
            bd_try_ignore_freq_hz_compat $p
        }
    }

    # If the user parameters are visible on the packaged IP, force the same
    # 16-bit stream width as the AXI DMA. Unsupported parameter names are ignored.
    set soc [get_bd_cells -quiet SOC_MLP_0]
    if {[llength $soc]} {
        foreach {cfg val} [list \
            CONFIG.AXIS_DATA_W {16} \
            CONFIG.AXIS_KEEP_W {2} \
        ] {
            catch {set_property $cfg $val $soc}
        }
    }
}

proc create_and_set_bd_wrapper {bd_name} {
    set bd_file [get_files ${bd_name}.bd]
    if {[llength $bd_file] == 0} {
        error "Cannot create wrapper: BD file not found for $bd_name"
    }

    generate_target all $bd_file

    set wrapper_path [make_wrapper -files $bd_file -top]
    if {[llength $wrapper_path] == 0} {
        set wrapper_path [file join [get_property DIRECTORY [current_project]] \
            [get_property NAME [current_project]].srcs sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
    }

    if {![file exists $wrapper_path]} {
        error "Wrapper was not generated: $wrapper_path"
    }

    add_files -norecurse -fileset sources_1 $wrapper_path
    set_property top ${bd_name}_wrapper [get_filesets sources_1]
    update_compile_order -fileset sources_1

    puts "INFO: wrapper generated: $wrapper_path"
    puts "INFO: synthesis top set to ${bd_name}_wrapper"
}

proc create_soc_mlp_block_design {bd_name ip_vlnv ip_clk_freq_hz ignore_freq_hz_compat} {
    if {[llength [get_bd_designs -quiet $bd_name]]} {
        puts "INFO: block design $bd_name already exists. Recreating it."
        close_bd_design $bd_name
        file delete -force [file join [get_property DIRECTORY [current_project]] [get_property NAME [current_project]].srcs sources_1 bd $bd_name]
    }

    create_bd_design $bd_name
    current_bd_design $bd_name

    set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]

    # Do not blindly apply the board preset when board_part is empty.
    # With Vivado 2025.2 this can enable unwanted PS masters.
    # The sequential DMA design uses only:
    #   - M_AXI_HPM0_FPD for AXI-Lite DMA control
    #   - S_AXI_HP0_FPD  for DMA DDR traffic
    # M_AXI_HPM0_LPD is intentionally disabled and must not appear.
    set current_board_part ""
    catch {set current_board_part [get_property board_part [current_project]]}
    catch {set current_board_part [get_property BOARD_PART [current_project]]}
    if {$current_board_part ne ""} {
        catch {apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps} board_msg
        if {[info exists board_msg] && $board_msg ne ""} { puts "INFO: PS board automation: $board_msg" }
    } else {
        puts "INFO: board_part is empty; skipping PS board automation preset."
    }

    set_property -dict [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__USE__IRQ0 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    ] $ps

    # Extra guards for ZynqMP PS presets: disable unused HPM masters.
    # CONFIG.PSU__USE__M_AXI_GP2 is the important one here: on ZynqMP it
    # corresponds to the unwanted M_AXI_HPM0_LPD path in many Vivado versions.
    # The other names are kept as compatibility aliases when the property exists.
    foreach {cfg val} [list \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__M_AXI_HPM1_FPD {0} \
        CONFIG.PSU__USE__M_AXI_HPM0_LPD {0} \
        CONFIG.PSU__USE__M_AXI_LPD {0} \
    ] {
        set_bd_property_if_exists $ps $cfg $val
    }

    # If this pin still exists, Vivado kept the unused LPD master alive.
    # Do not connect its clock as a workaround: fail early with a clear message.
    if {[llength [get_bd_intf_pins -quiet zynq_ultra_ps_e_0/M_AXI_HPM0_LPD]]} {
        puts "WARNING: M_AXI_HPM0_LPD is still present after PS configuration."
        puts "WARNING: The design does not use it; check the PS property name for this Vivado version."
    }

    set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_include_mm2s {1} \
        CONFIG.c_include_s2mm {1} \
        CONFIG.c_addr_width {40} \
        CONFIG.c_s_axis_s2mm_tdata_width {16} \
        CONFIG.c_m_axis_mm2s_tdata_width {16} \
        CONFIG.c_m_axi_mm2s_data_width {32} \
        CONFIG.c_m_axi_s2mm_data_width {32} \
        CONFIG.c_sg_length_width {23} \
    ] $dma

    set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

    set irq_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0]
    set_property CONFIG.NUM_PORTS {2} $irq_concat

    set sc_hp [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0]
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $sc_hp

    set sc_lite [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_1]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc_lite

    set mlp [create_bd_cell -type ip -vlnv $ip_vlnv SOC_MLP_0]

    # AXI-Stream data path:
    # DDR -> DMA MM2S -> SOC_MLP -> DMA S2MM -> DDR.
    maybe_connect_bd_intf axi_dma_0/M_AXIS_MM2S SOC_MLP_0/s_axis_s2mm axi_dma_mm2s_to_soc_mlp
    maybe_connect_bd_intf SOC_MLP_0/m_axis_mm2s axi_dma_0/S_AXIS_S2MM soc_mlp_to_axi_dma_s2mm

    # AXI memory-mapped data path.
    maybe_connect_bd_intf axi_dma_0/M_AXI_MM2S smartconnect_0/S00_AXI dma_mm2s_to_hp_sc
    maybe_connect_bd_intf axi_dma_0/M_AXI_S2MM smartconnect_0/S01_AXI dma_s2mm_to_hp_sc
    maybe_connect_bd_intf smartconnect_0/M00_AXI zynq_ultra_ps_e_0/S_AXI_HP0_FPD hp_sc_to_ps_hp0

    # AXI-Lite control path.
    maybe_connect_bd_intf zynq_ultra_ps_e_0/M_AXI_HPM0_FPD smartconnect_1/S00_AXI ps_hpm0_to_lite_sc
    maybe_connect_bd_intf smartconnect_1/M00_AXI axi_dma_0/S_AXI_LITE lite_sc_to_dma_lite

    # Clock/reset. SOC_MLP generated top uses clk/rstn.
    maybe_connect_bd_net pl_clk0 [list \
        zynq_ultra_ps_e_0/pl_clk0 \
        proc_sys_reset_0/slowest_sync_clk \
        axi_dma_0/s_axi_lite_aclk \
        axi_dma_0/m_axi_mm2s_aclk \
        axi_dma_0/m_axi_s2mm_aclk \
        smartconnect_0/aclk \
        smartconnect_1/aclk \
        zynq_ultra_ps_e_0/saxihp0_fpd_aclk \
        zynq_ultra_ps_e_0/maxihpm0_fpd_aclk \
        SOC_MLP_0/clk \
    ]

    maybe_connect_bd_net pl_resetn0 [list \
        zynq_ultra_ps_e_0/pl_resetn0 \
        proc_sys_reset_0/ext_reset_in \
    ]

    maybe_connect_bd_net peripheral_aresetn [list \
        proc_sys_reset_0/peripheral_aresetn \
        axi_dma_0/axi_resetn \
        smartconnect_0/aresetn \
        smartconnect_1/aresetn \
        SOC_MLP_0/rstn \
    ]

    maybe_connect_bd_net dma_irq_mm2s [list axi_dma_0/mm2s_introut xlconcat_0/In0]
    maybe_connect_bd_net dma_irq_s2mm [list axi_dma_0/s2mm_introut xlconcat_0/In1]
    maybe_connect_bd_net ps_irq0 [list xlconcat_0/dout zynq_ultra_ps_e_0/pl_ps_irq0]

    catch {assign_bd_address -offset 0xA0000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force} msg
    if {[info exists msg] && $msg ne ""} { puts "INFO: AXI-Lite address assignment: $msg" }

    catch {assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force} msg
    if {[info exists msg] && $msg ne ""} { puts "INFO: MM2S address assignment: $msg" }

    catch {assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force} msg
    if {[info exists msg] && $msg ne ""} { puts "INFO: S2MM address assignment: $msg" }

    # Vivado may report PS pl_clk0 as 99999001 Hz, not exactly 100000000 Hz.
    # Use the actual BD clock property everywhere on the stream path before validation.
    set pl_clk_freq [get_bd_pin_freq_hz zynq_ultra_ps_e_0/pl_clk0 $ip_clk_freq_hz]
    align_soc_mlp_axis_metadata $pl_clk_freq $ignore_freq_hz_compat

    validate_bd_design
    save_bd_design

    create_and_set_bd_wrapper $bd_name

    puts "INFO: block design generated: $bd_name"
}

if {$create_ip} {
    set component_xml [package_soc_mlp_ip $ip_repo_dir $ip_vendor $ip_library $ip_name $ip_version $top_name $ip_clk_freq_hz $ignore_freq_hz_compat]
} else {
    puts "WARNING: IP packaging skipped by --skip-ip. BD creation requires an existing $ip_vendor:$ip_library:$ip_name:$ip_version IP."
    set_property ip_repo_paths [list $ip_repo_dir] [current_project]
    update_ip_catalog -rebuild
}

if {$create_bd} {
    create_soc_mlp_block_design $bd_name "$ip_vendor:$ip_library:$ip_name:$ip_version" $ip_clk_freq_hz $ignore_freq_hz_compat
} else {
    puts "WARNING: block design generation skipped by --skip-bd. Synthesis top remains $top_name."
}

# Ensure default runs exist.
if {[get_runs -quiet synth_1] eq ""} {
    set synth_flow [lindex [get_flows "Vivado Synthesis*"] 0]
    if {$synth_flow ne ""} {
        create_run -name synth_1 -flow $synth_flow -strategy "Vivado Synthesis Defaults" -constrset constrs_1
    }
}
if {[get_runs -quiet impl_1] eq "" && [get_runs -quiet synth_1] ne ""} {
    set impl_flow [lindex [get_flows "Vivado Implementation*"] 0]
    if {$impl_flow ne ""} {
        create_run -name impl_1 -parent_run synth_1 -flow $impl_flow -strategy "Vivado Implementation Defaults" -constrset constrs_1
    }
}
if {[get_runs -quiet synth_1] ne ""} { current_run -synthesis [get_runs synth_1] }
if {[get_runs -quiet impl_1] ne ""}  { current_run -implementation [get_runs impl_1] }

if {$run_flow} {
    puts "INFO: Running synthesis/implementation/bitstream"
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs $jobs
    wait_on_run impl_1
}

puts ""
puts "INFO: Vivado project generated successfully."
puts "INFO: Project: $project_dir/$project_name.xpr"
puts "INFO: Topology: __TOPOLOGY__"
puts "INFO: BD: $bd_name"
puts "INFO: Next: open_project $project_dir/$project_name.xpr"
"""

    replacements = {
        "__LAYOUT_NAME__": info.layout_path.name,
        "__TOPOLOGY__": topology,
        "__PROJECT_NAME__": project_name,
        "__TOP_NAME__": top_name,
        "__PART_NAME__": part_name,
        "__BOARD_PART__": board_part_value,
        "__JOBS__": str(int(jobs)),
        "__RTL_LIST__": rtl_list,
    }
    tcl_text = template
    for key, value in replacements.items():
        tcl_text = tcl_text.replace(key, value)

    tcl_path.write_text(tcl_text, encoding="utf-8")
    return tcl_path

def generate_project(
    repo_root: str | Path,
    mem_layout_path: str | Path,
    *,
    clean_generated: bool = True,
    template_rel: str = "src/rtl/src_ultra_sequential",
    generated_rel: str = "src/rtl/generated/src_ultra_sequential",
    project_name: str = "SOC_MLP_ultra96",
    top_name: str = "SOC_MLP",
    part_name: str = "xczu3eg-sbva484-1-e",
    board_part: str | None = None,
    jobs: int = 8,
    dsp_pw: int = 48,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    info = parse_layout(mem_layout_path)
    rtl_dir = copy_and_render_ultra96_sequential_rtl(
        repo_root,
        info,
        template_rel=template_rel,
        generated_rel=generated_rel,
        clean_generated=clean_generated,
        dsp_pw=dsp_pw,
    )
    tcl_path = make_ultra96_sequential_tcl(
        repo_root,
        info,
        rtl_dir,
        project_name=project_name,
        top_name=top_name,
        part_name=part_name,
        board_part=board_part,
        jobs=jobs,
    )
    return {
        "mode": info.mode,
        "topology": info.topology,
        "num_linear": info.num_linear,
        "mem_dir": str(info.mem_dir),
        "layout_path": str(info.layout_path),
        "generated_rtl_dir": str(rtl_dir),
        "generated_tcl": str(tcl_path),
        "project_name": project_name,
        "part_name": part_name,
        "board_part": board_part or "",
        "top_name": top_name,
        "dsp_pw": dsp_pw,
        "weight_file": info.weight_file,
        "bias_file": info.bias_file,
        "activation_file": info.act_file,
        "generated_roms": {
            "weight": str(rtl_dir / "rom_weight_all.sv"),
            "bias": str(rtl_dir / "rom_bias_all.sv"),
            "activation": str(rtl_dir / "rom_act_all.sv"),
        },
        "total_weight_words": info.total_weight_words,
        "total_bias_words": info.total_bias_words,
        "total_activation_words": info.total_activation_words,
    }



# =============================================================================
# Ultra96 mode-specific rendering helpers
# =============================================================================

from dataclasses import field as _dataclass_field

ULTRA96_RTL_FILES_SEQUENTIAL = [
    "dsp.sv",
    "mem_a.sv",
    "mem_b.sv",
    "rom_weight_all.sv",
    "rom_bias_all.sv",
    "rom_act_all.sv",
    "mlp_sequential_ctrl.sv",
    "mlp_sequential_datapath.sv",
    "mlp_sequential.sv",
    "axi_input.sv",
    "axi_output.sv",
    "soc_mlp.sv",
]

ULTRA96_RTL_FILES_PIPELINE_STATIC = [
    "dsp.sv",
    "layer_ctrl.sv",
    "layer_datapath.sv",
    "phase_ctrl.sv",
    "pp_buffer.sv",
    "mlp_pipeline.sv",
    "axi_input.sv",
    "axi_output.sv",
    "soc_mlp.sv",
    "SOC_MLP.sv",
]

ULTRA96_PIPELINE_TEMPLATE_ONLY_FILES = {
    "layer.sv",
    "rom_weight.sv",
    "rom_bias.sv",
    "rom_act.sv",
}

KNOWN_PLACEHOLDERS.update({
    "module_layer_name",
    "module_weight_name",
    "module_bias_name",
    "module_act_name",
    "n_tab_expr",
    "sx_tab_expr",
    "sw_tab_expr",
    "sb_tab_expr",
    "sp_tab_expr",
    "act_tab_expr",
    "lutn_tab_expr",
    "intermediate_buffers",
    "layer_instances",
})

@dataclass(frozen=True)
class LayoutInfo:
    layout_path: Path
    mem_dir: Path
    mode: str
    num_linear: int
    topology: list[int]
    weight_file: str
    bias_file: str
    act_file: str
    total_weight_words: int
    total_bias_words: int
    total_activation_words: int
    weights: list[dict[str, Any]] = _dataclass_field(default_factory=list)
    biases: list[dict[str, Any]] = _dataclass_field(default_factory=list)
    activations: list[dict[str, Any]] = _dataclass_field(default_factory=list)

def infer_mode(layout: dict[str, Any], mode_override: str | None = None) -> str:
    if mode_override:
        mode = mode_override.lower().strip()
    else:
        raw = str(layout.get("mode") or layout.get("format") or "").lower()
        if "pipeline" in raw:
            mode = "pipeline"
        elif "concat" in raw or "sequential" in raw or "weight_all_mem" in layout:
            mode = "sequential"
        else:
            mode = "sequential"

    if mode in {"sequential", "seq", "concat"}:
        return "sequential"
    if mode in {"pipeline", "pipelined"}:
        return "pipeline"
    raise ValueError(f"Unknown FPGA mode: {mode!r}. Expected: sequential or pipeline.")


def _shape_of_weight(item: dict[str, Any]) -> tuple[int, int] | None:
    shape = item.get("shape")
    if shape and len(shape) == 2:
        # PyTorch Linear weight shape = [out_features, in_features]
        return int(shape[0]), int(shape[1])
    return None


def _file_basename(path_like: Any) -> str | None:
    if path_like is None:
        return None
    s = str(path_like).strip()
    if not s or s.lower() == "none":
        return None
    return Path(s).name


def _resolve_mem_path(mem_dir: Path, path_like: Any, default_name: str | None = None) -> Path:
    if path_like is None:
        if default_name is None:
            raise FileNotFoundError("No memory file was provided.")
        return mem_dir / default_name

    p = Path(str(path_like))
    if p.is_absolute():
        return p
    if (mem_dir / p).exists():
        return mem_dir / p
    return mem_dir / p.name


def parse_layout(layout_path: str | Path, mode_override: str | None = None) -> LayoutInfo:
    layout_path = Path(layout_path).resolve()
    layout = _read_json(layout_path)
    mem_dir = layout_path.parent
    mode = infer_mode(layout, mode_override)

    weights = sorted(layout.get("weights", []), key=lambda x: int(x.get("layer_idx", 0)))
    biases = sorted(layout.get("biases", []), key=lambda x: int(x.get("layer_idx", 0)))
    activations = sorted(layout.get("activations", []), key=lambda x: int(x.get("layer_idx", 0)))

    if not weights:
        raise ValueError("mem_layout.json does not contain a 'weights' section.")

    shapes = [_shape_of_weight(w) for w in weights]
    if any(s is None for s in shapes):
        raise ValueError("Each weights[] entry must contain shape=[N_OUT, N_IN].")

    first_out, first_in = shapes[0]  # type: ignore[index]
    topology = [first_in]
    for layer_idx, shape in enumerate(shapes):
        n_out, n_in = shape  # type: ignore[misc]
        if topology[-1] != n_in:
            raise ValueError(
                f"Inconsistent topology at layer {layer_idx}: "
                f"expected N_IN={topology[-1]}, read shape={shape}."
            )
        topology.append(n_out)

    return LayoutInfo(
        layout_path=layout_path,
        mem_dir=mem_dir,
        mode=mode,
        num_linear=len(weights),
        topology=topology,
        weight_file=_file_basename(layout.get("weight_all_mem")) or "weight_all.mem",
        bias_file=_file_basename(layout.get("bias_all_mem")) or "bias_all.mem",
        act_file=_file_basename(layout.get("act_all_mem")) or "act_all.mem",
        total_weight_words=int(layout.get("total_weight_words", sum(int(w.get("length", 0)) for w in weights))),
        total_bias_words=int(layout.get("total_bias_words", sum(int(b.get("length", 0)) for b in biases))),
        total_activation_words=int(layout.get("total_activation_words", sum(int(a.get("length", 0)) for a in activations))),
        weights=weights,
        biases=biases,
        activations=activations,
    )




# Sequential placeholder builder alias.
build_sequential_sv_variables = build_sv_variables

def _csv_expr(items: Iterable[str]) -> str:
    return ", ".join(items)


def _macro_list(prefix: str, count: int, *, start: int = 0) -> str:
    return _csv_expr(f"`{prefix}{i}" for i in range(start, start + count))


def _pipeline_buffer_name(idx: int, signal: str) -> str:
    return f"inter{idx}_{signal}"


def build_intermediate_buffers(info: LayoutInfo) -> str:
    lines: list[str] = []
    n = info.num_linear
    for b in range(max(0, n - 1)):
        depth_macro = _macro(b + 1)
        aw_name = f"INTER{b}_AW"
        lines.extend(
            [
                f"localparam int INTER{b}_DEPTH = {depth_macro};",
                f"localparam int {aw_name} = (INTER{b}_DEPTH > 1) ? $clog2(INTER{b}_DEPTH) : 1;",
                "",
                f"logic                 inter{b}_rd_en;",
                f"logic [{aw_name}-1:0] inter{b}_rd_addr;",
                f"logic signed [ACT_W-1:0] inter{b}_rd_data;",
                "",
                f"logic                 inter{b}_wr_en;",
                f"logic [{aw_name}-1:0] inter{b}_wr_addr;",
                f"logic signed [ACT_W-1:0] inter{b}_wr_data;",
                "",
                "pp_buffer #(",
                "    .ACT_W(ACT_W),",
                f"    .DEPTH(INTER{b}_DEPTH),",
                f"    .AW   ({aw_name})",
                f") u_inter_pp_{b} (",
                "    .clk        (clk),",
                "",
                "    .rd_bank_sel(phase_sel),",
                f"    .rd_en      (inter{b}_rd_en),",
                f"    .rd_addr    (inter{b}_rd_addr),",
                f"    .rd_data    (inter{b}_rd_data),",
                "",
                "    .wr_bank_sel(~phase_sel),",
                f"    .wr_en      (inter{b}_wr_en),",
                f"    .wr_addr    (inter{b}_wr_addr),",
                f"    .wr_data    (inter{b}_wr_data)",
                ");",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _layer_input_ports(layer_idx: int, num_layers: int) -> dict[str, str]:
    if num_layers == 1 or layer_idx == 0:
        return {
            "in_rd_en": "in_rd_en",
            "in_rd_addr": "in_rd_addr",
            "in_rd_data": "in_rd_data",
        }
    prev = layer_idx - 1
    return {
        "in_rd_en": f"inter{prev}_rd_en",
        "in_rd_addr": f"inter{prev}_rd_addr",
        "in_rd_data": f"inter{prev}_rd_data",
    }


def _layer_output_ports(layer_idx: int, num_layers: int) -> dict[str, str]:
    if num_layers == 1 or layer_idx == num_layers - 1:
        return {
            "out_wr_en": "out_wr_en",
            "out_wr_addr": "out_wr_addr",
            "out_wr_data": "out_wr_data",
        }
    return {
        "out_wr_en": f"inter{layer_idx}_wr_en",
        "out_wr_addr": f"inter{layer_idx}_wr_addr",
        "out_wr_data": f"inter{layer_idx}_wr_data",
    }


def build_layer_instances(info: LayoutInfo) -> str:
    lines: list[str] = []
    n = info.num_linear
    for l in range(n):
        p_in = _layer_input_ports(l, n)
        p_out = _layer_output_ports(l, n)
        has_sp = "1'b1" if l < n - 1 else "1'b0"
        sp_expr = _macro(l, "SP") if l < n - 1 else _macro(l + 1, "SX")

        lines.extend(
            [
                f"layer_{l} #(",
                f"    .ACT_TYPE         ({_macro(l, 'ACT')}),",
                f"    .N_IN             ({_macro(l)}),",
                f"    .N_OUT            ({_macro(l + 1)}),",
                "    .ACT_W            (ACT_W),",
                "    .WEIGHT_W         (WEIGHT_W),",
                "    .BIAS_W           (BIAS_W),",
                "    .LEAK_SHIFT       (LEAK_SHIFT_CFG),",
                f"    .HAS_SP           ({has_sp}),",
                f"    .Sx               ({_macro(l, 'SX')}),",
                f"    .Sw               ({_macro(l, 'SW')}),",
                f"    .Sb               ({_macro(l, 'SB')}),",
                f"    .Sp               ({sp_expr}),",
                f"    .Sx_out           ({_macro(l + 1, 'SX')}),",
                f"    .LUT_N            ({_macro(l, 'LUT_N')}),",
                "    .DSP_PW           (DSP_PW)",
                f") u_layer_{l} (",
                "    .clk        (clk),",
                "    .rstn       (rstn),",
                "    .start      (start_phase_pulse),",
                f"    .in_rd_en   ({p_in['in_rd_en']}),",
                f"    .in_rd_addr ({p_in['in_rd_addr']}),",
                f"    .in_rd_data ({p_in['in_rd_data']}),",
                f"    .out_wr_en  ({p_out['out_wr_en']}),",
                f"    .out_wr_addr({p_out['out_wr_addr']}),",
                f"    .out_wr_data({p_out['out_wr_data']}),",
                f"    .done       (layer_done_pulse[{l}])",
                ");",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_pipeline_sv_variables(info: LayoutInfo, scales: dict[str, int], *, dsp_pw: int = 32) -> dict[str, dict[str, str]]:
    """Build placeholder dictionaries for the pipeline templates."""
    n = info.num_linear
    act_width, w_width, b_width = _numeric_widths(scales, dsp_pw)

    common = {
        "dsp_pw_expr": str(int(dsp_pw)),
        "output_size_expr": _macro(n),
        "max_neurons_expr": str(int(max(info.topology))),
    }

    mlp_vars = {
        "n_tab_expr": _macro_list("N", n + 1),
        "sx_tab_expr": _macro_list("SX", n + 1),
        "sw_tab_expr": _macro_list("SW", n),
        "sb_tab_expr": _macro_list("SB", n),
        "sp_tab_expr": _macro_list("SP", max(0, n - 1)) if n > 1 else "0",
        "act_tab_expr": _macro_list("ACT", n),
        "lutn_tab_expr": _macro_list("LUT_N", n),
        "intermediate_buffers": build_intermediate_buffers(info),
        "layer_instances": build_layer_instances(info),
    }

    per_file: dict[str, dict[str, str]] = {
        "common": common,
        "mlp_pipeline.sv": mlp_vars,
        "soc_mlp.sv": common,
    }

    bias_by_idx = {int(b.get("layer_idx", i)): b for i, b in enumerate(info.biases)}
    act_by_idx = {int(a.get("layer_idx", i)): a for i, a in enumerate(info.activations)}

    for i, w_item in enumerate(info.weights):
        layer_idx = int(w_item.get("layer_idx", i))
        b_item = bias_by_idx.get(layer_idx)
        if b_item is None:
            raise ValueError(f"Missing bias entry for layer_idx={layer_idx}")

        w_path = _resolve_mem_path(info.mem_dir, w_item.get("file"), f"weight_layer{layer_idx}.mem")
        b_path = _resolve_mem_path(info.mem_dir, b_item.get("file"), f"bias_layer{layer_idx}.mem")

        w_words = read_mem_words(w_path, w_width)
        b_words = read_mem_words(b_path, b_width)

        _check_word_count(f"weight layer {layer_idx}", w_path, w_words, int(w_item.get("length", len(w_words))))
        _check_word_count(f"bias layer {layer_idx}", b_path, b_words, int(b_item.get("length", len(b_words))))

        layer_name = f"layer_{layer_idx}"
        weight_name = f"rom_weight_{layer_idx}"
        bias_name = f"rom_bias_{layer_idx}"
        act_name = f"rom_act_{layer_idx}"

        per_file[f"layer_{layer_idx}.sv"] = {
            "module_layer_name": layer_name,
            "module_weight_name": weight_name,
            "module_bias_name": bias_name,
            "module_act_name": act_name,
        }
        per_file[f"rom_weight_{layer_idx}.sv"] = {
            "module_weight_name": weight_name,
            "memory_weight": _format_memory_init(w_words, w_width),
        }
        per_file[f"rom_bias_{layer_idx}.sv"] = {
            "module_bias_name": bias_name,
            "memory_bias": _format_memory_init(b_words, b_width),
        }

        a_item = act_by_idx.get(layer_idx)
        act_len = int(a_item.get("length", 0)) if a_item is not None else 0
        act_file = a_item.get("file") if a_item is not None else None
        if act_len > 0:
            a_path = _resolve_mem_path(info.mem_dir, act_file, f"act_layer{layer_idx}.mem")
            a_words = read_mem_words(a_path, act_width)
            _check_word_count(f"activation layer {layer_idx}", a_path, a_words, act_len, allow_empty_as_zero=True)
            per_file[f"rom_act_{layer_idx}.sv"] = {
                "module_act_name": act_name,
                "memory_act_function": _format_memory_init(a_words, act_width),
            }

    return per_file




def _template_file(template_dir: Path, name: str) -> Path:
    """Return a template file, accepting SOC_MLP.sv/soc_mlp.sv case variants."""
    p = template_dir / name
    if p.exists():
        return p
    if name.lower() == "soc_mlp.sv":
        for alt in ("SOC_MLP.sv", "soc_mlp.sv"):
            p = template_dir / alt
            if p.exists():
                return p
    raise FileNotFoundError(f"Template file not found: {template_dir / name}")


def _generated_sv_files(out_dir: Path) -> list[str]:
    return sorted([p.name for p in out_dir.glob("*.sv")])


def copy_and_render_ultra96_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str,
    generated_rel: str,
    clean_generated: bool = True,
    dsp_pw: int = 48,
) -> Path:
    """Render Ultra96 RTL in sequential or pipeline mode."""
    repo_root = Path(repo_root).resolve()
    template_dir = repo_root / template_rel
    if not template_dir.exists():
        raise FileNotFoundError(f"Ultra96 RTL template directory not found: {template_dir}")

    out_dir = repo_root / generated_rel
    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_path = sync_scales_vh(template_dir, out_dir, info.mem_dir)
    scales = validate_scales_against_layout(scales_path, info)

    if info.mode == "sequential":
        return copy_and_render_ultra96_sequential_rtl(
            repo_root,
            info,
            template_rel=template_rel,
            generated_rel=generated_rel,
            clean_generated=False,
            dsp_pw=dsp_pw,
        )

    if info.mode != "pipeline":
        raise ValueError(f"Unsupported RTL mode: {info.mode}")

    per_file_vars = build_pipeline_sv_variables(info, scales, dsp_pw=dsp_pw)
    common = per_file_vars.get("common", {})

    # Render/copy all static SV templates except per-layer ROM/layer templates.
    for src in sorted(template_dir.glob("*.sv")):
        if src.name in ULTRA96_PIPELINE_TEMPLATE_ONLY_FILES:
            continue
        variables = dict(common)
        variables.update(per_file_vars.get(src.name, {}))
        if src.name.lower() == "soc_mlp.sv":
            variables.update(per_file_vars.get("soc_mlp.sv", {}))
            variables.update(per_file_vars.get("SOC_MLP.sv", {}))
        rendered = render_sv_placeholders(src.read_text(encoding="utf-8"), variables, filename=src.name)
        (out_dir / src.name).write_text(rendered, encoding="utf-8")

    # Required per-layer templates.
    layer_tpl = _template_file(template_dir, "layer.sv")
    weight_tpl = _template_file(template_dir, "rom_weight.sv")
    bias_tpl = _template_file(template_dir, "rom_bias.sv")
    act_tpl = template_dir / "rom_act.sv"

    for layer_idx in range(info.num_linear):
        for out_name, tpl in [
            (f"layer_{layer_idx}.sv", layer_tpl),
            (f"rom_weight_{layer_idx}.sv", weight_tpl),
            (f"rom_bias_{layer_idx}.sv", bias_tpl),
        ]:
            variables = dict(common)
            variables.update(per_file_vars.get(out_name, {}))
            rendered = render_sv_placeholders(tpl.read_text(encoding="utf-8"), variables, filename=out_name)
            (out_dir / out_name).write_text(rendered, encoding="utf-8")

        act_name = f"rom_act_{layer_idx}.sv"
        if act_name in per_file_vars:
            if not act_tpl.exists():
                raise FileNotFoundError(f"Activation ROM template not found: {act_tpl}")
            variables = dict(common)
            variables.update(per_file_vars[act_name])
            rendered = render_sv_placeholders(act_tpl.read_text(encoding="utf-8"), variables, filename=act_name)
            (out_dir / act_name).write_text(rendered, encoding="utf-8")

    return out_dir


def _patch_ultra96_tcl_for_generated_dir(tcl_path: Path, *, mode: str, generated_rel: str) -> None:
    """Patch the generated Vivado Tcl for mode-specific generated RTL dir."""
    text = tcl_path.read_text(encoding="utf-8")

    parts = Path(generated_rel).parts
    tcl_join = " ".join(f'"{p}"' for p in parts)
    text = re.sub(
        r'set rtl_dir \[file normalize \[file join \$repo_root "src" "rtl" "generated" "src_ultra_sequential"\]\]',
        f'set rtl_dir [file normalize [file join $repo_root {tcl_join}]]',
        text,
    )
    text = text.replace("# Mode      : sequential", f"# Mode      : {mode}")
    text = text.replace("Ultra96 sequential AXI", f"Ultra96 {mode} AXI")
    text = text.replace("src_ultra_sequential", Path(generated_rel).name)
    tcl_path.write_text(text, encoding="utf-8")


def generate_project(
    repo_root: str | Path,
    mem_layout_path: str | Path,
    *,
    clean_generated: bool = True,
    template_rel: str | None = None,
    generated_rel: str | None = None,
    project_name: str | None = None,
    top_name: str = "SOC_MLP",
    part_name: str = "xczu3eg-sbva484-1-e",
    board_part: str | None = None,
    jobs: int = 8,
    dsp_pw: int = 48,
    fpga_mode: str = "sequential",
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    info = parse_layout(mem_layout_path, mode_override=fpga_mode)

    if template_rel is None:
        template_rel = "src/rtl/src_ultra_pipeline" if info.mode == "pipeline" else "src/rtl/src_ultra_sequential"
    if generated_rel is None:
        generated_rel = "src/rtl/generated/src_ultra_pipeline" if info.mode == "pipeline" else "src/rtl/generated/src_ultra_sequential"
    if project_name is None:
        project_name = f"SOC_MLP_ultra96"

    rtl_dir = copy_and_render_ultra96_rtl(
        repo_root,
        info,
        template_rel=template_rel,
        generated_rel=generated_rel,
        clean_generated=clean_generated,
        dsp_pw=dsp_pw,
    )

    # Reuse the Vivado AXI/IP/BD Tcl generator for both RTL modes.
    # Provide the mode-specific generated SV list while writing the Tcl script.
    global ULTRA96_SEQUENTIAL_RTL_FILES
    old_rtl_files = ULTRA96_SEQUENTIAL_RTL_FILES
    try:
        ULTRA96_SEQUENTIAL_RTL_FILES = _generated_sv_files(rtl_dir)
        tcl_name = f"create_soc_mlp_project_vivado.tcl"
        tcl_path = make_ultra96_sequential_tcl(
            repo_root,
            info,
            rtl_dir,
            tcl_path=repo_root / "vivado" / tcl_name,
            project_name=project_name,
            top_name=top_name,
            part_name=part_name,
            board_part=board_part,
            jobs=jobs,
        )
    finally:
        ULTRA96_SEQUENTIAL_RTL_FILES = old_rtl_files

    _patch_ultra96_tcl_for_generated_dir(tcl_path, mode=info.mode, generated_rel=generated_rel)

    generated_roms: dict[str, Any]
    if info.mode == "pipeline":
        generated_roms = {
            "weight": [str(rtl_dir / f"rom_weight_{i}.sv") for i in range(info.num_linear)],
            "bias": [str(rtl_dir / f"rom_bias_{i}.sv") for i in range(info.num_linear)],
            "activation": sorted(str(p) for p in rtl_dir.glob("rom_act_*.sv")),
        }
    else:
        generated_roms = {
            "weight": str(rtl_dir / "rom_weight_all.sv"),
            "bias": str(rtl_dir / "rom_bias_all.sv"),
            "activation": str(rtl_dir / "rom_act_all.sv"),
        }

    return {
        "mode": info.mode,
        "topology": info.topology,
        "num_linear": info.num_linear,
        "mem_dir": str(info.mem_dir),
        "layout_path": str(info.layout_path),
        "generated_rtl_dir": str(rtl_dir),
        "generated_tcl": str(tcl_path),
        "project_name": project_name,
        "part_name": part_name,
        "board_part": board_part or "",
        "top_name": top_name,
        "dsp_pw": dsp_pw,
        "generated_roms": generated_roms,
        "total_weight_words": info.total_weight_words,
        "total_bias_words": info.total_bias_words,
        "total_activation_words": info.total_activation_words,
    }


NEXYS_SV_CORE_FILES = [
    "dsp.sv",
    "mem_a.sv",
    "mem_b.sv",
    "rom_weight_all.sv",
    "rom_bias_all.sv",
    "rom_act_all.sv",
    "mlp_sequential_ctrl.sv",
    "mlp_sequential_datapath.sv",
    "mlp_sequential.sv",
]

NEXYS_VHDL_FILES = [
    "uart_rx.vhd",
    "uart_tx.vhd",
    "uart_input.vhd",
    "uart_output.vhd",
    "soc_mlp_nexys.vhd",
]


def _nexys_config_package(info: LayoutInfo, scales: dict[str, int], dsp_pw: int) -> str:
    """Expose the scales.vh contract to the VHDL top level."""
    act_width, weight_width, bias_width = _numeric_widths(scales, dsp_pw)
    max_neurons = max(info.topology)
    vec_aw = max(1, (max_neurons - 1).bit_length())
    constants = {
        "ACT_W": act_width,
        "WEIGHT_W": weight_width,
        "BIAS_W": bias_width,
        "DSP_PW": dsp_pw,
        "INPUT_SIZE": info.topology[0],
        "OUTPUT_SIZE": info.topology[-1],
        "NUM_LINEAR": info.num_linear,
        "MAX_NEURONS": max_neurons,
        "VEC_AW": vec_aw,
    }
    lines = "\n".join(f"    constant {name:<11} : positive := {value};" for name, value in constants.items())
    return (
        "-- Auto-generated from mem_layout.json and scales.vh.\n"
        "package mlp_config_pkg is\n"
        f"{lines}\n"
        "end package mlp_config_pkg;\n"
    )


def generate_nexys_sequential(
    repo_root: str | Path,
    mem_layout_path: str | Path,
    *,
    generated_rel: str = "src/rtl/generated/src_nexys_sequential",
    clean_generated: bool = True,
    dsp_pw: int = 48,
) -> dict[str, Any]:
    """Render the sources to add to the Vivado project of the Nexys board.

    The MLP core comes from the Ultra96 sequential SystemVerilog templates, its
    AXI interface being replaced by the VHDL UART link of src_nexys_sequential.
    """
    repo_root = Path(repo_root).resolve()
    info = parse_layout(mem_layout_path, mode_override="sequential")
    core_dir = repo_root / "src" / "rtl" / "src_ultra_sequential"
    vhdl_dir = repo_root / "src" / "rtl" / "src_nexys_sequential"

    out_dir = repo_root / generated_rel
    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_path = sync_scales_vh(core_dir, out_dir, info.mem_dir)
    scales = validate_scales_against_layout(scales_path, info)
    per_file_vars = build_sv_variables(info, scales, dsp_pw=dsp_pw)
    common = per_file_vars.get("common", {})

    for name in NEXYS_SV_CORE_FILES:
        variables = {**common, **per_file_vars.get(name, {})}
        rendered = render_sv_placeholders((core_dir / name).read_text(encoding="utf-8"), variables, filename=name)
        (out_dir / name).write_text(rendered, encoding="utf-8")

    for name in NEXYS_VHDL_FILES:
        shutil.copy2(vhdl_dir / name, out_dir / name)
    (out_dir / "mlp_config_pkg.vhd").write_text(_nexys_config_package(info, scales, dsp_pw), encoding="utf-8")

    sim_dir = out_dir / "sim"
    sim_dir.mkdir(exist_ok=True)
    shutil.copy2(repo_root / "src" / "rtl" / "sim" / "tb_soc_mlp_nexys.vhd", sim_dir)
    for name in ("tb_inputs_stream.mem", "tb_expected_outputs.mem"):
        shutil.copy2(info.mem_dir / name, sim_dir / name)

    return {
        "mode": info.mode,
        "topology": info.topology,
        "num_linear": info.num_linear,
        "mem_dir": str(info.mem_dir),
        "layout_path": str(info.layout_path),
        "generated_rtl_dir": str(out_dir),
        "top_name": "soc_mlp_nexys",
        "dsp_pw": dsp_pw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ultra96 Vivado RTL and TCL from mem_layout.json.")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--layout", default="data/quantized_data/mem_layout.json", help="mem_layout.json path")
    parser.add_argument("--fpga-mode", choices=["sequential", "pipeline"], default="sequential", help="RTL mode")
    parser.add_argument("--template-rel", default=None, help="RTL template directory relative to repo root")
    parser.add_argument("--generated-rel", default=None, help="Generated RTL directory relative to repo root")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean generated RTL dir before rendering")
    parser.add_argument("--project-name", default=None, help="Vivado project name")
    parser.add_argument("--top", default="SOC_MLP", help="Top module")
    parser.add_argument("--part", default="xczu3eg-sbva484-1-e", help="Vivado FPGA part")
    parser.add_argument("--board-part", default=None, help="Optional Vivado board_part")
    parser.add_argument("--jobs", type=int, default=8, help="Vivado launch_runs jobs")
    parser.add_argument("--dsp-pw", type=int, default=48, help="DSP accumulator/result width")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    layout = Path(args.layout)
    if not layout.is_absolute():
        layout = repo / layout

    res = generate_project(
        repo_root=repo,
        mem_layout_path=layout,
        clean_generated=not args.no_clean,
        template_rel=args.template_rel,
        generated_rel=args.generated_rel,
        project_name=args.project_name,
        top_name=args.top,
        part_name=args.part,
        board_part=args.board_part,
        jobs=args.jobs,
        dsp_pw=args.dsp_pw,
        fpga_mode=args.fpga_mode,
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
