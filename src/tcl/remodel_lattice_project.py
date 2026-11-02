"""Generate a Lattice Radiant TCL project and concrete RTL from mem_layout.json.

This helper supports both Lattice/Radiant RTL flows:

- sequential:
    concatenated ROM memories generated from weight_all.mem, bias_all.mem
    and act_all.mem.

- pipeline:
    one generated layer module per MLP layer, with one generated weight ROM and
    one generated bias ROM per layer, plus one generated activation ROM per LUT
    layer when required.

The editable RTL files are treated as SystemVerilog templates. Only explicit
placeholders such as {memory_weight}, {module_weight_name}, {layer_instances},
etc. are replaced; normal SystemVerilog braces are left untouched.

Expected repository layout:
  <repo>/src/rtl/src_lattice_sequential/*.sv
  <repo>/src/rtl/src_lattice_pipeline/*.sv
  <repo>/data/quantized_data/mem_layout.json
  <repo>/data/quantized_data/scales.vh
  <repo>/data/quantized_data/*.mem
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


LATTICE_RTL_FILES_SEQUENTIAL = [
    "dsp.sv",
    "mem_a.sv",
    "mem_b.sv",
    "rom_weight_all.sv",
    "rom_bias_all.sv",
    "rom_act_all.sv",
    "mlp_sequential_ctrl.sv",
    "mlp_sequential_datapath.sv",
    "mlp_sequential.sv",
    "uart_rx.sv",
    "uart_tx.sv",
    "uart_input.sv",
    "uart_output.sv",
    "soc_mlp.sv",
]

# Static pipeline templates copied/rendered once.
# Per-layer generated files are created separately from layer.sv/rom_*.sv templates.
LATTICE_RTL_FILES_PIPELINE_STATIC = [
    "dsp.sv",
    "layer_ctrl.sv",
    "layer_datapath.sv",
    "phase_ctrl.sv",
    "pp_buffer.sv",
    "mlp_pipeline.sv",
    "uart_rx.sv",
    "uart_tx.sv",
    "uart_input.sv",
    "uart_output.sv",
    "soc_mlp.sv",
]

LATTICE_PIPELINE_TEMPLATE_ONLY_FILES = [
    "layer.sv",
    "rom_weight.sv",
    "rom_bias.sv",
    "rom_act.sv",
]

LATTICE_TCL_ONLY_FILES = [
    "scales.vh",
]

KNOWN_PLACEHOLDERS = {
    # Sequential memories and generated cases.
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

    # Pipeline structure.
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
    weights: list[dict[str, Any]] = field(default_factory=list)
    biases: list[dict[str, Any]] = field(default_factory=list)
    activations: list[dict[str, Any]] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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
            raise FileNotFoundError("No memory file provided.")
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
        raise ValueError(f"Macro `{name}` is missing from {scales_path}")
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
        raise ValueError("The Lattice UART transport requires ACT_W <= 16")
    if max(act_width, weight_width, bias_width) > header_dsp_width:
        raise ValueError("DSP_PW must cover ACT_W, WEIGHT_W and BIAS_W")
    if header_dsp_width != int(dsp_pw):
        raise ValueError(
            f"DSP_PW mismatch: scales.vh gives {header_dsp_width}, "
            f"but the Lattice profile requires {dsp_pw}."
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


def build_sequential_sv_variables(info: LayoutInfo, scales: dict[str, int], *, dsp_pw: int = 32) -> dict[str, dict[str, str]]:
    """Build per-file placeholder dictionaries for the sequential templates."""
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
            raise ValueError(f"Missing bias for layer_idx={layer_idx}")

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


def render_sv_placeholders(text: str, variables: dict[str, str], *, filename: str) -> str:
    """Replace only explicit {placeholder} tokens.

    This avoids Python str.format so SystemVerilog concatenations/replications such
    as {a, b} or {DATA_W{1'b1}} remain untouched.
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
        f"scales.vh not found in {mem_dir} and in {template_dir}."
    )


def _mode_rtl_files(mode: str) -> list[str]:
    if mode == "sequential":
        return LATTICE_RTL_FILES_SEQUENTIAL
    if mode == "pipeline":
        return LATTICE_RTL_FILES_PIPELINE_STATIC
    raise ValueError(f"Unknown mode: {mode}")


def copy_and_render_lattice_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str = "src/rtl/src_lattice_sequential",
    clean_generated: bool = True,
    dsp_pw: int = 32,
) -> Path:
    repo_root = Path(repo_root).resolve()
    template_dir = repo_root / template_rel
    if not template_dir.exists():
        raise FileNotFoundError(f"Lattice RTL template directory not found: {template_dir}")

    rtl_files = _mode_rtl_files(info.mode)
    missing = [f for f in rtl_files if not (template_dir / f).exists()]

    if info.mode == "pipeline":
        missing.extend([f for f in LATTICE_PIPELINE_TEMPLATE_ONLY_FILES if not (template_dir / f).exists()])

    if missing:
        raise FileNotFoundError(f"Missing Lattice RTL files in {template_dir}: {missing}")

    out_dir = repo_root / "src" / "rtl" / "generated" / f"src_lattice_{info.mode}"
    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_path = sync_scales_vh(template_dir, out_dir, info.mem_dir)
    scales = validate_scales_against_layout(scales_path, info)

    if info.mode == "sequential":
        per_file_vars = build_sequential_sv_variables(info, scales, dsp_pw=dsp_pw)
    else:
        per_file_vars = build_pipeline_sv_variables(info, scales, dsp_pw=dsp_pw)

    common = per_file_vars.get("common", {})

    for name in rtl_files:
        src = template_dir / name
        text = src.read_text(encoding="utf-8")
        variables = dict(common)
        variables.update(per_file_vars.get(name, {}))
        rendered = render_sv_placeholders(text, variables, filename=name)
        (out_dir / name).write_text(rendered, encoding="utf-8")

    if info.mode == "pipeline":
        layer_template = (template_dir / "layer.sv").read_text(encoding="utf-8")
        weight_template = (template_dir / "rom_weight.sv").read_text(encoding="utf-8")
        bias_template = (template_dir / "rom_bias.sv").read_text(encoding="utf-8")
        act_template = (template_dir / "rom_act.sv").read_text(encoding="utf-8")

        for i in range(info.num_linear):
            for out_name, template_text in [
                (f"layer_{i}.sv", layer_template),
                (f"rom_weight_{i}.sv", weight_template),
                (f"rom_bias_{i}.sv", bias_template),
            ]:
                variables = dict(common)
                variables.update(per_file_vars.get(out_name, {}))
                rendered = render_sv_placeholders(template_text, variables, filename=out_name)
                (out_dir / out_name).write_text(rendered, encoding="utf-8")

            act_out_name = f"rom_act_{i}.sv"
            if act_out_name in per_file_vars:
                variables = dict(common)
                variables.update(per_file_vars[act_out_name])
                rendered = render_sv_placeholders(act_template, variables, filename=act_out_name)
                (out_dir / act_out_name).write_text(rendered, encoding="utf-8")

    # In sequential mode, copy/render any additional helper .sv file found in the
    # template directory. In pipeline mode, template-only files are rendered through
    # the dedicated per-layer generation loop.
    if info.mode == "sequential":
        for src in sorted(template_dir.glob("*.sv")):
            if src.name in rtl_files:
                continue
            variables = dict(common)
            variables.update(per_file_vars.get(src.name, {}))
            rendered = render_sv_placeholders(src.read_text(encoding="utf-8"), variables, filename=src.name)
            (out_dir / src.name).write_text(rendered, encoding="utf-8")

    return out_dir


def copy_and_patch_lattice_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str = "src/rtl/src_lattice_sequential",
    text_template_rel: str | None = None,
    clean_generated: bool = True,
    dsp_pw: int = 32,
) -> Path:
    """Render the Lattice RTL templates and return the generated RTL directory.

    `text_template_rel` is accepted by the function signature but is ignored;
    placeholders are read directly from the RTL template files.
    """
    return copy_and_render_lattice_rtl(
        repo_root,
        info,
        template_rel=template_rel,
        clean_generated=clean_generated,
        dsp_pw=dsp_pw,
    )


def _discover_pdc_files(repo_root: Path, template_rel: str) -> list[Path]:
    candidates: list[Path] = []
    for rel in [template_rel, "constraints", "lattice", "."]:
        base = repo_root / rel
        if base.exists():
            candidates.extend(sorted(base.glob("*.pdc")))

    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _rtl_files_for_tcl(rtl_dir: Path) -> list[str]:
    names = ["scales.vh"]
    names.extend(sorted(p.name for p in rtl_dir.glob("*.sv")))
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if (rtl_dir / name).exists() and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def make_lattice_tcl(
    repo_root: str | Path,
    info: LayoutInfo,
    rtl_dir: Path,
    *,
    template_rel: str = "src/rtl/src_lattice_sequential",
    tcl_path: str | Path | None = None,
    project_name: str = "SOC_MLP_lattice",
    impl_name: str = "impl_1",
    device: str = "iCE40UP5K-SG48I",
    synthesis: str = "synplify",
    top_name: str = "SOC_MLP",
    run_flow: bool = False,
) -> Path:
    repo_root = Path(repo_root).resolve()
    rtl_dir = Path(rtl_dir).resolve()
    tcl_path = Path(tcl_path or (repo_root / "lattice" / "create_soc_mlp_project_lattice.tcl"))
    tcl_path.parent.mkdir(parents=True, exist_ok=True)

    pdc_files = _discover_pdc_files(repo_root, template_rel)
    pdc_rel = []
    for p in pdc_files:
        try:
            pdc_rel.append(p.resolve().relative_to(repo_root).as_posix())
        except ValueError:
            pdc_rel.append(str(p.resolve()))

    tcl_source_files = _rtl_files_for_tcl(rtl_dir)

    rtl_list = "\n".join(f"    {f}" for f in tcl_source_files)
    pdc_list = "\n".join(f"    {p}" for p in pdc_rel)
    topology = " -> ".join(map(str, info.topology))

    try:
        rtl_dir_rel = rtl_dir.relative_to(repo_root).as_posix()
    except ValueError:
        rtl_dir_rel = str(rtl_dir)

    tcl_text = f"""# -----------------------------------------------------------------------------
# Auto-generated Lattice Radiant project script.
# Layout    : {info.layout_path.name}
# Topology  : {topology}
# Mode      : {info.mode}
# -----------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]

set project_name "{project_name}"
set impl_name    "{impl_name}"
set device       "{device}"
set synthesis    "{synthesis}"
set top_name     "{top_name}"
set run_flow     {int(run_flow)}

foreach arg $argv {{
    if {{$arg eq "--run"}} {{ set run_flow 1 }}
}}

set project_dir [file normalize [file join $repo_root "lattice" "build" $project_name]]
set rtl_dir     [file normalize [file join $repo_root "{rtl_dir_rel}"]]

puts "INFO: repo_root    = $repo_root"
puts "INFO: project_name = $project_name"
puts "INFO: project_dir  = $project_dir"
puts "INFO: impl_name    = $impl_name"
puts "INFO: device       = $device"
puts "INFO: synthesis    = $synthesis"
puts "INFO: top          = $top_name"
puts "INFO: mode         = {info.mode}"
puts "INFO: rtl_dir      = $rtl_dir"

if {{[file exists $project_dir]}} {{
    file delete -force $project_dir
}}
file mkdir $project_dir

# Radiant command names can differ slightly between versions. The script keeps
# the generated source list explicit and stable so it can also be reused manually.
prj_create -name $project_name -impl $impl_name -dev $device -synthesis $synthesis -dir $project_dir

set rtl_files {{
{rtl_list}
}}

foreach src $rtl_files {{
    set src_path [file normalize [file join $rtl_dir $src]]
    puts "INFO: add source $src_path"
    prj_add_source $src_path
}}

set pdc_files {{
{pdc_list}
}}

foreach pdc $pdc_files {{
    if {{[file pathtype $pdc] eq "absolute"}} {{
        set pdc_path [file normalize $pdc]
    }} else {{
        set pdc_path [file normalize [file join $repo_root $pdc]]
    }}
    if {{[file exists $pdc_path]}} {{
        puts "INFO: add PDC $pdc_path"
        prj_add_source $pdc_path
    }}
}}

# Set the intended top when supported by the installed Radiant version.
catch {{prj_set_impl_opt -impl $impl_name top $top_name}}
catch {{prj_set_top $top_name}}

if {{$run_flow}} {{
    puts "INFO: Running implementation flow"
    prj_run Synthesis -impl $impl_name
    prj_run Map       -impl $impl_name
    prj_run PAR       -impl $impl_name
    prj_run Export    -impl $impl_name
}}
"""

    tcl_path.write_text(tcl_text, encoding="utf-8")
    return tcl_path


def generate_project(
    repo_root: str | Path,
    mem_layout_path: str | Path,
    *,
    fpga_mode: str | None = None,
    clean_generated: bool = True,
    template_rel: str = "src/rtl/src_lattice_sequential",
    text_template_rel: str | None = None,
    project_name: str = "SOC_MLP_lattice",
    impl_name: str = "impl_1",
    device: str = "iCE40UP5K-SG48I",
    synthesis: str = "synplify",
    top_name: str = "SOC_MLP",
    dsp_pw: int = 32,
    run_flow: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    info = parse_layout(mem_layout_path, mode_override=fpga_mode)

    rtl_dir = copy_and_render_lattice_rtl(
        repo_root,
        info,
        template_rel=template_rel,
        clean_generated=clean_generated,
        dsp_pw=dsp_pw,
    )
    tcl_path = make_lattice_tcl(
        repo_root,
        info,
        rtl_dir,
        template_rel=template_rel,
        project_name=project_name,
        impl_name=impl_name,
        device=device,
        synthesis=synthesis,
        top_name=top_name,
        run_flow=run_flow,
    )

    generated_roms: dict[str, Any]
    if info.mode == "sequential":
        generated_roms = {
            "weight": str(rtl_dir / "rom_weight_all.sv"),
            "bias": str(rtl_dir / "rom_bias_all.sv"),
            "activation": str(rtl_dir / "rom_act_all.sv"),
        }
    else:
        generated_roms = {
            "weights": [str(p) for p in sorted(rtl_dir.glob("rom_weight_*.sv"))],
            "biases": [str(p) for p in sorted(rtl_dir.glob("rom_bias_*.sv"))],
            "activations": [str(p) for p in sorted(rtl_dir.glob("rom_act_*.sv"))],
            "layers": [str(p) for p in sorted(rtl_dir.glob("layer_*.sv"))],
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
        "impl_name": impl_name,
        "device": device,
        "synthesis": synthesis,
        "top_name": top_name,
        "dsp_pw": dsp_pw,
        "weight_file": info.weight_file,
        "bias_file": info.bias_file,
        "activation_file": info.act_file,
        "generated_roms": generated_roms,
        "total_weight_words": info.total_weight_words,
        "total_bias_words": info.total_bias_words,
        "total_activation_words": info.total_activation_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lattice/Radiant RTL and TCL from mem_layout.json.")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--layout", default="data/quantized_data/mem_layout.json", help="mem_layout.json path")
    parser.add_argument("--fpga-mode", default=None, choices=["sequential", "pipeline"], help="Override FPGA mode from mem_layout.json")
    parser.add_argument("--template-rel", default=None, help="RTL template directory relative to repo root")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean generated RTL dir before rendering")
    parser.add_argument("--text-template-rel", default=None, help="Accepted but ignored; placeholders are read from RTL templates")
    parser.add_argument("--project-name", default="SOC_MLP_lattice", help="Radiant project name")
    parser.add_argument("--impl", default="impl_1", help="Radiant implementation name")
    parser.add_argument("--device", default="iCE40UP5K-SG48I", help="Lattice device OPN")
    parser.add_argument("--synthesis", default="synplify", help="Radiant synthesis tool")
    parser.add_argument("--top", default="SOC_MLP", help="Top module")
    parser.add_argument("--dsp-pw", type=int, default=32, help="DSP accumulator/result width propagated from SOC_MLP")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    layout = Path(args.layout)
    if not layout.is_absolute():
        layout = repo / layout

    layout_dict = _read_json(layout)
    mode = infer_mode(layout_dict, args.fpga_mode)
    template_rel = args.template_rel or ("src/rtl/src_lattice_pipeline" if mode == "pipeline" else "src/rtl/src_lattice_sequential")

    res = generate_project(
        repo_root=repo,
        mem_layout_path=layout,
        fpga_mode=mode,
        clean_generated=not args.no_clean,
        template_rel=template_rel,
        text_template_rel=args.text_template_rel,
        project_name=args.project_name,
        impl_name=args.impl,
        device=args.device,
        synthesis=args.synthesis,
        top_name=args.top,
        dsp_pw=args.dsp_pw,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
