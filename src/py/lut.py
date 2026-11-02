"""Search the best LUT address width N for nonlinear activations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

from activations import (
    activation_float,
    activation_needs_lut,
    analyze_last_segment_coverage_from_q,
    build_activation_module,
    default_guard_info_for_analytic,
    print_last_segment_guard_debug,
)
from config import ActivationExport
from export import format_layer_file_tag, save_activation_lut_artifacts
from fixed_point import dequantize_from_int, quantize_to_int
from utils import _read


@dataclass(frozen=True)
class LUTSearchConfig:
    save_artifacts: bool = False
    show_tables: bool = True
    show_best_tables: bool = True
    print_debug: bool = False


def auto_candidate_addr_bits(input_bits: int, min_n: int = 4, max_cap: int = 10, max_words: int = 1024) -> list[int]:
    max_n_by_words = int(np.floor(np.log2(max_words))) if max_words is not None else input_bits - 1
    max_n = min(input_bits - 1, max_cap, max_n_by_words)
    min_n = min(min_n, max_n)
    return list(range(min_n, max_n + 1))


def select_neighbor_addr_bits(candidate_bits: list[int], best_addr_bits: int, window: int = 2) -> list[int]:
    ordered = sorted(set(int(v) for v in candidate_bits))
    idx = ordered.index(int(best_addr_bits))
    lo = max(0, idx - int(window))
    hi = min(len(ordered), idx + int(window) + 1)
    return ordered[lo:hi]


def print_activation_search_debug(
    scored_df: pd.DataFrame,
    *,
    layer_name: str,
    activation_name: str,
    memory_weight: float,
    top_k: int = 10,
):
    cols = [
        "addr_bits",
        "mae",
        "max_abs_err",
        "mae_lsb_out",
        "max_abs_err_lsb_out",
        "memory_bits",
        "mae_normalized",
        "memory_bits_normalized",
        "score",
        "touch_last_segment",
        "touch_count",
        "touch_ratio",
        "headroom_q",
        "max_pre_q",
        "max_safe_q",
    ]
    cols = [col for col in cols if col in scored_df.columns]

    best = scored_df.iloc[0]

    print("\n" + "=" * 100)
    print(f"[Recherche meilleur N] {layer_name} | activation={activation_name}")
    print(
        "Selection criterion : "
        f"score = mae_normalized + {memory_weight:.3f} * memory_bits_normalized"
    )
    print(
        f"Meilleur candidat retenu : "
        f"N={int(best['addr_bits'])} | "
        f"mae={best['mae']:.8f} | "
        f"max_abs_err={best['max_abs_err']:.8f} | "
        f"memory_bits={int(best['memory_bits'])} | "
        f"score={best['score']:.6f}"
    )
    print("Top candidates sorted by score :")
    print(scored_df[cols].head(top_k).to_string(index=False))


def choose_best_activation_candidate(
    df: pd.DataFrame,
    memory_weight: float = 0.35,
) -> pd.Series:
    return score_activation_candidates(df, memory_weight=memory_weight).iloc[0]


def _activation_report_columns() -> list[str]:
    return [
        "layer_name",
        "activation_name",
        "implementation",
        "addr_bits",
        "input_exp",
        "output_exp",
        "mae",
        "max_abs_err",
        "mae_lsb_out",
        "max_abs_err_lsb_out",
        "memory_bits",
        "mae_normalized",
        "memory_bits_normalized",
        "score",
        "touch_last_segment",
        "touch_count",
        "touch_ratio",
        "headroom_q",
        "max_pre_q",
        "max_safe_q",
        "is_best",
    ]


def score_activation_candidates(
    df: pd.DataFrame,
    memory_weight: float = 0.35,
    error_col: str = "mae_lsb_out",
) -> pd.DataFrame:
    """
    Score LUT candidates using accuracy gain versus memory cost.

    The idea is:
    - the smallest LUT is the memory reference;
    - the best MAE is the accuracy reference;
    - a candidate is good only if its MAE improvement justifies its memory cost.

    Lower score is better.
    """

    tmp = df.copy()
    eps = 1e-12

    if error_col not in tmp.columns:
        error_col = "mae"

    tmp = tmp.sort_values("memory_bits").reset_index(drop=True)

    memory_min = float(tmp["memory_bits"].min())
    memory_max = float(tmp["memory_bits"].max())

    error_at_min_memory = float(
        tmp.loc[tmp["memory_bits"].idxmin(), error_col]
    )
    error_best = float(tmp[error_col].min())

    max_quality_gain = error_at_min_memory - error_best
    max_memory_extra = memory_max - memory_min

    if abs(max_quality_gain) < eps:
        tmp["quality_gain_normalized"] = 0.0
    else:
        tmp["quality_gain_normalized"] = (
            error_at_min_memory - tmp[error_col]
        ) / max_quality_gain

    if abs(max_memory_extra) < eps:
        tmp["memory_extra_normalized"] = 0.0
    else:
        tmp["memory_extra_normalized"] = (
            tmp["memory_bits"] - memory_min
        ) / max_memory_extra

    tmp["score"] = (
        1.0
        - tmp["quality_gain_normalized"]
        + float(memory_weight) * tmp["memory_extra_normalized"]
    )

    # Compatibility with the old column names
    tmp["mae_normalized"] = 1.0 - tmp["quality_gain_normalized"]
    tmp["memory_bits_normalized"] = tmp["memory_extra_normalized"]

    return tmp.sort_values(
        ["score", error_col, "mae", "max_abs_err", "memory_bits", "addr_bits"]
    ).reset_index(drop=True)


def run_search_best_activation_n(
    basic_results: dict[str, Any],
    *,
    save_artifacts: bool = False,
    show_tables: bool = True,
    show_best_tables: bool = True,
    print_debug: bool = False,
) -> dict[str, Any]:
    """
    Automatically search the best N (= addr_bits) for each activation LUT.

    - Analytical activations (relu, leaky_relu, identity) are evaluated with a pure integer implementation.
    - LUT activations (sigmoid, tanh, etc.) test all candidate N values.
    - The best N is selected using an accuracy / memory trade-off.
    """
    qcfg = basic_results["qcfg"]
    paths = basic_results["paths"]
    tensors = basic_results["tensors"]
    quantized_layers = basic_results["quantized_layers"]

    activation_dir = paths.output_dir / qcfg.activation_artifact_subdir
    if save_artifacts:
        activation_dir.mkdir(parents=True, exist_ok=True)

    activation_modules: dict[int, Any] = {}
    activation_candidate_rows: list[dict[str, Any]] = []
    activation_best_rows: list[dict[str, Any]] = []

    n_hidden = sum(0 if _read(layer, "is_output") else 1 for layer in quantized_layers)

    for hidden_idx in range(n_hidden):
        layer_info = quantized_layers[hidden_idx]
        layer_name = _read(layer_info, "name")
        layer_tag = format_layer_file_tag(layer_name)
        act_name = _read(layer_info, "activation_name")
        sx = int(_read(layer_info, "preact_exp"))
        sy = int(_read(layer_info, "act_exp"))

        pre_float = tensors[f"layer_{hidden_idx}_preact"]
        pre_q = quantize_to_int(pre_float, sx, num_bits=qcfg.act_bits).astype(np.int64)
        ideal_float = activation_float(pre_float, act_name, negative_slope=qcfg.activation_negative_slope)
        ideal_q = quantize_to_int(ideal_float, sy, num_bits=qcfg.act_bits).astype(np.int64)

        if not activation_needs_lut(act_name):
            module = build_activation_module(
                act_name,
                act_bits=qcfg.act_bits,
                input_exp=sx,
                output_exp=sy,
                qcfg=qcfg,
                addr_bits=None,
            )
            approx_q = module.forward_int(pre_q).astype(np.int64)
            approx_float = dequantize_from_int(approx_q, sy)
            abs_err = np.abs(approx_float - ideal_float)
            abs_q_err = np.abs(approx_q - ideal_q.astype(np.int64))

            report = ActivationExport(
                layer_name=layer_name,
                activation_name=act_name,
                implementation="analytic_int",
                act_bits=qcfg.act_bits,
                addr_bits=None,
                signed_input=True,
                input_exp=sx,
                output_exp=sy,
                x_min=float(pre_q.min() / (2 ** sx)),
                x_max=float(pre_q.max() / (2 ** sx)),
                memory_words=0,
                memory_bits=0,
                mae=float(abs_err.mean()),
                max_abs_err=float(abs_err.max()),
                mae_lsb_out=float(abs_q_err.mean()),
                max_abs_err_lsb_out=float(abs_q_err.max()),
                last_segment_strategy="not_applicable",
                paths={},
            )

            row = asdict(report)
            row["layer_tag"] = layer_tag
            row["is_best"] = True
            row["selected_for_comparison"] = True
            row.update(default_guard_info_for_analytic(pre_q))
            row["mae_normalized"] = np.nan
            row["memory_bits_normalized"] = np.nan
            row["score"] = np.nan

            if print_debug:
                print("\n" + "=" * 90)
                print(f"[Recherche meilleur N] {layer_name} | activation={act_name}")
                print("Analytical integer activation: no LUT and no N search.")
                print(
                    f"mae={row['mae']:.8f} | "
                    f"max_abs_err={row['max_abs_err']:.8f} | "
                    f"memory_bits={row['memory_bits']}"
                )

            activation_modules[hidden_idx] = module
            activation_candidate_rows.append(row.copy())
            activation_best_rows.append(row.copy())
            continue

        candidate_addr_bits = auto_candidate_addr_bits(
            input_bits=qcfg.act_bits,
            min_n=qcfg.lut_addr_bits_min,
            max_cap=qcfg.lut_addr_bits_max_cap,
            max_words=qcfg.lut_max_words,
        )

        local_rows: list[dict[str, Any]] = []
        module_by_n: dict[int, Any] = {}

        for addr_bits in candidate_addr_bits:
            module = build_activation_module(
                act_name,
                act_bits=qcfg.act_bits,
                input_exp=sx,
                output_exp=sy,
                qcfg=qcfg,
                addr_bits=addr_bits,
            )

            guard_info = analyze_last_segment_coverage_from_q(
                pre_q,
                input_bits=qcfg.act_bits,
                addr_bits=addr_bits,
            )

            approx_q = module.forward_int(pre_q).astype(np.int64)
            approx_float = dequantize_from_int(approx_q, sy)
            abs_err = np.abs(approx_float - ideal_float)
            abs_q_err = np.abs(approx_q - ideal_q.astype(np.int64))

            memory_words = len(module.y_knots_q)
            memory_bits = memory_words * qcfg.act_bits

            report_paths = {}

            if save_artifacts:
                candidate_dir = activation_dir / layer_tag / f"N_{addr_bits}"

                report_paths = save_activation_lut_artifacts(
                    candidate_dir=candidate_dir,
                    module=module,
                    qcfg=qcfg,
                    layer_name=layer_name,
                    layer_tag=layer_tag,
                    act_name=act_name,
                    addr_bits=addr_bits,
                    sx=sx,
                    sy=sy,
                    memory_words=memory_words,
                    memory_bits=memory_bits,
                    abs_err=abs_err,
                    save_debug_csv=False,
                )

            report = ActivationExport(
                layer_name=layer_name,
                activation_name=act_name,
                implementation="paper_fit",
                act_bits=qcfg.act_bits,
                addr_bits=addr_bits,
                signed_input=True,
                input_exp=sx,
                output_exp=sy,
                x_min=float(module.knots_df["x"].min()),
                x_max=float(module.knots_df["x"].max()),
                memory_words=memory_words,
                memory_bits=memory_bits,
                mae=float(abs_err.mean()),
                max_abs_err=float(abs_err.max()),
                mae_lsb_out=float(abs_q_err.mean()),
                max_abs_err_lsb_out=float(abs_q_err.max()),
                last_segment_strategy=qcfg.last_segment_strategy,
                paths=report_paths,
            )

            row = asdict(report)
            row["layer_tag"] = layer_tag
            row["is_best"] = False
            row["selected_for_comparison"] = False
            row.update(guard_info)

            local_rows.append(row.copy())
            module_by_n[addr_bits] = module

        local_df = pd.DataFrame(local_rows).sort_values("addr_bits").reset_index(drop=True)
        local_df["guard_valid"] = ~local_df["touch_last_segment"].astype(bool)

        # The sequential RTL intentionally leaves the positive last segment
        # unsafe (next address wraps into the negative half).  V2 never fixes
        # this in RTL: candidates that touch it are simply invalid.
        valid_local_df = local_df.loc[local_df["guard_valid"]].copy()
        if valid_local_df.empty:
            raise ValueError(
                f"No valid LUT candidate for {layer_name} ({act_name}): "
                "all candidate address widths touch the forbidden last segment."
            )

        scored_local_df = score_activation_candidates(
            valid_local_df,
            memory_weight=qcfg.activation_score_memory_weight,
        )

        best = scored_local_df.iloc[0]
        best_addr_bits = int(best["addr_bits"])
        valid_addr_bits = sorted(int(v) for v in valid_local_df["addr_bits"].tolist())

        compare_addr_bits = select_neighbor_addr_bits(
            valid_addr_bits,
            best_addr_bits,
            window=qcfg.lut_auto_window,
        )

        if print_debug:
            print("\nForbidden-zone coverage detail :")
            for _, dbg_row in local_df.sort_values("addr_bits").iterrows():
                print_last_segment_guard_debug(
                    layer_name=layer_name,
                    activation_name=act_name,
                    addr_bits=int(dbg_row["addr_bits"]),
                    guard_info=dbg_row.to_dict(),
                )

        activation_modules[hidden_idx] = module_by_n[best_addr_bits]

        scored_by_n = {int(r["addr_bits"]): r for _, r in scored_local_df.iterrows()}
        for row in local_rows:
            row_addr_bits = int(row["addr_bits"])
            scored_row = scored_by_n.get(row_addr_bits)

            enriched_row = row.copy()
            enriched_row["guard_valid"] = not bool(row["touch_last_segment"])
            enriched_row["is_best"] = row_addr_bits == best_addr_bits
            enriched_row["selected_for_comparison"] = row_addr_bits in compare_addr_bits
            if scored_row is None:
                enriched_row["mae_normalized"] = np.nan
                enriched_row["memory_bits_normalized"] = np.nan
                enriched_row["score"] = np.inf
            else:
                enriched_row["mae_normalized"] = float(scored_row["mae_normalized"])
                enriched_row["memory_bits_normalized"] = float(scored_row["memory_bits_normalized"])
                enriched_row["score"] = float(scored_row["score"])

            activation_candidate_rows.append(enriched_row.copy())

            if enriched_row["is_best"]:
                activation_best_rows.append(enriched_row.copy())

    activation_candidate_df = pd.DataFrame(activation_candidate_rows).sort_values(
        ["layer_name", "addr_bits"], na_position="first"
    ).reset_index(drop=True)
    activation_best_df = pd.DataFrame(activation_best_rows).sort_values(
        ["layer_name", "addr_bits"], na_position="first"
    ).reset_index(drop=True)
    activation_report_df = activation_candidate_df.copy()

    activation_guard_df = activation_candidate_df[
    [
        col for col in [
            "layer_name",
            "activation_name",
            "addr_bits",
            "touch_last_segment",
            "touch_count",
            "total_count",
            "touch_ratio",
            "headroom_q",
            "max_pre_q",
            "max_safe_q",
            "guard_start_q",
            "reserved_codes",
            "memory_bits",
            "mae",
            "max_abs_err",
            "score",
        ]
        if col in activation_candidate_df.columns]].copy()
    
    if show_tables:
        guard_display = activation_guard_df.copy()

        if "touch_ratio" in guard_display.columns:
            guard_display["touch_ratio"] = guard_display["touch_ratio"].map(
                lambda x: f"{100.0 * x:.4f}%" if pd.notna(x) else "-"
            )

        guard_display = guard_display.sort_values(["layer_name", "addr_bits"], na_position="first").reset_index(drop=True)
        display(guard_display)

    if show_best_tables and len(activation_best_df):
        cols = [col for col in _activation_report_columns() if col in activation_best_df.columns]
        display(activation_best_df[cols].sort_values(["layer_name", "addr_bits"], na_position="first"))

    return {
        **basic_results,
        "activation_modules": activation_modules,
        "activation_candidate_df": activation_candidate_df,
        "activation_best_df": activation_best_df,
        "activation_report_df": activation_report_df,
        "activation_guard_df": activation_guard_df,
    }


def search_best_lut_size(
    quant_results: dict[str, Any],
    cfg: LUTSearchConfig | None = None,
) -> dict[str, Any]:
    """Friendly wrapper around the LUT search step."""
    cfg = cfg or LUTSearchConfig()
    return run_search_best_activation_n(
        quant_results,
        save_artifacts=cfg.save_artifacts,
        show_tables=cfg.show_tables,
        show_best_tables=cfg.show_best_tables,
        print_debug=cfg.print_debug,
    )

