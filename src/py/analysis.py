"""Plots, tables and PDF reports for quantization analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

from fixed_point import int_bounds
from utils import _cfg_value, _read, _to_numpy
from activations import activation_float


def collect_activation_stats(model, x_np, activation_plan, activation_float_fn):
    x = torch.tensor(x_np, dtype=torch.float32)
    stats = []
    current = x.clone()

    for layer_index, layer in enumerate(model.hidden_layers):
        linear_input = _to_numpy(current).reshape(-1)
        linear_output_t = layer(current)
        linear_output = _to_numpy(linear_output_t).reshape(-1)

        act_name = activation_plan[layer_index]["name"]
        activation_output = activation_float_fn(_to_numpy(linear_output_t), act_name).reshape(-1)

        stats.append(
            {
                "layer_name": f"Layer {layer_index}",
                "activation_name": act_name,
                "linear_input": linear_input,
                "linear_output": linear_output,
                "activation_output": activation_output,
            }
        )

        current = torch.tensor(activation_output.reshape(linear_output_t.shape), dtype=torch.float32)

    final_linear_input = _to_numpy(current).reshape(-1)
    final_linear_output = _to_numpy(model.out(current)).reshape(-1)
    stats.append(
        {
            "layer_name": "Output",
            "activation_name": "none",
            "linear_input": final_linear_input,
            "linear_output": final_linear_output,
            "activation_output": None,
        }
    )
    return stats


def _build_weight_bias_stats(layer_name: str, weight_tensor, bias_tensor) -> dict[str, Any]:
    weights = _to_numpy(weight_tensor).reshape(-1)
    biases = _to_numpy(bias_tensor).reshape(-1)
    return {
        "layer_name": layer_name,
        "weights": weights,
        "biases": biases,
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_shape": tuple(_to_numpy(weight_tensor).shape),
        "weight_sparsity": float(np.mean(np.isclose(weights, 0.0)) * 100.0),
        "bias_mean": float(np.mean(biases)),
        "bias_median": float(np.median(biases)),
        "bias_min": float(np.min(biases)),
        "bias_max": float(np.max(biases)),
        "bias_shape": tuple(_to_numpy(bias_tensor).shape),
    }


def collect_layer_param_stats(model) -> list[dict[str, Any]]:
    layer_stats = [
        _build_weight_bias_stats(f"Layer {layer_index}", layer.weight, layer.bias)
        for layer_index, layer in enumerate(model.hidden_layers)
    ]
    layer_stats.append(_build_weight_bias_stats("Output", model.out.weight, model.out.bias))
    return layer_stats


def _add_histogram_with_stats(
    ax,
    values,
    *,
    bins: int,
    color,
    xlabel: str,
    ylabel: str,
    title: str,
    show_negative_ratio: bool = False,
    legend_loc: str = "best",
):
    values = _to_numpy(values).reshape(-1)

    ax.hist(values, bins=bins, alpha=0.7, color=color, density=True, edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(values), color="red", linestyle="--", linewidth=2, label=f"Moy: {np.mean(values):.4f}")
    ax.axvline(np.median(values), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(values):.4f}")
    ax.axvline(np.min(values), color="blue", linestyle=":", linewidth=1.5, label=f"Min: {np.min(values):.4f}")
    ax.axvline(np.max(values), color="blue", linestyle=":", linewidth=1.5, label=f"Max: {np.max(values):.4f}")

    if show_negative_ratio:
        neg_ratio = (values < 0).mean() * 100
        title = f"{title} | Negative values: {neg_ratio:.1f}%"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc=legend_loc)
    ax.grid(True, alpha=0.3)


def plot_weight_bias_distributions(layer_stats: list[dict[str, Any]]):
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(layer_stats)))
    n_layers = len(layer_stats)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 8 * n_rows))
    fig.suptitle("Detailed per-layer distributions - Weights and Biases", fontsize=18, fontweight="bold", y=0.98)

    for index, stats in enumerate(layer_stats):
        col = index % n_cols
        row_offset = (index // n_cols) * 2

        ax_weights = plt.subplot(2 * n_rows, n_cols, row_offset * n_cols + col + 1)
        _add_histogram_with_stats(
            ax_weights,
            stats["weights"],
            bins=50,
            color=colors[index],
            xlabel="Weight values",
            ylabel="Density",
            title=f"{stats['layer_name']} - WEIGHTS\nShape: {stats['weight_shape']}, Sparsity: {stats['weight_sparsity']:.2f}%",
            legend_loc="upper right",
        )

        ax_biases = plt.subplot(2 * n_rows, n_cols, (row_offset + 1) * n_cols + col + 1)
        _add_histogram_with_stats(
            ax_biases,
            stats["biases"],
            bins=30,
            color=colors[index],
            xlabel="Bias values",
            ylabel="Density",
            title=f"{stats['layer_name']} - BIAIS\nShape: {stats['bias_shape']}",
            legend_loc="upper right",
        )

    plt.tight_layout()
    plt.show()
    print("✅ Per-layer distribution plots generated")
    return fig


def plot_activation_distributions(activation_stats: list[dict[str, Any]]):
    fig_w_per_col = 6
    fig_h_per_subplot = 4
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(activation_stats)))

    n_layers = len(activation_stats)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(fig_w_per_col * n_cols, fig_h_per_subplot * 3 * n_rows))
    fig.suptitle(
        "Activation distributions - Linear inputs/outputs and activations",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for index, stats in enumerate(activation_stats):
        col = index % n_cols
        row_offset = (index // n_cols) * 3

        linear_input = stats["linear_input"]
        ax1 = plt.subplot(3 * n_rows, n_cols, row_offset * n_cols + col + 1)
        _add_histogram_with_stats(
            ax1,
            linear_input,
            bins=50,
            color=colors[index],
            xlabel="Linear input values (±)",
            ylabel="Density",
            title=f"{stats['layer_name']} - LINEAR INPUT\nRange: [{np.min(linear_input):.3f}, {np.max(linear_input):.3f}]",
        )

        linear_output = stats["linear_output"]
        ax2 = plt.subplot(3 * n_rows, n_cols, (row_offset + 1) * n_cols + col + 1)
        _add_histogram_with_stats(
            ax2,
            linear_output,
            bins=50,
            color=colors[index],
            xlabel="Linear output values (±)",
            ylabel="Density",
            title=f"{stats['layer_name']} - SORTIE LINEAR\nRange: [{np.min(linear_output):.3f}, {np.max(linear_output):.3f}]",
            show_negative_ratio=True,
        )

        ax3 = plt.subplot(3 * n_rows, n_cols, (row_offset + 2) * n_cols + col + 1)
        activation_output = stats["activation_output"]
        if activation_output is None:
            ax3.text(
                0.5,
                0.5,
                "No activation on the output layer",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
                style="italic",
            )
            ax3.set_title(f"{stats['layer_name']} - NO ACTIVATION")
            ax3.axis("off")
        else:
            _add_histogram_with_stats(
                ax3,
                activation_output,
                bins=50,
                color=colors[index],
                xlabel=f"Values after activation ({stats['activation_name']})",
                ylabel="Density",
                title=(
                    f"{stats['layer_name']} - SORTIE Activation\n"
                    f"Act: {stats['activation_name']} | Range: [{np.min(activation_output):.3f}, {np.max(activation_output):.3f}]"
                ),
                show_negative_ratio=True,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    print("✅ Activation distribution plots generated")
    return fig


def _quant_report(arr, exponent, bits, int_bounds_fn):
    arr = _to_numpy(arr)
    exponent = int(exponent)

    float_min = float(np.min(arr))
    float_max = float(np.max(arr))
    scale_factor = 2 ** exponent
    qmin_bound, qmax_bound = int_bounds_fn(bits)

    # Analysis reports the true rounded integer range.  V2 does not hide an
    # invalid scale by saturating/clipping it.
    q = np.round(arr * scale_factor)

    qmin = int(np.min(q))
    qmax = int(np.max(q))
    amplitude = min(qmax_bound, -qmin_bound)
    max_abs_q = max(abs(qmin), abs(qmax))
    util_sym = 100.0 * max_abs_q / amplitude if amplitude > 0 else 0.0

    return {
        "shift_exponent": exponent,
        "scale_factor": int(scale_factor),
        "bits": int(bits),
        "float32_min": float_min,
        "float32_max": float_max,
        "int_min": qmin,
        "int_max": qmax,
        "utilisation_sym_%": util_sym,
    }


def _build_best_activation_map(activation_best_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    best_map: dict[str, dict[str, Any]] = {}
    if activation_best_df is None or len(activation_best_df) == 0:
        return best_map
    for _, row in activation_best_df.iterrows():
        best_map[row["layer_name"]] = row.to_dict()
    return best_map


def build_quantification_dataframe(
    *,
    tensors: dict[str, Any],
    shift_exp: dict[str, int],
    quantized_layers: list[Any],
    qcfg: Any,
    int_bounds_fn,
    activation_best_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    best_activation_map = _build_best_activation_map(activation_best_df)
    rows = [
        {
            "tensor": "input",
            "layer": "-",
            "kind": "input",
            "activation": "-",
            "implementation": "-",
            "selected_N": "-",
            **_quant_report(tensors["input"], shift_exp["input"], _cfg_value(qcfg, "act_bits"), int_bounds_fn),
        }
    ]

    for layer_index, layer in enumerate(quantized_layers):
        layer_name = _read(layer, "name")
        is_output = bool(_read(layer, "is_output"))

        rows.append(
            {
                "tensor": f"{layer_name}_weights",
                "layer": layer_name,
                "kind": "weights",
                "activation": "-",
                "implementation": "-",
                "selected_N": "-",
                **_quant_report(_read(layer, "w_float"), _read(layer, "weight_exp"), _cfg_value(qcfg, "weight_bits"), int_bounds_fn),
            }
        )
        rows.append(
            {
                "tensor": f"{layer_name}_bias",
                "layer": layer_name,
                "kind": "bias",
                "activation": "-",
                "implementation": "-",
                "selected_N": "-",
                **_quant_report(_read(layer, "b_float"), _read(layer, "bias_exp"), _cfg_value(qcfg, "bias_bits"), int_bounds_fn),
            }
        )

        if is_output:
            rows.append(
                {
                    "tensor": f"{layer_name}_output",
                    "layer": layer_name,
                    "kind": "output",
                    "activation": "-",
                    "implementation": "-",
                    "selected_N": "-",
                    **_quant_report(tensors["output"], _read(layer, "output_exp"), _cfg_value(qcfg, "act_bits"), int_bounds_fn),
                }
            )
            continue

        act_meta = best_activation_map.get(layer_name, {})
        selected_n = act_meta.get("addr_bits", np.nan)
        selected_n = "-" if pd.isna(selected_n) else int(selected_n)

        rows.append(
            {
                "tensor": f"{layer_name}_preact",
                "layer": layer_name,
                "kind": "preact",
                "activation": _read(layer, "activation_name", "-"),
                "implementation": act_meta.get("implementation", "-"),
                "selected_N": selected_n,
                **_quant_report(
                    tensors[f"layer_{layer_index}_preact"], _read(layer, "preact_exp"), _cfg_value(qcfg, "act_bits"), int_bounds_fn
                ),
            }
        )
        rows.append(
            {
                "tensor": f"{layer_name}_act",
                "layer": layer_name,
                "kind": "act",
                "activation": _read(layer, "activation_name", "-"),
                "implementation": act_meta.get("implementation", "-"),
                "selected_N": selected_n,
                **_quant_report(
                    tensors[f"layer_{layer_index}_act"], _read(layer, "act_exp"), _cfg_value(qcfg, "act_bits"), int_bounds_fn
                ),
            }
        )

    return pd.DataFrame(rows)


def format_quantification_dataframe(df_quant_table: pd.DataFrame) -> pd.DataFrame:
    df_display = df_quant_table.copy()
    for column in ["float32_min", "float32_max"]:
        df_display[column] = df_display[column].map(lambda x: f"{x:.6f}")
    for column in ["int_min", "int_max", "shift_exponent", "bits"]:
        df_display[column] = df_display[column].map(lambda x: f"{x}")
    df_display["scale_factor"] = df_display["scale_factor"].map(lambda x: f"{x}")
    df_display["utilisation_sym_%"] = df_display["utilisation_sym_%"].map(lambda x: f"{x:.1f}%")
    return df_display


def build_model_info_lines(
    *,
    model,
    config: dict[str, Any],
    activation_plan: list[dict[str, Any]],
    activation_best_df: pd.DataFrame | None,
    x,
    output_dim: int,
    qcfg: Any,
) -> list[str]:
    best_activation_map = _build_best_activation_map(activation_best_df)

    activation_lines = []
    for spec in activation_plan:
        layer_name = f"layer_{spec['layer_index']}"
        best_meta = best_activation_map.get(layer_name, {})
        if best_meta:
            if best_meta.get("implementation") == "paper_fit":
                activation_lines.append(
                    f"{layer_name}: {spec['name']} | impl={best_meta['implementation']} | N_best={int(best_meta['addr_bits'])}"
                )
            else:
                activation_lines.append(f"{layer_name}: {spec['name']} | impl={best_meta['implementation']}")
        else:
            activation_lines.append(f"{layer_name}: {spec['name']}")

    return [
        f"Architecture : {tuple(config['layers'])}",
        f"Hidden activations : {', '.join(activation_lines)}",
        f"Input dimension : {x.shape[1]}",
        f"Output dimension : {output_dim}",
        f"Total number of parameters : {sum(param.numel() for param in model.parameters()):,}",
        f"Dataset : {config.get('dataset_name', 'Not specified')}",
        (
            "V2 widths act/weights/bias/DSP : "
            f"{_cfg_value(qcfg, 'act_bits')}/{_cfg_value(qcfg, 'weight_bits')}/"
            f"{_cfg_value(qcfg, 'bias_bits')}/{_cfg_value(qcfg, 'dsp_bits')} bits"
        ),
        (
            f"Calibration : mode={_cfg_value(qcfg, 'calibration_mode')}, "
            f"percentile={_cfg_value(qcfg, 'calibration_percentile')}"
        ),
    ]


def plot_quantification_table(df_quant_display: pd.DataFrame, nn_info_lines: list[str]):
    fig_height = max(6.0, 0.5 * len(df_quant_display) + 4.2)
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis("off")
    fig.suptitle("Quantization summary table", fontsize=18, fontweight="bold", y=0.985)

    ax.text(
        0.01,
        0.965,
        "\n".join(nn_info_lines),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef5ff", edgecolor="#aac6e3"),
    )

    col_labels = [
        "Tenseur", "Layer", "Kind", "Activation", "Impl.", "N sel.",
        "Shift exponent", "Facteur scale", "Bits", "Float32 min",
        "Float32 max", "Int min", "Int max", "% Utilisation sym.",
    ]

    cell_text = df_quant_display[
        [
            "tensor", "layer", "kind", "activation", "implementation", "selected_N",
            "shift_exponent", "scale_factor", "bits", "float32_min", "float32_max",
            "int_min", "int_max", "utilisation_sym_%",
        ]
    ].values.tolist()

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.74])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#d9edf7")
            cell.set_height(cell.get_height() * 1.15)
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    print("✅ Quantization summary table generated")
    return fig


def run_print_distribution(
    basic_results: dict[str, Any],
    *,
    activation_best_df: pd.DataFrame | None = None,
    save_pdf: bool = False,
    output_pdf_path: str | Path | None = None,
    show_table_dataframe: bool = True,
) -> dict[str, Any]:
    """
    Generate only the visual outputs :
    - distributions weights / biases
    - activation distributions
    - summary table
    - export PDF optionnel

    This function is designed to be called from a second notebook cell.
    """
    paths = basic_results["paths"]
    qcfg = basic_results["qcfg"]
    if activation_best_df is None:
        activation_best_df = basic_results.get("activation_best_df")
    dataset = basic_results["dataset"]
    model_bundle = basic_results["model_bundle"]
    tensors = basic_results["tensors"]
    shift_exp = basic_results["shift_exp"]
    quantized_layers = basic_results["quantized_layers"]

    layer_stats = collect_layer_param_stats(model_bundle.model)
    activation_stats = collect_activation_stats(
        model_bundle.model,
        dataset.x,
        model_bundle.activation_plan,
        lambda x, act_name: activation_float(x, act_name, negative_slope=qcfg.activation_negative_slope),
    )
    print(f"✅ Statistics ready: {len(layer_stats)} weight/bias layers, {len(activation_stats)} activation layers")

    fig_wb = plot_weight_bias_distributions(layer_stats)
    fig_act = plot_activation_distributions(activation_stats)

    df_quant_table = build_quantification_dataframe(
        tensors=tensors,
        shift_exp=shift_exp,
        quantized_layers=quantized_layers,
        qcfg=qcfg,
        int_bounds_fn=int_bounds,
        activation_best_df=activation_best_df,
    )
    df_quant_display = format_quantification_dataframe(df_quant_table)

    if show_table_dataframe:
        display(df_quant_display)

    nn_info_lines = build_model_info_lines(
        model=model_bundle.model,
        config=model_bundle.config,
        activation_plan=model_bundle.activation_plan,
        activation_best_df=activation_best_df,
        x=dataset.x,
        output_dim=int(model_bundle.model.out.out_features),
        qcfg=qcfg,
    )
    fig_table = plot_quantification_table(df_quant_display, nn_info_lines)

    pdf_path = None
    if save_pdf:
        pdf_path = Path(output_pdf_path or (paths.output_dir / qcfg.distribution_pdf_name))
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig_wb, bbox_inches="tight")
            pdf.savefig(fig_act, bbox_inches="tight")
            pdf.savefig(fig_table, bbox_inches="tight")
        print(f"✅ PDF exported : {pdf_path}")

    return {
        **basic_results,
        "layer_stats": layer_stats,
        "activation_stats": activation_stats,
        "df_quant_table": df_quant_table,
        "df_quant_display": df_quant_display,
        "fig_wb": fig_wb,
        "fig_act": fig_act,
        "fig_table": fig_table,
        "pdf_path": pdf_path,
    }

