#!/usr/bin/env python3
"""
Plot runtime vs bond dimension for peaked-circuit results (CPU + GPU),
with a top x-axis showing median overlap percentages by bond dimension.

Reads JSONL files from:
  mps_tests/data/peaked-circuits-results/

By default uses dominant_overlap_percent as "overlap" metadata for the top axis.
Color style follows the statevector CPU/GPU convention:
  - CPU: shades of blue
  - GPU: shades of red

Outputs:
  - Clean runtime summary plot (median + IQR + min/max whiskers)
  - Optional simple fit overlay (linear fit in log2(chi) space)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPS_TESTS_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(MPS_TESTS_DIR, "data", "peaked-circuits-results")
PLOTS_DIR = os.path.join(MPS_TESTS_DIR, "plots")
DEFAULT_OUTPUT = os.path.join(PLOTS_DIR, "fig_peaked_overlap_vs_bond_dimension.png")
DEFAULT_OUTPUT_FIT = os.path.join(PLOTS_DIR, "fig_peaked_overlap_vs_bond_dimension_fit.png")

CPU_COLOR = "#1f77b4"
GPU_COLOR = "#d62728"


def _rzz_key(circuit_key: str) -> tuple[int, str]:
    m = re.search(r"_RZZ(\d+)_", circuit_key or "")
    if m:
        return int(m.group(1)), circuit_key
    return 10**9, circuit_key


def _load_rows(path: str, overlap_field: str) -> dict[str, dict[int, dict[str, float]]]:
    """
    Return:
      circuit_key -> {bond_dimension -> {"overlap": x, "runtime_s": y}}
    using medians for duplicate bond_dimension rows.
    """
    raw_ov: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_rt: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            if overlap_field not in r:
                continue
            if not isinstance(r.get("run_time_ms"), (int, float)) or r.get("run_time_ms", 0) <= 0:
                continue
            circuit_key = r.get("circuit_key")
            bond_dim = r.get("bond_dimension")
            if not circuit_key or not isinstance(bond_dim, int):
                continue
            raw_ov[circuit_key][bond_dim].append(float(r[overlap_field]))
            raw_rt[circuit_key][bond_dim].append(float(r["run_time_ms"]) / 1000.0)

    out: dict[str, dict[int, dict[str, float]]] = {}
    for circuit_key, by_bd in raw_ov.items():
        out[circuit_key] = {}
        for bd, vals in by_bd.items():
            out[circuit_key][bd] = {
                "overlap": float(np.median(np.asarray(vals, dtype=float))),
                "runtime_s": float(np.median(np.asarray(raw_rt[circuit_key][bd], dtype=float))),
            }
    return out


def _read_device_files(results_dir: str, overlap_field: str) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict[int, dict[str, float]]]]:
    cpu_data: dict[str, dict[int, dict[str, float]]] = {}
    gpu_data: dict[str, dict[int, dict[str, float]]] = {}

    if not os.path.exists(results_dir):
        return cpu_data, gpu_data

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(results_dir, fname)
        if fname.startswith("CPU_"):
            device_rows = _load_rows(fpath, overlap_field)
            cpu_data.update(device_rows)
        elif fname.startswith("GPU_"):
            device_rows = _load_rows(fpath, overlap_field)
            gpu_data.update(device_rows)
    return cpu_data, gpu_data


def _aggregate_by_bond_runtime(data: dict[str, dict[int, dict[str, float]]]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for by_bd in data.values():
        for bd, vals in by_bd.items():
            out[int(bd)].append(float(vals["runtime_s"]))
    return out


def _aggregate_by_bond_overlap(data: dict[str, dict[int, dict[str, float]]]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for by_bd in data.values():
        for bd, vals in by_bd.items():
            out[int(bd)].append(float(vals["overlap"]))
    return out


def _apply_post_saturation_fill(
    data: dict[str, dict[int, dict[str, float]]],
    *,
    saturation_value: float = 100.0,
    tol: float = 1e-9,
) -> dict[str, dict[int, dict[str, float]]]:
    """
    If a circuit reaches saturation (overlap ~= 100) at bond dimension b_sat,
    treat all *missing* larger bond dimensions as saturated too.

    This addresses early-saturating circuits that stop providing additional points
    at higher bond dimensions due to run interruptions, while preserving measured
    values where present.
    """
    if not data:
        return data

    global_bds = sorted({bd for by_bd in data.values() for bd in by_bd.keys()})
    if not global_bds:
        return data

    out: dict[str, dict[int, dict[str, float]]] = {}
    for circuit_key, by_bd in data.items():
        filled = dict(by_bd)
        sat_bds = sorted(bd for bd, vals in by_bd.items() if vals["overlap"] >= saturation_value - tol)
        if sat_bds:
            b_sat = sat_bds[0]
            for bd in global_bds:
                if bd > b_sat and bd not in filled:
                    # Fill missing overlap as saturated while preserving runtime as missing.
                    filled[bd] = {"overlap": saturation_value, "runtime_s": np.nan}
        out[circuit_key] = filled
    return out


def _plot_clean_device(
    ax,
    data: dict[str, dict[int, dict[str, float]]],
    *,
    line_color: str,
    faint_color: str,
    label: str,
) -> int:
    """
    Runtime publication style:
      - median runtime line
      - IQR shaded band (Q25-Q75)
      - min/max whiskers
    """
    if not data:
        return 0

    keys = sorted(data.keys(), key=_rzz_key)
    agg = _aggregate_by_bond_runtime(data)
    bds = sorted(agg.keys())
    med = np.array([np.median(agg[b]) for b in bds], dtype=float)
    q25 = np.array([np.percentile(agg[b], 25) for b in bds], dtype=float)
    q75 = np.array([np.percentile(agg[b], 75) for b in bds], dtype=float)
    mins = np.array([np.min(agg[b]) for b in bds], dtype=float)
    maxs = np.array([np.max(agg[b]) for b in bds], dtype=float)

    # Min/max whiskers
    lower_whisk = med - mins
    upper_whisk = maxs - med
    ax.errorbar(
        bds,
        med,
        yerr=np.vstack([lower_whisk, upper_whisk]),
        fmt="none",
        ecolor=line_color,
        elinewidth=1.1,
        capsize=3,
        capthick=1.0,
        alpha=0.35,
        zorder=1,
    )

    # IQR shaded band (requested blue/red regions)
    ax.fill_between(
        bds,
        q25,
        q75,
        color=line_color,
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )

    # Median line + markers
    ax.plot(
        bds,
        med,
        color=line_color,
        linewidth=2.8,
        alpha=0.96,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=3,
        label=label,
    )
    return len(keys)


def _median_series_runtime(data: dict[str, dict[int, dict[str, float]]]) -> tuple[np.ndarray, np.ndarray]:
    agg = _aggregate_by_bond_runtime(data)
    if not agg:
        return np.array([], dtype=float), np.array([], dtype=float)
    bds = np.array(sorted(agg.keys()), dtype=float)
    meds = np.array([np.median(agg[int(b)]) for b in bds], dtype=float)
    return bds, meds


def _median_overlap_labels(
    cpu_data: dict[str, dict[int, dict[str, float]]],
    gpu_data: dict[str, dict[int, dict[str, float]]],
    ticks: list[int],
) -> list[str]:
    cpu_agg = _aggregate_by_bond_overlap(cpu_data)
    gpu_agg = _aggregate_by_bond_overlap(gpu_data)
    labels: list[str] = []
    for bd in ticks:
        parts = []
        if bd in cpu_agg and len(cpu_agg[bd]) > 0:
            parts.append(f"C:{np.median(cpu_agg[bd]):.1f}%")
        if bd in gpu_agg and len(gpu_agg[bd]) > 0:
            parts.append(f"G:{np.median(gpu_agg[bd]):.1f}%")
        labels.append(" / ".join(parts) if parts else "")
    return labels


def _fit_log2_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 2 or y.size < 2:
        return None
    lx = np.log2(x)
    coef = np.polyfit(lx, y, deg=1)
    return float(coef[0]), float(coef[1])  # slope, intercept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot peaked overlap vs bond dimension (CPU + GPU)."
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory containing peaked-circuit-results JSONL files.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-fit",
        default=DEFAULT_OUTPUT_FIT,
        help="Output PNG path for simple-fit overlay figure.",
    )
    parser.add_argument(
        "--overlap-field",
        default="dominant_overlap_percent",
        choices=["dominant_overlap_percent", "weighted_overlap_percent", "target_hit_rate_percent"],
        help="Which overlap metric to plot on Y-axis.",
    )
    parser.add_argument(
        "--with-fit",
        action="store_true",
        help="Also create a second figure with simple linear fit in log2(bond_dimension).",
    )
    args = parser.parse_args()

    cpu_data, gpu_data = _read_device_files(os.path.abspath(args.results_dir), args.overlap_field)
    cpu_data = _apply_post_saturation_fill(cpu_data)
    gpu_data = _apply_post_saturation_fill(gpu_data)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    n_cpu = _plot_clean_device(
        ax,
        cpu_data,
        line_color=CPU_COLOR,
        faint_color=plt.get_cmap("Blues")(0.65),
        label="CPU median (IQR shaded + min/max)",
    )
    n_gpu = _plot_clean_device(
        ax,
        gpu_data,
        line_color=GPU_COLOR,
        faint_color=plt.get_cmap("Reds")(0.65),
        label="GPU median (IQR shaded + min/max)",
    )

    y_label_map = {
        "dominant_overlap_percent": "Overlap (%)",
        "weighted_overlap_percent": "Weighted Overlap (%)",
        "target_hit_rate_percent": "Target Hit Rate (%)",
    }
    y_label = y_label_map.get(args.overlap_field, f"{args.overlap_field} (%)")

    ax.set_xlabel(r"Bond Dimension ($\chi$)", fontsize=12)
    ax.set_ylabel("Run Time (s)", fontsize=12)
    ax.set_title("Peaked Circuits: Runtime vs Bond Dimension", fontsize=14)
    ax.grid(True, which="both", alpha=0.30, linestyle="--")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    xticks = sorted({*(_aggregate_by_bond_runtime(cpu_data).keys()), *(_aggregate_by_bond_runtime(gpu_data).keys())})
    if not xticks:
        xticks = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Top x-axis: show median overlap percentages at each bond dimension.
    ax_top = ax.twiny()
    ax_top.set_xscale("log", base=2)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels(_median_overlap_labels(cpu_data, gpu_data, xticks), fontsize=8)
    ax_top.set_xlabel("Median Overlap (%)", fontsize=10, labelpad=8)

    if n_cpu + n_gpu > 0:
        handles = [
            Line2D([0], [0], color=CPU_COLOR, linewidth=3.0, marker="o", markersize=5, label="CPU median"),
            Line2D([0], [0], color=GPU_COLOR, linewidth=3.0, marker="o", markersize=5, label="GPU median"),
            Line2D([0], [0], color=CPU_COLOR, linewidth=8.0, alpha=0.20, label="IQR shaded region"),
            Line2D([0], [0], color="0.35", linewidth=1.1, alpha=0.35, label="Min/Max"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.94)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid rows found in peaked-circuit-results",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    plt.tight_layout()
    clean_out = os.path.abspath(args.output)
    fig.savefig(clean_out, dpi=300)
    plt.close(fig)

    fit_out = None
    if args.with_fit and (n_cpu + n_gpu) > 0:
        fig_fit, ax_fit = plt.subplots(figsize=(11.5, 6.8))
        _plot_clean_device(
            ax_fit,
            cpu_data,
            line_color=CPU_COLOR,
            faint_color=plt.get_cmap("Blues")(0.65),
            label="CPU median (IQR shaded + min/max)",
        )
        _plot_clean_device(
            ax_fit,
            gpu_data,
            line_color=GPU_COLOR,
            faint_color=plt.get_cmap("Reds")(0.65),
            label="GPU median (IQR shaded + min/max)",
        )

        cpu_x, cpu_y = _median_series_runtime(cpu_data)
        gpu_x, gpu_y = _median_series_runtime(gpu_data)
        cpu_fit = _fit_log2_linear(cpu_x, cpu_y)
        gpu_fit = _fit_log2_linear(gpu_x, gpu_y)

        if cpu_fit is not None:
            m, b = cpu_fit
            xx = np.logspace(np.log2(np.min(cpu_x)), np.log2(np.max(cpu_x)), 120, base=2.0)
            yy = m * np.log2(xx) + b
            ax_fit.plot(
                xx,
                yy,
                linestyle="--",
                linewidth=2.1,
                color=CPU_COLOR,
                alpha=0.95,
                label=rf"CPU fit: runtime={m:.2f}·log2($\chi$)+{b:.1f}",
            )
        if gpu_fit is not None:
            m, b = gpu_fit
            xx = np.logspace(np.log2(np.min(gpu_x)), np.log2(np.max(gpu_x)), 120, base=2.0)
            yy = m * np.log2(xx) + b
            ax_fit.plot(
                xx,
                yy,
                linestyle="--",
                linewidth=2.1,
                color=GPU_COLOR,
                alpha=0.95,
                label=rf"GPU fit: runtime={m:.2f}·log2($\chi$)+{b:.1f}",
            )

        ax_fit.set_xlabel(r"Bond Dimension ($\chi$)", fontsize=12)
        ax_fit.set_ylabel("Run Time (s)", fontsize=12)
        ax_fit.set_title("Peaked Circuits: Runtime vs Bond Dimension (with simple fit)", fontsize=14)
        ax_fit.grid(True, which="both", alpha=0.30, linestyle="--")
        ax_fit.set_yscale("log")
        ax_fit.set_xscale("log", base=2)
        ax_fit.set_xticks(xticks)
        ax_fit.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_fit.legend(loc="lower right", fontsize=8.8, framealpha=0.94)
        plt.tight_layout()
        fit_out = os.path.abspath(args.output_fit)
        fig_fit.savefig(fit_out, dpi=300)
        plt.close(fig_fit)

    print(
        f"Saved clean plot: {clean_out}\n"
        f"Saved fit plot: {fit_out if fit_out else '(disabled; pass --with-fit)'}\n"
        f"CPU circuits plotted: {n_cpu}\n"
        f"GPU circuits plotted: {n_gpu}\n"
        f"Metric: {args.overlap_field}"
    )


if __name__ == "__main__":
    main()

