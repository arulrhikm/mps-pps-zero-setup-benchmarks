#!/usr/bin/env python3
"""
Plot MPS runtime vs two-qubit gate count — **CPU and GPU on one axes** (paper-style).

Expects JSONL from ``experiments/peaked_circuits_2q_scaling.py`` run twice::

    python experiments/peaked_circuits_2q_scaling.py --device mps.cpu
    python experiments/peaked_circuits_2q_scaling.py --device mps.gpu

Reads:
  mps_tests/data/peaked_2q_scaling_mps_cpu.jsonl
  mps_tests/data/peaked_2q_scaling_mps_gpu.jsonl

One panel: median ``run_time_ms`` vs ``num_two_qubit_gates`` per ``circuit_key`` for each device.
CPU markers: circles; GPU markers: squares. Shared τ colormap when ``sweep_tau`` is present.
Y-axis defaults to **log** scale so CPU and GPU are visible on the same figure; pass ``--linear-y``
for linear milliseconds.

Output:
  mps_tests/plots/fig_peaked_2q_gate_scaling.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")

CPU_FILE = os.path.join(DATA_DIR, "peaked_2q_scaling_mps_cpu.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "peaked_2q_scaling_mps_gpu.jsonl")
OUT_PNG = os.path.join(PLOT_DIR, "fig_peaked_2q_gate_scaling.png")

CPU_COLOR = "#2171b5"
GPU_COLOR = "#cb181d"


def load_rows(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                r = json.loads(s)
                if "error" not in r and r.get("run_time_ms"):
                    rows.append(r)
            except json.JSONDecodeError:
                continue
    return rows


def collect_tau_values(rows: list[dict]) -> list[float]:
    return [float(r["sweep_tau"]) for r in rows if "sweep_tau" in r]


def aggregate_by_circuit(
    rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per circuit_key: median run_time_ms, num_two_qubit_gates, n_trials, sweep_tau (nan if absent)."""
    by_key: dict[str, list[float]] = defaultdict(list)
    n2_by_key: dict[str, int] = {}
    tau_by_key: dict[str, float] = {}
    for r in rows:
        k = r.get("circuit_key")
        if not k or "num_two_qubit_gates" not in r:
            continue
        by_key[k].append(float(r["run_time_ms"]))
        n2_by_key[k] = int(r["num_two_qubit_gates"])
        if "sweep_tau" in r:
            tau_by_key[k] = float(r["sweep_tau"])

    keys = sorted(
        n2_by_key,
        key=lambda x: (tau_by_key.get(x, float("inf")), n2_by_key[x], x),
    )
    xs = np.array([n2_by_key[k] for k in keys], dtype=float)
    ys = np.array([float(np.median(by_key[k])) for k in keys], dtype=float)
    ns = np.array([len(by_key[k]) for k in keys], dtype=int)
    taus = np.array([tau_by_key.get(k, np.nan) for k in keys], dtype=float)
    return xs, ys, ns, taus


def draw_series(
    ax,
    rows: list[dict],
    *,
    marker: str,
    tau_norm: mcolors.Normalize | None,
    fallback_color: str,
    zorder: int,
) -> tuple[object | None, bool]:
    """
    Scatter + dashed polyline for one device.
    Returns (mappable for colorbar or None, has_tau).
    """
    if not rows:
        return None, False
    xs, ys, _ns, taus = aggregate_by_circuit(rows)
    has_tau = bool(np.any(~np.isnan(taus)))
    sc = None
    if has_tau and tau_norm is not None:
        sc = ax.scatter(
            xs,
            ys,
            c=taus,
            cmap="viridis",
            norm=tau_norm,
            s=52,
            alpha=0.92,
            marker=marker,
            edgecolors="white",
            linewidths=0.65,
            zorder=zorder,
        )
    elif has_tau:
        sc = ax.scatter(
            xs,
            ys,
            c=taus,
            cmap="viridis",
            s=52,
            alpha=0.92,
            marker=marker,
            edgecolors="white",
            linewidths=0.65,
            zorder=zorder,
        )
    else:
        ax.scatter(
            xs,
            ys,
            c=fallback_color,
            s=48,
            alpha=0.88,
            marker=marker,
            edgecolors="white",
            linewidths=0.55,
            zorder=zorder,
        )
    if len(xs) >= 2:
        line_c = "0.35" if has_tau else fallback_color
        ax.plot(
            xs,
            ys,
            color=line_c,
            alpha=0.55,
            linewidth=1.25,
            linestyle="--",
            zorder=zorder - 1,
        )
    return sc, has_tau


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot peaked 2Q MPS scaling: CPU and GPU on one figure."
    )
    parser.add_argument(
        "--cpu",
        type=str,
        default=CPU_FILE,
        help="Path to peaked_2q_scaling_mps_cpu.jsonl",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=GPU_FILE,
        help="Path to peaked_2q_scaling_mps_gpu.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUT_PNG,
        help="Output PNG path",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear y-axis (default: log scale for CPU+GPU visibility)",
    )
    args = parser.parse_args()

    cpu_file = os.path.abspath(args.cpu)
    gpu_file = os.path.abspath(args.gpu)
    out_png = os.path.abspath(args.output)

    if not os.path.exists(cpu_file):
        print(f"WARNING: CPU data missing: {cpu_file}", file=sys.stderr)
        print(
            "  Run: python experiments/peaked_circuits_2q_scaling.py --device mps.cpu",
            file=sys.stderr,
        )
    if not os.path.exists(gpu_file):
        print(f"WARNING: GPU data missing: {gpu_file}", file=sys.stderr)
        print(
            "  Run: python experiments/peaked_circuits_2q_scaling.py --device mps.gpu",
            file=sys.stderr,
        )

    cpu_rows = load_rows(cpu_file)
    gpu_rows = load_rows(gpu_file)

    taus_all = collect_tau_values(cpu_rows) + collect_tau_values(gpu_rows)
    tau_norm: mcolors.Normalize | None = None
    if taus_all:
        tau_norm = mcolors.Normalize(vmin=min(taus_all), vmax=max(taus_all))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")

    sc_cpu, _ = draw_series(
        ax, cpu_rows, marker="o", tau_norm=tau_norm, fallback_color=CPU_COLOR, zorder=3
    )
    sc_gpu, _ = draw_series(
        ax, gpu_rows, marker="s", tau_norm=tau_norm, fallback_color=GPU_COLOR, zorder=2
    )

    mappable = sc_cpu if sc_cpu is not None else sc_gpu
    if mappable is not None and tau_norm is not None:
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label(r"Sweep $\tau$ (filename)", fontsize=10)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CPU_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=9,
            label="MPS CPU (median, 5 trials / point)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=GPU_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=9,
            label="MPS GPU (median, 5 trials / point)",
            linestyle="None",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.95, fontsize=9)

    ax.set_xlabel("Two-Qubit Gates (Decomposed)", fontsize=11)
    y_label = "Median run time (ms)"
    if not args.linear_y:
        ax.set_yscale("log")
        y_label += " (log scale)"
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(
        "Peaked circuits: CPU vs GPU runtime vs two-qubit gate count (1 shot)",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not cpu_rows and not gpu_rows:
        ax.text(
            0.5,
            0.5,
            "No data — run peaked_circuits_2q_scaling.py for CPU and GPU",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved: {out_png}\n"
        f"  CPU: {cpu_file} ({len(cpu_rows)} rows)\n"
        f"  GPU: {gpu_file} ({len(gpu_rows)} rows)"
    )


if __name__ == "__main__":
    main()
