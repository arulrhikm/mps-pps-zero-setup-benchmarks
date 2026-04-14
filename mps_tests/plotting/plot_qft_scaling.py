#!/usr/bin/env python3
"""
QFT scaling — MPS CPU vs GPU
===========================
Runtime vs qubit count for fixed bond dimension and QFT approximation degree
(same circuit family as archived experiment 6).

Data:
  mps_tests/data/qft_scaling_cpu.jsonl
  mps_tests/data/qft_scaling_gpu.jsonl

Default slice: χ=64, approximation_degree=0 (exact QFT). Median runtime per n;
CPU/GPU curves use markers connected by lines (log-y).

Output:
  mps_tests/plots/fig6_qft_scaling.png
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.linewidth":    1.2,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.30,
    "legend.framealpha": 0.92,
    "legend.fontsize":   10,
    "figure.dpi":        150,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")

CPU_FILE = os.path.join(DATA_DIR, "qft_scaling_cpu.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "qft_scaling_gpu.jsonl")
OUTPUT_PNG = os.path.join(PLOT_DIR, "fig6_qft_scaling.png")

CPU_COLOR = "#1f77b4"
GPU_COLOR = "#d62728"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                r = json.loads(s)
                if "error" not in r:
                    rows.append(r)
            except json.JSONDecodeError:
                continue
    return rows


def aggregate(rows, bond_dim, approx_degree):
    """num_qubits -> median seconds; IQR if n >= 3."""
    by_n = defaultdict(list)
    for r in rows:
        if r.get("bond_dimension") != bond_dim:
            continue
        if r.get("approximation_degree") != approx_degree:
            continue
        n = r.get("num_qubits")
        rt = r.get("run_time_ms")
        if n is None or rt is None or rt <= 0:
            continue
        by_n[n].append(float(rt) / 1000.0)

    out = {}
    for n in sorted(by_n):
        vals = np.asarray(by_n[n], dtype=float)
        cnt = int(vals.size)
        med = float(np.median(vals))
        if cnt >= 3:
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            has_iqr = True
        else:
            q25 = q75 = med
            has_iqr = False
        out[n] = dict(median=med, q25=q25, q75=q75, n_trials=cnt, has_iqr=has_iqr)
    return out


def plot_series(ax, agg, color, marker, label):
    if not agg:
        return
    ns = np.array(sorted(agg), dtype=float)
    meds = np.array([agg[int(n)]["median"] for n in ns])
    ax.semilogy(
        ns, meds, marker=marker, color=color, linestyle="-",
        linewidth=1.6, markersize=6, markeredgewidth=0.6,
        markeredgecolor="white", zorder=5, label=label,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot QFT CPU vs GPU scaling")
    parser.add_argument("--bond-dimension", type=int, default=64)
    parser.add_argument("--approximation-degree", type=int, default=0)
    parser.add_argument("--cpu-file", type=str, default=CPU_FILE)
    parser.add_argument("--gpu-file", type=str, default=GPU_FILE)
    parser.add_argument("--output", type=str, default=OUTPUT_PNG)
    args = parser.parse_args()

    cpu_rows = load_jsonl(args.cpu_file)
    gpu_rows = load_jsonl(args.gpu_file)
    cpu_agg = aggregate(cpu_rows, args.bond_dimension, args.approximation_degree)
    gpu_agg = aggregate(gpu_rows, args.bond_dimension, args.approximation_degree)

    print(f"CPU n points: {sorted(cpu_agg)}")
    print(f"GPU n points: {sorted(gpu_agg)}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_series(ax, cpu_agg, CPU_COLOR, "o", "MPS CPU")
    plot_series(ax, gpu_agg, GPU_COLOR, "^", "MPS GPU")

    for ref_n in (32, 64, 96):
        ax.axvline(ref_n, color="grey", lw=0.8, ls=":", alpha=0.55)

    ax.set_xlabel("Number of Qubits  $n$", fontsize=13)
    ax.set_ylabel("Runtime  $T$  (s)", fontsize=13)
    deg = args.approximation_degree
    chi = args.bond_dimension
    ax.set_title(
        "QFT runtime scaling: MPS CPU vs GPU\n"
        rf"($\chi = {chi}$, Qiskit QFT approximation degree $k = {deg}$)",
        fontsize=12,
        pad=14,
        linespacing=1.2,
    )
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(16))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:g} s" if v >= 1 else f"{v * 1000:.0f} ms")
    )
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="#ccc")
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.22)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
