#!/usr/bin/env python3
"""
Overlay NVIDIA speedup points on the existing 3-bar Qiskit speedup chart.

Bars are the existing baseline values from the paper draft:
  delta = [1e-4, 5e-5, 2.5e-5] -> [56, 108, 177]

NVIDIA points are computed from measured PPS-Qiskit means divided by
optimized GPU means from:
  data/pps_qiskit_benchmark.jsonl
  data/pps_gpu_optimized_benchmark.jsonl
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

QISKIT_FILE = os.path.join(DATA_DIR, "pps_qiskit_benchmark.jsonl")
GPU_OPT_FILE = os.path.join(DATA_DIR, "pps_gpu_optimized_benchmark.jsonl")

OUT_PNG = os.path.join(PLOT_DIR, "pps_gpu_speedup_over_qiskit_with_nvidia.png")


def load_mean_runtime_by_delta(path: str) -> dict[float, float]:
    by_delta: dict[float, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            r = json.loads(s)
            if "error" in r:
                continue
            delta = float(r["delta"])
            if "run_time_ms" in r:
                t_s = float(r["run_time_ms"]) / 1000.0
            elif "run_time_s" in r:
                t_s = float(r["run_time_s"])
            else:
                continue
            by_delta[delta].append(t_s)
    return {d: float(np.mean(ts)) for d, ts in by_delta.items() if ts}


def main() -> None:
    deltas = np.array([1e-4, 5e-5, 2.5e-5], dtype=float)
    baseline_bars = np.array([56, 108, 177], dtype=float)  # current chart values

    qiskit_mean = load_mean_runtime_by_delta(QISKIT_FILE)
    gpu_opt_mean = load_mean_runtime_by_delta(GPU_OPT_FILE)

    nvidia_speedup = []
    for d in deltas:
        if d not in qiskit_mean or d not in gpu_opt_mean:
            raise RuntimeError(f"Missing delta={d} in benchmark data")
        nvidia_speedup.append(qiskit_mean[d] / gpu_opt_mean[d])
    nvidia_speedup = np.array(nvidia_speedup, dtype=float)

    x = np.arange(len(deltas))

    plt.figure(figsize=(7.2, 6.2))
    bar_color = "#76B900"  # NVIDIA green-like
    plt.bar(x, baseline_bars, color=bar_color, width=0.48, label="Current chart values")

    # Existing bar labels
    for xi, yi in zip(x, baseline_bars):
        plt.text(xi, yi + 3, f"{yi:.0f}", ha="center", va="bottom", fontsize=10)

    # NVIDIA measured points overlay
    plt.plot(
        x,
        nvidia_speedup,
        color="#FF7F0E",
        marker="o",
        markersize=7,
        linewidth=2.0,
        label="NVIDIA (measured, optimized GPU)",
    )
    for xi, yi in zip(x, nvidia_speedup):
        plt.text(xi, yi + max(8, yi * 0.02), f"{yi:.0f}x", ha="center", va="bottom", fontsize=9, color="#FF7F0E")

    plt.xticks(x, [f"{d:g}" for d in deltas], fontsize=11)
    plt.ylabel("X Factor Speedup", fontsize=11)
    plt.xlabel("Coefficient Cutoff", fontsize=11)
    plt.title("GPU Speedup Over Qiskit PauliProp\nfor 127 Qubit Utility Circuit", fontsize=18, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_PNG)
    for d, sp in zip(deltas, nvidia_speedup):
        print(f"delta={d:g} -> NVIDIA speedup={sp:.2f}x")


if __name__ == "__main__":
    main()

