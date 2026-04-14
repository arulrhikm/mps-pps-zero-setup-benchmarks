#!/usr/bin/env python3
"""
Qubit scaling  (MPS CPU vs GPU)
===============================
Runtime vs number of qubits at fixed d=10, chi=256.
Quadratic fits  T = a*n^2 + b  with coefficient-ratio annotation.

Data files:
  mps_tests/data/qubit_scaling_cpu.jsonl
  mps_tests/data/qubit_scaling_gpu.jsonl
"""
import matplotlib
matplotlib.use("Agg")

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.linewidth":    1.2,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.30,
    "legend.framealpha": 0.92,
    "legend.fontsize":   9,
    "figure.dpi":        150,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CPU_FILE = os.path.join(DATA_DIR, "qubit_scaling_cpu.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "qubit_scaling_gpu.jsonl")

OUTPUT_PNG = os.path.join(PLOT_DIR, "fig4_qubit_scaling.png")

CPU_COLOR = "#1f77b4"
GPU_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def aggregate_by_qubits(rows):
    """Median per n; IQR error bars only when n_trials >= 3 (measured)."""
    by_n = defaultdict(list)
    for r in rows:
        n  = r.get("num_qubits")
        rt = r.get("run_time_ms")
        if n is not None and rt is not None and rt > 0:
            by_n[n].append(rt / 1000.0)

    result = {}
    for n in sorted(by_n):
        vals = np.asarray(by_n[n], dtype=float)
        cnt  = int(vals.size)
        med  = float(np.median(vals))
        if cnt >= 3:
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            has_iqr = True
        else:
            q25 = q75 = med
            has_iqr = False
        result[n] = dict(median=med, q25=q25, q75=q75,
                         n_trials=cnt, has_iqr=has_iqr)
    return result


def quadratic(x, a, b):
    return a * np.asarray(x, dtype=float) ** 2 + b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_agg = aggregate_by_qubits(load_jsonl(CPU_FILE))
    gpu_agg = aggregate_by_qubits(load_jsonl(GPU_FILE))

    print(f"CPU qubits: {sorted(cpu_agg)}")
    print(f"GPU qubits: {sorted(gpu_agg)}")

    # Common qubit set for fitting (use 4-step grid present in both)
    common = sorted(set(cpu_agg) & set(gpu_agg))
    print(f"Common qubits for fit: {common}")

    def _arrays(agg, keys):
        ns   = np.array(keys, dtype=float)
        meds = np.array([agg[n]["median"] for n in keys])
        return ns, meds

    cpu_keys = sorted(cpu_agg)
    gpu_keys = sorted(gpu_agg)

    # Quadratic fits on common points
    cx, cy = _arrays(cpu_agg, common)
    gx, gy = _arrays(gpu_agg, common)
    popt_cpu, _ = curve_fit(quadratic, cx, cy, p0=[0.01, 0])
    popt_gpu, _ = curve_fit(quadratic, gx, gy, p0=[0.01, 0])
    ratio = popt_cpu[0] / popt_gpu[0]

    print(f"\n-- Quadratic fits (common points) --")
    print(f"  CPU: {popt_cpu[0]:.4f} n^2 + {popt_cpu[1]:.2f}")
    print(f"  GPU: {popt_gpu[0]:.4f} n^2 + {popt_gpu[1]:.2f}")
    print(f"  Ratio: {ratio:.1f}x")

    x_fit = np.linspace(min(common) - 2, max(common) + 2, 300)

    # -- Figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))

    def _plot_points(agg, keys, color, marker, label):
        xs = np.array(keys, dtype=float)
        ys = np.array([agg[k]["median"] for k in keys])
        ax.plot(
            xs, ys,
            linestyle="None", marker=marker, color=color, markersize=7,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=5, label=label,
        )

    _plot_points(cpu_agg, cpu_keys, CPU_COLOR, "o", "MPS CPU")
    _plot_points(gpu_agg, gpu_keys, GPU_COLOR, "^", "MPS GPU")

    ax.plot(x_fit, quadratic(x_fit, *popt_cpu), "--", color=CPU_COLOR,
            linewidth=2.2, alpha=0.75, zorder=4,
            label=rf"CPU fit: ${popt_cpu[0]:.2f}\,n^2 + {popt_cpu[1]:.0f}$")
    ax.plot(x_fit, quadratic(x_fit, *popt_gpu), "--", color=GPU_COLOR,
            linewidth=2.2, alpha=0.75, zorder=4,
            label=rf"GPU fit: ${popt_gpu[0]:.2f}\,n^2 + {popt_gpu[1]:.0f}$")

    ax.text(0.97, 0.05,
            f"Coefficient ratio: {ratio:.1f}$\\times$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color="#444",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#bbb", alpha=0.92))

    ax.set_xlabel("Number of Qubits  $n$", fontsize=13)
    ax.set_ylabel("Runtime  $T$  (s)", fontsize=13)
    ax.set_title(
        r"Qubit scaling  ($d = 10$, $\chi = 256$)",
        fontsize=12, pad=10)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
    ax.legend(loc="upper left", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nSaved:\n  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
