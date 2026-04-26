#!/usr/bin/env python3
"""
Depth scaling  (MPS CPU vs GPU)
===============================
Runtime vs circuit depth at fixed n=40, chi=256.
Linear fits  T = a*d + b  with slope-ratio annotation.

Data files:
  mps_tests/data/depth_scaling_cpu256.jsonl
  mps_tests/data/depth_scaling_gpu256.jsonl
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

CPU_FILE = os.path.join(DATA_DIR, "depth_scaling_cpu256.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "depth_scaling_gpu256.jsonl")

OUTPUT_PNG = os.path.join(PLOT_DIR, "fig5_depth_scaling.png")

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


def aggregate_by_depth(rows):
    """Median per depth; IQR error bars only when n_trials >= 3."""
    by_d = defaultdict(list)
    for r in rows:
        d  = r.get("depth")
        rt = r.get("run_time_ms")
        if d is not None and rt is not None and rt > 0:
            by_d[d].append(rt / 1000.0)

    result = {}
    for d in sorted(by_d):
        vals = np.asarray(by_d[d], dtype=float)
        cnt  = int(vals.size)
        med  = float(np.median(vals))
        if cnt >= 3:
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            has_iqr = True
        else:
            q25 = q75 = med
            has_iqr = False
        result[d] = dict(median=med, q25=q25, q75=q75,
                         n_trials=cnt, has_iqr=has_iqr)
    return result


def linear(x, a, b):
    return a * np.asarray(x, dtype=float) + b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_agg = aggregate_by_depth(load_jsonl(CPU_FILE))
    gpu_agg = aggregate_by_depth(load_jsonl(GPU_FILE))

    print(f"CPU depths: {sorted(cpu_agg)}")
    print(f"GPU depths: {sorted(gpu_agg)}")

    def _arrays(agg):
        keys = sorted(agg)
        ds   = np.array(keys, dtype=float)
        meds = np.array([agg[d]["median"] for d in keys])
        return ds, meds

    cpu_keys = sorted(cpu_agg)
    gpu_keys = sorted(gpu_agg)
    cpu_x, cpu_y = _arrays(cpu_agg)
    gpu_x, gpu_y = _arrays(gpu_agg)

    # Linear fits
    popt_cpu, _ = curve_fit(linear, cpu_x, cpu_y)
    popt_gpu, _ = curve_fit(linear, gpu_x, gpu_y)
    slope_ratio = popt_cpu[0] / popt_gpu[0]

    print(f"\n-- Linear fits --")
    print(f"  CPU: slope={popt_cpu[0]:.2f}, intercept={popt_cpu[1]:.2f}")
    print(f"  GPU: slope={popt_gpu[0]:.2f}, intercept={popt_gpu[1]:.2f}")
    print(f"  Slope ratio: {slope_ratio:.1f}x")

    lo_x = min(cpu_x.min(), gpu_x.min()) - 2
    hi_x = max(cpu_x.max(), gpu_x.max()) + 2
    x_fit = np.linspace(lo_x, hi_x, 300)

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

    ax.plot(x_fit, linear(x_fit, *popt_cpu), "--", color=CPU_COLOR,
            linewidth=2.2, alpha=0.75, zorder=4,
            label=f"CPU fit: slope {popt_cpu[0]:.1f}")
    ax.plot(x_fit, linear(x_fit, *popt_gpu), "--", color=GPU_COLOR,
            linewidth=2.2, alpha=0.75, zorder=4,
            label=f"GPU fit: slope {popt_gpu[0]:.1f}")

    ax.text(0.97, 0.05,
            f"Slope ratio: {slope_ratio:.1f}$\\times$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color="#444",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#bbb", alpha=0.92))

    ax.set_xlabel("Circuit Depth ($d$)", fontsize=13)
    ax.set_ylabel("Run Time (s)", fontsize=13)
    ax.set_title(
        r"Depth scaling  ($n = 40$, $\chi = 256$)",
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
