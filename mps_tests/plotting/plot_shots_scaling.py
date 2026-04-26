#!/usr/bin/env python3
"""
MPS shots scaling — build vs sampling (stacked bars)

Reads:
  mps_tests/data/shots_scaling_cpu.jsonl
  mps_tests/data/shots_scaling_gpu.jsonl

Fixed circuit filter: n=40, depth=10, bond_dimension=256 (matches experiment defaults).
Per shot count: median T_build and T_sampling (seconds); stacked bars.

Output:
  mps_tests/plots/fig_shots_build_sampling.png  (CPU | GPU side by side)
"""
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")

CPU_FILE = os.path.join(DATA_DIR, "shots_scaling_cpu.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "shots_scaling_gpu.jsonl")
OUT_COMBINED = os.path.join(PLOT_DIR, "fig_shots_build_sampling.png")

NUM_QUBITS = 40
DEPTH = 10
BOND_DIM = 256

CPU_COLOR_BUILD = "#6baed6"
CPU_COLOR_SAMP = "#2171b5"
GPU_COLOR_BUILD = "#fcae91"
GPU_COLOR_SAMP = "#cb181d"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print(f"  WARNING: missing {path}")
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


def aggregate(rows):
    """
    shots -> build/sampling/total per trial (seconds).
    Median + IQR when n >= 3 for each statistic.
    """
    by = defaultdict(list)
    for r in rows:
        if r.get("num_qubits") != NUM_QUBITS:
            continue
        if r.get("depth") != DEPTH or r.get("bond_dimension") != BOND_DIM:
            continue
        if r.get("mps_build_time_ms") is None:
            continue
        rt = r.get("run_time_ms")
        if rt is None or rt <= 0:
            continue
        sh = r["shots"]
        b = float(r["mps_build_time_ms"]) / 1000.0
        sp = float(r["sampling_time_ms"]) / 1000.0
        by[sh].append((b, sp, b + sp))

    out = {}
    for sh in sorted(by):
        arr = np.array(by[sh], dtype=float)
        n = arr.shape[0]
        bs, ss, tt = arr[:, 0], arr[:, 1], arr[:, 2]

        def stat1d(a):
            med = float(np.median(a))
            if n >= 3:
                lo = float(np.percentile(a, 25))
                hi = float(np.percentile(a, 75))
                return med, lo, hi, True
            return med, med, med, False

        bm, bl, bh, bi = stat1d(bs)
        sm, sl, shh, si = stat1d(ss)
        tm, tl, th, ti = stat1d(tt)
        out[sh] = {
            "build_med": bm,
            "build_lo": bl,
            "build_hi": bh,
            "build_iqr": bi,
            "samp_med": sm,
            "samp_lo": sl,
            "samp_hi": shh,
            "samp_iqr": si,
            "tot_med": tm,
            "tot_lo": tl,
            "tot_hi": th,
            "tot_iqr": ti,
            "n": n,
        }
    return out


def draw_panel(ax, agg, title, c_build, c_samp):
    if not agg:
        ax.set_title(title + " (no data)")
    else:
        shots = sorted(agg)
        x = np.arange(len(shots), dtype=float)
        build = np.array([agg[s]["build_med"] for s in shots])
        samp = np.array([agg[s]["samp_med"] for s in shots])
        w = 0.62
        ax.bar(
            x,
            build,
            w,
            color=c_build,
            edgecolor="white",
            linewidth=0.6,
            label=r"$T_{\mathrm{build}}$",
        )
        ax.bar(
            x,
            samp,
            w,
            bottom=build,
            color=c_samp,
            edgecolor="white",
            linewidth=0.6,
            label=r"$T_{\mathrm{sampling}}$",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in shots], fontsize=10)
        ymax = float(np.max(build + samp)) * 1.06
        ax.set_ylim(0.0, ymax)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(title, fontsize=12)

    ax.set_xlabel("Shots", fontsize=11)
    ax.set_ylabel("Run Time (s)", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_combined(cpu_agg, gpu_agg, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=False)
    draw_panel(axes[0], cpu_agg, "MPS CPU", CPU_COLOR_BUILD, CPU_COLOR_SAMP)
    draw_panel(axes[1], gpu_agg, "MPS GPU", GPU_COLOR_BUILD, GPU_COLOR_SAMP)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cpu_agg = aggregate(load_jsonl(CPU_FILE))
    gpu_agg = aggregate(load_jsonl(GPU_FILE))

    print(f"CPU shot buckets: {sorted(cpu_agg)}")
    print(f"GPU shot buckets: {sorted(gpu_agg)}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_combined(cpu_agg, gpu_agg, OUT_COMBINED)
    print(f"Saved: {OUT_COMBINED}")


if __name__ == "__main__":
    main()
