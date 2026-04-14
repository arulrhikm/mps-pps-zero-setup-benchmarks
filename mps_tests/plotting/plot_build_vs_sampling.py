#!/usr/bin/env python3
"""
MPS build time vs bond dimension (single panel)
===============================================
Log-log plot of T_build with power-law fits for CPU and GPU.
Sampling is negligible vs build at shots=1; stated in the subtitle only.

Data:
  mps_tests/data/bond_scaling_cpu.jsonl
  mps_tests/data/bond_scaling_gpu.jsonl
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

CPU_FILE = os.path.join(DATA_DIR, "bond_scaling_cpu.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "bond_scaling_gpu.jsonl")

OUTPUT_PNG = os.path.join(PLOT_DIR, "fig3_build_vs_sampling.png")

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


def aggregate_build(rows):
    """Group by bond_dimension using mps_build_time_ms; IQR only if n >= 3."""
    build_by = defaultdict(list)

    for r in rows:
        chi   = r.get("bond_dimension")
        build = r.get("mps_build_time_ms")
        if chi is None or build is None:
            continue
        build_by[chi].append(build / 1000.0)

    result = {}
    for chi in sorted(build_by):
        bv = np.asarray(build_by[chi], dtype=float)
        n = int(bv.size)
        b_med = float(np.median(bv))
        if n >= 3:
            b_q25 = float(np.percentile(bv, 25))
            b_q75 = float(np.percentile(bv, 75))
            has_iqr = True
        else:
            b_q25 = b_q75 = b_med
            has_iqr = False
        result[chi] = dict(
            build_med=b_med, build_q25=b_q25, build_q75=b_q75,
            n_trials=n, has_iqr=has_iqr,
        )
    return result


def fit_power_law(chis, meds):
    """Unweighted log-log fit on measured medians (no synthetic weights)."""
    def _model(lx, log_a, alpha):
        return log_a + alpha * lx
    try:
        popt, pcov = curve_fit(_model, np.log(chis), np.log(meds), p0=[0.0, 2.0])
        perr = np.sqrt(np.diag(pcov))
        return np.exp(popt[0]), popt[1], perr[1]
    except Exception as exc:
        print(f"  Fit failed: {exc}")
        return None, None, None


def power_law(x, a, alpha):
    return a * np.asarray(x, dtype=float) ** alpha


def main():
    cpu_data = aggregate_build(load_jsonl(CPU_FILE))
    gpu_data = aggregate_build(load_jsonl(GPU_FILE))

    print(f"CPU chi (with build_time): {sorted(cpu_data)}")
    print(f"GPU chi (with build_time): {sorted(gpu_data)}")

    def _fit(data, label):
        chis = np.array(sorted(data), dtype=float)
        meds = np.array([data[c]["build_med"] for c in chis])
        a, alpha, ae = fit_power_law(chis, meds)
        if a is not None:
            print(f"  {label} T_build = {a:.4g} * chi^{alpha:.3f}  +/-{ae:.3f}")
        return a, alpha, ae

    print("\n-- Build-phase fits --")
    cpu_a, cpu_alpha, cpu_ae = _fit(cpu_data, "CPU")
    gpu_a, gpu_alpha, gpu_ae = _fit(gpu_data, "GPU")

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for data, color, marker, lbl, a, alpha, ae in [
        (cpu_data, CPU_COLOR, "o", "CPU", cpu_a, cpu_alpha, cpu_ae),
        (gpu_data, GPU_COLOR, "^", "GPU", gpu_a, gpu_alpha, gpu_ae),
    ]:
        if not data:
            continue
        chis = np.array(sorted(data), dtype=float)
        bmed = np.array([data[c]["build_med"] for c in chis])
        ntri = [data[c]["n_trials"] for c in chis]

        ax.plot(
            chis, bmed,
            linestyle="None", marker=marker, color=color, markersize=9,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=5,
            label=rf"{lbl} $T_{{\mathrm{{build}}}}$",
        )

        for c, m, n in zip(chis, bmed, ntri):
            ax.annotate(f"n={n}", xy=(c, m),
                          xytext=(0, 11), textcoords="offset points",
                          fontsize=6, color=color, ha="center", alpha=0.75)

        if a is not None:
            xs = np.logspace(np.log10(chis.min()), np.log10(chis.max()), 200)
            ax.plot(xs, power_law(xs, a, alpha), "--", color=color,
                    linewidth=2.2, zorder=4,
                    label=rf"{lbl} fit: $T_{{\mathrm{{build}}}} "
                          rf"\propto \chi^{{{alpha:.2f}\pm{ae:.2f}}}$")

    if cpu_alpha and gpu_alpha:
        diff = cpu_alpha - gpu_alpha
        ax.text(0.97, 0.04,
                f"Build exponents:\n"
                f"  CPU $\\chi^{{{cpu_alpha:.2f}}}$,  "
                f"GPU $\\chi^{{{gpu_alpha:.2f}}}$\n"
                f"  GPU advantage $\\chi^{{{diff:.2f}}}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, color="#444",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#bbb", alpha=0.92))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Bond Dimension  $\chi$", fontsize=12)
    ax.set_ylabel(r"$T_{\mathrm{build}}$  (s)", fontsize=12)
    ax.set_title(
        r"MPS build time vs $\chi$  (shots$=1$; sampling not plotted)",
        fontsize=12, pad=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(loc="upper left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nSaved:\n  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
