#!/usr/bin/env python3
"""
Bond dimension scaling — MPS (CPU / GPU) and Quantum Rings
============================================================
Same figure recipe as ``plot_bond_scaling.py``, plus Quantum Rings wall-clock
runtime from the cross-platform benchmark (threshold = bond dimension).

Data
----
  mps_tests/data/bond_scaling_cpu.jsonl
  mps_tests/data/bond_scaling_gpu.jsonl
  crossplatform_tests/data/quantum_ring_mps_results.jsonl

Output
------
  mps_tests/plots/fig2_bond_dimension_scaling_with_quantum_rings.png
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
    "legend.fontsize":   8,
    "figure.dpi":        150,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPS_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(MPS_ROOT, "data")
PLOT_DIR = os.path.join(MPS_ROOT, "plots")
REPO_ROOT = os.path.dirname(MPS_ROOT)

CPU_FILE = os.path.join(DATA_DIR, "bond_scaling_cpu.jsonl")
GPU_FILE = os.path.join(DATA_DIR, "bond_scaling_gpu.jsonl")
QR_FILE = os.path.join(
    REPO_ROOT, "crossplatform_tests", "data", "quantum_ring_mps_results.jsonl"
)

OUTPUT_PNG = os.path.join(PLOT_DIR, "fig2_bond_dimension_scaling_with_quantum_rings.png")

CPU_COLOR = "#1f77b4"
GPU_COLOR = "#d62728"
QR_COLOR = "#2ca02c"

NUM_QUBITS = 40
DEPTH = 10
EXTRAP_CHI = 5000


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


def load_quantum_rings_rows(path):
    """QV bond sweep: same n, d as MPS bond scaling."""
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
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            if r.get("depth") != DEPTH:
                continue
            if r.get("num_qubits") is not None and r.get("num_qubits") != NUM_QUBITS:
                continue
            rows.append(r)
    return rows


def aggregate_by_chi(rows, time_field="run_time_ms"):
    by_chi = defaultdict(list)
    for r in rows:
        chi = r.get("bond_dimension")
        rt = r.get(time_field)
        if chi is not None and rt is not None and rt > 0:
            by_chi[chi].append(rt / 1000.0)

    result = {}
    for chi in sorted(by_chi):
        vals = np.asarray(sorted(by_chi[chi]), dtype=float)
        n = int(vals.size)
        med = float(np.median(vals))
        if n >= 3:
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            has_iqr = True
        else:
            q25 = q75 = med
            has_iqr = False
        result[chi] = dict(
            median=med, q25=q25, q75=q75, n_trials=n, has_iqr=has_iqr)
    return result


def fit_power_law(chi_arr, med_arr, iqr_arr=None, has_iqr_mask=None):
    log_chi = np.log(chi_arr)
    log_med = np.log(med_arr)

    def _model(lx, log_a, alpha):
        return log_a + alpha * lx

    kw = {}
    if iqr_arr is not None and has_iqr_mask is not None:
        sigma = np.ones_like(med_arr, dtype=float)
        for i in range(len(med_arr)):
            if has_iqr_mask[i] and iqr_arr[i] > 0:
                sigma[i] = max(1e-6, np.log1p(iqr_arr[i] / med_arr[i]))
            else:
                sigma[i] = 1.0
        kw = dict(sigma=sigma, absolute_sigma=False)
    try:
        popt, pcov = curve_fit(_model, log_chi, log_med, p0=[0.0, 2.0], **kw)
        perr = np.sqrt(np.diag(pcov))
        return np.exp(popt[0]), popt[1], perr[1]
    except Exception as exc:
        print(f"  Fit failed: {exc}")
        return None, None, None


def power_law(x, a, alpha):
    return a * np.asarray(x, dtype=float) ** alpha


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    cpu_agg = aggregate_by_chi(load_jsonl(CPU_FILE))
    gpu_agg = aggregate_by_chi(load_jsonl(GPU_FILE))
    qr_agg = aggregate_by_chi(load_quantum_rings_rows(QR_FILE))

    print(f"CPU chi: {sorted(cpu_agg)}")
    print(f"GPU chi: {sorted(gpu_agg)}")
    print(f"Quantum Rings chi: {sorted(qr_agg)}")

    def _fit(agg, label):
        chis = np.array(sorted(agg), dtype=float)
        meds = np.array([agg[c]["median"] for c in chis])
        iqrs = np.array([agg[c]["q75"] - agg[c]["q25"] for c in chis])
        mask = np.array([agg[c]["has_iqr"] for c in chis], dtype=bool)
        a, alpha, alpha_err = fit_power_law(chis, meds, iqrs, mask)
        if a is not None:
            print(f"  {label}: T = {a:.4g} * chi^{alpha:.3f}  +/-{alpha_err:.3f}")
        return a, alpha, alpha_err

    print("\n-- Power-law fits --")
    cpu_a, cpu_alpha, cpu_ae = _fit(cpu_agg, "CPU")
    gpu_a, gpu_alpha, gpu_ae = _fit(gpu_agg, "GPU")
    qr_a, qr_alpha, qr_ae = _fit(qr_agg, "Quantum Rings")
    if cpu_alpha and gpu_alpha:
        print(f"  GPU advantage exponent (vs CPU): {cpu_alpha - gpu_alpha:.3f}")

    fig, ax = plt.subplots(figsize=(9, 6.5))

    def _plot_series(agg, color, marker, label, ms=9):
        if not agg:
            return
        chis = np.array(sorted(agg), dtype=float)
        meds = np.array([agg[c]["median"] for c in chis])
        ntr = [agg[c]["n_trials"] for c in chis]
        ax.plot(
            chis, meds,
            linestyle="None", marker=marker, color=color, markersize=ms,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=5, label=label,
        )
        for c, m, n in zip(chis, meds, ntr):
            ax.annotate(f"n={n}", xy=(c, m),
                        xytext=(0, 11), textcoords="offset points",
                        fontsize=6, color=color, ha="center", alpha=0.75)

    _plot_series(cpu_agg, CPU_COLOR, "o", "MPS CPU")
    _plot_series(gpu_agg, GPU_COLOR, "^", "MPS GPU")
    _plot_series(qr_agg, QR_COLOR, "s", "Quantum Rings (wall clock, CUSTOM)")

    def _plot_fit(a, alpha, ae, color, chi_lo, chi_hi, label):
        if a is None:
            return
        xs = np.logspace(np.log10(chi_lo), np.log10(chi_hi), 200)
        ax.plot(xs, power_law(xs, a, alpha), "--", color=color,
                linewidth=2.2, zorder=4, label=label)

    def _plot_extrap(a, alpha, color, chi_start, chi_end, label):
        if a is None:
            return
        xs = np.logspace(np.log10(chi_start), np.log10(chi_end), 200)
        ax.plot(xs, power_law(xs, a, alpha), ":", color=color,
                linewidth=1.8, zorder=3, label=label)
        t_end = power_law(chi_end, a, alpha)
        hrs = t_end / 3600.0
        txt = f"{hrs:.1f} h" if hrs >= 1 else f"{t_end:.0f} s"
        ax.annotate(
            txt,
            xy=(chi_end, t_end),
            xytext=(0, -12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
            clip_on=False,
        )

    if cpu_agg:
        lo, hi = min(cpu_agg), max(cpu_agg)
        _plot_fit(cpu_a, cpu_alpha, cpu_ae, CPU_COLOR, lo, hi,
                  rf"CPU fit: $T \propto \chi^{{{cpu_alpha:.2f}\pm{cpu_ae:.2f}}}$"
                  if cpu_alpha else None)
        _plot_extrap(
            cpu_a, cpu_alpha, CPU_COLOR, hi, EXTRAP_CHI,
            f"CPU extrap. to $\\chi$={EXTRAP_CHI}",
        )

    if gpu_agg:
        lo, hi = min(gpu_agg), max(gpu_agg)
        _plot_fit(gpu_a, gpu_alpha, gpu_ae, GPU_COLOR, lo, hi,
                  rf"GPU fit: $T \propto \chi^{{{gpu_alpha:.2f}\pm{gpu_ae:.2f}}}$"
                  if gpu_alpha else None)
        _plot_extrap(
            gpu_a, gpu_alpha, GPU_COLOR, hi, EXTRAP_CHI,
            f"GPU extrap. to $\\chi$={EXTRAP_CHI}",
        )

    if qr_agg:
        lo, hi = min(qr_agg), max(qr_agg)
        _plot_fit(qr_a, qr_alpha, qr_ae, QR_COLOR, lo, hi,
                  rf"QR fit: $T \propto \chi^{{{qr_alpha:.2f}\pm{qr_ae:.2f}}}$"
                  if qr_alpha else None)
        _plot_extrap(
            qr_a, qr_alpha, QR_COLOR, hi, EXTRAP_CHI,
            f"QR extrap. to $\\chi$={EXTRAP_CHI}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, EXTRAP_CHI * 1.55)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 1.35)
    ax.set_xlabel(r"Bond Dimension ($\chi$)", fontsize=13)
    ax.set_ylabel("Run Time (s)", fontsize=13)
    ax.set_title(
        f"Bond dimension scaling  ($n={NUM_QUBITS}$, $d={DEPTH}$)  "
        f"— MPS and Quantum Rings",
        fontsize=12, pad=10,
    )
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    if cpu_alpha and gpu_alpha:
        diff = cpu_alpha - gpu_alpha
        ax.text(0.97, 0.04,
                f"MPS GPU vs CPU:\n$S(\\chi) \\propto \\chi^{{{diff:.2f}}}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#444",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#bbb", alpha=0.92))

    ax.legend(loc="upper left", fontsize=7.5, ncol=1,
              framealpha=0.92, edgecolor="#ccc")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.22)
    print(f"\nSaved:\n  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
