"""
plot_experiment6_bond_scaling.py
=================================
QFT runtime vs bond dimension for both MPS CPU and GPU.

  x-axis: bond dimension (χ = 64, 256, 512, 768, 1024)
  y-axis: runtime (seconds, log scale)
  lines:  one per qubit count (selected representative values)
  panels: MPS CPU (left) | MPS GPU (right)

Data source: experiment6_qft_scaling_{cpu,gpu}.jsonl
             (merged file containing all bond dims at approximation_degree=0)
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CPU_FILE = os.path.join(BASE, "experiment6b_qft_bond_cpu.jsonl")
GPU_FILE = os.path.join(BASE, "experiment6b_qft_bond_gpu.jsonl")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "plots", "experiment6_qft_bond_scaling.png")

APPROX_DEG   = 0
SHOW_QUBITS  = [16, 32, 48, 64, 80, 96]   # representative qubit counts to plot

# ── Color map for qubit counts ─────────────────────────────────────────────────
CMAP   = plt.cm.plasma
COLORS = {n: CMAP(i / (len(SHOW_QUBITS) - 1)) for i, n in enumerate(SHOW_QUBITS)}

# ── Load & group ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
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


def group_by_qubit(rows, degree):
    """Return {num_qubits: {bond_dim: mean_runtime_s}}"""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("approximation_degree") == degree:
            buckets[r["num_qubits"]][r["bond_dimension"]].append(r["run_time_ms"])
    result = {}
    for n, bond_map in buckets.items():
        result[n] = {X: np.mean(v) / 1000.0 for X, v in bond_map.items()}
    return result


cpu_data = group_by_qubit(load_jsonl(CPU_FILE), APPROX_DEG)
gpu_data = group_by_qubit(load_jsonl(GPU_FILE), APPROX_DEG)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.subplots_adjust(wspace=0.06)
plt.rcParams.update({"font.family": "sans-serif"})

for ax, data, title in zip(axes, [cpu_data, gpu_data], ["MPS CPU", "MPS GPU"]):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")

    # Collect which bond dims actually have data in this panel
    available_bonds = sorted({
        X for n in SHOW_QUBITS if n in data
        for X in data[n]
    })

    plotted_any = False
    for n in SHOW_QUBITS:
        if n not in data:
            continue
        bond_map = data[n]
        xs = sorted(b for b in bond_map if b in available_bonds)
        ys = [bond_map[X] for X in xs]
        if len(xs) < 2:
            continue
        ax.semilogy(xs, ys, "o-",
                    color=COLORS[n], lw=2.0, markersize=6,
                    markeredgewidth=0.6, markeredgecolor="white",
                    label=f"{n}q")
        plotted_any = True

    # X axis: linear, ticks only where data exists
    ax.set_xscale("linear")
    ax.set_xticks(available_bonds)
    ax.set_xticklabels([str(b) for b in available_bonds], fontsize=9)
    padding = (available_bonds[-1] - available_bonds[0]) * 0.08
    ax.set_xlim(available_bonds[0] - padding, available_bonds[-1] + padding)

    ax.set_xlabel(r"Bond Dimension  ($\chi$)", fontsize=12, labelpad=6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:g}s" if v >= 1 else f"{v*1000:.0f}ms")
    )
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
    ax.grid(True, which="major", axis="x", ls="-", lw=0.8, alpha=0.2)
    if plotted_any:
        ax.legend(title="Qubit count", title_fontsize=9,
                  fontsize=9, loc="upper left", ncol=2)

axes[0].set_ylabel("Runtime (log scale)", fontsize=12, labelpad=8)

fig.suptitle(
    "Experiment 6: QFT Runtime vs Bond Dimension — CPU vs GPU\n"
    r"Exact QFT (approximation\_degree = 0)",
    fontsize=13, y=1.01,
)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_FILE}")
plt.show()
