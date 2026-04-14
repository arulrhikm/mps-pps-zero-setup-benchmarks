"""
plot_experiment6_qft_scaling.py
================================
QFT runtime scaling: MPS CPU vs MPS GPU.
  - bond_dimension = 64, approximation_degree = 0 (exact QFT)
  - x-axis: number of qubits (4 → 96)
  - y-axis: runtime (seconds, log scale)
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CPU_FILE = os.path.join(BASE, "experiment6_qft_scaling_cpu.jsonl")
GPU_FILE = os.path.join(BASE, "experiment6_qft_scaling_gpu.jsonl")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "plots", "experiment6_qft_scaling.png")

BOND_DIM    = 64
APPROX_DEG  = 0

# ── Load ──────────────────────────────────────────────────────────────────────
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


def extract(rows, bond_dim, degree):
    buckets = defaultdict(list)
    for r in rows:
        if r.get("bond_dimension") == bond_dim and r.get("approximation_degree") == degree:
            buckets[r["num_qubits"]].append(r["run_time_ms"])
    xs = sorted(buckets)
    ys = [np.mean(buckets[n]) / 1000.0 for n in xs]
    return np.array(xs, dtype=float), np.array(ys)


x_cpu, y_cpu = extract(load_jsonl(CPU_FILE), BOND_DIM, APPROX_DEG)
x_gpu, y_gpu = extract(load_jsonl(GPU_FILE), BOND_DIM, APPROX_DEG)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

CPU_COLOR = "#2196F3"
GPU_COLOR = "#FF5722"

ax.semilogy(x_cpu, y_cpu, "o-", color=CPU_COLOR, lw=2.2, markersize=6,
            markeredgewidth=0.6, markeredgecolor="white", label="MPS CPU")
ax.semilogy(x_gpu, y_gpu, "s-", color=GPU_COLOR, lw=2.2, markersize=6,
            markeredgewidth=0.6, markeredgecolor="white", label="MPS GPU")

# Statevector wall shading (up to ~32q)
ax.axvspan(0, 32, alpha=0.05, color="#888888", zorder=0)

# Reference lines
y_top = max(y_cpu.max(), y_gpu.max()) * 1.6
for ref_n in [32, 64, 96]:
    ax.axvline(ref_n, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.text(ref_n + 0.6, y_top, f"{ref_n}q",
            fontsize=8, color="grey", va="top")

ax.set_xlabel("Number of Qubits", fontsize=12, labelpad=6)
ax.set_ylabel("Runtime (log scale)", fontsize=12, labelpad=8)
ax.set_title(
    "Experiment 6: QFT High-Qubit Scaling — MPS CPU vs GPU\n"
    r"Exact QFT  |  $\chi$ = 64  |  $n$ = 4–96 qubits",
    fontsize=13, pad=12,
)
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(ticker.MultipleLocator(16))
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"{v:g}s" if v >= 1 else f"{v*1000:.0f}ms")
)
ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
ax.set_facecolor("#FAFAFA")
ax.legend(fontsize=11, loc="upper left")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_FILE}")
plt.show()
