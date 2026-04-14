"""
plot_figure6_qft_scaling.py  (Section 5.5)
============================================
Figure 6: QFT runtime up to 96 qubits, MPS CPU vs GPU, chi=64.
  - Exact QFT only (approximation_degree=0, bond_dimension=64)
  - Log-scale y-axis
  - Mean over 5 trials (no error bars)
  - Speedup annotations at key qubit counts (32q, 96q)
  - Single figure (no duplicate)
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CPU_FILE = os.path.join(BASE, "experiment6_qft_scaling_cpu.jsonl")
GPU_FILE = os.path.join(BASE, "experiment6_qft_scaling_gpu.jsonl")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
OUT_FILE = os.path.join(OUT_DIR, "figure6_qft_scaling.png")

CPU_COLOR = "#2196F3"
GPU_COLOR = "#FF5722"
BOND_DIM = 64
APPROX_DEG = 0


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


def aggregate(rows):
    """Filter to degree-0 / bond-64 and aggregate over trials."""
    buckets = defaultdict(list)
    for r in rows:
        if r.get("bond_dimension") != BOND_DIM:
            continue
        if r.get("approximation_degree") != APPROX_DEG:
            continue
        buckets[r["num_qubits"]].append(r["run_time_ms"] / 1000.0)
    xs = sorted(buckets)
    means = np.array([np.mean(buckets[n]) for n in xs])
    stds = np.array([np.std(buckets[n], ddof=1) if len(buckets[n]) > 1 else 0.0
                      for n in xs])
    return np.array(xs, dtype=float), means, stds


# ── Load & aggregate ─────────────────────────────────────────────────────────
x_cpu, y_cpu, _ = aggregate(load_jsonl(CPU_FILE))
x_gpu, y_gpu, _ = aggregate(load_jsonl(GPU_FILE))

# ── Speedup at key points ────────────────────────────────────────────────────
cpu_map = dict(zip(x_cpu, y_cpu))
gpu_map = dict(zip(x_gpu, y_gpu))
speedup_points = [32, 96]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(x_cpu, y_cpu, linestyle="-", linewidth=1.6, marker="o", color=CPU_COLOR,
        markersize=6, markeredgewidth=0.6,
        markeredgecolor="white", label="MPS CPU")
ax.plot(x_gpu, y_gpu, linestyle="-", linewidth=1.6, marker="s", color=GPU_COLOR,
        markersize=6, markeredgewidth=0.6,
        markeredgecolor="white", label="MPS GPU")

ax.set_yscale("log")

# Statevector practical-limit shading
ax.axvspan(0, 32, alpha=0.05, color="#888888", zorder=0)

# Reference lines and speedup annotations
for ref_n in speedup_points:
    if ref_n in cpu_map and ref_n in gpu_map:
        spd = cpu_map[ref_n] / gpu_map[ref_n]
        y_mid = np.sqrt(cpu_map[ref_n] * gpu_map[ref_n])
        ax.annotate(
            f"{spd:.1f}$\\times$",
            xy=(ref_n, y_mid), fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray",
                      alpha=0.9),
        )
    ax.axvline(ref_n, color="grey", lw=0.8, ls=":", alpha=0.5)

y_top = max(y_cpu.max(), y_gpu.max()) * 1.6
for ref_n in [32, 64, 96]:
    ax.text(ref_n + 0.8, y_top, f"{ref_n}q", fontsize=8, color="grey", va="top")

ax.set_xlabel("Number of Qubits ($n$)", fontsize=12, labelpad=6)
ax.set_ylabel("Runtime (log scale)", fontsize=12, labelpad=8)
ax.set_title(
    "Figure 6: QFT High-Qubit Scaling — MPS CPU vs GPU\n"
    r"Exact QFT ($\mathrm{degree}=0$)  |  $\chi = 64$  |  5 trials per point",
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
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_FILE}")

print("\nSpeedup summary:")
for n in speedup_points:
    if n in cpu_map and n in gpu_map:
        print(f"  {n}q: CPU={cpu_map[n]:.3f}s  GPU={gpu_map[n]:.3f}s  "
              f"speedup={cpu_map[n]/gpu_map[n]:.1f}x")
plt.show()
