"""
pps_gpu_speedup.py
══════════════════
Standalone GPU speedup figure — shows the speedup factor (CPU/GPU) at
each shared δ value, plotted against the number of Pauli terms.

  Main plot:  Speedup  vs  N_P   (GPU vs each other backend)
  Annotates the peak speedup and the "embarrassingly parallel" argument.

Output:  pps_gpu_speedup.{pdf,png}
"""

import numpy as np
import matplotlib.pyplot as plt
import pps_plot_utils as U

# ── Load ───────────────────────────────────────────────────────────────────
U.apply_style()
print("Loading benchmarks...")
data = U.load_all_backends()

if "PPS-GPU" not in data:
    raise SystemExit("GPU data required for speedup plot")

# ── Compute speedup vs each other backend ─────────────────────────────────
gpu = data["PPS-GPU"]
comparisons = {}

for label in ("PPS-CPU", "PPS-Qiskit", "PauliPropagation.jl"):
    if label not in data:
        continue
    other = data[label]
    common = sorted(set(gpu["delta"]) & set(other["delta"]), reverse=True)
    if len(common) < 2:
        continue
    np_vals, speedups = [], []
    for cd in common:
        gi = np.argmin(np.abs(gpu["delta"] - cd))
        oi = np.argmin(np.abs(other["delta"] - cd))
        tg = gpu["time_s_mean"][gi]
        to = other["time_s_mean"][oi]
        sp = to / tg
        np_vals.append(U.delta_to_paulis(cd)[0])
        speedups.append(sp)

    comparisons[label] = {
        "np": np.array(np_vals),
        "speedup": np.array(speedups),
    }

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7.5))
fig.subplots_adjust(top=0.9, bottom=0.12, left=0.1, right=0.95)

markers = {"PPS-CPU": "s", "PPS-Qiskit": "^", "PauliPropagation.jl": "D"}
colors  = {"PPS-CPU": "#EA580C", "PPS-Qiskit": "#16A34A",
           "PauliPropagation.jl": "#7C3AED"}

for label, comp in comparisons.items():
    ax.plot(
        comp["np"], comp["speedup"],
        marker=markers.get(label, "o"),
        linestyle="-",
        linewidth=1.6,
        color=colors.get(label, "gray"),
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=7,
        label=f"vs {U.BACKENDS[label].get('label', label)}",
        zorder=3,
    )

    # annotate peak speedup
    peak_i = int(np.argmax(comp["speedup"]))
    # Vertical staggering in log-space
    offset_map = {"PPS-CPU": 1.25, "PPS-Qiskit": 1.6, "PauliPropagation.jl": 1.25}
    offset = offset_map.get(label, 1.2)
    
    ax.annotate(
        f"{comp['speedup'][peak_i]:.0f}×",
        xy=(comp["np"][peak_i], comp["speedup"][peak_i]),
        xytext=(comp["np"][peak_i] * 0.4, comp["speedup"][peak_i] * offset),
        fontsize=10, fontweight="bold",
        ha="right", va="bottom",
        color=colors.get(label, "gray"),
        arrowprops=dict(arrowstyle="-|>", color=colors.get(label, "gray"),
                        lw=1, mutation_scale=10),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

ax.axhline(1, color="0.6", ls=":", lw=1, zorder=0)
ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < ax.get_xlim()[1]
        else ax.get_xlim()[1],
        1.05, "no speedup", fontsize=8, color="0.5", va="bottom")

ax.set_xscale("log")
ax.set_yscale("log")

# Ensure headroom for annotations
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 5.0)

ax.set_xlabel("Number of Pauli terms  $N_P$")
ax.set_ylabel("Speedup   (other / GPU)")
ax.set_title("GPU speedup grows with problem size",
             loc="center", fontweight="bold")
ax.grid(True, which="major", ls="--", alpha=0.35)
ax.grid(True, which="minor", ls=":", alpha=0.15)
ax.legend(loc="upper left", framealpha=0.93)

# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(U.PLOT_DIR / "pps_gpu_speedup.png")
print("\nSaved pps_gpu_speedup.png")
plt.close(fig)