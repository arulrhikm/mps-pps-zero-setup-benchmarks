"""
pps_runtime_comparison.py
═════════════════════════
Standalone plot showing PPS runtime vs truncation threshold δ
for all backends.

Output:  plots/pps_runtime_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pps_plot_utils as U

# ── Load ───────────────────────────────────────────────────────────────────
U.apply_style()
print("Loading benchmarks...")
data = U.load_all_backends()

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")

for label, cfg in U.BACKENDS.items():
    if label not in data:
        continue
    d = data[label]
    ax.plot(
        d["delta"], d["time_s_mean"],
        marker=cfg["marker"],
        linestyle="-",
        linewidth=1.6,
        color=cfg["color"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=7,
        label=cfg.get("label", label),
        zorder=cfg["zorder"],
    )

ax.set_yscale("log")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_ylabel("Runtime  (seconds)")
ax.set_xlabel(r"Truncation threshold  $\delta$")
ax.set_title("Runtime vs truncation threshold",
             loc="center", fontweight="bold", pad=35)
ax.grid(True, which="major", ls="--", alpha=0.35)
ax.grid(True, which="minor", ls=":", alpha=0.15)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left")

# Secondary top axis — Pauli counts
ax_p = U.add_pauli_top_axis(ax)

# ── Shade GPU-only regime ─────────────────────────────────────────────────
if "PPS-GPU" in data:
    gpu_min = data["PPS-GPU"]["delta"].min()
    other_mins = [data[l]["delta"].min() for l in data if l != "PPS-GPU"]
    cpu_boundary = min(other_mins) if other_mins else gpu_min
    if gpu_min < cpu_boundary:
        ax.axvspan(gpu_min, cpu_boundary,
                   alpha=0.06, color=U.BACKENDS["PPS-GPU"]["color"],
                   zorder=0)

# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(U.PLOT_DIR / "pps_runtime_comparison.png")
print("\nSaved pps_runtime_comparison.png")
plt.close(fig)
