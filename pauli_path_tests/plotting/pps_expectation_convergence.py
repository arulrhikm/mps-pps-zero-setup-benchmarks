"""
pps_expectation_convergence.py
══════════════════════════════
Shows how the PPS estimate of ⟨Z_62⟩ converges toward the exact value
as the truncation threshold δ is lowered.

  Main plot:   ⟨Z_62⟩_PPS  vs  δ   for every backend
               Horizontal band for O_exact ± tolerance
  Key insight: GPU accesses a convergence regime unreachable by CPU,
               demonstrating that this is a *capability* gap, not just speed.

Output:  pps_expectation_convergence.{pdf,png}
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

for label_key, cfg in U.BACKENDS.items():
    if label_key not in data:
        continue
    d = data[label_key]

    # Unify colors and markers for unified method labels
    color  = cfg["color"]
    label  = cfg.get("method", label_key)
    marker = cfg["marker"]
    if "BlueQubit" in label:
        color  = "#2563EB"
        marker = "o"
    
    ax.plot(
        d["delta"], d["exp_mean"],
        marker=marker,
        linestyle="-",
        linewidth=1.6,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=7,
        label=label,
        zorder=cfg["zorder"],
    )

# Exact reference
ax.axhline(U.O_EXACT, color="black", ls="-", lw=1.2, alpha=0.7, zorder=0)
ax.axhspan(U.O_EXACT - 0.005, U.O_EXACT + 0.005,
           color="black", alpha=0.06, zorder=0)
ax.text(0.98, U.O_EXACT - 0.002,
        f"$O_{{\\mathrm{{exact}}}} = {U.O_EXACT}$",
        transform=ax.get_yaxis_transform(),
        fontsize=10, va="top", ha="right", color="0.4")

ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel(r"Truncation threshold  $\delta$")
ax.set_ylabel(r"$\langle Z_{62} \rangle_{\mathrm{PPS}}$")
ax.set_title(r"Convergence of $\langle Z_{62}\rangle$ estimate "
             "with decreasing $\\delta$",
             loc="center", fontweight="bold", pad=32)
ax.set_xlim(1.1e-2, 4e-6)
ax.grid(True, which="major", ls="--", alpha=0.35)
ax.grid(True, which="minor", ls=":", alpha=0.15)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="best")

# Secondary top axis
ax2 = U.add_pauli_top_axis(ax)

# ── Shade GPU-only regime ─────────────────────────────────────────────────
if "PPS-GPU" in data:
    gpu_min = data["PPS-GPU"]["delta"].min()
    other_mins = [data[l]["delta"].min() for l in data if l != "PPS-GPU"]
    cpu_boundary = min(other_mins) if other_mins else gpu_min
    if gpu_min < cpu_boundary:
        ax.axvspan(gpu_min, cpu_boundary,
                   alpha=0.06, color=U.BACKENDS["PPS-GPU"]["color"],
                   zorder=0)
        mid = np.sqrt(gpu_min * cpu_boundary)
        ax.text(mid, ax.get_ylim()[0] + 0.003,
                "GPU-only regime",
                ha="center", va="bottom", fontsize=8.5,
                color=U.BACKENDS["PPS-GPU"]["color"],
                fontstyle="italic", fontweight="bold")

# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(U.PLOT_DIR / "pps_expectation_convergence.png")
print("\nSaved pps_expectation_convergence.png")
plt.close(fig)