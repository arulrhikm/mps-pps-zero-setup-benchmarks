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

# Optional replacement for PPS-GPU using optimized benchmark file.
optimized_gpu_fp = U.DATA_DIR / "pps_gpu_optimized_benchmark.jsonl"
if optimized_gpu_fp.exists():
    optimized_gpu_raw = U.load_jsonl(optimized_gpu_fp)
    data["PPS-GPU"] = U.aggregate(optimized_gpu_raw)
    print(
        "  [ok]   PPS-GPU (optimized)      "
        f"{len(data['PPS-GPU']['delta']):2d} delta-values  "
        f"({len(optimized_gpu_raw)} records)"
    )

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")

plot_data = {}
for label, cfg in U.BACKENDS.items():
    if label not in data:
        continue
    d = data[label]
    if label == "PPS-GPU":
        d = U.thin_dense_gpu_tail(d)
    plot_data[label] = d
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
if plot_data:
    ax.set_xlim(*U.inverted_delta_xlim(plot_data))
ax.set_ylabel("Runtime  (seconds)")
ax.set_xlabel(r"Truncation threshold  $\delta$")
ax.set_title("Runtime vs truncation threshold",
             loc="center", fontweight="bold", pad=10)
ax.grid(True, which="major", ls="--", alpha=0.35)
ax.grid(True, which="minor", ls=":", alpha=0.15)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left")

# Secondary top axis — Pauli counts
ax_p = U.add_pauli_top_axis(
    ax,
    U.unique_deltas_from_plot_data(plot_data) if plot_data else None,
)

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
fig.savefig(U.PLOT_DIR / "pps_runtime_comparison_optimized.png")
print("\nSaved pps_runtime_comparison_optimized.png")
plt.close(fig)
