"""
pps_accuracy_comparison.py
══════════════════════════
Accuracy error vs truncation threshold δ.

BlueQubit: PPS GPU on AMD (MI300X), same δ subsample as the runtime plot
(`thin_dense_gpu_tail`). Qiskit and PauliPropagation.jl use all available
points (they stop where benchmarks end). x-axis spans to ~1.5B Pauli terms.

Output:  plots/pps_accuracy_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pps_plot_utils as U

# ── Load ───────────────────────────────────────────────────────────────────
U.apply_style()
print("Loading benchmarks...")
data = U.load_all_backends(
    backend_keys={"PPS-GPU", "PPS-Qiskit", "PauliPropagation.jl"},
    pps_gpu_benchmark_filename="pps_gpu_benchmark_mi300x.jsonl",
)

plot_data = {
    k: (U.thin_dense_gpu_tail(data[k]) if k == "PPS-GPU" else data[k])
    for k in data
}

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")

for label_key, cfg in U.BACKENDS.items():
    if label_key not in plot_data:
        continue
    d = plot_data[label_key]
    eps_mean = np.abs(U.O_EXACT - d["exp_mean"])

    color = cfg["color"]
    label = cfg.get("method", label_key)
    marker = cfg["marker"]
    if "BlueQubit" in label:
        color = "#2563EB"
        marker = "o"

    ax.plot(
        d["delta"],
        eps_mean,
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

ax.set_yscale("log")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlim(*U.inverted_delta_xlim(plot_data))
ax.set_ylabel(r"Error ($\epsilon = |O_{\mathrm{exact}} - O_{\mathrm{PPS}}|$)")
ax.set_xlabel(r"Truncation Threshold ($\delta$)")
ax.set_title(
    "Accuracy Error vs truncation threshold",
    loc="center",
    fontweight="bold",
    pad=10,
)
ax.grid(True, which="major", ls="--", alpha=0.35)
ax.grid(True, which="minor", ls=":", alpha=0.15)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left")

# Secondary top axis — Pauli counts
ax_p = U.add_pauli_top_axis(ax, U.unique_deltas_from_plot_data(plot_data))

# ── Shade GPU-only regime (finest δ reached only on GPU) ─────────────────
if "PPS-GPU" in data:
    gpu_min = float(np.min(data["PPS-GPU"]["delta"]))
    other_mins = [float(np.min(data[l]["delta"])) for l in data if l != "PPS-GPU"]
    cpu_boundary = min(other_mins) if other_mins else gpu_min
    if gpu_min < cpu_boundary:
        ax.axvspan(
            gpu_min,
            cpu_boundary,
            alpha=0.06,
            color=U.BACKENDS["PPS-GPU"]["color"],
            zorder=0,
        )

# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(U.PLOT_DIR / "pps_accuracy_comparison.png")
print("\nSaved pps_accuracy_comparison.png")
plt.close(fig)
