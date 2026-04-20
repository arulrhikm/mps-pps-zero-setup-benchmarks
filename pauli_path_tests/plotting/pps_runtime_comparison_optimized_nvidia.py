"""
pps_runtime_comparison_optimized_nvidia.py
══════════════════════════════════════════
Same layout as ``pps_runtime_comparison_optimized.png`` (optimized PPS-GPU,
CPU, Qiskit, Julia) plus **three measured NVIDIA points** estimated from
your measured speedups over PPS-Qiskit at three cutoffs (bar-chart reference):

  δ = 1e-4   → 56×
  δ = 5e-5   → 108×
  δ = 2.5e-5 → 177×

Runtime at each measured δ is ``t_qiskit_mean / measured_speedup``.
No interpolation or extrapolation is used.

Output: plots/pps_runtime_comparison_optimized_nvidia.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pps_plot_utils as U

# NVIDIA speedup vs PPS-Qiskit (measured), δ → factor
_NVIDIA_SPEEDUP_MEASURED = {
    1e-4: 56.0,
    5e-5: 108.0,
    2.5e-5: 177.0,
}


def _thin_dense_gpu_tail(d: dict) -> dict:
    """
    Keep all points except the ultra-dense low-delta tail, where we keep
    representative anchors to reduce marker clutter.
    """
    delta = d["delta"]
    # Remove points in the mid-band (149M .. 670M) to reduce clutter.
    drop_mid_band = {9e-06, 8e-06, 7e-06, 6e-06, 5e-06}
    keep_tail = {4.5e-06, 2.89453125e-06}
    mask = (~np.isin(delta, list(drop_mid_band))) & ((delta >= 1e-05) | np.isin(delta, list(keep_tail)))
    return {k: np.asarray(v)[mask] for k, v in d.items()}


def build_nvidia_series(qiskit_agg: dict) -> dict:
    """Return only the three measured NVIDIA points derived from Qiskit means."""
    d_map = {float(d): i for i, d in enumerate(qiskit_agg["delta"])}
    deltas = []
    t_nv = []
    for delta, speedup in sorted(_NVIDIA_SPEEDUP_MEASURED.items(), reverse=True):
        if delta not in d_map:
            raise SystemExit(f"Missing PPS-Qiskit mean runtime for measured delta={delta}")
        qi = d_map[delta]
        deltas.append(delta)
        t_nv.append(float(qiskit_agg["time_s_mean"][qi]) / speedup)
    return {"delta": np.array(deltas, dtype=float), "time_s_mean": np.array(t_nv, dtype=float)}


# ── Load ───────────────────────────────────────────────────────────────────
U.apply_style()
print("Loading benchmarks...")
data = U.load_all_backends()

optimized_gpu_fp = U.DATA_DIR / "pps_gpu_optimized_benchmark.jsonl"
if optimized_gpu_fp.exists():
    optimized_gpu_raw = U.load_jsonl(optimized_gpu_fp)
    data["PPS-GPU"] = U.aggregate(optimized_gpu_raw)
    print(
        "  [ok]   PPS-GPU (optimized)      "
        f"{len(data['PPS-GPU']['delta']):2d} delta-values  "
        f"({len(optimized_gpu_raw)} records)"
    )

if "PPS-Qiskit" not in data:
    raise SystemExit("PPS-Qiskit data required to estimate NVIDIA runtimes")

nvidia_series = build_nvidia_series(data["PPS-Qiskit"])
print(
    f"  [ok]   NVIDIA (from Qiskit / speedup)  {len(nvidia_series['delta']):2d} delta-values"
)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")

for label, cfg in U.BACKENDS.items():
    if label not in data:
        continue
    d = data[label]
    if label == "PPS-GPU":
        d = _thin_dense_gpu_tail(d)
    ax.plot(
        d["delta"],
        d["time_s_mean"],
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

# NVIDIA (three measured points only)
ax.plot(
    nvidia_series["delta"],
    nvidia_series["time_s_mean"],
    marker="s",
    linestyle="None",
    color="#76B900",
    markeredgecolor="white",
    markeredgewidth=0.6,
    markersize=8,
    label="NVIDIA PPS",
    zorder=5,
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_ylabel("Runtime  (seconds)")
ax.set_xlabel(r"Truncation threshold  $\delta$")
ax.set_title(
    "Runtime vs truncation threshold",
    loc="center",
    fontweight="bold",
    pad=35,
)
ax.grid(True, which="major", ls="--", alpha=0.35)
ax.grid(True, which="minor", ls=":", alpha=0.15)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left")

ax_top = U.add_pauli_top_axis(ax)

if "PPS-GPU" in data:
    gpu_min = data["PPS-GPU"]["delta"].min()
    other_mins = [data[l]["delta"].min() for l in data if l != "PPS-GPU"]
    cpu_boundary = min(other_mins) if other_mins else gpu_min
    if gpu_min < cpu_boundary:
        ax.axvspan(
            gpu_min,
            cpu_boundary,
            alpha=0.06,
            color=U.BACKENDS["PPS-GPU"]["color"],
            zorder=0,
        )

out = U.PLOT_DIR / "pps_runtime_comparison_optimized_nvidia.png"
fig.savefig(out)
print(f"\nSaved {out.name}")
plt.close(fig)
