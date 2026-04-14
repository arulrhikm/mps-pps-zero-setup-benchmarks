"""
plot_all_figs.py  —  Reproduce plots from JSONL data files.

Produces:
  fig1_2_gpu_reproduction.png  — EV evolution + runtime (Pauli count proxy) over depth (Figs 1 & 2)
  fig3_5_gpu_reproduction.png  — Runtime (Pauli count proxy) + EV norm-decay vs num_ops (Figs 3, 4 & 5)
  fig6_gpu_reproduction.png    — Convergence vs δ, random θ_X, multiple T (Fig 6)
  fig7_gpu_reproduction.png    — Convergence vs δ, fixed θ_X=0.4, multiple T (Fig 7)
  fig8_gpu_reproduction.png    — <Z_62> vs δ sweep, multiple θ_X (Fig 8)

API note: the BlueQubit JobResult does NOT expose raw Pauli coefficient
histograms or Pauli counts (confirmed from SDK docs). The faithful quantitative
proxies used here are:
  • runtime (ms, log scale) -> proportional to # active Pauli terms (Fig 3 / Fig 2 analogue)
  • expectation_value -> norm-decay proxy (Fig 4 / 5 analogue), or observable evolution (Fig 1)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

# JSONL files live next to this script (or under ./data if present)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_alt = os.path.join(_SCRIPT_DIR, "data")
DATA_DIR = _alt if os.path.isdir(_alt) else _SCRIPT_DIR

# ─── Colour helpers ────────────────────────────────────────────────────────────
VIRIDIS   = plt.get_cmap("viridis")
PLASMA    = plt.get_cmap("plasma")
COOLWARM  = plt.get_cmap("coolwarm")

def vir(n, total):
    return VIRIDIS(0.15 + 0.7 * n / max(total - 1, 1))

def pla(n, total):
    return PLASMA(0.1 + 0.75 * n / max(total - 1, 1))


def load_jsonl(fname):
    rows = []
    with open(os.path.join(DATA_DIR, fname), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Figs 1 & 2  —  EV evolution (observable decay) + runtime (Pauli count proxy)
# API limitation: raw Pauli histograms are not returned by JobResult.
# Proxies: EV tracks coefficient redistribution; runtime ∝ # active Paulis.
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Figs 1 & 2 …")
rows12 = load_jsonl("fig1_2_pps_gpu.jsonl")
rows12 = [r for r in rows12 if "error" not in r]
rows12.sort(key=lambda r: r["trotter_step"])

steps    = [r["trotter_step"]    for r in rows12]
num_ops  = [r["num_operations"]  for r in rows12]
ev       = [r["expectation_value"] for r in rows12]
rt_ms    = [r["run_time_ms"]       for r in rows12]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    r"Figs 1 & 2 — PPS GPU: Pauli spreading proxy ($\delta = 5\times10^{-5}$, 127 qubits, random $\theta_X$)",
    fontsize=12, fontweight="bold"
)

# Left: EV vs num_ops  (Fig 1 analogue: observable tracks coefficient redistribution)
ax = axes[0]
ax.plot(num_ops, ev, color="#1B3FA0", linestyle="None", marker="o", markersize=5,
        label=r"$\langle Z_{62}\rangle$")
# mark the 6 paper snapshot gate counts
for snap in [902, 1353, 1804, 2706, 4059, 5412]:
    ax.axvline(snap, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
ax.annotate("Fig 1 snapshots", xy=(902, 0.05), xytext=(1100, 0.18),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.set_xlabel("number of operations", fontsize=11)
ax.set_ylabel(r"$\langle Z_{62}\rangle$ (expectation value)", fontsize=11)
ax.set_title(r"$\langle Z_{62}\rangle$ evolution over circuit depth"
             "\n(spreads as Pauli coefficients redistribute — Fig 1 analogue)", fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Right: runtime vs num_ops  (Fig 2 analogue: runtime ∝ # active Paulis)
ax = axes[1]
ax.plot(num_ops, rt_ms, color="#2A9D8F", linestyle="None", marker="s", markersize=5)
ax.set_xlabel("number of operations", fontsize=11)
ax.set_ylabel("PPS runtime (ms)", fontsize=11)
ax.set_title("Runtime proxy for active Pauli count\n"
             r"(runtime $\propto$ # Paulis — Fig 2 analogue)", fontsize=10)
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out12 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_2_gpu_reproduction.png")
plt.savefig(out12, dpi=200, bbox_inches="tight")
print(f"  -> saved {out12}")
plt.show()



# ══════════════════════════════════════════════════════════════════════════════
# Figs 3, 4 & 5  —  Runtime (Pauli count proxy) + EV norm-decay vs num_operations
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Figs 3, 4 & 5 …")
rows35 = load_jsonl("fig3_5_pps_gpu.jsonl")
rows35 = [r for r in rows35 if "error" not in r]

by_delta = defaultdict(list)
for r in rows35:
    by_delta[r["delta"]].append(r)

deltas_sorted = sorted(by_delta.keys(), reverse=True)  # large δ first
n_deltas = len(deltas_sorted)

# viridis palette: purple (small δ, many Paulis) -> teal (large δ, fewer Paulis)
cmap35 = plt.get_cmap("viridis")
colors35 = {d: cmap35(0.1 + 0.8 * i / max(n_deltas - 1, 1))
            for i, d in enumerate(reversed(deltas_sorted))}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Figs 3 & 4/5 — PPS GPU: Pauli growth proxy & norm decay (127 qubits, T=20, random θ_X)",
    fontsize=12, fontweight="bold"
)

# Left: runtime (log) vs num_operations — Fig 3 analogue
ax = axes[0]
for d in deltas_sorted:
    grp = sorted(by_delta[d], key=lambda r: r["num_operations"])
    x = [r["num_operations"] for r in grp]
    y = [r["run_time_ms"]     for r in grp]
    ax.plot(x, y, color=colors35[d], linestyle="None", marker="o", markersize=3.5,
            label=rf"$\delta = {d:.1e}$")

ax.set_xlabel("number of operations", fontsize=11)
ax.set_ylabel("PPS runtime (ms)   [proxy for # active Paulis]", fontsize=10)
ax.set_title(
    r"Pauli count growth — runtime proxy (Fig 3 analogue)" "\n"
    r"Smaller $\delta$ keeps more Paulis $\Rightarrow$ higher curve",
    fontsize=10
)
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3, which="both")

# Right: EV vs num_operations — Fig 4/5 analogue
ax = axes[1]
for d in deltas_sorted:
    grp = sorted(by_delta[d], key=lambda r: r["num_operations"])
    x = [r["num_operations"]   for r in grp]
    y = [r["expectation_value"] for r in grp]
    ax.plot(x, y, color=colors35[d], linestyle="None", marker="o", markersize=3.5,
            label=rf"$\delta = {d:.1e}$")

ax.set_xlabel("number of operations", fontsize=11)
ax.set_ylabel(r"$\langle Z_{62}\rangle$", fontsize=11)
ax.set_title(
    r"Expectation value vs depth (Figs 4/5 analogue)" "\n"
    r"Larger $\delta$ truncates more $\Rightarrow$ faster decay",
    fontsize=10
)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out35 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_5_gpu_reproduction.png")
plt.savefig(out35, dpi=200, bbox_inches="tight")
print(f"  -> saved {out35}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6  —  Convergence vs δ, random θ_X, multiple Trotter steps T
# Two rows: top ε_tol=0.01, bottom ε_tol=0.001 (matching whitepaper Fig 6)
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 6 …")
rows6 = load_jsonl("fig6_pps_gpu.jsonl")
rows6_ok = [r for r in rows6 if "error" not in r]

by_T6 = defaultdict(list)
for r in rows6_ok:
    by_T6[r["trotter_steps"]].append(r)

T_vals6 = sorted(by_T6.keys())
n_T6 = len(T_vals6)
T_colors6 = {T: vir(i, n_T6) for i, T in enumerate(T_vals6)}


def apply_tolerance(grp_sorted, eps_tol, margin=4):
    """Return the subset of data points shown when using convergence tolerance eps_tol.
    Simulates the PPS stopping criterion: stop at first δ where |O_k - O_{k-1}| < eps_tol
    AFTER the EV has become meaningfully non-zero (skip the trivial all-zero plateau).
    grp_sorted must be sorted by δ DESCENDING (largest δ first = leftmost on plot).
    """
    evs = [r["expectation_value"] for r in grp_sorted]

    # Phase 1: find where EV first becomes non-trivial (> 5% or > eps_tol*10)
    nontrivial_threshold = max(0.05, eps_tol * 10)
    nontrivial_start = len(evs) - 1  # if never non-trivial, show all
    for i, ev in enumerate(evs):
        if abs(ev) > nontrivial_threshold:
            nontrivial_start = i
            break

    # Phase 2: from nontrivial_start, find first convergence within eps_tol
    stop = len(evs)  # default: show all
    for i in range(nontrivial_start + 1, len(evs)):
        if abs(evs[i] - evs[i - 1]) < eps_tol:
            stop = min(i + margin, len(evs))
            break
    return grp_sorted[:stop]


TOLS = [(0.01, r"$\varepsilon_{{\rm tol}} = 10^{{-2}}$"),
         (0.001, r"$\varepsilon_{{\rm tol}} = 10^{{-3}}$")]

fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
fig6.suptitle(
    "Fig 6 — PPS GPU: Convergence vs δ (random θ_X, 127 qubits)",
    fontsize=13, fontweight="bold", y=1.01
)

for row_idx, (eps_tol, tol_label) in enumerate(TOLS):
    ax1, ax2 = axes6[row_idx]

    for T in T_vals6:
        grp = sorted(by_T6[T], key=lambda r: r["delta"], reverse=True)  # large δ first
        grp = apply_tolerance(grp, eps_tol)
        deltas_T = [r["delta"] for r in grp]
        evs_T    = [r["expectation_value"] for r in grp]
        rts_T    = [r["run_time_ms"] / 1000 for r in grp]
        color    = T_colors6[T]
        lbl      = f"T = {T}"
        ax1.plot(deltas_T, evs_T, color=color, linestyle="None", marker="o", ms=5, label=lbl)
        ax2.plot(deltas_T, rts_T, color=color, linestyle="None", marker="o", ms=5, label=lbl)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\delta_k$", fontsize=11)
        ax.legend(fontsize=9, title="Depth")
        ax.grid(True, alpha=0.3, which="both", ls=":")
        ax.set_title(f"Fixed random θ_X,  {tol_label}", fontsize=10)

    ax1.set_ylabel(r"$\mathcal{O}_k \approx \langle Z_{62}\rangle$", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Runtime (s)", fontsize=11)
    ax2.set_yscale("log")

fig6.tight_layout()
out6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig6_gpu_reproduction.png")
fig6.savefig(out6, dpi=200, bbox_inches="tight")
print(f"  -> saved {out6}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7  —  Convergence vs δ, fixed θ_X = 0.4, multiple T
# Two rows: top ε_tol=0.01, bottom ε_tol=0.001 (matching whitepaper Fig 7)
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 7 …")
rows7 = load_jsonl("fig7_pps_gpu.jsonl")
rows7_ok = [r for r in rows7 if "error" not in r]

by_T7 = defaultdict(list)
for r in rows7_ok:
    by_T7[r["trotter_steps"]].append(r)

T_vals7 = sorted(by_T7.keys())
n_T7 = len(T_vals7)
T_colors7 = {T: vir(i, n_T7) for i, T in enumerate(T_vals7)}

fig7, axes7 = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
fig7.suptitle(
    r"Fig 7 — PPS GPU: Convergence vs δ (fixed $\theta_X = 0.4$, 127 qubits)",
    fontsize=13, fontweight="bold", y=1.01
)

for row_idx, (eps_tol, tol_label) in enumerate(TOLS):
    ax1, ax2 = axes7[row_idx]

    for T in T_vals7:
        grp = sorted(by_T7[T], key=lambda r: r["delta"], reverse=True)  # large δ first
        grp = apply_tolerance(grp, eps_tol)
        deltas_T = [r["delta"] for r in grp]
        evs_T    = [r["expectation_value"] for r in grp]
        rts_T    = [r["run_time_ms"] / 1000 for r in grp]
        color    = T_colors7[T]
        lbl      = f"T = {T}"
        ax1.plot(deltas_T, evs_T, color=color, linestyle="None", marker="o", ms=5, label=lbl)
        ax2.plot(deltas_T, rts_T, color=color, linestyle="None", marker="o", ms=5, label=lbl)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\delta_k$", fontsize=11)
        ax.legend(fontsize=9, title="Depth")
        ax.grid(True, alpha=0.3, which="both", ls=":")
        ax.set_title(f"Fixed θ_X = 0.4,  {tol_label}", fontsize=10)

    ax1.set_ylabel(r"$\mathcal{O}_k \approx \langle Z_{62}\rangle$", fontsize=11)
    ax2.set_ylabel("Runtime (s)", fontsize=11)
    ax2.set_yscale("log")

fig7.tight_layout()
out7 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig7_gpu_reproduction.png")
fig7.savefig(out7, dpi=200, bbox_inches="tight")
print(f"  -> saved {out7}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8  —  <Z_62> vs δ, multiple θ_X angles
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 8 …")
rows8 = load_jsonl("fig8_pps_gpu.jsonl")
rows8_ok = [r for r in rows8 if "error" not in r and "expectation_value" in r]

by_rx8 = defaultdict(list)
for r in rows8_ok:
    by_rx8[r["rx_angle"]].append(r)

rx_angles = sorted(by_rx8.keys())

colors8  = {0.3: "#E89B2D", 0.4: "#1B3FA0", 0.6: "#C93030", 0.7: "#2A9D8F"}
markers8 = {0.3: "o",       0.4: "s",       0.6: "D",       0.7: "^"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Fig 8 — PPS GPU: IBM 127-qubit Trotter sweep", fontsize=13, fontweight="bold")

ax1, ax2 = axes

for rx in rx_angles:
    grp = sorted(by_rx8[rx], key=lambda r: r["delta_index"])
    deltas = [r["delta"] for r in grp]
    evs    = [r["expectation_value"] for r in grp]
    rts    = [r["run_time_ms"] / 1000 for r in grp]
    dl     = [-np.log10(d) for d in deltas]
    color  = colors8.get(rx, "#555555")
    marker = markers8.get(rx, "o")
    lbl    = rf"$\theta_X = {rx}$"
    ax1.plot(dl, evs, color=color, marker=marker, markersize=6, linestyle="None", label=lbl)
    ax2.plot(dl, rts, color=color, marker=marker, markersize=6, linestyle="None", label=lbl)

    # convergence band on EV plot
    last = [v for v in evs[-5:] if v is not None]
    if last:
        mid = np.mean(last)
        spread = max(np.std(last) * 2, 0.02)
        ax1.axhspan(mid - spread, mid + spread, color=color, alpha=0.10)

garnet_dl = -np.log10(2e-3)
ax1.axvline(garnet_dl, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.axvline(garnet_dl, color="gray", linestyle="--", linewidth=1, alpha=0.7)

ax1.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
ax1.set_ylabel(r"$\langle Z_{62}\rangle$", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
ax2.set_ylabel("runtime (s)", fontsize=11)
ax2.set_yscale("log")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig8_gpu_reproduction.png")
plt.savefig(out8, dpi=200, bbox_inches="tight")
print(f"  -> saved {out8}")
plt.show()

print("\nAll plots complete.")
