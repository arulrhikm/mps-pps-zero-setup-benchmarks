"""
MPS Predictor – Analytical Formula Plot
=========================================
Generates  mps_predictor_cpu_analytical.png

Left panel  : x = S = n·d·χ²  (raw complexity proxy, matches other plots)
              Overlays the fixed formula  T = 1e-4·(6·su4s·n + gates + shots)·χ²

Right panel : x = (su4s·n + gates + shots)·χ²   ← the full formula's natural x-axis
              Fits a WLS linear model  T = c1·x + c2  (w = 1/x)
              Draws a single, smooth fit line  (not the jagged formula trace)
              95% PI band  margin = t · σ_w · √x

Usage:
    python mps_predictor_analytical.py
"""

import json, os
import numpy as np
from scipy.optimize import nnls
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── paths ──────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MPS_ROOT  = os.path.dirname(BASE)
DATA_FILE = os.path.join(MPS_ROOT, "data", "all_mps_data_with_su4.jsonl")
PLOTS_DIR = os.path.join(MPS_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── constants ───────────────────────────────────────────────────────────────
FORMULA_COEFF = 1e-4   # fixed analytical prefactor
CI_REL_FLOOR  = 0.05   # lower CI never drops below 5% of ŷ on log scale

# ─── data ────────────────────────────────────────────────────────────────────

def load_cpu_data():
    cpu = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                need = ("num_qubits", "depth", "bond_dimension",
                        "run_time_ms", "num_su4s", "num_gates", "shots")
                if any(r.get(k) is None for k in need):
                    continue
                n = r["num_qubits"]
                d = r["depth"]
                X = r["bond_dimension"]
                r["S"]         = n * d * X**2                           # raw proxy (left panel)
                r["x_formula"] = (r["num_su4s"] * n + r["num_gates"]    # full-formula x (right panel)
                                  + r["shots"]) * X**2
                sf = r.get("source_file", "").lower()
                if "gpu" not in sf:
                    cpu.append(r)
            except Exception:
                continue
    return cpu


def formula_ms(r):
    """Fixed analytical formula:  T = 1e-4 · (6·su4s·n + gates + shots) · χ²"""
    return FORMULA_COEFF * (6 * r["num_su4s"] * r["num_qubits"]
                            + r["num_gates"] + r["shots"]) * r["bond_dimension"]**2


# ─── WLS fit ─────────────────────────────────────────────────────────────────

def fit_wls(x, y):
    """
    Fit  T = c1·x + c2  with WLS weights  w_i = 1/x_i  (assumes Var ∝ x²).
    Returns (coeffs, r2, pred, sigma_w).
    """
    w  = 1.0 / np.maximum(x, 1.0)
    sw = np.sqrt(w)
    A  = np.column_stack([sw * x, sw])
    b  = sw * y
    coeffs, _ = nnls(A, b)
    pred  = coeffs[0] * x + coeffs[1]
    resid = y - pred
    sigma_w = float(np.sqrt(np.sum((sw * resid)**2) / max(len(x) - 2, 1)))
    ss_tot  = float(np.sum((y - y.mean())**2))
    r2      = float(1.0 - np.sum(resid**2) / max(ss_tot, 1.0))
    return coeffs, r2, pred, sigma_w


def pct_errors(pred, actual):
    return np.abs(pred - actual) / np.maximum(actual, 1.0) * 100


# ─── plotting ────────────────────────────────────────────────────────────────

def plot_left(ax, cpu_data, formula_vals):
    """Left panel: x = S = n·d·χ², overlay fixed formula line."""
    S      = np.array([r["S"]          for r in cpu_data])
    actual = np.array([r["run_time_ms"] for r in cpu_data]) / 1000

    ax.scatter(S, actual, s=14, alpha=0.30, color="steelblue",
               edgecolors="none", label="Actual", zorder=2)

    # Sort by S for a clean formula trace
    order    = np.argsort(S)
    S_sorted = S[order]
    f_sorted = formula_vals[order] / 1000

    pct = pct_errors(formula_vals, np.array([r["run_time_ms"] for r in cpu_data]))

    ax.plot(S_sorted, f_sorted, color="#E53935", lw=2.0, zorder=3,
            label=f"Formula (1e-4·(6·su4s·n+gates+shots)·χ²)\n"
                  f"Med err {np.median(pct):.1f}%  |  P90 {np.percentile(pct, 90):.1f}%")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("S = n · d · χ²  (raw complexity proxy)", fontsize=11)
    ax.set_ylabel("Runtime (sec)", fontsize=11)
    ax.set_title("Left: raw S-axis  —  formula overlay", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")


def plot_right(ax, cpu_data):
    """
    Right panel: x = (su4s·n + gates + shots)·χ²
    WLS linear fit  T = c1·x + c2  drawn as a single smooth line + 95% PI.
    """
    x      = np.array([r["x_formula"]  for r in cpu_data])
    actual = np.array([r["run_time_ms"] for r in cpu_data])

    coeffs, r2, pred, sigma_w = fit_wls(x, actual)
    c1, c2 = coeffs
    n      = len(cpu_data)
    t_val  = sp_stats.t.ppf(0.975, df=max(n - 2, 1))

    pct = pct_errors(pred, actual)

    # Scatter
    ax.scatter(x, actual / 1000, s=14, alpha=0.28, color="steelblue",
               edgecolors="none", label="Actual", zorder=2)

    # Smooth fit line + 95% PI
    x_sm   = np.linspace(x.min(), x.max(), 600)
    pred_ms = np.maximum(c1 * x_sm + c2, 0)
    margin  = t_val * sigma_w * np.sqrt(x_sm)          # ms

    y_sm  = pred_ms / 1000
    y_hi  = (pred_ms + margin) / 1000
    y_lo  = np.maximum(pred_ms - margin, pred_ms * CI_REL_FLOOR) / 1000

    ax.fill_between(x_sm, y_lo, y_hi, color="#1E88E5", alpha=0.15, zorder=1)
    ax.plot(x_sm, y_sm, color="#1E88E5", lw=2.5, zorder=3,
            label=(f"WLS fit:  T = {c1:.3e}·x + {c2:.0f}  (ms)\n"
                   f"R²={r2:.3f}  |  Med err {np.median(pct):.1f}%  |  P90 {np.percentile(pct, 90):.1f}%"))

    # Annotation box
    txt = (f"n = {n}\n"
           f"c₁ = {c1:.3e}\n"
           f"c₂ = {c2:.0f} ms\n"
           f"R²  = {r2:.3f}\n"
           f"Med = {np.median(pct):.1f}%\n"
           f"P90 = {np.percentile(pct, 90):.1f}%")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#ccc", alpha=0.95))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("(su4s·n + gates + shots) · χ²  (full formula variable)", fontsize=11)
    ax.set_ylabel("Runtime (sec)", fontsize=11)
    ax.set_title("Right: WLS linear fit in full-formula x-space\n"
                 "(lower CI floored at 5% of ŷ)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cpu_data = load_cpu_data()
    print(f"Loaded {len(cpu_data)} CPU records")

    formula_vals = np.array([formula_ms(r) for r in cpu_data])
    actual       = np.array([r["run_time_ms"] for r in cpu_data])
    pct_f = pct_errors(formula_vals, actual)
    print(f"Fixed formula  —  Med: {np.median(pct_f):.1f}%  "
          f"Mean: {np.mean(pct_f):.1f}%  P90: {np.percentile(pct_f, 90):.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        r"CPU:  $T = 10^{-4} \cdot (6 \cdot \mathrm{su4s} \cdot n + \mathrm{gates} + \mathrm{shots}) \cdot \chi^2$",
        fontsize=15, fontweight="bold"
    )

    plot_left(axes[0],  cpu_data, formula_vals)
    plot_right(axes[1], cpu_data)

    out = os.path.join(PLOTS_DIR, "mps_predictor_cpu_analytical.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
