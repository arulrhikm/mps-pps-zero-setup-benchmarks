"""
CPU Analytical Formula Fit Plot
===============================
Formula: T = 1e-4 * (6 * su4s * n + gates + shots) * X^2
Plots actual data vs this fixed formula.
"""

import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
MPS_ROOT = os.path.dirname(BASE)
DATA_FILE = os.path.join(MPS_ROOT, "data", "all_mps_data_with_su4.jsonl")
PLOTS_DIR = os.path.join(MPS_ROOT, "plots")

def load_cpu_data():
    cpu = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                r = json.loads(line)
                sf = r.get("source_file","").lower()
                if "gpu" not in sf:
                    r["S"] = r["num_qubits"] * r["depth"] * r["bond_dimension"]**2
                    cpu.append(r)
            except: continue
    return cpu

if __name__ == "__main__":
    cpu_data = load_cpu_data()
    print(f"Loaded {len(cpu_data)} CPU records")

    actual = np.array([r["run_time_ms"] for r in cpu_data])
    S = np.array([r["S"] for r in cpu_data])
    
    # Calculate the formula values: 1e-4 * (6 * su4s * n + gates + shots) * X^2
    formula_vals = []
    for r in cpu_data:
        v = 1e-4 * (6 * r["num_su4s"] * r["num_qubits"] + r["num_gates"] + r["shots"]) * (r["bond_dimension"]**2)
        formula_vals.append(v)
    formula_vals = np.array(formula_vals)

    # Create Plot: formula prediction vs S with smooth fit line
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle("CPU: Analytical Formula Fit (SU(4)-based)", fontsize=16, fontweight="bold", y=1.02)

    # Metrics for fixed 1e-4
    pct_err = np.abs(formula_vals - actual) / np.maximum(actual, 1) * 100
    median_err = np.median(pct_err)

    # ── Panel 1: S vs Runtime with smooth formula overlay ──
    ax = axes[0]
    ax.set_facecolor("#f5f5f5")
    ax.scatter(S, actual / 1000, s=20, alpha=0.35, color="steelblue", label="Actual Data", zorder=2)
    
    # Smooth formula line: fit a linear regression of formula_vals vs S,
    # then plot with linspace for a clean curve
    from scipy.stats import linregress
    slope_f, intercept_f, _, _, _ = linregress(S, formula_vals)
    x_smooth = np.linspace(S.min(), S.max(), 500)
    y_smooth = (slope_f * x_smooth + intercept_f) / 1000
    ax.plot(x_smooth, y_smooth, color="#E53935", lw=2.5, zorder=4, 
            label=f"Formula (1e-4): Med Err {median_err:.1f}%")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("S = n \u00b7 d \u00b7 \u03c7\u00b2", fontsize=12)
    ax.set_ylabel("Runtime (sec)", fontsize=12)
    ax.set_title("S vs Runtime (with formula trend)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=11, loc="upper left")

    # ── Panel 2: Actual vs Predicted scatter ──
    ax = axes[1]
    ax.set_facecolor("#f5f5f5")
    mask = (actual > 0) & (formula_vals > 0)
    a_sec = actual[mask] / 1000
    f_sec = formula_vals[mask] / 1000
    
    ax.scatter(a_sec, f_sec, s=20, alpha=0.35, color="steelblue", edgecolors="none", zorder=3)
    all_v = np.concatenate([a_sec, f_sec])
    mn, mx = all_v.min() * 0.5, all_v.max() * 2
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, alpha=0.5, label="Perfect prediction", zorder=2)
    xx = np.logspace(np.log10(mn), np.log10(mx), 50)
    ax.fill_between(xx, xx*0.5, xx*2, alpha=0.06, color="green", zorder=1, label="\u00b1 2\u00d7 band")
    
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel("Actual Runtime (sec)", fontsize=12)
    ax.set_ylabel("Formula Prediction (sec)", fontsize=12)
    ax.set_title("Actual vs Formula Prediction", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=10, loc="upper left")
    
    txt = f"Median: {median_err:.1f}%\nMean:  {np.mean(pct_err):.1f}%\nP90:   {np.percentile(pct_err, 90):.1f}%"
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.95))

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "refined_fit_cpu_analytical.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
