"""
Refined MPS Runtime Predictor
==============================
Key insight from NNLS: gates & shots coefficients are zero.
The model simplifies to:  T = c1 * (su4s * n * X^2) + c2

Compares:
  1. Single global:   T = c1 * su4_n_X2 + c2              (2 params)
  2. 3-regime:        Same form, fit per regime            (6 params)

All coefficients constrained >= 0.
"""

import json, os, numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
MPS_ROOT = os.path.dirname(BASE)
DATA_FILE = os.path.join(MPS_ROOT, "data", "all_mps_data_with_su4.jsonl")
PLOTS_DIR = os.path.join(MPS_ROOT, "plots")

# ─────────────────────────── data ─────────────────────────────────────────
def load_data():
    gpu, cpu = [], []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                r = json.loads(line)
                need = ("num_qubits","depth","bond_dimension","run_time_ms",
                        "num_su4s","num_gates","shots")
                if any(r.get(k) is None for k in need): continue
                r["S"] = r["num_qubits"] * r["depth"] * r["bond_dimension"]**2
                r["su4_n_X2"] = r["num_su4s"] * r["num_qubits"] * r["bond_dimension"]**2
                sf = r.get("source_file","").lower()
                (gpu if "gpu" in sf else cpu).append(r)
            except: continue
    return gpu, cpu

# ─────────────────────────── fit ──────────────────────────────────────────
def fit_2param(recs):
    """Fit T = c1 * su4_n_X2 + c2  with c1, c2 >= 0."""
    n = len(recs)
    A = np.zeros((n, 2))
    y = np.zeros(n)
    for i, r in enumerate(recs):
        A[i, 0] = r["su4_n_X2"]
        A[i, 1] = 1.0
        y[i] = r["run_time_ms"]
    coeffs, _ = nnls(A, y)
    pred = A @ coeffs
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return coeffs, r2, pred

def predict_2param(r, coeffs):
    return max(0, coeffs[0] * r["su4_n_X2"] + coeffs[1])

def metrics(pred, actual):
    pct = np.abs(pred - actual) / np.maximum(actual, 1.0) * 100
    return np.median(pct), np.mean(pct), np.percentile(pct, 90)

# ─────────────────────────── regime split ─────────────────────────────────
def split_regimes(recs, n_regimes):
    """Quantile-based split on S = n*d*X^2."""
    S = np.array([r["S"] for r in recs])
    bounds = [np.percentile(S, 100*i/n_regimes) for i in range(1, n_regimes)]
    groups = [[] for _ in range(n_regimes)]
    for r in recs:
        placed = False
        for i, b in enumerate(bounds):
            if r["S"] < b:
                groups[i].append(r)
                placed = True
                break
        if not placed:
            groups[-1].append(r)
    return groups, bounds

# ─────────────────────────── cross-validation ─────────────────────────────
def cv_error_regimes(recs, n_regimes, n_folds=5):
    """K-fold cross-validated median % error for n_regimes model."""
    np.random.seed(42)
    indices = np.random.permutation(len(recs))
    fold_size = len(recs) // n_folds
    all_pct = []
    
    for fold in range(n_folds):
        test_idx = set(indices[fold*fold_size:(fold+1)*fold_size])
        train = [recs[i] for i in range(len(recs)) if i not in test_idx]
        test  = [recs[i] for i in range(len(recs)) if i in test_idx]
        if len(test) == 0: continue
        
        if n_regimes == 1:
            coeffs, _, _ = fit_2param(train)
            for r in test:
                p = predict_2param(r, coeffs)
                a = r["run_time_ms"]
                all_pct.append(abs(p - a) / max(a, 1) * 100)
        else:
            groups_tr, bounds = split_regimes(train, n_regimes)
            regime_coeffs = []
            global_coeffs, _, _ = fit_2param(train)
            for grp in groups_tr:
                if len(grp) >= 3:
                    c, _, _ = fit_2param(grp)
                    regime_coeffs.append(c)
                else:
                    regime_coeffs.append(global_coeffs)
            
            for r in test:
                # Find regime
                ri = len(bounds)  # default: last
                for i, b in enumerate(bounds):
                    if r["S"] < b:
                        ri = i
                        break
                p = predict_2param(r, regime_coeffs[ri])
                a = r["run_time_ms"]
                all_pct.append(abs(p - a) / max(a, 1) * 100)
    
    arr = np.array(all_pct)
    return np.median(arr), np.mean(arr), np.percentile(arr, 90)

# ─────────────────────────── main ─────────────────────────────────────────
if __name__ == "__main__":
    gpu_data, cpu_data = load_data()
    print(f"Loaded {len(gpu_data)} GPU, {len(cpu_data)} CPU records\n")

    for hw, recs in [("GPU", gpu_data), ("CPU", cpu_data)]:
        print(f"{'='*75}")
        print(f"  {hw}  (N={len(recs)})")
        print(f"{'='*75}")
        
        actual = np.array([r["run_time_ms"] for r in recs])

        # ── Global 2-param: T = c1*su4_n_X2 + c2 ──
        coeffs_g, r2_g, pred_g = fit_2param(recs)
        m_g = metrics(pred_g, actual)

        # ── 3-regime 2-param ──
        groups_3, bounds_3 = split_regimes(recs, 3)
        pred_3 = np.zeros(len(recs))
        coeffs_3 = []
        idx = 0
        for grp in groups_3:
            c, _, p = fit_2param(grp)
            coeffs_3.append(c)
            for j in range(len(grp)):
                pred_3[idx] = p[j]
                idx += 1
        m_3 = metrics(pred_3, actual)
        
        # ── Cross-validation ──
        print(f"\n  Cross-validated (5-fold):")
        for nr in [1, 2, 3, 4, 5]:
            cv_med, cv_mean, cv_p90 = cv_error_regimes(recs, nr)
            print(f"    {nr} regime(s):  median={cv_med:.1f}%  mean={cv_mean:.1f}%  p90={cv_p90:.1f}%")
        
        # ── In-sample comparison ──
        print(f"\n  In-sample results:")
        print(f"  {'Model':<35s} {'Params':>6s} {'Med%':>7s} {'Mean%':>7s} {'P90%':>7s}")
        print(f"  {'-'*65}")
        print(f"  {'Global: c1*su4*n*X² + c2':<35s} {'2':>6s} {m_g[0]:>6.1f}% {m_g[1]:>6.1f}% {m_g[2]:>6.1f}%")
        print(f"  {'3-regime: c1*su4*n*X² + c2':<35s} {'6':>6s} {m_3[0]:>6.1f}% {m_3[1]:>6.1f}% {m_3[2]:>6.1f}%")
        
        # ── Coefficients ──
        print(f"\n  Global: c1={coeffs_g[0]:.6e}, c2={coeffs_g[1]:.0f} ms, R²={r2_g:.4f}")
        for gi, (c, grp) in enumerate(zip(coeffs_3, groups_3)):
            lo = min(r["S"] for r in grp)
            hi = max(r["S"] for r in grp)
            print(f"  R{gi+1} (S={lo:,.0f}..{hi:,.0f}, N={len(grp)}): c1={c[0]:.6e}, c2={c[1]:.0f} ms")

        # ── PLOT ──
        n_panels = 2
        fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 7))
        if n_panels == 1: axes = [axes]
        fig.patch.set_facecolor("#fafafa")
        fig.suptitle(f"{hw} Runtime Predictor Comparison (SU(4)-based)", fontsize=16, fontweight="bold", y=1.02)
        
        models_plot = [
            ("Global\nc1\u00b7su4\u00b7n\u00b7\u03c7\u00b2 + c2", pred_g, m_g, "#1E88E5", "2 params"),
            ("3-regime\nc1\u00b7su4\u00b7n\u00b7\u03c7\u00b2 + c2", pred_3, m_3, "#2E7D32", "6 params"),
        ]
        
        for ax, (title, pred, m, color, plabel) in zip(axes, models_plot):
            ax.set_facecolor("#f5f5f5")
            a_s, p_s = actual / 1000, pred / 1000
            mask = (a_s > 0) & (p_s > 0)
            a, p = a_s[mask], p_s[mask]
            
            if len(a) == 0:
                ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center")
                continue
            
            all_v = np.concatenate([a, p])
            mn, mx = all_v.min() * 0.5, all_v.max() * 2
            
            ax.scatter(a, p, s=16, alpha=0.4, c=color, edgecolors="none", zorder=3)
            ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, alpha=0.4, zorder=2)
            xx = np.logspace(np.log10(mn), np.log10(mx), 50)
            ax.fill_between(xx, xx*0.5, xx*2, alpha=0.06, color="green", zorder=1)
            
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
            ax.set_xlabel("Actual (sec)", fontsize=12)
            ax.set_ylabel("Predicted (sec)", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.2, which="both")
            
            txt = f"Median: {m[0]:.1f}%\nMean:  {m[1]:.1f}%\nP90:   {m[2]:.1f}%\n({plabel})"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=11,
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.95))
        
        plt.tight_layout()
        fname = f"refined_predictor_{hw.lower()}.png"
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
        print(f"\n  Saved {fname}")
        plt.close()
    
    # ── S vs Runtime plot (showing regime boundaries) ──
    for hw, recs in [("GPU", gpu_data), ("CPU", cpu_data)]:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#f5f5f5")
        
        S_all = np.array([r["S"] for r in recs])
        y_all = np.array([r["run_time_ms"] for r in recs]) / 1000
        # Sort by su4_n_X2 for smooth prediction lines
        su4_all = np.array([r["su4_n_X2"] for r in recs])
        
        ax.scatter(su4_all, y_all, s=12, alpha=0.35, color="steelblue", label="Actual", zorder=2)
        
        groups, bounds = split_regimes(recs, 3)
        colors_r = ["#E53935", "#1E88E5", "#2E7D32"]
        
        for gi, grp in enumerate(groups):
            c, r2, _ = fit_2param(grp)
            su4_g = np.array([r["su4_n_X2"] for r in grp])
            
            _, _, p = fit_2param(grp)
            m = metrics(p, np.array([r["run_time_ms"] for r in grp]))
            
            # Smooth fit line: T = c[0] * su4_n_X2 + c[1], so generate
            # evenly spaced x values and compute analytically
            x_smooth = np.linspace(su4_g.min(), su4_g.max(), 300)
            y_smooth = (c[0] * x_smooth + c[1]) / 1000
            
            ax.plot(x_smooth, y_smooth, color=colors_r[gi], lw=2.5, zorder=3,
                    label=f"R{gi+1} (N={len(grp)}, med={m[0]:.1f}%, R²={r2:.3f})")
        
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("su4s \u00b7 n \u00b7 \u03c7\u00b2", fontsize=12)
        ax.set_ylabel("Runtime (sec)", fontsize=12)
        ax.set_title(f"{hw}: T = c1 \u00b7 su4s \u00b7 n \u00b7 \u03c7\u00b2 + c2  (3 regimes, all coeff \u2265 0)",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.25, which="both")
        
        plt.tight_layout()
        fname = f"refined_fit_{hw.lower()}.png"
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    
    # ── Pseudocode ──
    pc = []
    for hw, recs in [("GPU", gpu_data), ("CPU", cpu_data)]:
        groups, bounds = split_regimes(recs, 3)
        pc.append(f"# {hw} Runtime Predictor (simplified, all coefficients >= 0)")
        pc.append(f"# Model: T_ms = c1 * num_su4s * n * X^2 + c2")
        pc.append(f"# S = n * d * X^2 selects the regime")
        pc.append(f"#")
        pc.append(f"def predict_{hw.lower()}_runtime_ms(n, d, X, num_su4s):")
        pc.append(f"    S = n * d * X**2")
        
        for gi, grp in enumerate(groups):
            c, r2, pred_g = fit_2param(grp)
            m = metrics(pred_g, np.array([r["run_time_ms"] for r in grp]))
            
            if gi == 0:
                pc.append(f"    if S < {bounds[0]:,.0f}:  # R{gi+1} (N={len(grp)}, med_err={m[0]:.1f}%, R²={r2:.3f})")
            elif gi < len(bounds):
                pc.append(f"    elif S < {bounds[gi]:,.0f}:  # R{gi+1} (N={len(grp)}, med_err={m[0]:.1f}%, R²={r2:.3f})")
            else:
                pc.append(f"    else:  # R{gi+1} (N={len(grp)}, med_err={m[0]:.1f}%, R²={r2:.3f})")
            
            pc.append(f"        return max(0, {c[0]:.6e} * num_su4s * n * X**2 + {c[1]:.0f})")
        pc.append("")
    
    pc_text = "\n".join(pc) + "\n"
    outpath = os.path.join(PLOTS_DIR, "refined_predictor_pseudocode.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(pc_text)
    print(f"\nSaved refined_predictor_pseudocode.txt")
    print("\n" + pc_text)
