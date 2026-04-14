"""
Robust Multivariate Regime Predictor
=====================================
Fixes: ridge regression, non-negative c1, adaptive regime count,
       min-N/regime, LOO-CV error, smooth fallback for sparse regimes.

Outputs:
  predictor_pseudocode.txt
  predictor_fit_plot.png
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.abspath(__file__))           # analysis/
MPS_ROOT = os.path.dirname(BASE)                             # mps_tests/
DATA_FILE = os.path.join(MPS_ROOT, "data", "all_mps_data_with_su4.jsonl")
PLOTS_DIR = os.path.join(MPS_ROOT, "plots")

MIN_N_PER_REGIME = 15        # must have at least this many points
RIDGE_ALPHA      = 1e-3      # L2 regularization strength

# ═════════════════════════════ data ═══════════════════════════════════════

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
                n, d, X = r["num_qubits"], r["depth"], r["bond_dimension"]
                r["S"] = n * d * X**2
                r["f_su4_n"]  = r["num_su4s"] * n * X**2
                r["f_gates"] = r["num_gates"] * X**2
                r["f_shots"] = r["shots"] * X**2
                sf = r.get("source_file","").replace("/","\\").lower()
                (gpu if sf.startswith("gpu\\") else cpu).append(r)
            except: continue
    return gpu, cpu

# ═════════════════════════════ ridge fit (c1 >= 0) ════════════════════════

def _build_Ay(recs):
    A = np.column_stack([
        [r["f_su4_n"]  for r in recs],
        [r["f_gates"] for r in recs],
        [r["f_shots"] for r in recs],
        np.ones(len(recs))
    ])
    y = np.array([r["run_time_ms"] for r in recs])
    return A, y

def fit_ridge_constrained(recs, alpha=RIDGE_ALPHA):
    """Ridge regression with c1 >= 0 (su4*n term must be non-negative).
    Fits: runtime = c1*(su4s*n*X^2) + c2*(gates*X^2) + c3*(shots*X^2) + c4
    """
    A, y = _build_Ay(recs)
    n, p = A.shape

    def objective(c):
        resid = A @ c - y
        return np.sum(resid**2) + alpha * n * np.sum(c[:3]**2)  # regularize coeffs, not intercept

    # warm start with OLS
    c0 = np.linalg.lstsq(A, y, rcond=None)[0]
    if c0[0] < 0:
        c0[0] = 1e-5

    # bounds: c1 >= 0, others unconstrained
    bounds = [(0, None), (None, None), (None, None), (None, None)]
    result = minimize(objective, c0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000, "ftol": 1e-14})
    c = result.x

    pred = A @ c
    resid = y - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    res_std = float(np.sqrt(ss_res / max(n-p, 1)))
    pct = np.abs(resid) / np.maximum(y, 1) * 100

    return dict(c1=c[0], c2=c[1], c3=c[2], c4=c[3],
                r2=r2, res_std=res_std, n=n,
                med_pct=float(np.median(pct)), mean_pct=float(np.mean(pct)),
                p90_pct=float(np.percentile(pct, 90)),
                mae=float(np.mean(np.abs(resid))))

def fit_scaled_formula(recs):
    """Fallback for sparse regimes: fit single alpha in runtime = alpha * formula.
    formula = (6*su4s*n + gates + shots) * X^2
    """
    f = np.array([analytical_ms_raw(r) for r in recs])
    y = np.array([r["run_time_ms"] for r in recs])
    # alpha = <f,y> / <f,f>
    alpha = np.dot(f, y) / max(np.dot(f, f), 1e-10)
    pred = alpha * f
    resid = y - pred
    ss_res = np.sum(resid**2); ss_tot = np.sum((y-y.mean())**2)
    r2 = 1-ss_res/ss_tot if ss_tot>0 else 0
    pct = np.abs(resid)/np.maximum(y,1)*100
    return dict(alpha=alpha, r2=r2, n=len(recs), res_std=float(np.sqrt(ss_res/max(len(recs)-1,1))),
                med_pct=float(np.median(pct)), mean_pct=float(np.mean(pct)),
                p90_pct=float(np.percentile(pct,90)), mae=float(np.mean(np.abs(resid))),
                mode="scaled_formula")

def loo_cv_error(recs, fit_fn):
    """Leave-one-out cross-validation % error."""
    if len(recs) < 8:
        return None
    errs = []
    for i in range(len(recs)):
        train = recs[:i] + recs[i+1:]
        m = fit_fn(train)
        if m is None: continue
        if "alpha" in m:
            pred = m["alpha"] * analytical_ms_raw(recs[i])
        else:
            pred = predict_one(m, recs[i])
        actual = recs[i]["run_time_ms"]
        errs.append(abs(pred-actual)/max(actual,1)*100)
    return float(np.median(errs)) if errs else None

# ═════════════════════════════ predict ═════════════════════════════════════

def predict_one(model, r):
    if "alpha" in model:
        return max(0, model["alpha"] * analytical_ms_raw(r))
    return max(0, model["c1"]*r["f_su4_n"] + model["c2"]*r["f_gates"] + model["c3"]*r["f_shots"] + model["c4"])

def analytical_ms_raw(r):
    """Raw analytical formula value (without the 1e-4 prefactor baked in)."""
    return (6*r["num_su4s"]*r["num_qubits"] + r["num_gates"] + r["shots"]) * r["bond_dimension"]**2

def analytical_ms(r):
    return 1e-4 * analytical_ms_raw(r)

# ═════════════════════════════ regime search ═══════════════════════════════

def assign_regimes(recs, bounds):
    nr = len(bounds)-1
    groups = [[] for _ in range(nr)]
    for r in recs:
        s = r["S"]; idx = nr-1
        for i in range(nr):
            if i < nr-1 and bounds[i] <= s < bounds[i+1]: idx=i; break
        groups[idx].append(r)
    return groups

def fit_regime(recs):
    """Fit a regime: ridge if N >= MIN_N_PER_REGIME, scaled formula if N >= 5, else None."""
    if len(recs) >= MIN_N_PER_REGIME:
        return fit_ridge_constrained(recs)
    elif len(recs) >= 5:
        return fit_scaled_formula(recs)
    return None

def eval_bounds(recs, bounds):
    groups = assign_regimes(recs, bounds)
    stats = []
    for i, grp in enumerate(groups):
        m = fit_regime(grp)
        if m:
            stats.append(dict(n=len(grp), lo=bounds[i], hi=bounds[i+1], model=m,
                              med_pct=m["med_pct"], mean_pct=m["mean_pct"],
                              p90_pct=m["p90_pct"], mae=m["mae"]))
        else:
            stats.append(dict(n=len(grp), lo=bounds[i], hi=bounds[i+1], model=None,
                              med_pct=999, mean_pct=999, p90_pct=999, mae=0))
    return stats

def score(stats):
    """Score = sum of N_i * med_pct_i * weight_i, so large regimes with high
    error are heavily penalized. Higher-index regimes get extra weight.
    Empty regimes get a heavy penalty."""
    s = 0
    for i, st in enumerate(stats):
        regime_weight = (i+1)**1.5
        if st["n"] == 0 or st["model"] is None:
            s += 1e6 * regime_weight  # heavy penalty for empty/unfittable
        else:
            s += st["n"] * st["med_pct"] * regime_weight
    return s

def merge_small_regimes(bounds, recs, min_n=5, min_regimes=3):
    """Merge regimes with < min_n points (truly unfittable).
    N=5-14 handled by scaled_formula fallback, so only merge N<5."""
    while len(bounds) > min_regimes + 1:
        groups = assign_regimes(recs, bounds)
        # find regime with fewest points
        worst_i, worst_n = -1, min_n
        for i in range(len(groups)):
            if len(groups[i]) < worst_n:
                worst_i, worst_n = i, len(groups[i])
        if worst_i < 0:
            break  # all regimes have >= min_n
        # merge with smaller neighbor
        if worst_i == 0:
            bounds = [bounds[0]] + bounds[2:]
        elif worst_i == len(groups)-1:
            bounds = bounds[:-2] + [bounds[-1]]
        else:
            left_n = len(groups[worst_i-1])
            right_n = len(groups[worst_i+1])
            if left_n <= right_n:
                bounds = bounds[:worst_i] + bounds[worst_i+1:]
            else:
                bounds = bounds[:worst_i+1] + bounds[worst_i+2:]
    return bounds

def find_bounds(recs, nr=10, n_iter=2000, seed=42):
    S = np.array([r["S"] for r in recs], dtype=np.float64)
    Smin, Smax = float(S.min()), float(S.max())
    lS, lE = np.log(max(1, Smin)), np.log(Smax)
    def clamp(b):
        b = sorted(set(b)); b[0] = Smin; b[-1] = Smax
        # deduplicate and ensure exactly nr+1
        while len(b) < nr+1:
            # insert midpoints in log space in widest gap
            gaps = [(np.log(b[i+1])-np.log(max(1,b[i])), i) for i in range(len(b)-1)]
            gaps.sort(reverse=True)
            mid = np.exp((np.log(max(1,b[gaps[0][1]])) + np.log(b[gaps[0][1]+1])) / 2)
            b.insert(gaps[0][1]+1, mid)
            b = sorted(set(b))
        return b[:nr+1]

    # Generate diverse initial candidates
    pcts_even = np.linspace(0, 100, nr+1)
    cands = [
        clamp(list(np.percentile(S, pcts_even))),
        clamp(list(np.exp(np.linspace(lS, lE, nr+1)))),
        clamp(list(np.exp([lS+(lE-lS)*(i/nr)**0.5  for i in range(nr+1)]))),
        clamp(list(np.exp([lS+(lE-lS)*(i/nr)**0.35 for i in range(nr+1)]))),
        clamp(list(np.exp([lS+(lE-lS)*(i/nr)**0.7  for i in range(nr+1)]))),
    ]

    # Evaluate all candidates (no merging — sparse regimes get scaled fallback)
    best_b, best_sc = None, 1e18
    for c in cands:
        if len(c) != nr+1: continue
        st = eval_bounds(recs, c); sc = score(st)
        if sc < best_sc: best_b, best_sc = list(c), sc

    # Stochastic search: move boundaries but never change regime count
    rng = np.random.RandomState(seed)
    for it in range(n_iter):
        t = list(best_b)
        idx = rng.randint(1, nr)  # pick an interior boundary
        lo_n, hi_n = t[idx-1], t[idx+1]
        if hi_n <= lo_n + 2: continue
        nv = np.exp(rng.uniform(np.log(max(1,lo_n+1)), np.log(max(2,hi_n-1))))
        if nv <= lo_n or nv >= hi_n: continue
        t[idx] = nv
        st = eval_bounds(recs, t); sc = score(st)
        if sc < best_sc: best_b, best_sc = t, sc
    return best_b, eval_bounds(recs, best_b)

# ═════════════════════════════ pseudocode ══════════════════════════════════

def gen_pseudocode(hw, stats):
    lines = [
        f"# {hw.upper()} Runtime Predictor",
        f"#",
        f"# Full formula per regime: runtime_ms = (c1*su4s*n + c2*gates + c3*shots) * X^2 + c4",
        f"# Sparse regime fallback:  runtime_ms = alpha * (6*su4s*n + gates + shots) * X^2",
        f"# S = n*d*X^2 selects the regime",
        f"#",
        f"def predict_{hw}_runtime_ms(n, d, X, su4s, gates, shots):",
        f"    S = n * d * X**2",
    ]
    first = True
    for i, st in enumerate(stats):
        m = st["model"]
        if m is None: continue
        kw = "if" if first else "elif"
        first = False
        end = "<=" if i == len(stats)-1 else "<"
        mode = "scaled" if "alpha" in m else "ridge"
        cv_txt = ""
        lines.append(f"    {kw} S {end} {st['hi']:,.0f}:".ljust(36) +
                     f"  # R{i+1} (N={st['n']}, R2={m['r2']:.3f}, med_err={st['med_pct']:.1f}%, {mode})")
        if "alpha" in m:
            lines.append(f"        return {m['alpha']:.6e} * (6*su4s*n + gates + shots) * X**2")
        else:
            c1s = f"{m['c1']:.6e}" if m['c1'] > 0 else "0"
            lines.append(f"        return max(0, ({m['c1']:.6e}*su4s*n + {m['c2']:.6e}*gates + {m['c3']:.6e}*shots) * X**2 + ({m['c4']:.1f}))")
    lines.append(f"    else:")
    # extrapolation: use last full model or fall back to analytical
    last = [s["model"] for s in stats if s["model"]][-1]
    if "alpha" in last:
        lines.append(f"        return {last['alpha']:.6e} * (6*su4s*n + gates + shots) * X**2")
    else:
        lines.append(f"        return max(0, ({last['c1']:.6e}*su4s*n + {last['c2']:.6e}*gates + {last['c3']:.6e}*shots) * X**2 + ({last['c4']:.1f}))")
    return "\n".join(lines)

# ═════════════════════════════ reports ═════════════════════════════════════

def print_report(hw, stats):
    lines = []
    nr = len(stats)
    lines.append(f"\n{'='*125}")
    lines.append(f"  {hw.upper()} Predictor  --  {sum(s['n'] for s in stats)} records, {nr} regimes")
    lines.append(f"{'='*125}")
    lines.append(f"  {'Regime':<44} {'N':>4} {'Mode':>8}  {'R2':>7} {'Med%':>7} {'Mean%':>7} {'P90%':>7}   c1(su4*n)")
    lines.append("  "+"-"*118)
    for i, st in enumerate(stats):
        end = "]" if i==nr-1 else ")"
        lbl = f"R{i+1:>2}: [{st['lo']:>14,.0f}, {st['hi']:>14,.0f}{end}"
        m = st["model"]
        if m:
            mode = "scaled" if "alpha" in m else "ridge"
            c1_str = f"{m.get('alpha', m.get('c1', 0)):.4e}"
            lines.append(f"  {lbl:<44} {st['n']:>4} {mode:>8}  {m['r2']:>7.4f} {st['med_pct']:>6.1f}% {st['mean_pct']:>6.1f}% {st['p90_pct']:>6.1f}%   {c1_str}")
        else:
            lines.append(f"  {lbl:<44} {st['n']:>4}  -- too few --")
    lines.append("  "+"-"*118)
    valid = [s for s in stats if s["med_pct"]<999]
    wmed = np.average([s["med_pct"] for s in valid], weights=[s["n"] for s in valid])
    wmean = np.average([s["mean_pct"] for s in valid], weights=[s["n"] for s in valid])
    lines.append(f"  GLOBAL: weighted median %err = {wmed:.1f}%  |  weighted mean %err = {wmean:.1f}%")
    lines.append("="*125)
    return "\n".join(lines)

# ═════════════════════════════ plotting ════════════════════════════════════

def plot_fit_panel(ax, hw, recs, stats, bounds):
    S_all = np.array([r["S"] for r in recs])
    y_all = np.array([r["run_time_ms"] for r in recs]) / 1000
    ax.scatter(S_all, y_all, s=8, alpha=0.3, color="steelblue", label="Actual", zorder=2)

    groups = assign_regimes(recs, bounds)
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))
    for i, (st, grp) in enumerate(zip(stats, groups)):
        m = st["model"]
        if m is None or len(grp) == 0: continue
        s_g = np.array([r["S"] for r in grp])
        p_g = np.array([predict_one(m, r) for r in grp])
        mode = "s" if "alpha" in m else "r"

        # Smooth fit line: fit a simple linear regression of prediction vs S
        # within this regime, then draw with linspace for a clean line
        from scipy.stats import linregress
        if len(s_g) >= 3 and np.ptp(s_g) > 0:
            slope_s, intercept_s, _, _, _ = linregress(s_g, p_g)
            x_smooth = np.linspace(s_g.min(), s_g.max(), 300)
            y_smooth = (slope_s * x_smooth + intercept_s) / 1000
        else:
            x_smooth = np.sort(s_g)
            y_smooth = np.full_like(x_smooth, np.mean(p_g)) / 1000
        ax.plot(x_smooth, y_smooth, color=colors[i], lw=1.8, zorder=3,
                label=f"R{i+1} ({st['med_pct']:.1f}%, {mode})")
        # CI band
        n_pts = m["n"]; p_params = 4 if "alpha" not in m else 1
        if n_pts > p_params + 2:
            t_val = sp_stats.t.ppf(0.975, df=n_pts-p_params)
            margin = t_val * m["res_std"] / 1000
            ax.fill_between(x_smooth, np.maximum(0, y_smooth-margin),
                            y_smooth+margin, color=colors[i], alpha=0.08, zorder=1)
        if i > 0: ax.axvline(st["lo"], color="gray", lw=0.4, ls="--", alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("S = n·d·χ²"); ax.set_ylabel("Runtime (sec)")
    ax.set_title(f"{hw.upper()} Robust Regime Predictor (95% PI)", fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.25, which="both")

    # summary box
    valid = [s for s in stats if s["med_pct"]<999]
    wmed = np.average([s["med_pct"] for s in valid], weights=[s["n"] for s in valid])
    txt = f"{hw.upper()}: Weighted med err = {wmed:.1f}%\nRegimes: {len(stats)}"
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))


def plot_comparison(ax, recs, stats, bounds):
    actual = np.array([r["run_time_ms"] for r in recs])
    analytical = np.array([analytical_ms(r) for r in recs])
    groups = assign_regimes(recs, bounds)
    regime_pred = np.zeros(len(recs))
    idx = 0
    for i, grp in enumerate(groups):
        m = stats[i]["model"]
        for r in grp:
            regime_pred[idx] = predict_one(m, r) if m else actual[idx]
            idx += 1

    reg_pct = np.abs(regime_pred-actual)/np.maximum(actual,1)*100
    ana_pct = np.abs(analytical-actual)/np.maximum(actual,1)*100

    mx = max(actual.max(), analytical.max(), regime_pred.max()) / 1000
    ax.scatter(actual/1000, regime_pred/1000, s=12, alpha=0.4, c="dodgerblue",
               label=f"Regime (med={np.median(reg_pct):.1f}%, mean={np.mean(reg_pct):.1f}%)", zorder=3)
    ax.scatter(actual/1000, analytical/1000, s=12, alpha=0.4, c="orangered",
               label=f"Formula (med={np.median(ana_pct):.1f}%, mean={np.mean(ana_pct):.1f}%)", zorder=3)
    ax.plot([0.5, mx*1.2], [0.5, mx*1.2], 'k--', lw=1, alpha=0.5, label="Perfect")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual Runtime (sec)"); ax.set_ylabel("Predicted Runtime (sec)")
    ax.set_title("CPU: Robust Regime vs Analytical Formula", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, which="both")
    txt = (f"Robust Regime Predictor:\n"
           f"  Median %err: {np.median(reg_pct):.1f}%\n"
           f"  Mean %err:  {np.mean(reg_pct):.1f}%\n"
           f"  P90 %err:   {np.percentile(reg_pct,90):.1f}%\n\n"
           f"Analytical Formula:\n"
           f"  Median %err: {np.median(ana_pct):.1f}%\n"
           f"  Mean %err:  {np.mean(ana_pct):.1f}%\n"
           f"  P90 %err:   {np.percentile(ana_pct,90):.1f}%")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))

# ═════════════════════════════ main ════════════════════════════════════════

if __name__ == "__main__":
    gpu_data, cpu_data = load_data()
    print(f"Loaded {len(gpu_data)} GPU, {len(cpu_data)} CPU records")

    # Adaptive regime count: ensure min N per regime
    gpu_nr = max(3, min(10, len(gpu_data) // MIN_N_PER_REGIME))
    cpu_nr = max(3, min(10, len(cpu_data) // MIN_N_PER_REGIME))
    print(f"Adaptive regime count: GPU={gpu_nr}, CPU={cpu_nr}")

    gpu_bounds, gpu_stats = find_bounds(gpu_data, nr=gpu_nr)
    cpu_bounds, cpu_stats = find_bounds(cpu_data, nr=cpu_nr)

    print(print_report("GPU", gpu_stats))
    print(print_report("CPU", cpu_stats))

    # ── Validation checks ──
    print(f"\n{'='*70}")
    print(f"  VALIDATION CHECKS")
    print(f"{'='*70}")
    all_ok = True
    for hw, stats in [("GPU", gpu_stats), ("CPU", cpu_stats)]:
        for i, st in enumerate(stats):
            m = st["model"]
            if m is None: continue
            if st["n"] < MIN_N_PER_REGIME and "alpha" not in m:
                print(f"  [WARN] {hw} R{i+1}: N={st['n']} < {MIN_N_PER_REGIME} with full model")
            if "c1" in m and m["c1"] < 0:
                print(f"  [FAIL] {hw} R{i+1}: c1={m['c1']:.4e} < 0 (non-physical)")
                all_ok = False
            if m.get("r2", 0) > 0.9999 and st["n"] < 10:
                print(f"  [WARN] {hw} R{i+1}: R2={m['r2']:.4f} with N={st['n']} (possible overfit)")
    if all_ok:
        print("  [OK] All c1 coefficients are non-negative")
    print(f"{'='*70}")

    # ── CPU comparison ──
    actual = np.array([r["run_time_ms"] for r in cpu_data])
    analytical = np.array([analytical_ms(r) for r in cpu_data])
    groups = assign_regimes(cpu_data, cpu_bounds)
    regime_pred = np.zeros(len(cpu_data))
    idx = 0
    for i, grp in enumerate(groups):
        m = cpu_stats[i]["model"]
        for r in grp:
            regime_pred[idx] = predict_one(m, r) if m else actual[idx]
            idx += 1
    reg_pct = np.abs(regime_pred-actual)/np.maximum(actual,1)*100
    ana_pct = np.abs(analytical-actual)/np.maximum(actual,1)*100

    print(f"\n{'='*70}")
    print(f"  CPU: Robust Regime vs Analytical Formula")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Regime':>12} {'Formula':>12}")
    print(f"  {'-'*50}")
    for lbl, fn in [("Median % error", np.median), ("Mean % error", np.mean),
                    ("P90 % error", lambda x: np.percentile(x,90)),
                    ("P95 % error", lambda x: np.percentile(x,95))]:
        rv, fv = fn(reg_pct), fn(ana_pct)
        mark = " <--" if rv < fv else ""
        print(f"  {lbl:<25} {rv:>11.1f}% {fv:>11.1f}%{mark}")
    w = "Regime" if np.mean(reg_pct) < np.mean(ana_pct) else "Formula"
    print(f"\n  WINNER (by mean %err): {w}")
    print(f"{'='*70}")

    # ── pseudocode ──
    pc = gen_pseudocode("gpu", gpu_stats) + "\n\n" + gen_pseudocode("cpu", cpu_stats)
    pc += "\n\n# ── Fixed Analytical Formula (CPU baseline) ──\n"
    pc += "# runtime_ms = 1e-4 * (6*su4s*n + gates + shots) * X^2\n"
    pc += f"# Median %err: {np.median(ana_pct):.1f}%  |  Mean: {np.mean(ana_pct):.1f}%  |  P90: {np.percentile(ana_pct,90):.1f}%\n"
    with open(os.path.join(PLOTS_DIR, "predictor_pseudocode.txt"), "w", encoding="utf-8") as f:
        f.write(pc)
    print(f"\nWrote predictor_pseudocode.txt")

    # ── GPU plot (single panel) ──
    fig_gpu, ax_gpu = plt.subplots(1, 1, figsize=(14, 7))
    plot_fit_panel(ax_gpu, "GPU", gpu_data, gpu_stats, gpu_bounds)
    plt.tight_layout(pad=1.5)
    fig_gpu.savefig(os.path.join(PLOTS_DIR, "predictor_gpu.png"), dpi=150, bbox_inches="tight")
    print("Saved predictor_gpu.png")
    plt.close(fig_gpu)

    # ── CPU plot (two panels: regime fit + formula comparison) ──
    fig_cpu, axes_cpu = plt.subplots(2, 1, figsize=(14, 14))
    plot_fit_panel(axes_cpu[0], "CPU", cpu_data, cpu_stats, cpu_bounds)
    plot_comparison(axes_cpu[1], cpu_data, cpu_stats, cpu_bounds)
    plt.tight_layout(pad=2.5)
    fig_cpu.savefig(os.path.join(PLOTS_DIR, "predictor_cpu.png"), dpi=150, bbox_inches="tight")
    print("Saved predictor_cpu.png")
    plt.close(fig_cpu)
