"""
GPU & CPU Regime-Based MPS Runtime Predictor
=============================================
Loads all_mps_data.jsonl, labels each record GPU or CPU by its source
directory, then fits a separate piecewise-linear predictor for each.

Scaling factor:  S = n * d * X**2

Regime boundaries are tuned so large-S regimes achieve low median % error.

Usage:
    python predict_runtime_regimes.py                       # evaluate both
    python predict_runtime_regimes.py <n> <d> <X>           # predict (both)
    python predict_runtime_regimes.py <n> <d> <X> gpu       # GPU only
    python predict_runtime_regimes.py <n> <d> <X> cpu       # CPU only
"""

import json, os, sys
import numpy as np
from scipy import stats as sp_stats

_BASE = os.path.dirname(os.path.abspath(__file__))
_MPS_ROOT = os.path.dirname(_BASE)
DATA_FILE = os.path.join(_MPS_ROOT, "data", "all_mps_data.jsonl")

# ───────────────────────────── data ────────────────────────────────────────

def _hw_label(source_file):
    s = source_file.replace("/", "\\").lower()
    if s.startswith("gpu\\"):
        return "gpu"
    return "cpu"  # cpu\\, quantum_volume_scaling.jsonl, or other top-level = CPU

def load_data(filepath=DATA_FILE):
    gpu, cpu = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                n, d, X, rt = (r.get("num_qubits"), r.get("depth"),
                               r.get("bond_dimension"), r.get("run_time_ms"))
                if None in (n, d, X, rt):
                    continue
                r["S"] = n * d * X ** 2
                hw = _hw_label(r.get("source_file", ""))
                (gpu if hw == "gpu" else cpu).append(r)
            except (json.JSONDecodeError, TypeError):
                continue
    return gpu, cpu

# ───────────────────────────── model ───────────────────────────────────────

def fit_linear(x, y):
    """Fit y = a*x + b.  x, y are numpy arrays.  Returns model dict."""
    if len(x) < 3:
        return None
    if np.ptp(x) == 0:
        return dict(slope=0.0, intercept=float(y.mean()), r2=0.0,
                    std_err=0.0, res_std=float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                    n=len(x), x_mean=float(x[0]), x_var=0.0)
    slope, intercept, r_val, _, std_err = sp_stats.linregress(x, y)
    resid = y - (slope * x + intercept)
    res_std = float(np.std(resid, ddof=2)) if len(resid) > 2 else float(np.std(resid))
    return dict(slope=slope, intercept=intercept, r2=r_val ** 2,
                std_err=std_err, res_std=res_std, n=len(x),
                x_mean=float(x.mean()), x_var=float(x.var()))

def predict_ci(model, S, confidence=0.95):
    if model is None:
        return None
    y = model["slope"] * S + model["intercept"]
    n = model["n"]
    if n <= 2:
        return dict(pred=max(0, y), lo=0, hi=max(0, y) * 3, margin=max(0, y) * 2)
    t = sp_stats.t.ppf((1 + confidence) / 2, df=n - 2)
    xdev2 = (S - model["x_mean"]) ** 2
    denom = max(n * model["x_var"], 1)
    se = model["res_std"] * np.sqrt(1 + 1 / n + xdev2 / denom)
    margin = t * se
    return dict(pred=max(0, y), lo=max(0, y - margin), hi=y + margin, margin=margin)

# ──────────── vectorised regime evaluation (fast) ─────────────────────────

def _eval_split(all_S, all_y, bounds):
    """
    Given pre-sorted arrays all_S, all_y and boundary list, compute the
    weighted score and per-regime stats.  Pure numpy, no per-record dicts.
    """
    n_reg = len(bounds) - 1
    score = 0.0
    stats = []
    for i in range(n_reg):
        lo, hi = bounds[i], bounds[i + 1]
        if i < n_reg - 1:
            mask = (all_S >= lo) & (all_S < hi)
        else:
            mask = (all_S >= lo) & (all_S <= hi)
        xs, ys = all_S[mask], all_y[mask]
        if len(xs) < 3:
            stats.append(dict(n=int(mask.sum()), med_pct=999.0, mean_pct=999.0,
                              p90_pct=999.0, coverage=0.0, lo=lo, hi=hi,
                              mae=0.0, model=None))
            score += 999.0 * ((i + 1) ** 2.5)
            continue
        m = fit_linear(xs, ys)
        pred = m["slope"] * xs + m["intercept"]
        errs = np.abs(pred - ys)
        pct = errs / np.maximum(ys, 1) * 100
        # CI coverage
        n_pts = m["n"]
        t = sp_stats.t.ppf(0.975, df=max(n_pts - 2, 1))
        xdev2 = (xs - m["x_mean"]) ** 2
        denom = max(n_pts * m["x_var"], 1)
        se = m["res_std"] * np.sqrt(1 + 1 / n_pts + xdev2 / denom)
        margin = t * se
        within = np.sum((ys >= pred - margin) & (ys <= pred + margin))
        med = float(np.median(pct))
        stats.append(dict(
            n=int(mask.sum()), med_pct=med,
            mean_pct=float(np.mean(pct)),
            p90_pct=float(np.percentile(pct, 90)),
            mae=float(np.mean(errs)),
            coverage=float(within / len(xs) * 100),
            lo=lo, hi=hi, model=m))
        score += med * ((i + 1) ** 2.5)
    return score, stats

# ──────────────────── boundary optimisation ────────────────────────────────

def find_best_boundaries(recs, n_regimes=10, n_iter=800, seed=42):
    all_S = np.array([r["S"] for r in recs], dtype=np.float64)
    all_y = np.array([r["run_time_ms"] for r in recs], dtype=np.float64)
    S_min, S_max = float(all_S.min()), float(all_S.max())

    def clamp(b):
        b = sorted(b); b[0] = S_min; b[-1] = S_max; return b

    # Seed candidates
    candidates = []
    q = np.linspace(0, 100, n_regimes + 1)
    candidates.append(clamp(list(np.percentile(all_S, q))))
    candidates.append(clamp(list(np.exp(
        np.linspace(np.log(max(1, S_min)), np.log(S_max), n_regimes + 1)))))
    # Dense at large S
    logS, logE = np.log(max(1, S_min)), np.log(S_max)
    pts = [logS + (logE - logS) * (i / n_regimes) ** 0.5 for i in range(n_regimes + 1)]
    candidates.append(clamp(list(np.exp(pts))))
    # Dense at large S (stronger)
    pts2 = [logS + (logE - logS) * (i / n_regimes) ** 0.35 for i in range(n_regimes + 1)]
    candidates.append(clamp(list(np.exp(pts2))))

    best_bounds, best_score, best_stats = None, 1e18, None
    for cand in candidates:
        sc, st = _eval_split(all_S, all_y, cand)
        if sc < best_score:
            best_bounds, best_score, best_stats = list(cand), sc, st

    rng = np.random.RandomState(seed)
    for _ in range(n_iter):
        trial = list(best_bounds)
        idx = rng.randint(1, n_regimes)
        lo_n, hi_n = trial[idx - 1], trial[idx + 1]
        new = np.exp(rng.uniform(np.log(max(1, lo_n + 1)),
                                  np.log(max(2, hi_n - 1))))
        if new <= lo_n or new >= hi_n:
            continue
        trial[idx] = new
        sc, st = _eval_split(all_S, all_y, trial)
        if sc < best_score:
            best_bounds, best_score, best_stats = trial, sc, st

    return best_bounds, best_stats

# ──────────────────────────── report ───────────────────────────────────────

def _lbl(i, lo, hi, last):
    end = "]" if last else ")"
    return f"R{i+1:>2}: [{lo:>14,.0f}, {hi:>14,.0f}{end}"

def run_report(hw_name, recs, n_regimes):
    if len(recs) < 10:
        print(f"\n  {hw_name}: only {len(recs)} records -- skipping.\n")
        return None, None

    print(f"\n{'=' * 115}")
    print(f"  {hw_name}  ({len(recs)} records)  --  {n_regimes} regimes")
    print(f"{'=' * 115}")

    bounds, reg_stats = find_best_boundaries(recs, n_regimes=n_regimes)
    models = [s["model"] for s in reg_stats]

    # Model params
    print(f"\n  {'Regime':<42} {'N':>4} {'Slope':>12} {'Intercept':>14} "
          f"{'R2':>7} {'Resid std':>14}")
    print("  " + "-" * 105)
    for i, st in enumerate(reg_stats):
        lbl = _lbl(i, st["lo"], st["hi"], i == len(reg_stats) - 1)
        m = st["model"]
        if m:
            print(f"  {lbl:<42} {st['n']:>4} {m['slope']:>12.6f} "
                  f"{m['intercept']:>14.1f} {m['r2']:>7.4f} {m['res_std']:>14.1f}")
        else:
            print(f"  {lbl:<42} {st['n']:>4}  -- too few --")

    # Accuracy
    print(f"\n  {'Regime':<42} {'N':>4} {'MAE (ms)':>12} {'Med%':>7} "
          f"{'Mean%':>7} {'P90%':>7} {'95%CI':>7}")
    print("  " + "-" * 105)
    all_pct, tot_w, tot_n = [], 0, 0
    for i, st in enumerate(reg_stats):
        lbl = _lbl(i, st["lo"], st["hi"], i == len(reg_stats) - 1)
        if st["med_pct"] >= 999:
            print(f"  {lbl:<42} {st['n']:>4}  -- insufficient --"); continue
        print(f"  {lbl:<42} {st['n']:>4} {st['mae']:>12,.0f} "
              f"{st['med_pct']:>6.1f}% {st['mean_pct']:>6.1f}% "
              f"{st['p90_pct']:>6.1f}% {st['coverage']:>6.1f}%")

    # Global
    all_S = np.array([r["S"] for r in recs])
    all_y = np.array([r["run_time_ms"] for r in recs])
    _, gstats = _eval_split(all_S, all_y, bounds)
    g_pct, g_w, g_n = [], 0, 0
    for st in gstats:
        if st["model"] is None: continue
        g_n += st["n"]
        g_w += int(st["coverage"] / 100 * st["n"])
    for st in gstats:
        if st["med_pct"] < 999:
            g_pct.extend([st["med_pct"]] * st["n"])   # weight by count
    cov = g_w / g_n * 100 if g_n else 0

    # compute true global pcts
    g_all_pct = []
    for i, st in enumerate(gstats):
        m = st["model"]
        if m is None: continue
        lo, hi = st["lo"], st["hi"]
        mask = (all_S >= lo) & (all_S <= hi)
        xs, ys = all_S[mask], all_y[mask]
        pred = m["slope"] * xs + m["intercept"]
        pct = np.abs(pred - ys) / np.maximum(ys, 1) * 100
        g_all_pct.extend(pct.tolist())
    ga = np.array(g_all_pct)
    print("  " + "-" * 105)
    print(f"  {'GLOBAL':<42} {len(ga):>4} {'':>12} {np.median(ga):>6.1f}% "
          f"{np.mean(ga):>6.1f}% {np.percentile(ga, 90):>6.1f}% {cov:>6.1f}%")
    print(f"\n  Median % err = {np.median(ga):.1f}%  |  "
          f"P90 = {np.percentile(ga, 90):.1f}%  |  "
          f"95% CI coverage = {cov:.1f}%")
    print("=" * 115)
    return models, bounds


def predict_single(n, d, X, models, bounds, hw):
    S = n * d * X ** 2
    n_reg = len(bounds) - 1
    idx = n_reg - 1
    for i in range(n_reg):
        if i < n_reg - 1 and bounds[i] <= S < bounds[i + 1]:
            idx = i; break
        elif i == n_reg - 1 and S <= bounds[i + 1]:
            idx = i
    if S < bounds[0]:
        idx = 0
    m = models[idx]
    print(f"\n  {'=' * 65}")
    print(f"  {hw.upper()} PREDICTION")
    print(f"  {'=' * 65}")
    print(f"    Qubits (n):         {n}")
    print(f"    Depth (d):          {d}")
    print(f"    Bond Dimension (X): {X}")
    print(f"    Scaling Factor S:   {S:,}")
    print(f"    Regime:             R{idx+1}  [{bounds[idx]:,.0f} - {bounds[idx+1]:,.0f}]")
    if m is None:
        print(f"    WARNING: too few data points."); return
    p = predict_ci(m, S)
    print(f"\n    Predicted Runtime:  {p['pred']:,.0f} ms  ({p['pred']/1000:.2f} sec)")
    print(f"    95% Prediction Interval:")
    print(f"      Lower: {p['lo']:,.0f} ms  ({p['lo']/1000:.2f} sec)")
    print(f"      Upper: {p['hi']:,.0f} ms  ({p['hi']/1000:.2f} sec)")
    print(f"    Model R2:          {m['r2']:.4f}")
    print(f"  {'=' * 65}")

# ───────────────────────────── main ────────────────────────────────────────

if __name__ == "__main__":
    gpu_data, cpu_data = load_data()
    print(f"Loaded {len(gpu_data)} GPU records, {len(cpu_data)} CPU records")

    gpu_models, gpu_bounds = run_report("GPU", gpu_data, n_regimes=10)
    cpu_models, cpu_bounds = run_report("CPU", cpu_data, n_regimes=10)

    if len(sys.argv) >= 4:
        try:
            n, d, X = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
            hw = sys.argv[4].lower() if len(sys.argv) >= 5 else "both"
        except ValueError:
            print("Error: args must be integers"); sys.exit(1)
        if hw in ("gpu", "both") and gpu_models:
            predict_single(n, d, X, gpu_models, gpu_bounds, "gpu")
        if hw in ("cpu", "both") and cpu_models:
            predict_single(n, d, X, cpu_models, cpu_bounds, "cpu")
    elif len(sys.argv) == 1:
        print(f"\n{'=' * 65}")
        print(f"  SAMPLE PREDICTIONS")
        print(f"{'=' * 65}")
        for n, d, X in [(30, 20, 64), (40, 20, 128), (60, 20, 256)]:
            if gpu_models: predict_single(n, d, X, gpu_models, gpu_bounds, "gpu")
            if cpu_models: predict_single(n, d, X, cpu_models, cpu_bounds, "cpu")
