"""
MPS Runtime Predictor
======================
THE main predictor for MPS runtime scaling on GPU and CPU.

Loads all_mps_data_with_su4.jsonl, labels each record GPU or CPU,
fits piecewise-linear models using  T = c1 * (su4s * n * X²) + c2,
outputs a single comparison figure + pseudocode.

Key design choices
------------------
* Regimes are split by PERCENTILE of  x = su4s·n·χ²  (the model's own
  x-variable), so the regime boundaries align exactly with the x-axis of
  the "smooth" plot and lines never cross.
* Fitting uses WLS with weights  w_i = 1 / x_i  (inverse of the regressor).
  This assumes Var(T_i) ∝ x_i²  (roughly constant percentage error), so the
  prediction interval width grows as  √x*  while the prediction grows as x*,
  giving a relative PI that shrinks as  1/√x*  → tighter at high x.
* The 95 % PI is displayed as a shaded band on the "smooth" plot and is
  used in the predict_single CLI output.

Usage:
    python mps_predictor.py                       # fit + plot + pseudocode
    python mps_predictor.py <n> <d> <X>           # predict (both)
    python mps_predictor.py <n> <d> <X> gpu       # GPU only
    python mps_predictor.py <n> <d> <X> cpu       # CPU only
"""

import json, os, sys
import numpy as np
from scipy.optimize import nnls
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════ paths ═════════════════════════════════════════
BASE     = os.path.dirname(os.path.abspath(__file__))
MPS_ROOT = os.path.dirname(BASE)
DATA_FILE = os.path.join(MPS_ROOT, "data", "all_mps_data_with_su4.jsonl")
PLOTS_DIR = os.path.join(MPS_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ═══════════════════════════ data ══════════════════════════════════════════

def _hw_label(source_file):
    s = source_file.replace("/", "\\").lower()
    return "gpu" if s.startswith("gpu\\") else "cpu"

def load_data(filepath=DATA_FILE):
    gpu, cpu = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                r = json.loads(line)
                need = ("num_qubits", "depth", "bond_dimension", "run_time_ms",
                        "num_su4s", "num_gates")
                if any(r.get(k) is None for k in need): continue
                n, d, X = r["num_qubits"], r["depth"], r["bond_dimension"]
                r["S"]        = n * d * X**2          # used only for raw plot x-axis
                r["su4_n_X2"] = r["num_su4s"] * n * X**2   # model variable
                hw = _hw_label(r.get("source_file", ""))
                (gpu if hw == "gpu" else cpu).append(r)
            except (json.JSONDecodeError, TypeError):
                continue
    return gpu, cpu

# ═══════════════════════════ model ═════════════════════════════════════════

def fit_wls(recs):
    """
    Fit  T = c1 * su4_n_X2 + c2  using WLS with weights w_i = 1 / su4_n_X2_i.

    Variance assumption:  Var(T_i) ∝ x_i²  (roughly constant percentage error).
    → WLS via row-scaling:  scale each row by sqrt(w_i) before calling NNLS.
    → Weighted residual std σ_w satisfies:
         PI at x*  =  pred ± t · σ_w · √x*
      so the *relative* PI width  ∝ 1/√x*,  shrinking at large x.

    Returns
    -------
    coeffs : array [c1, c2]
    r2     : unweighted R² (for human-readable reporting)
    pred   : array of fitted values at training x
    sigma_w: weighted residual std (use with √x* to get PI margin)
    """
    n = len(recs)
    x = np.array([r["su4_n_X2"] for r in recs], dtype=float)
    y = np.array([r["run_time_ms"] for r in recs], dtype=float)

    w  = 1.0 / np.maximum(x, 1.0)   # w_i = 1/x_i
    sw = np.sqrt(w)                  # row-scale factor

    # WLS as scaled OLS:  (√W · A) @ coeffs = √W · y
    A_scaled = np.column_stack([sw * x, sw])
    y_scaled = sw * y
    coeffs, _ = nnls(A_scaled, y_scaled)

    pred  = coeffs[0] * x + coeffs[1]
    resid = y - pred

    # Weighted residual std
    sigma_w = float(np.sqrt(np.sum((sw * resid)**2) / max(n - 2, 1)))

    # Unweighted R² (informational)
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float(1.0 - np.sum(resid**2) / max(ss_tot, 1.0))

    return coeffs, r2, pred, sigma_w


def wls_pi_margin(x_new, sigma_w, n, alpha=0.05):
    """95 % WLS prediction interval half-width at new point x_new."""
    t = sp_stats.t.ppf(1 - alpha / 2, df=max(n - 2, 1))
    return t * sigma_w * float(np.sqrt(max(x_new, 0.0)))


def metrics(pred, actual):
    pct = np.abs(pred - actual) / np.maximum(actual, 1.0) * 100
    return np.median(pct), np.mean(pct), np.percentile(pct, 90)

# ═══════════════════════════ regime split ══════════════════════════════════

def split_regimes(recs, n_regimes):
    """
    Split records into n_regimes groups by PERCENTILE of su4_n_X2.
    Boundaries are in su4_n_X2 space so they align exactly with the
    model's x-axis — no more overlapping lines.
    """
    x = np.array([r["su4_n_X2"] for r in recs])
    bounds = [float(np.percentile(x, 100 * i / n_regimes))
              for i in range(1, n_regimes)]
    groups = [[] for _ in range(n_regimes)]
    for r in recs:
        placed = False
        for i, b in enumerate(bounds):
            if r["su4_n_X2"] < b:
                groups[i].append(r); placed = True; break
        if not placed:
            groups[-1].append(r)
    return groups, bounds

# ═══════════════════════════ reporting ════════════════════════════════════

def run_report(hw_name, recs, n_regimes=3):
    """Fit regimes, print report, return result dict."""
    if len(recs) < 10:
        print(f"\n  {hw_name}: only {len(recs)} records -- skipping.\n")
        return None
    print(f"\n{'='*90}")
    print(f"  {hw_name}  ({len(recs)} records)  —  {n_regimes} regimes  "
          f"[split + fit on su4s·n·χ², WLS w=1/x]")
    print(f"  Model: T = c1 · su4s · n · χ² + c2      "
          f"PI margin = t · σ_w · √(su4s·n·χ²*)   (→ tighter at high x)")
    print(f"{'='*90}")

    groups, bounds = split_regimes(recs, n_regimes)
    actual = np.array([r["run_time_ms"] for r in recs])
    regime_coeffs = []
    pred_all = np.zeros(len(recs))
    idx = 0

    print(f"\n  {'Regime':<40} {'N':>5} {'c1':>14} {'c2 (ms)':>12} "
          f"{'R²':>7} {'σ_w':>12} {'Med%':>7} {'P90%':>7}")
    print("  " + "-" * 100)
    for gi, grp in enumerate(groups):
        c, r2, p, sigma_w = fit_wls(grp)
        regime_coeffs.append((c, sigma_w, len(grp)))
        m = metrics(p, np.array([r["run_time_ms"] for r in grp]))
        for j in range(len(grp)):
            pred_all[idx] = p[j]; idx += 1
        lo = min(r["su4_n_X2"] for r in grp)
        hi = max(r["su4_n_X2"] for r in grp)
        label = (f"R{gi+1} (x={lo:>12,.0f}..{hi:>12,.0f})")
        print(f"  {label:<40} {len(grp):>5} "
              f"{c[0]:>14.6e} {c[1]:>12.0f} {r2:>7.4f} "
              f"{sigma_w:>12.2f} {m[0]:>6.1f}% {m[2]:>6.1f}%")

    m_all = metrics(pred_all, actual)
    print("  " + "-" * 100)
    print(f"  {'GLOBAL':<40} {len(actual):>5} {'':>14} {'':>12} {'':>7} "
          f"{'':>12} {m_all[0]:>6.1f}% {m_all[2]:>6.1f}%")
    print(f"{'='*90}")

    return dict(groups=groups, bounds=bounds, coeffs=regime_coeffs,
                pred=pred_all, actual=actual, metrics_all=m_all)

# ═══════════════════════════ plotting ═════════════════════════════════════

_REGIME_COLORS = ["#E53935", "#FB8C00", "#FDD835",
                  "#1E88E5", "#2E7D32", "#8E24AA"]


def _plot_raw_colored(ax, recs, result, title_extra=""):
    """
    Left panel: x = S = n·d·χ².  Points are coloured by regime membership
    (regime = which su4_n_X2 percentile bucket they fall in).  No misleading
    vertical lines in this coordinate — boundaries live in a different space.
    """
    for gi, grp in enumerate(result["groups"]):
        S_g = np.array([r["S"]          for r in grp])
        y_g = np.array([r["run_time_ms"] for r in grp]) / 1000
        c, r2, p, sigma_w = fit_wls(grp)
        m = metrics(p, np.array([r["run_time_ms"] for r in grp]))
        ax.scatter(S_g, y_g, s=20, alpha=0.45, color=_REGIME_COLORS[gi],
                   edgecolors="none", zorder=2,
                   label=f"R{gi+1} (N={len(grp)}, med={m[0]:.1f}%, R²={r2:.3f})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("S = n · d · χ²  (raw complexity proxy)", fontsize=11)
    ax.set_ylabel("Runtime (sec)", fontsize=11)
    ax.set_title("Points coloured by regime\n(boundaries live in su4s·n·χ² space)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")


# Relative floor: lower CI never drops below this fraction of the prediction.
# Prevents the band from vanishing to the log-scale bottom in high-noise / low-R² regimes.
_CI_REL_FLOOR = 0.05   # 5 % of ŷ
_R2_POOR_THRESHOLD = 0.20   # regimes below this get hatched fill


def _plot_smooth_ci(ax, recs, result, hw_label="", alpha_ci=0.05):
    """
    Right panel: x = su4s·n·χ².  Regime boundaries are exact vertical lines.
    Each regime shows its WLS fit line with 95% PI band.

    Lower CI floor at _CI_REL_FLOOR * ŷ so the band never disappears on a log
    scale (it would clip to 0 → −∞ otherwise).  Regimes with R² < _R2_POOR_THRESHOLD
    get hatched fill to signal that the linear model fits poorly there.
    """
    su4_all = np.array([r["su4_n_X2"]    for r in recs])
    y_all   = np.array([r["run_time_ms"] for r in recs]) / 1000
    ax.scatter(su4_all, y_all, s=18, alpha=0.22, color="steelblue",
               edgecolors="none", label="Actual", zorder=2)

    for gi, grp in enumerate(result["groups"]):
        c, r2, p, sigma_w = fit_wls(grp)
        x_g = np.array([r["su4_n_X2"] for r in grp])
        m   = metrics(p, np.array([r["run_time_ms"] for r in grp]))
        col = _REGIME_COLORS[gi]
        n_g = len(grp)
        poor_fit = r2 < _R2_POOR_THRESHOLD

        x_lo, x_hi = x_g.min(), x_g.max()
        x_sm = np.linspace(x_lo, x_hi, 400)
        pred_ms = np.maximum(c[0] * x_sm + c[1], 0)   # predicted value in ms
        y_sm    = pred_ms / 1000

        # WLS 95 % PI band:  margin = t · σ_w · √x
        t_val  = sp_stats.t.ppf(1 - alpha_ci / 2, df=max(n_g - 2, 1))
        margin = t_val * sigma_w * np.sqrt(x_sm)        # in ms

        # Lower bound: clamp to _CI_REL_FLOOR of ŷ so it never hits 0 on log scale.
        # Without this, max(pred - margin, 0) → 0 → −∞ on log scale for noisy regimes.
        y_hi = (pred_ms + margin) / 1000
        y_lo = np.maximum(pred_ms - margin, pred_ms * _CI_REL_FLOOR) / 1000

        label_r2 = f"R{gi+1} N={n_g}, med={m[0]:.1f}%, R²={r2:.3f}"
        if poor_fit:
            label_r2 += "  ⚠ poor fit"

        if poor_fit:
            # Hatch + lower alpha: the CI is wide because the linear model
            # doesn't explain the overhead-dominated region, not because the
            # circuit truly has that range of uncertainty.
            ax.fill_between(x_sm, y_lo, y_hi, color=col, alpha=0.08,
                            hatch="///", edgecolor=col, linewidth=0.0, zorder=1)
        else:
            ax.fill_between(x_sm, y_lo, y_hi, color=col, alpha=0.15, zorder=1)

        ax.plot(x_sm, y_sm, color=col, lw=2.2, zorder=3, label=label_r2)

    # Regime boundary vlines (exact in su4_n_X2 space)
    for b in result["bounds"]:
        ax.axvline(b, color="gray", lw=1.0, ls="--", alpha=0.55,
                   label=f"boundary x={b:,.0f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("su4s · n · χ²  (model variable)", fontsize=11)
    ax.set_ylabel("Runtime (sec)", fontsize=11)
    ax.set_title(
        "WLS fit + 95% PI band per regime\n"
        "(lower CI floored at 5% of ŷ;  ⚠ hatching = poor linear fit, R²<0.20)",
        fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")


def plot_all(gpu_data, cpu_data, gpu_result, cpu_result):
    for hw, data, result in [("GPU", gpu_data, gpu_result),
                              ("CPU", cpu_data, cpu_result)]:
        if result is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)
        fig.patch.set_facecolor("#fafafa")
        fig.suptitle(
            f"{hw}:  T = c₁ · su4s · n · χ² + c₂\n"
            f"Regimes split + fit on  su4s·n·χ²  "
            f"(WLS w=1/x → tighter CI at high x)",
            fontsize=14, fontweight="bold"
        )

        _plot_raw_colored(axes[0], data, result)
        _plot_smooth_ci(axes[1],   data, result, hw_label=hw)

        out = os.path.join(PLOTS_DIR, f"mps_predictor_{hw.lower()}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

# ═══════════════════════════ pseudocode generation ════════════════════════

def gen_pseudocode(hw, result):
    if result is None: return ""
    lines = [
        f"# {hw.upper()} Runtime Predictor",
        f"# Model:  T_ms = c1 * num_su4s * n * X**2 + c2",
        f"# Regimes split on x = num_su4s * n * X**2  (WLS, w=1/x)",
        f"# 95% PI at x*:  pred ± t * sigma_w * sqrt(x*)   (tighter at large x)",
        f"#",
        f"def predict_{hw.lower()}_runtime_ms(n, d, X, num_su4s):",
        f"    x = num_su4s * n * X**2",
    ]
    groups, bounds, coeffs = result["groups"], result["bounds"], result["coeffs"]
    for gi, grp in enumerate(groups):
        c, sigma_w, n_grp = coeffs[gi]
        _, _, p, _ = fit_wls(grp)
        m = metrics(p, np.array([r["run_time_ms"] for r in grp]))
        t_val  = sp_stats.t.ppf(0.975, df=max(n_grp - 2, 1))
        if gi == 0:
            lines.append(f"    if x < {bounds[0]:,.0f}:  "
                         f"# R{gi+1} (N={n_grp}, med_err={m[0]:.1f}%)")
        elif gi < len(bounds):
            lines.append(f"    elif x < {bounds[gi]:,.0f}:  "
                         f"# R{gi+1} (N={n_grp}, med_err={m[0]:.1f}%)")
        else:
            lines.append(f"    else:  # R{gi+1} (N={n_grp}, med_err={m[0]:.1f}%)")
        lines.append(f"        pred   = max(0, {c[0]:.6e} * x + {c[1]:.0f})")
        lines.append(f"        margin = {t_val:.4f} * {sigma_w:.4f} * x**0.5  "
                     f"# 95% PI half-width (ms)")
        lines.append(f"        return pred, pred - margin, pred + margin")
    lines.append("")
    return "\n".join(lines)

# ═══════════════════════════ single prediction ════════════════════════════

def predict_single(n, d, X, result, hw):
    if result is None:
        print(f"  {hw.upper()}: no model available"); return
    su4_est   = d * (n // 2)
    x_new     = su4_est * n * X**2
    S         = n * d * X**2
    groups, bounds, coeffs = result["groups"], result["bounds"], result["coeffs"]

    # Identify regime
    idx = len(groups) - 1
    for i, b in enumerate(bounds):
        if x_new < b: idx = i; break

    c, sigma_w, n_grp = coeffs[idx]
    pred   = max(0.0, c[0] * x_new + c[1])
    margin = wls_pi_margin(x_new, sigma_w, n_grp)
    rel_pi = margin / max(pred, 1) * 100

    print(f"\n  {'='*62}")
    print(f"  {hw.upper()} PREDICTION  (WLS, w=1/x, 95% PI)")
    print(f"  {'='*62}")
    print(f"    Qubits (n):            {n}")
    print(f"    Depth (d):             {d}")
    print(f"    Bond dim (χ):          {X}")
    print(f"    Est. SU(4) gates:      {su4_est:,}")
    print(f"    x = su4s·n·χ²:         {x_new:,.0f}")
    print(f"    S = n·d·χ²:            {S:,}")
    print(f"    Regime:                R{idx+1}")
    print(f"\n    Predicted:             {pred:,.0f} ms  ({pred/1000:.2f} s)")
    print(f"    95% PI:               "
          f"[{max(0, pred-margin):,.0f}, {pred+margin:,.0f}] ms")
    print(f"    PI width (relative):   ±{rel_pi:.1f}%")
    print(f"  {'='*62}")

# ═══════════════════════════ main ═════════════════════════════════════════

if __name__ == "__main__":
    gpu_data, cpu_data = load_data()
    print(f"Loaded {len(gpu_data)} GPU records, {len(cpu_data)} CPU records")

    gpu_result = run_report("GPU", gpu_data, n_regimes=6)
    cpu_result = run_report("CPU", cpu_data, n_regimes=6)

    plot_all(gpu_data, cpu_data, gpu_result, cpu_result)

    pc = gen_pseudocode("gpu", gpu_result) + "\n" + gen_pseudocode("cpu", cpu_result)
    out_pc = os.path.join(PLOTS_DIR, "mps_predictor_pseudocode.txt")
    with open(out_pc, "w", encoding="utf-8") as f:
        f.write(pc)
    print(f"Saved {out_pc}")
    print("\n" + pc)

    if len(sys.argv) >= 4:
        try:
            n, d, X = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
            hw = sys.argv[4].lower() if len(sys.argv) >= 5 else "both"
        except ValueError:
            print("Error: args must be integers"); sys.exit(1)
        if hw in ("gpu", "both"): predict_single(n, d, X, gpu_result, "gpu")
        if hw in ("cpu", "both"): predict_single(n, d, X, cpu_result, "cpu")
    elif len(sys.argv) == 1:
        print(f"\n{'='*62}")
        print(f"  SAMPLE PREDICTIONS")
        print(f"{'='*62}")
        for n, d, X in [(30, 20, 64), (40, 20, 128), (60, 20, 256)]:
            predict_single(n, d, X, gpu_result, "gpu")
            predict_single(n, d, X, cpu_result, "cpu")
