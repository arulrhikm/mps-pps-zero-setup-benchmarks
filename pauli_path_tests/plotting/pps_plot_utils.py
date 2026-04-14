"""
pps_plot_utils.py  –  Shared data loading & styling for PPS benchmark plots.

All plotting scripts import from here so the colour palette, JSONL parsing,
delta-to-Pauli mapping, and aggregation logic live in exactly one place.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"       # ../data from plots/
PLOT_DIR = Path(__file__).parent.parent / "plots"       # ../plots from plotting/

# ── Visual identity ────────────────────────────────────────────────────────
BACKENDS = {
    "PPS-GPU": {
        "file":   "pps_gpu_benchmark.jsonl",
        "color":  "#2563EB",   # blue
        "marker": "o",
        "ls":     "-",
        "zorder": 4,
        "label":  "PPS-GPU (BlueQubit)",
        "method": "PPS (BlueQubit)",
    },
    "PPS-CPU": {
        "file":   "pps_cpu_benchmark.jsonl",
        "color":  "#EA580C",   # orange (distinct for performance plots)
        "marker": "o",         # SAME circle for unified method plots
        "ls":     "-",
        "zorder": 3,
        "label":  "PPS-CPU (BlueQubit)",
        "method": "PPS (BlueQubit)",
    },
    "PPS-Qiskit": {
        "file":   "pps_qiskit_benchmark.jsonl",
        "color":  "#16A34A",   # green
        "marker": "^",
        "ls":     "-",
        "zorder": 2,
        "method": "PPS-Qiskit",
    },
    "PauliPropagation.jl": {
        "file":   "pps_julia_benchmark.jsonl",
        "color":  "#7C3AED",   # purple
        "marker": "D",
        "ls":     "--",
        "zorder": 1,
        "method": "PauliPropagation.jl",
    },
}

# δ → Max Pauli terms (from GPU benchmark tables)
DELTA_TO_PAULIS = {
    0.01:    664,
    0.005:   2_265,
    0.001:   33_143,
    0.0005:  115_667,
    0.0001:  2_166_786,
    5e-05:   7_566_013,
    2.5e-05: 27_570_927,
    1e-05:   149_411_103,
    9e-06:   181_362_123,
    8e-06:   226_590_774,
    7e-06:   291_693_307,
    6e-06:   390_125_351,
    5e-06:   549_853_786,
    4.5e-06: 662_823_961,
}

# Exact reference value for ⟨Z_62⟩ on the 127-qubit kicked Ising model.
# ── UPDATE THIS with your best high-χ MPS or published value ──
O_EXACT = 0.2955


# ── Matplotlib house style ─────────────────────────────────────────────────
def apply_style():
    """Call once at top of each script."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.labelsize":   13,
        "axes.titlesize":   13,
        "legend.fontsize":  9.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.grid":        False,       # we add grids manually
    })


# ── JSONL helpers ──────────────────────────────────────────────────────────
def load_jsonl(filepath):
    """Load a JSONL benchmark file, skipping # comment headers."""
    records = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate(records):
    """
    Group by delta → compute mean ± std for runtime and expectation value.
    Returns dict of numpy arrays sorted with δ descending (large first).
    """
    by_delta = defaultdict(lambda: {"times_ms": [], "exp_vals": []})
    for r in records:
        if "error" in r:
            continue
        
        # Normalize runtime to ms
        rt_ms = r.get("run_time_ms")
        if rt_ms is None:
            rt_s = r.get("run_time_s")
            if rt_s is not None:
                rt_ms = rt_s * 1000.0
        
        # Expectation value normalization
        ev = r.get("expectation_value")
        
        if rt_ms is not None and ev is not None:
            by_delta[r["delta"]]["times_ms"].append(rt_ms)
            by_delta[r["delta"]]["exp_vals"].append(ev)

    deltas = sorted(by_delta.keys(), reverse=True)
    return {
        "delta":       np.array(deltas),
        "time_s_mean": np.array([np.mean(by_delta[d]["times_ms"]) / 1000 for d in deltas]),
        "time_s_std":  np.array([np.std(by_delta[d]["times_ms"],  ddof=1) / 1000
                                 if len(by_delta[d]["times_ms"]) > 1 else 0
                                 for d in deltas]),
        "exp_mean":    np.array([np.mean(by_delta[d]["exp_vals"]) for d in deltas]),
        "exp_std":     np.array([np.std(by_delta[d]["exp_vals"],  ddof=1)
                                 if len(by_delta[d]["exp_vals"]) > 1 else 0
                                 for d in deltas]),
    }


def load_all_backends():
    """Load & aggregate every backend whose file exists. Returns dict."""
    data = {}
    for label, cfg in BACKENDS.items():
        fp = DATA_DIR / cfg["file"]
        if not fp.exists():
            print(f"  [skip] {fp.name} not found")
            continue
        raw = load_jsonl(fp)
        data[label] = aggregate(raw)
        print(f"  [ok]   {label:22s}  {len(data[label]['delta']):2d} delta-values  "
              f"({len(raw)} records)")
    return data


def delta_to_paulis(delta_arr):
    """Map an array of δ values to Pauli counts via nearest-key lookup."""
    out = []
    for d in np.atleast_1d(delta_arr):
        closest = min(DELTA_TO_PAULIS.keys(), key=lambda x: abs(x - d))
        out.append(DELTA_TO_PAULIS[closest])
    return np.array(out, dtype=float)


def pauli_label(n):
    """Pretty-print a Pauli count: 664 → '664', 2.2M → '2M', etc."""
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


def add_pauli_top_axis(ax_host):
    """
    Add a secondary top x-axis showing Max Pauli counts,
    mirroring the log-scaled δ axis of ax_host.
    Returns the twin axis.
    """
    import matplotlib.pyplot as plt
    ax2 = ax_host.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax_host.get_xlim())

    # Choose a readable subset of ticks
    tick_deltas = [1e-2, 1e-3, 1e-4, 1e-5, 4.5e-6]
    # Only keep those inside the current xlim
    lo, hi = sorted(ax_host.get_xlim())
    tick_deltas = [d for d in tick_deltas if lo * 0.9 <= d <= hi * 1.1]

    ax2.set_xticks(tick_deltas)
    ax2.set_xticklabels([pauli_label(delta_to_paulis(d)[0]) for d in tick_deltas],
                        fontsize=9)
    ax2.set_xlabel("Max Pauli Terms", fontsize=11, labelpad=6)
    ax2.tick_params(direction="in", length=4)
    ax2.minorticks_off()
    return ax2