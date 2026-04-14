"""
Quantum Volume Benchmark Plot: CPU vs GPU statevector (GPU SV)

Reads JSONL from ../data/.
Run from this directory:  python qv_plot_combined.py

- Target depths: [10, 30, 60, 90, 120, 150] for CPU; [30, 60, 90, 120, 150] for GPU (no GPU depth 10)
- Uses first 5 trials per (num_qubits, depth) config
- Y-axis: runtime in ms, log scale
- CPU lines: shades of blue; GPU SV lines: shades of red
- Plots mean runtime over trials as discrete markers (no error bars, no point-to-point lines)
- Saves qv_benchmark_plot.png under ../plots/
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_DEPTHS     = [10, 30, 60, 90, 120, 150]
PLOT_DEPTHS       = [30, 60, 90, 120, 150] # Depths to show on plot (shared between CPU/GPU)
GPU_DEPTHS        = [30, 60, 90, 120, 150]
MAX_TRIALS        = 5
MIN_QUBITS        = 16
MAX_QUBITS_CPU    = 34
MAX_QUBITS_GPU    = 34   # include full GPU statevector sweep in jsonl

CPU_FILES = [
    os.path.join(DATA_DIR, "quantum_volume_runs_cpu_updated.jsonl"),   # depths 10, 20, 30, 40
    os.path.join(DATA_DIR, "quantum_volume_runs_cpu_extra.jsonl"),     # depths 60, 90, 120, 150
]
GPU_FILES = [
    os.path.join(DATA_DIR, "quantum_volume_runs_gpu_updated.jsonl"),   # depths 30, 60, 90, ...
]
PLOTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "plots")
OUTPUT_IMAGE = os.path.join(PLOTS_DIR, "qv_benchmark_plot.png")

# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(files, target_depths, min_q, max_q, max_trials):
    """Returns dict: (num_qubits, depth) -> list of run_time_ms (first max_trials)"""
    raw = {}   # (q, d) -> list of (trial, run_time_ms)
    for fpath in files:
        if not os.path.exists(fpath):
            print(f"[warn] File not found: {fpath}")
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                d = json.loads(line)
                q    = d['num_qubits']
                dep  = d['depth']
                t    = d['trial']
                rt   = d['run_time_ms']
                if q < min_q or q > max_q:
                    continue
                if dep not in target_depths:
                    continue
                key = (q, dep)
                if key not in raw:
                    raw[key] = []
                raw[key].append((t, rt))

    # Keep only first max_trials (by trial index)
    result = {}
    for key, runs in raw.items():
        runs.sort(key=lambda x: x[0])        # sort by trial number
        kept = [rt for _, rt in runs[:max_trials]]
        result[key] = kept
    return result


# ── Statistics per depth ─────────────────────────────────────────────────────
def group_by_depth(data, depths):
    """Group data: depth -> sorted list of (num_qubits, mean_ms, std_ms)"""
    groups = {d: [] for d in depths}
    for (q, dep), times in data.items():
        if dep not in groups:
            continue
        if len(times) == 0:
            continue
        arr = np.array(times, dtype=float)
        groups[dep].append((q, arr.mean(), arr.std()))
    for dep in groups:
        groups[dep].sort(key=lambda x: x[0])
    return groups


# ── Load ─────────────────────────────────────────────────────────────────────
cpu_data  = load_data(CPU_FILES, TARGET_DEPTHS, MIN_QUBITS, MAX_QUBITS_CPU, MAX_TRIALS)
gpu_data  = load_data(GPU_FILES, GPU_DEPTHS,    MIN_QUBITS, MAX_QUBITS_GPU, MAX_TRIALS)

cpu_groups = group_by_depth(cpu_data, TARGET_DEPTHS)
gpu_groups = group_by_depth(gpu_data, GPU_DEPTHS)

print(f"CPU configs loaded: { {d: len(v) for d, v in cpu_groups.items()} }")
print(f"GPU configs loaded: { {d: len(v) for d, v in gpu_groups.items()} }")

# ── Colors ───────────────────────────────────────────────────────────────────
# Use PLOT_DEPTHS for shared indexing
n_depths = len(PLOT_DEPTHS)

blue_cmap = plt.cm.Blues
red_cmap  = plt.cm.Reds

# Use range 0.40 → 0.90 so colours don't go too pale or too dark
cpu_colors = [blue_cmap(0.40 + 0.50 * i / max(n_depths - 1, 1)) for i in range(n_depths)]
gpu_colors = [red_cmap( 0.40 + 0.50 * i / max(n_depths - 1, 1)) for i in range(n_depths)]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

# CPU lines
for i, dep in enumerate(PLOT_DEPTHS):
    pts = cpu_groups.get(dep, [])
    if not pts:
        continue
    qs    = [p[0] for p in pts]
    means = [p[1] for p in pts]
    ax.plot(qs, means,
            marker='o', linestyle='None', markersize=7,
            markeredgecolor='white', markeredgewidth=0.5,
            color=cpu_colors[i],
            label=f'CPU depth={dep}')

# GPU lines
for i, dep in enumerate(PLOT_DEPTHS):
    pts = gpu_groups.get(dep, [])
    if not pts:
        continue
    qs    = [p[0] for p in pts]
    means = [p[1] for p in pts]
    ax.plot(qs, means,
            marker='s', linestyle='None', markersize=7,
            markeredgecolor='white', markeredgewidth=0.5,
            color=gpu_colors[i],
            label=f'GPU SV depth={dep}')

ax.set_yscale('log')
ax.set_xlabel('Number of Qubits', fontsize=13)
ax.set_ylabel('Run Time (ms)', fontsize=13)
ax.set_title(
    'Quantum volume runtime: CPU vs GPU SV\n'
    'Depths [30, 60, 90, 120, 150]  ·  n=5 trials',
    fontsize=13
)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, which='both', alpha=0.25)
ax.tick_params(axis='both', labelsize=11)

os.makedirs(PLOTS_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=150)
print(f"Saved plot to {OUTPUT_IMAGE}")

# ── Speedup Table ──────────────────────────────────────────────────────────
print("\n" + "="*40)
print("       GPU SV SPEEDUP VS CPU (at Q=33)")
print("="*40)

speedup_table = []
# Calculate speedup for the highest qubit count where both have data
# Usually this is 33 qubits
target_q = 33

for dep in PLOT_DEPTHS:
    cpu_pts = cpu_groups.get(dep, [])
    gpu_pts = gpu_groups.get(dep, [])
    
    cpu_val = next((p[1] for p in cpu_pts if p[0] == target_q), None)
    gpu_val = next((p[1] for p in gpu_pts if p[0] == target_q), None)
    
    if cpu_val and gpu_val:
        speedup = cpu_val / gpu_val
        speedup_table.append({
            "Depth": dep,
            "CPU Time (ms)": f"{cpu_val:.1f}",
            "GPU SV (ms)": f"{gpu_val:.1f}",
            "Speedup (x)": f"{speedup:.2f}x"
        })

if speedup_table:
    df = pd.DataFrame(speedup_table)
    print(df.to_string(index=False))
else:
    print("Could not calculate speedups (missing data for Q=33)")

