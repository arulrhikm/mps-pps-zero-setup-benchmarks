import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

BASE = 'c:/Users/arulr/Projects/BlueQubit/mps_tests'

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def aggregate(data, key):
    groups = defaultdict(list)
    for d in data:
        groups[d[key]].append(d['run_time_ms'])
    xs = sorted(groups.keys())
    ys = [np.mean(groups[x]) for x in xs]
    return np.array(xs, dtype=float), np.array(ys) / 1000  # seconds

def add_fit(ax, x, y, color, label_prefix):
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ax.plot(x, fit, '--', color=color, alpha=0.6, lw=1.5,
            label=f'{label_prefix} fit (slope={slope:.1f} s/unit)')
    return slope

# ─── Experiment 1: Qubit Scaling ───
cpu1 = load_jsonl(f'{BASE}/cpu/experiment1_qubit_scaling_cpu.jsonl')
gpu1 = load_jsonl(f'{BASE}/gpu/experiment1_qubit_scaling_gpu.jsonl')
x_cpu1, y_cpu1 = aggregate(cpu1, 'num_qubits')
x_gpu1, y_gpu1 = aggregate(gpu1, 'num_qubits')

def poly_fit_r2(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    p = np.poly1d(coeffs)
    y_fit = p(x)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return p, r2

# Try linear (deg=1) and quadratic (deg=2) for each
_, r2_cpu_lin = poly_fit_r2(x_cpu1, y_cpu1, 1)
p_cpu1, r2_cpu_quad = poly_fit_r2(x_cpu1, y_cpu1, 2)
_, r2_gpu_lin = poly_fit_r2(x_gpu1, y_gpu1, 1)
p_gpu1, r2_gpu_quad = poly_fit_r2(x_gpu1, y_gpu1, 2)

# Extract coefficients: p(x) = a*x^2 + b*x + c
a_cpu, b_cpu, c_cpu = p_cpu1.coeffs
a_gpu, b_gpu, c_gpu = p_gpu1.coeffs

x_smooth = np.linspace(x_cpu1.min(), x_cpu1.max(), 200)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_cpu1, y_cpu1, linestyle='None', marker='o', color='#2196F3', markersize=5, label='MPS CPU')
ax.plot(x_gpu1, y_gpu1, linestyle='None', marker='s', color='#FF5722', markersize=5, label='MPS GPU')
ax.plot(x_smooth, p_cpu1(x_smooth), '--', color='#2196F3', alpha=0.7, lw=2,
        label=f'CPU: ${a_cpu:.2f}n^2 {b_cpu:+.1f}n {c_cpu:+.0f}$  ($R^2$={r2_cpu_quad:.4f})')
ax.plot(x_smooth, p_gpu1(x_smooth), '--', color='#FF5722', alpha=0.7, lw=2,
        label=f'GPU: ${a_gpu:.2f}n^2 {b_gpu:+.1f}n {c_gpu:+.0f}$  ($R^2$={r2_gpu_quad:.4f})')
ax.set_xlabel('Number of Qubits (n)', fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_title('Experiment 1: MPS Qubit Scaling — CPU vs GPU\n(depth=16, $\\chi$=256)', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, ls='--', alpha=0.5)
fig.tight_layout()
fig.savefig('cpu_vs_gpu_experiment1_qubit_scaling.png', dpi=150)
print(f"Exp1 CPU: T = {a_cpu:.2f}n² + {b_cpu:.1f}n + {c_cpu:.0f}, R²={r2_cpu_quad:.4f}")
print(f"Exp1 GPU: T = {a_gpu:.2f}n² + {b_gpu:.1f}n + {c_gpu:.0f}, R²={r2_gpu_quad:.4f}")

# ─── Experiment 2: Bond Dimension Scaling ───
# Use polynomial fits with exponents from log-log analysis
from scipy.optimize import curve_fit

cpu2 = load_jsonl(f'{BASE}/cpu/experiment2_bond_scaling_cpu_updated.jsonl')
gpu2 = load_jsonl(f'{BASE}/gpu/experiment2_bond_scaling_gpu_updated.jsonl')
x_cpu2, y_cpu2 = aggregate(cpu2, 'bond_dimension')
x_gpu2, y_gpu2 = aggregate(gpu2, 'bond_dimension')

# Exponents from log-log analysis (chi >= 40 regime)
p_cpu = 1.91  # from experiment2_bond_scaling_cpu_loglog
p_gpu = 1.17  # from experiment2_bond_scaling_gpu_loglog

def power_model(x, a, b, p):
    return a * x**p + b

# Fit T = a * chi^p + b with fixed exponent p
popt_cpu, _ = curve_fit(lambda x, a, b: power_model(x, a, b, p_cpu), x_cpu2, y_cpu2, p0=[1e-3, 0])
popt_gpu, _ = curve_fit(lambda x, a, b: power_model(x, a, b, p_gpu), x_gpu2, y_gpu2, p0=[1e-3, 0])

x_smooth = np.linspace(x_cpu2.min(), x_cpu2.max(), 200)
fit_cpu2 = power_model(x_smooth, *popt_cpu, p_cpu)
fit_gpu2 = power_model(x_smooth, *popt_gpu, p_gpu)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_cpu2, y_cpu2, linestyle='None', marker='o', color='#2196F3', markersize=5, label='MPS CPU')
ax.plot(x_gpu2, y_gpu2, linestyle='None', marker='s', color='#FF5722', markersize=5, label='MPS GPU')
ax.plot(x_smooth, fit_cpu2, '--', color='#2196F3', alpha=0.7, lw=2,
        label=f'CPU fit: $T \\propto \\chi^{{{p_cpu:.2f}}}$')
ax.plot(x_smooth, fit_gpu2, '--', color='#FF5722', alpha=0.7, lw=2,
        label=f'GPU fit: $T \\propto \\chi^{{{p_gpu:.2f}}}$')
ax.set_xlabel('Bond Dimension ($\\chi$)', fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_title(f'Experiment 2: MPS Bond Dimension Scaling — CPU vs GPU\n(n=40 qubits, depth=16) | CPU: $\\chi^{{{p_cpu}}}$, GPU: $\\chi^{{{p_gpu}}}$', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, ls='--', alpha=0.5)
fig.tight_layout()
fig.savefig('cpu_vs_gpu_experiment2_bond_scaling.png', dpi=150)
print(f"Exp2: CPU exponent={p_cpu}, GPU exponent={p_gpu}")

# ─── Experiment 3: Depth Scaling ───
cpu3 = load_jsonl(f'{BASE}/cpu/experiment3_depth_scaling_cpu.jsonl')
gpu3 = load_jsonl(f'{BASE}/gpu/experiment3_depth_scaling_gpu.jsonl')
x_cpu3, y_cpu3 = aggregate(cpu3, 'depth')
x_gpu3, y_gpu3 = aggregate(gpu3, 'depth')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_cpu3, y_cpu3, linestyle='None', marker='o', color='#2196F3', markersize=5, label='MPS CPU')
ax.plot(x_gpu3, y_gpu3, linestyle='None', marker='s', color='#FF5722', markersize=5, label='MPS GPU')
s_cpu = add_fit(ax, x_cpu3, y_cpu3, '#2196F3', 'CPU')
s_gpu = add_fit(ax, x_gpu3, y_gpu3, '#FF5722', 'GPU')
ax.set_xlabel('Circuit Depth', fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_title(f'Experiment 3: MPS Depth Scaling — CPU vs GPU\n(n=40 qubits, $\\chi$=128) | GPU speedup: {s_cpu/s_gpu:.1f}x', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, ls='--', alpha=0.5)
fig.tight_layout()
fig.savefig('cpu_vs_gpu_experiment3_depth_scaling.png', dpi=150)
print(f"Exp3: CPU slope={s_cpu:.1f}, GPU slope={s_gpu:.1f}, speedup={s_cpu/s_gpu:.1f}x")

print("\nDone! All 3 MPS comparison plots generated.")
