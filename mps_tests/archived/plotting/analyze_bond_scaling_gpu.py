import json
import matplotlib.pyplot as plt
import numpy as np

# Load GPU data
data_file = 'c:/Users/arulr/Projects/BlueQubit/mps_tests/gpu/experiment2_bond_scaling_gpu_updated.jsonl'
bond_dims, runtimes = [], []
with open(data_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        try:
            e = json.loads(line)
            bond_dims.append(e['bond_dimension'])
            runtimes.append(e['run_time_ms'])
        except json.JSONDecodeError: continue

chi = np.array(bond_dims, dtype=float)
t = np.array(runtimes, dtype=float)
idx = np.argsort(chi); chi, t = chi[idx], t[idx]

# Filter: only use chi >= 40 to avoid the constant-overhead regime at small bond dims
mask = chi >= 40
chi_fit = chi[mask]
t_fit = t[mask]

log_chi_all = np.log10(chi)
log_t_all = np.log10(t)
log_chi = np.log10(chi_fit)
log_t = np.log10(t_fit)

p, c = np.polyfit(log_chi, log_t, 1)
fit = p * log_chi + c

# R² calculation
ss_res = np.sum((log_t - fit)**2)
ss_tot = np.sum((log_t - np.mean(log_t))**2)
r2 = 1 - ss_res / ss_tot

print(f"GPU: T ~ chi^{p:.4f}  (slope p = {p:.4f}, R² = {r2:.4f}, fit for chi >= 40)")

plt.figure(figsize=(10, 6))
plt.scatter(log_chi_all, log_t_all, color='gray', s=20, alpha=0.4, label='All data')
plt.scatter(log_chi, log_t, color='black', s=20, label='Fit region ($\\chi \\geq 40$)')
plt.plot(log_chi, fit, 'r--', lw=2, label=f'Fit: $T \\propto \\chi^{{{p:.2f}}}$ (p={p:.4f}, $R^2$={r2:.4f})')
plt.xlabel('log$_{10}$($\\chi$)', fontsize=12)
plt.ylabel('log$_{10}$(Runtime ms)', fontsize=12)
plt.title(f'MPS Bond Scaling (GPU)\n$T \\propto \\chi^{{{p:.2f}}}$, $R^2 = {r2:.4f}$ (fit for $\\chi \\geq 40$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig('experiment2_bond_scaling_gpu_loglog.png')
print("Saved: experiment2_bond_scaling_gpu_loglog.png")
