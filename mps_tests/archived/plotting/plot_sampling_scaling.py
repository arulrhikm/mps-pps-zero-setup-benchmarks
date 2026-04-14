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

cpu = load_jsonl(f'{BASE}/cpu/experiment4_sampling_scaling_cpu.jsonl')
gpu = load_jsonl(f'{BASE}/gpu/experiment4_sampling_scaling_gpu.jsonl')

EXPECTED_SHOTS = {100, 500, 1000, 2000}

def get_complete_configs(data):
    """Group by (n, d, chi) -> {shots: runtime_ms}, keep only complete (all 4 shots)."""
    configs = defaultdict(dict)
    for d in data:
        key = (d['num_qubits'], d['depth'], d['bond_dimension'])
        configs[key][d['shots']] = d['run_time_ms']
    return {k: v for k, v in configs.items() if EXPECTED_SHOTS.issubset(v.keys())}

def get_time_per_shot(shots_dict):
    """Fit T = A + B*shots, return B (ms/shot)."""
    shots = np.array(sorted(shots_dict.keys()), dtype=float)
    times = np.array([shots_dict[s] for s in sorted(shots_dict.keys())], dtype=float)
    slope, _ = np.polyfit(shots, times, 1)
    return slope  # ms per shot

cpu_configs = get_complete_configs(cpu)
gpu_configs = get_complete_configs(gpu)

# Extract time-per-shot for each config
cpu_tps = {k: get_time_per_shot(v) for k, v in cpu_configs.items()}
gpu_tps = {k: get_time_per_shot(v) for k, v in gpu_configs.items()}

print(f"CPU: {len(cpu_tps)} complete configs")
print(f"GPU: {len(gpu_tps)} complete configs")

# Plotting helper
def plot_tps(tps_dict, x_index, x_label, group_labels, device, color_base, filename):
    """
    x_index: 0=qubits, 1=depth, 2=bond_dim in the (n,d,chi) key
    group_labels: the other two indices to group by
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by the non-x indices
    groups = defaultdict(lambda: ([], []))
    for key, tps in tps_dict.items():
        x_val = key[x_index]
        other = tuple(key[i] for i in range(3) if i != x_index)
        groups[other][0].append(x_val)
        groups[other][1].append(tps)
    
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(groups)))
    for i, (other, (xs, ys)) in enumerate(sorted(groups.items())):
        idx = np.argsort(xs)
        xs = np.array(xs)[idx]
        ys = np.array(ys)[idx]
        label_parts = []
        other_keys = [j for j in range(3) if j != x_index]
        names = ['n', 'd', '$\\chi$']
        for j, ok in enumerate(other_keys):
            label_parts.append(f'{names[ok]}={other[j]}')
        ax.plot(xs, ys, linestyle='None', marker='o', color=colors[i], markersize=6,
                label=', '.join(label_parts))
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Time per Shot (ms/shot)', fontsize=12)
    ax.set_title(f'Experiment 4: MPS Sampling — Time per Shot vs {x_label}\n({device}, complete trials only)', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.5)
    ax.axhline(y=0, color='gray', lw=0.5, ls=':')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved: {filename}")

# 1. CPU: Time per shot vs Depth
plot_tps(cpu_tps, 1, 'Circuit Depth', None, 'CPU', 'blue',
         'experiment4_tps_cpu_vs_depth.png')

# 2. GPU: Time per shot vs Depth
plot_tps(gpu_tps, 1, 'Circuit Depth', None, 'GPU', 'red',
         'experiment4_tps_gpu_vs_depth.png')

# 3. CPU: Time per shot vs Qubits
plot_tps(cpu_tps, 0, 'Number of Qubits', None, 'CPU', 'blue',
         'experiment4_tps_cpu_vs_qubits.png')

# 4. GPU: Time per shot vs Qubits
plot_tps(gpu_tps, 0, 'Number of Qubits', None, 'GPU', 'red',
         'experiment4_tps_gpu_vs_qubits.png')

# 5. CPU: Time per shot vs Bond Dimension
plot_tps(cpu_tps, 2, 'Bond Dimension ($\\chi$)', None, 'CPU', 'blue',
         'experiment4_tps_cpu_vs_bond.png')

# 6. GPU: Time per shot vs Bond Dimension
plot_tps(gpu_tps, 2, 'Bond Dimension ($\\chi$)', None, 'GPU', 'red',
         'experiment4_tps_gpu_vs_bond.png')

print("\nDone! 6 sampling scaling plots generated.")
