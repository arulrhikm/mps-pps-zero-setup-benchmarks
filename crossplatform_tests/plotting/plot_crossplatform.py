"""
Cross-platform runtime figures (Braket SV1, Quantum Rings, BlueQubit CPU/GPU).

Each series is the mean run time over trials at fixed (backend, n, depth).
Measured points are markers only (no chords between them); there are no error bars.
"""

import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths relative to this script (works from any cwd)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSSPLATFORM_ROOT = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(CROSSPLATFORM_ROOT, "plots")
DATA_DIR = os.path.join(CROSSPLATFORM_ROOT, "data")
STATEVECTOR_DATA = os.path.join(
    os.path.dirname(CROSSPLATFORM_ROOT), "statevector_tests", "data"
)


def load_jsonl(filepath, backend_name):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            entry = json.loads(line)
            if entry.get('job_id') == 'error' or entry.get('status') == 'error':
                continue
            entry['backend'] = backend_name
            data.append(entry)
    return pd.DataFrame(data)

def load_all_data():
    df_braket = load_jsonl(
        os.path.join(DATA_DIR, "braket_sv1_results.jsonl"), "Braket SV1"
    )
    df_qr = load_jsonl(
        os.path.join(DATA_DIR, "quantum_rings_results.jsonl"), "QuantumRings"
    )
    df_bq_gpu = load_jsonl(
        os.path.join(STATEVECTOR_DATA, "quantum_volume_runs_gpu_updated.jsonl"),
        "BlueQubit GPU",
    )
    df_bq_cpu_1 = load_jsonl(
        os.path.join(STATEVECTOR_DATA, "quantum_volume_runs_cpu_updated.jsonl"),
        "BlueQubit CPU",
    )

    # Check if extra CPU file exists, sometimes the user has split data
    cpu_extra = os.path.join(STATEVECTOR_DATA, "quantum_volume_runs_cpu_extra.jsonl")
    if os.path.exists(cpu_extra):
        df_bq_cpu_2 = load_jsonl(cpu_extra, "BlueQubit CPU")
        df_bq_cpu = pd.concat([df_bq_cpu_1, df_bq_cpu_2], ignore_index=True)
    else:
        df_bq_cpu = df_bq_cpu_1
    
    df_all = pd.concat([df_braket, df_qr, df_bq_cpu, df_bq_gpu], ignore_index=True)

    # Remove bad trials (outliers): drop runs with run_time_ms > 1.5× group median (for cleaner means)
    # (pandas 3+ groupby.apply drops grouping keys; loop keeps all columns)
    pieces = []
    for _, group in df_all.groupby(["backend", "num_qubits", "depth"], sort=False):
        if len(group) <= 2:
            pieces.append(group)
            continue
        median_time = group["run_time_ms"].median()
        filtered = group[group["run_time_ms"] <= 1.5 * median_time]
        pieces.append(group if len(filtered) == 0 else filtered)
    df_cleaned = pd.concat(pieces, ignore_index=True)
    return df_cleaned

def plot_runtime_scaling(df, target_depth):
    plt.figure(figsize=(10, 6))
    
    # Filter to specified depth
    df_filter = df[df['depth'] == target_depth]
    
    # Aggregate over trials
    agg = (
        df_filter.groupby(["backend", "num_qubits"], as_index=False)["run_time_ms"]
        .mean()
        .rename(columns={"run_time_ms": "mean"})
    )
    
    if agg.empty:
        print(f"No data for depth {target_depth}")
        return
        
    backends = ['Braket SV1', 'QuantumRings', 'BlueQubit CPU', 'BlueQubit GPU']
    colors = {'Braket SV1': 'purple', 'QuantumRings': 'green', 'BlueQubit CPU': 'blue', 'BlueQubit GPU': 'red'}
    markers = {'Braket SV1': 's', 'QuantumRings': '^', 'BlueQubit CPU': 'o', 'BlueQubit GPU': 'v'}
    
    for backend in backends:
        b_data = agg[agg['backend'] == backend].sort_values('num_qubits')
        if b_data.empty:
            continue
        plt.plot(
            b_data["num_qubits"],
            b_data["mean"],
            label=backend,
            color=colors[backend],
            marker=markers[backend],
            markersize=8,
            linestyle="None",
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    plt.yscale('log')
    plt.xlabel('Number of Qubits ($n$)', fontsize=14)
    plt.ylabel('Runtime (ms)', fontsize=14)
    plt.title(f'Quantum Volume Circuit Runtime (Depth={target_depth})', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filename = f"fig_crossplatform_runtime_d{target_depth}.png"
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def plot_runtime_per_gate(df):
    plt.figure(figsize=(10, 6))
    
    # Calculate runtime per gate
    df['runtime_per_gate_ms'] = df['run_time_ms'] / df['num_gates']
    
    # Aggregate over ALL depths and trials to get a smooth curve per qubit
    # SV gate time depends strictly on num_qubits, so grouping all depths is numerically valid and rich.
    agg = (
        df.groupby(["backend", "num_qubits"], as_index=False)["runtime_per_gate_ms"]
        .mean()
        .rename(columns={"runtime_per_gate_ms": "mean"})
    )
    
    backends = ['Braket SV1', 'QuantumRings', 'BlueQubit CPU', 'BlueQubit GPU']
    colors = {'Braket SV1': 'purple', 'QuantumRings': 'green', 'BlueQubit CPU': 'blue', 'BlueQubit GPU': 'red'}
    markers = {'Braket SV1': 's', 'QuantumRings': '^', 'BlueQubit CPU': 'o', 'BlueQubit GPU': 'v'}
    
    for backend in backends:
        b_data = agg[agg['backend'] == backend].sort_values('num_qubits')
        if b_data.empty:
            continue
        plt.plot(
            b_data["num_qubits"],
            b_data["mean"],
            label=backend,
            color=colors[backend],
            marker=markers[backend],
            markersize=8,
            linestyle="None",
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    plt.yscale('log')
    plt.xlabel('Number of Qubits ($n$)', fontsize=14)
    plt.ylabel('Runtime per Gate (ms)', fontsize=14)
    plt.title('Statevector Gate Application Time', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "fig_crossplatform_runtime_per_gate.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    df = load_all_data()
    plot_runtime_scaling(df, target_depth=30)
    plot_runtime_scaling(df, target_depth=60)
    plot_runtime_per_gate(df)
