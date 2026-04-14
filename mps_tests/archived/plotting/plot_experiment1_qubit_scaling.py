#!/usr/bin/env python3
"""
Plot Experiment 1: Qubit Scaling
Analyzes scaling behavior with fixed depth=16 and bond_dimension=16
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(filename):
    """Load data from JSONL file, filtering out specific outliers"""
    data = []
    outliers_removed = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                entry = json.loads(line)
                # Filter out outliers: trial 0 at 16 qubits (first one) and trial 0 at 68 qubits
                if (entry['num_qubits'] == 16 and entry['trial'] == 0 and entry['run_time_ms'] > 10000):
                    outliers_removed += 1
                    continue
                if (entry['num_qubits'] == 68 and entry['trial'] == 0 and entry['run_time_ms'] > 30000):
                    outliers_removed += 1
                    continue
                data.append(entry)
    
    print(f"Loaded {len(data)} data points ({outliers_removed} outliers removed)")
    return data

def linear_model(x, a, b):
    """Linear scaling model: a * x + b"""
    return a * x + b

def quadratic_model(x, a, b):
    """Quadratic scaling model: a * x^2 + b"""
    return a * x**2 + b

def main():
    import os
    # Load experiment data (with outliers filtered)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "experiment1_qubit_scaling.jsonl"),
        os.path.join(here, "..", "cpu", "experiment1_qubit_scaling_cpu.jsonl"),
        os.path.join(here, "..", "experiment1_qubit_scaling_cpu_old.jsonl"),
    ]
    filename = next((c for c in candidates if os.path.exists(c)), None)
    if not filename:
        print("Error: no experiment1 JSONL found (tried local, ../cpu/, ../experiment1_*_old.jsonl)")
        return
    data = load_data(filename)
    
    # Group by num_qubits and calculate mean runtime
    qubit_runtimes = {}
    for entry in data:
        n_qubits = entry['num_qubits']
        runtime_ms = entry['run_time_ms']
        
        if n_qubits not in qubit_runtimes:
            qubit_runtimes[n_qubits] = []
        qubit_runtimes[n_qubits].append(runtime_ms)
    
    # Calculate mean runtime for each qubit count
    qubits = sorted(qubit_runtimes.keys())
    mean_runtimes = [np.mean(qubit_runtimes[q]) for q in qubits]

    # Convert to seconds for better readability
    mean_runtimes_sec = np.array(mean_runtimes) / 1000.0
    qubits_array = np.array(qubits)
    
    # Fit linear and quadratic models
    popt_linear, _ = curve_fit(linear_model, qubits_array, mean_runtimes_sec)
    popt_quad, _ = curve_fit(quadratic_model, qubits_array, mean_runtimes_sec)
    
    # Calculate R² for both fits
    r2_linear = 1 - np.var(mean_runtimes_sec - linear_model(qubits_array, *popt_linear)) / np.var(mean_runtimes_sec)
    r2_quad = 1 - np.var(mean_runtimes_sec - quadratic_model(qubits_array, *popt_quad)) / np.var(mean_runtimes_sec)
    
    # Generate fitted curves
    qubits_fit = np.linspace(min(qubits), max(qubits), 200)
    runtime_fit_linear = linear_model(qubits_fit, *popt_linear)
    runtime_fit_quad = quadratic_model(qubits_fit, *popt_quad)
    
    # Create single linear plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot data (mean runtime)
    ax.plot(qubits, mean_runtimes_sec,
            linestyle='None', marker='o', markersize=6, label='Measured data',
            color='#2E86AB', alpha=0.7, zorder=5)
    
    # Plot both fits
    ax.plot(qubits_fit, runtime_fit_linear, '-', linewidth=2.5, 
             label=f'Linear: {popt_linear[0]:.3f}·n + {popt_linear[1]:.2f} (R²={r2_linear:.4f})', 
             color='#F18F01', alpha=0.9)
    ax.plot(qubits_fit, runtime_fit_quad, '--', linewidth=2.5, 
             label=f'Quadratic: {popt_quad[0]:.2e}·n² + {popt_quad[1]:.2f} (R²={r2_quad:.4f})', 
             color='#00A878', alpha=0.8)
    
    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Experiment 1: Qubit Scaling (Fixed χ=16, d=16)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('experiment1_qubit_scaling.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'experiment1_qubit_scaling.png'")
    print(f"\nFit comparison:")
    print(f"  Linear:    {popt_linear[0]:.3f}·n + {popt_linear[1]:.2f}, R² = {r2_linear:.6f}")
    print(f"  Quadratic: {popt_quad[0]:.2e}·n² + {popt_quad[1]:.2f}, R² = {r2_quad:.6f}")
    print(f"\nDifference in R²: {abs(r2_quad - r2_linear):.6f}")
    if abs(r2_quad - r2_linear) < 0.01:
        print("-> Both fits are nearly equivalent; linear is simpler")
    else:
        print(f"-> {'Quadratic' if r2_quad > r2_linear else 'Linear'} fit is better")
    
    plt.show()

if __name__ == '__main__':
    main()
