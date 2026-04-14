#!/usr/bin/env python3
"""
Plot Experiment 3: Depth Scaling
Expected: Linear scaling O(d) where d is the circuit depth
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(filename):
    """Load data from JSONL file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                data.append(json.loads(line))
    return data

def linear_model(x, a, b):
    """Linear scaling model: a * x + b"""
    return a * x + b

def main():
    import os
    # Load experiment data
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "experiment3_depth_scaling.jsonl"),
        os.path.join(here, "..", "cpu", "experiment3_depth_scaling_cpu.jsonl"),
        os.path.join(here, "..", "experiment3_depth_scaling_cpu.jsonl"),
    ]
    filename = next((c for c in candidates if os.path.exists(c)), None)
    if not filename:
        print("Error: no experiment3 JSONL found")
        return
    data = load_data(filename)
    
    # Group by depth and num_qubits
    depth_data = {}
    for entry in data:
        depth = entry['depth']
        n_qubits = entry['num_qubits']
        runtime_ms = entry['run_time_ms']
        
        if n_qubits not in depth_data:
            depth_data[n_qubits] = {}
        if depth not in depth_data[n_qubits]:
            depth_data[n_qubits][depth] = []
        depth_data[n_qubits][depth].append(runtime_ms)
    
    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    qubit_counts = sorted(depth_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Linear scale plot
    for n_qubits, color in zip(qubit_counts, colors):
        depths = sorted(depth_data[n_qubits].keys())
        mean_runtimes = [np.mean(depth_data[n_qubits][d]) for d in depths]
        
        # Convert to seconds
        mean_runtimes_sec = np.array(mean_runtimes) / 1000.0
        depths_array = np.array(depths)
        
        # Fit linear model
        popt, _ = curve_fit(linear_model, depths_array, mean_runtimes_sec)
        
        # Generate fitted curve
        depth_fit = np.linspace(min(depths), max(depths), 200)
        runtime_fit = linear_model(depth_fit, *popt)
        
        # Plot data points
        ax1.plot(depths, mean_runtimes_sec, linestyle='None', marker='o', markersize=8,
                label=f'{n_qubits} qubits (data)', color=color, alpha=0.7)
        # Plot fit
        ax1.plot(depth_fit, runtime_fit, '--', linewidth=2, 
                label=f'{n_qubits} qubits (fit: {popt[0]:.2f}·d + {popt[1]:.2f})', 
                color=color, alpha=0.5)
    
    ax1.set_xlabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Experiment 3: Depth Scaling (Linear O(d))', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Log-log plot to verify linear scaling (slope = 1)
    for n_qubits, color in zip(qubit_counts, colors):
        depths = sorted(depth_data[n_qubits].keys())
        mean_runtimes = [np.mean(depth_data[n_qubits][d]) for d in depths]
        mean_runtimes_sec = np.array(mean_runtimes) / 1000.0
        
        ax2.plot(depths, mean_runtimes_sec, linestyle='None', marker='o', markersize=8,
                label=f'{n_qubits} qubits', color=color, alpha=0.7)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Circuit Depth (log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime (seconds, log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Experiment 3: Depth Scaling (Log-Log, slope ≈ 1)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig('experiment3_depth_scaling.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'experiment3_depth_scaling.png'")
    print(f"Expected scaling: O(d) - Linear")
    
    # Create individual plots for each qubit count
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (n_qubits, ax, color) in enumerate(zip(qubit_counts, axes, colors)):
        depths = sorted(depth_data[n_qubits].keys())
        mean_runtimes = [np.mean(depth_data[n_qubits][d]) for d in depths]
        mean_runtimes_sec = np.array(mean_runtimes) / 1000.0
        depths_array = np.array(depths)
        
        # Fit linear model
        popt, pcov = curve_fit(linear_model, depths_array, mean_runtimes_sec)
        perr = np.sqrt(np.diag(pcov))
        
        # Generate fitted curve
        depth_fit = np.linspace(min(depths), max(depths), 200)
        runtime_fit = linear_model(depth_fit, *popt)
        
        # Plot
        ax.plot(depths, mean_runtimes_sec, linestyle='None', marker='o', markersize=10,
                label='Measured data', color=color, alpha=0.7)
        ax.plot(depth_fit, runtime_fit, '--', linewidth=2, 
                label=f'Linear fit: {popt[0]:.2f}·d + {popt[1]:.2f}', 
                color='black', alpha=0.6)
        ax.set_xlabel('Circuit Depth', fontsize=11, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=11, fontweight='bold')
        ax.set_title(f'{n_qubits} Qubits (R² = {1 - np.var(mean_runtimes_sec - linear_model(depths_array, *popt))/np.var(mean_runtimes_sec):.4f})', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    fig2.suptitle('Experiment 3: Depth Scaling - Individual Qubit Counts', 
                  fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('experiment3_depth_scaling_individual.png', dpi=300, bbox_inches='tight')
    print(f"Individual plots saved as 'experiment3_depth_scaling_individual.png'")
    
    plt.show()

if __name__ == '__main__':
    main()
