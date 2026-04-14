#!/usr/bin/env python3
"""
Plot Experiment 2: Bond Dimension Scaling
Usage: python plot_experiment2_bond_scaling.py <input_file.jsonl>
Example: python plot_experiment2_bond_scaling.py experiment2_bond_scaling_cpu_updated.jsonl
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os

def load_data(filename):
    """Load data from JSONL file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def cubic_model(x, a, b):
    """Cubic scaling model: a * x^3 + b"""
    return a * x**3 + b

def quadratic_model(x, a, b):
    """Quadratic scaling model: a * x^2 + b"""
    return a * x**2 + b

def main():
    # Get input file from command line argument
    if len(sys.argv) < 2:
        print("Usage: python plot_experiment2_bond_scaling.py <input_file.jsonl>")
        print("Example: python plot_experiment2_bond_scaling.py experiment2_bond_scaling_cpu_updated.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Generate output filenames based on input file
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_plot.png"
    
    print(f"Loading data from: {input_file}")
    
    # Load experiment data
    data = load_data(input_file)
    
    if not data:
        print("Error: No data found in file")
        sys.exit(1)
    
    # Group by bond_dimension and num_qubits
    bond_data = {}
    for entry in data:
        bond_dim = entry['bond_dimension']
        n_qubits = entry['num_qubits']
        runtime_ms = entry['run_time_ms']
        
        if n_qubits not in bond_data:
            bond_data[n_qubits] = {}
        if bond_dim not in bond_data[n_qubits]:
            bond_data[n_qubits][bond_dim] = []
        bond_data[n_qubits][bond_dim].append(runtime_ms)
    
    qubit_counts = sorted(bond_data.keys())
    print(f"Found data for {len(qubit_counts)} qubit count(s): {qubit_counts}")
    
    # If only one qubit count, create a single plot
    if len(qubit_counts) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    else:
        # Create subplots for multiple qubit counts
        n_plots = len(qubit_counts)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, n_qubits in enumerate(qubit_counts):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        
        bond_dims = sorted(bond_data[n_qubits].keys())
        mean_runtimes = [np.mean(bond_data[n_qubits][b]) for b in bond_dims]
        
        # Convert to seconds
        mean_runtimes_sec = np.array(mean_runtimes) / 1000.0
        bond_dims_array = np.array(bond_dims)
        
        # Fit cubic model
        popt_cubic, _ = curve_fit(cubic_model, bond_dims_array, mean_runtimes_sec)
        r2_cubic = 1 - np.var(mean_runtimes_sec - cubic_model(bond_dims_array, *popt_cubic)) / np.var(mean_runtimes_sec)
        
        # Fit quadratic model
        popt_quad, _ = curve_fit(quadratic_model, bond_dims_array, mean_runtimes_sec)
        r2_quad = 1 - np.var(mean_runtimes_sec - quadratic_model(bond_dims_array, *popt_quad)) / np.var(mean_runtimes_sec)
        
        # Generate fitted curves
        bond_fit = np.linspace(min(bond_dims), max(bond_dims), 200)
        runtime_fit_cubic = cubic_model(bond_fit, *popt_cubic)
        runtime_fit_quad = quadratic_model(bond_fit, *popt_quad)
        
        # Determine which fit is better
        better_fit = "Cubic (chi^3)" if r2_cubic > r2_quad else "Quadratic (chi^2)"
        
        # Plot data points
        ax.plot(bond_dims, mean_runtimes_sec, linestyle='None', marker='o', markersize=8,
                label='Measured data', color=color, alpha=0.7)
        
        # Plot quadratic fit
        ax.plot(bond_fit, runtime_fit_quad, '--', linewidth=2, 
                label=f'Quadratic: {popt_quad[0]:.2e}·χ² (R²={r2_quad:.4f})', 
                color='#2E86AB', alpha=0.8)
        
        # Plot cubic fit
        ax.plot(bond_fit, runtime_fit_cubic, '-', linewidth=2.5, 
                label=f'Cubic: {popt_cubic[0]:.2e}·χ³ (R²={r2_cubic:.4f})', 
                color='#F18F01', alpha=0.9)
        
        ax.set_xlabel('Bond Dimension (χ)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=11, fontweight='bold')
        
        if len(qubit_counts) == 1:
            ax.set_title(f'Bond Dimension Scaling ({n_qubits} Qubits)\nBetter fit: {better_fit}', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{n_qubits} Qubits (Better: {better_fit})', fontsize=12, fontweight='bold')
        
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Print comparison
        print(f"\n{n_qubits} Qubits - Fit Comparison:")
        print(f"  Quadratic (chi^2): R^2 = {r2_quad:.6f}")
        print(f"  Cubic (chi^3):     R^2 = {r2_cubic:.6f}")
        print(f"  Better fit: {better_fit}")
    
    # Hide unused subplots
    if len(qubit_counts) > 1:
        for idx in range(len(qubit_counts), len(axes)):
            axes[idx].set_visible(False)
    
    if len(qubit_counts) > 1:
        fig.suptitle('Experiment 2: Bond Dimension Scaling', 
                     fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    
    plt.show()

if __name__ == '__main__':
    main()
