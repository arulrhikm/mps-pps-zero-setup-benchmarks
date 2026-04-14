#!/usr/bin/env python3
"""
Compare CPU vs GPU Bond Dimension Scaling (Experiment 2)
Plots both datasets on the same graph for direct comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPU_FILE = os.path.join(SCRIPT_DIR, '..', 'cpu', 'experiment2_bond_scaling_cpu_updated.jsonl')
GPU_FILE = os.path.join(SCRIPT_DIR, '..', 'gpu', 'experiment2_bond_scaling_gpu_updated.jsonl')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'experiment2_cpu_vs_gpu_comparison.png')

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

def quadratic_model(x, a, b):
    """Quadratic scaling model: a * x^2 + b"""
    return a * x**2 + b

def extract_scaling_data(data):
    """Extract bond dimensions and mean runtimes from data"""
    bond_data = {}
    for entry in data:
        bond_dim = entry['bond_dimension']
        runtime_ms = entry['run_time_ms']
        if bond_dim not in bond_data:
            bond_data[bond_dim] = []
        bond_data[bond_dim].append(runtime_ms)
    
    bond_dims = sorted(bond_data.keys())
    mean_runtimes_sec = np.array([np.mean(bond_data[b]) for b in bond_dims]) / 1000.0
    return np.array(bond_dims), mean_runtimes_sec

def main():
    print("Loading CPU data...")
    cpu_data = load_data(CPU_FILE)
    print(f"  Loaded {len(cpu_data)} CPU data points")
    
    print("Loading GPU data...")
    gpu_data = load_data(GPU_FILE)
    print(f"  Loaded {len(gpu_data)} GPU data points")
    
    # Extract scaling data
    cpu_bond_dims, cpu_runtimes = extract_scaling_data(cpu_data)
    gpu_bond_dims, gpu_runtimes = extract_scaling_data(gpu_data)
    
    # Fit quadratic models
    popt_cpu, _ = curve_fit(quadratic_model, cpu_bond_dims, cpu_runtimes)
    popt_gpu, _ = curve_fit(quadratic_model, gpu_bond_dims, gpu_runtimes)
    
    r2_cpu = 1 - np.var(cpu_runtimes - quadratic_model(cpu_bond_dims, *popt_cpu)) / np.var(cpu_runtimes)
    r2_gpu = 1 - np.var(gpu_runtimes - quadratic_model(gpu_bond_dims, *popt_gpu)) / np.var(gpu_runtimes)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot CPU data
    ax.plot(cpu_bond_dims, cpu_runtimes, linestyle='None', marker='o', markersize=8,
            label='CPU (measured)', color='#2E86AB', alpha=0.7)
    cpu_fit_x = np.linspace(min(cpu_bond_dims), max(cpu_bond_dims), 200)
    ax.plot(cpu_fit_x, quadratic_model(cpu_fit_x, *popt_cpu), '-', linewidth=2,
            label=f'CPU fit: {popt_cpu[0]:.2e}·χ² (R²={r2_cpu:.4f})', 
            color='#2E86AB', alpha=0.9)
    
    # Plot GPU data
    ax.plot(gpu_bond_dims, gpu_runtimes, linestyle='None', marker='s', markersize=8,
            label='GPU (measured)', color='#F18F01', alpha=0.7)
    gpu_fit_x = np.linspace(min(gpu_bond_dims), max(gpu_bond_dims), 200)
    ax.plot(gpu_fit_x, quadratic_model(gpu_fit_x, *popt_gpu), '--', linewidth=2,
            label=f'GPU fit: {popt_gpu[0]:.2e}·χ² (R²={r2_gpu:.4f})', 
            color='#F18F01', alpha=0.9)
    
    ax.set_xlabel('Bond Dimension (χ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 2: CPU vs GPU Bond Dimension Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{OUTPUT_FILE}'")
    
    # ===== SECOND PLOT: Bond dimension <= 256 only =====
    max_bond_filter = 256
    output_file_256 = os.path.join(SCRIPT_DIR, 'experiment2_cpu_vs_gpu_comparison_256.png')
    
    # Filter data
    cpu_mask = cpu_bond_dims <= max_bond_filter
    gpu_mask = gpu_bond_dims <= max_bond_filter
    cpu_bond_256 = cpu_bond_dims[cpu_mask]
    cpu_runtime_256 = cpu_runtimes[cpu_mask]
    gpu_bond_256 = gpu_bond_dims[gpu_mask]
    gpu_runtime_256 = gpu_runtimes[gpu_mask]
    
    # Fit quadratic models to filtered data
    popt_cpu_256, _ = curve_fit(quadratic_model, cpu_bond_256, cpu_runtime_256)
    popt_gpu_256, _ = curve_fit(quadratic_model, gpu_bond_256, gpu_runtime_256)
    r2_cpu_256 = 1 - np.var(cpu_runtime_256 - quadratic_model(cpu_bond_256, *popt_cpu_256)) / np.var(cpu_runtime_256)
    r2_gpu_256 = 1 - np.var(gpu_runtime_256 - quadratic_model(gpu_bond_256, *popt_gpu_256)) / np.var(gpu_runtime_256)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot CPU data
    ax2.plot(cpu_bond_256, cpu_runtime_256, linestyle='None', marker='o', markersize=8,
            label='CPU (measured)', color='#2E86AB', alpha=0.7)
    cpu_fit_x_256 = np.linspace(min(cpu_bond_256), max(cpu_bond_256), 200)
    ax2.plot(cpu_fit_x_256, quadratic_model(cpu_fit_x_256, *popt_cpu_256), '-', linewidth=2,
            label=f'CPU fit: {popt_cpu_256[0]:.2e}·χ² (R²={r2_cpu_256:.4f})', 
            color='#2E86AB', alpha=0.9)
    
    # Plot GPU data
    ax2.plot(gpu_bond_256, gpu_runtime_256, linestyle='None', marker='s', markersize=8,
            label='GPU (measured)', color='#F18F01', alpha=0.7)
    gpu_fit_x_256 = np.linspace(min(gpu_bond_256), max(gpu_bond_256), 200)
    ax2.plot(gpu_fit_x_256, quadratic_model(gpu_fit_x_256, *popt_gpu_256), '--', linewidth=2,
            label=f'GPU fit: {popt_gpu_256[0]:.2e}·χ² (R²={r2_gpu_256:.4f})', 
            color='#F18F01', alpha=0.9)
    
    ax2.set_xlabel('Bond Dimension (χ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Experiment 2: CPU vs GPU Bond Dimension Scaling (χ ≤ {max_bond_filter})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file_256, dpi=300, bbox_inches='tight')
    print(f"Plot (χ ≤ 256) saved as '{output_file_256}'")
    
    # Print comparison summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"CPU Quadratic Coefficient: {popt_cpu[0]:.6e}")
    print(f"GPU Quadratic Coefficient: {popt_gpu[0]:.6e}")
    print(f"GPU is {popt_cpu[0]/popt_gpu[0]:.2f}x faster than CPU (scaling coefficient ratio)")
    
    # Compare at specific bond dimensions
    print("\nRuntime Comparison at Key Bond Dimensions:")
    common_bonds = sorted(set(cpu_bond_dims) & set(gpu_bond_dims))
    for bond in [64, 256, 512, 672]:
        if bond in common_bonds:
            cpu_idx = np.where(cpu_bond_dims == bond)[0][0]
            gpu_idx = np.where(gpu_bond_dims == bond)[0][0]
            speedup = cpu_runtimes[cpu_idx] / gpu_runtimes[gpu_idx]
            print(f"  χ={bond:3d}: CPU={cpu_runtimes[cpu_idx]:8.2f}s, GPU={gpu_runtimes[gpu_idx]:8.2f}s, Speedup={speedup:.2f}x")
    
    plt.show()

if __name__ == '__main__':
    main()
