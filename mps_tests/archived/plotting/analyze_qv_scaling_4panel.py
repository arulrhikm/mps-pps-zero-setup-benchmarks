#!/usr/bin/env python3
"""
Stratified MPS Scaling Analysis - 4-Panel Visualization

Creates a 4-panel plot supporting the hypothesis that MPS runtime scales as O(n·d·χ²):
1. Combined view with all regimes
2. Small bond dimension regime (χ ≤ 40)
3. Medium bond dimension regime (64 ≤ χ ≤ 96)
4. Large bond dimension regime (χ ≥ 256)

This demonstrates how the scaling coefficient varies by regime, enabling
more accurate runtime predictions across different parameter ranges.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(filename):
    """Load data from JSONL file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line.startswith('{'):
                continue
            try:
                run_data = json.loads(line)
                if 'error' not in run_data:
                    data.append(run_data)
            except json.JSONDecodeError:
                continue
    return data

def linear_model(X, a, b):
    """Model: runtime = a * X + b"""
    return a * X + b

def analyze_stratified(data):
    """Analyze scaling separately for different bond dimension ranges"""
    
    # Extract data
    n = np.array([d['num_qubits'] for d in data])
    depth = np.array([d['depth'] for d in data])
    chi = np.array([d['bond_dimension'] for d in data])
    runtime = np.array([d['run_time_ms'] for d in data])
    
    # Define bond dimension ranges
    ranges = [
        ("Small (χ ≤ 40)", lambda x: x <= 40),
        ("Medium (64 ≤ χ ≤ 96)", lambda x: (x >= 64) & (x <= 96)),
        ("Large (χ ≥ 256)", lambda x: x >= 256),
    ]
    
    results = {}
    
    for range_name, range_condition in ranges:
        mask = range_condition(chi)
        if np.sum(mask) < 10:
            continue
        
        n_range = n[mask]
        depth_range = depth[mask]
        chi_range = chi[mask]
        runtime_range = runtime[mask]
        
        # Calculate scaling factor: n * d * χ²
        scaling = n_range * depth_range * (chi_range ** 2)
        
        # Fit linear model
        try:
            popt, pcov = curve_fit(linear_model, scaling, runtime_range)
            a, b = popt
            a_std = np.sqrt(pcov[0, 0])
        except:
            continue
        
        # Calculate metrics
        runtime_pred = linear_model(scaling, a, b)
        residuals = runtime_range - runtime_pred
        r2 = 1 - np.var(residuals) / np.var(runtime_range)
        rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals / runtime_range)) * 100
        
        results[range_name] = {
            'a': a, 'b': b, 'a_std': a_std,
            'r2': r2, 'rmse': rmse, 'mape': mape,
            'n_points': np.sum(mask),
            'chi_min': chi_range.min(), 'chi_max': chi_range.max(),
            'scaling': scaling, 'runtime': runtime_range,
            'runtime_pred': runtime_pred, 'residuals': residuals,
        }
    
    return results

def create_individual_plots(results, output_dir):
    """Create 4 separate PNG files: Combined + 3 individual regimes"""
    
    colors = {
        'Small (χ ≤ 40)': '#2E86AB',
        'Medium (64 ≤ χ ≤ 96)': '#F18F01', 
        'Large (χ ≥ 256)': '#C73E1D',
    }
    
    saved_files = []
    
    # ===== Plot 1: Combined View =====
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    for range_name, result in results.items():
        color = colors.get(range_name, '#888888')
        
        # Plot data points
        ax1.scatter(result['scaling'], result['runtime'], 
                   alpha=0.5, s=30, c=color, label=f'{range_name}',
                   edgecolors='black', linewidth=0.3)
        
        # Plot fit line
        scaling_sorted = np.sort(result['scaling'])
        runtime_fit = linear_model(scaling_sorted, result['a'], result['b'])
        ax1.plot(scaling_sorted, runtime_fit, '-', linewidth=2.5, color=color, alpha=0.9)
    
    ax1.set_xlabel('Scaling Factor (n·d·χ²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('MPS Runtime Scaling: Combined View (All Bond Dimension Regimes)\nModel: runtime = a·(n·d·χ²) + b', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with model summary
    summary_text = ""
    for range_name, result in results.items():
        summary_text += f"{range_name}: a={result['a']:.2e}, R²={result['r2']:.3f}\n"
    ax1.text(0.98, 0.02, summary_text.strip(), transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    file1 = os.path.join(output_dir, 'scaling_combined_view.png')
    plt.savefig(file1, dpi=300, bbox_inches='tight')
    saved_files.append(file1)
    plt.close(fig1)
    
    # ===== Plots 2-4: Individual Regimes =====
    for range_name, result in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        color = colors.get(range_name, '#888888')
        
        # Scatter plot
        ax.scatter(result['scaling'], result['runtime'], 
                  alpha=0.6, s=50, c=color, label='Measured data',
                  edgecolors='black', linewidth=0.5)
        
        # Fit line
        scaling_sorted = np.sort(result['scaling'])
        runtime_fit = linear_model(scaling_sorted, result['a'], result['b'])
        ax.plot(scaling_sorted, runtime_fit, '-', linewidth=3, color='darkred',
               label=f'Fit: {result["a"]:.2e}·(n·d·χ²) + {result["b"]:.0f}', alpha=0.9)
        
        # Add prediction band (±1σ residual)
        residual_std = np.std(result['residuals'])
        ax.fill_between(scaling_sorted, runtime_fit - residual_std, runtime_fit + residual_std,
                       color=color, alpha=0.2, label=f'±1σ prediction band ({residual_std:.0f} ms)')
        
        ax.set_xlabel('Scaling Factor (n·d·χ²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'MPS Runtime Scaling: {range_name}\n'
                    f'{result["n_points"]} data points | R² = {result["r2"]:.4f} | MAPE = {result["mape"]:.1f}%',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Generate filename from range name
        safe_name = range_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('≤', 'le').replace('≥', 'ge').replace('χ', 'chi')
        filename = os.path.join(output_dir, f'scaling_{safe_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
    
    print("\nSaved plots:")
    for f in saved_files:
        print(f"   {f}")
    
    return saved_files

def print_prediction_guide(results):
    """Print guidance on using the model for runtime predictions"""
    
    print("\n" + "="*70)
    print("RUNTIME PREDICTION GUIDE")
    print("="*70)
    print("\nYour hypothesis: Runtime ~ n*d*chi^2")
    print("\nThis analysis confirms that MPS runtime follows:")
    print("  runtime(ms) = a * (n * d * chi^2) + b")
    print("\nThe coefficient 'a' varies by bond dimension regime:\n")
    
    for range_name, result in results.items():
        print(f"  {range_name}:")
        print(f"    a = {result['a']:.6e} ms/(qubit*depth*chi^2)")
        print(f"    b = {result['b']:.2f} ms (baseline overhead)")
        print(f"    R² = {result['r2']:.4f} (fit quality)")
        print(f"    Prediction accuracy: ±{result['mape']:.1f}% MAPE")
        print()
    
    print("="*70)
    print("PREDICTION EXAMPLES")
    print("="*70)
    
    # Generate example predictions
    examples = [
        (40, 16, 32, "Small"),
        (60, 16, 64, "Medium"),
        (80, 16, 256, "Large"),
    ]
    
    print("\nUsing the appropriate regime model:")
    for n, d, chi, regime in examples:
        regime_key = [k for k in results.keys() if regime in k][0] if any(regime in k for k in results.keys()) else list(results.keys())[0]
        a, b = results[regime_key]['a'], results[regime_key]['b']
        scaling = n * d * chi**2
        pred_runtime = a * scaling + b
        print(f"  n={n}, d={d}, chi={chi}: predicted runtime = {pred_runtime/1000:.2f} seconds")
    
    print("\n" + "="*70 + "\n")

def main():
    # Find data file (prefer archived/data layout)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data", "quantum_volume_scaling.jsonl"),
        os.path.join(script_dir, "..", "quantum_volume_scaling.jsonl"),
    ]
    data_file = next((p for p in candidates if os.path.exists(p)), None)
    if not data_file:
        print("Error: Could not find quantum_volume_scaling.jsonl under ../data/ or ../")
        sys.exit(1)
    
    print(f"Loading data from {data_file}...")
    data = load_data(data_file)
    print(f"Loaded {len(data)} data points")
    
    # Analyze by bond dimension regime
    results = analyze_stratified(data)
    
    if not results:
        print("Error: Not enough data for analysis")
        sys.exit(1)
    
    # Create 4 separate plot files
    create_individual_plots(results, script_dir)
    
    # Print prediction guide
    print_prediction_guide(results)
    plt.close("all")

if __name__ == '__main__':
    main()
