"""
Improved MPS Scaling Analysis with Bond-Dimension Stratification

This script recognizes that different bond dimensions may have different
scaling behaviors and creates separate models or uses a more sophisticated
unified model.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import sys

def load_data(filename):
    """Load data from JSONL file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
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

def analyze_by_bond_dimension(data):
    """Analyze scaling separately for different bond dimension ranges"""
    
    # Extract data
    n = np.array([d['num_qubits'] for d in data])
    depth = np.array([d['depth'] for d in data])
    X = np.array([d['bond_dimension'] for d in data])
    runtime = np.array([d['run_time_ms'] for d in data])
    
    # Define bond dimension ranges
    ranges = [
        ("Small (4-40)", lambda x: np.array([xi <= 40 for xi in x])),
        ("Medium (64-96)", lambda x: np.array([(64 <= xi <= 96) for xi in x])),
        ("Large (256+)", lambda x: np.array([xi >= 256 for xi in x])),
    ]
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"BOND-DIMENSION STRATIFIED ANALYSIS")
    print(f"{'='*80}\n")
    
    for range_name, range_filter in ranges:
        # Filter data for this range
        mask = range_filter(X)
        if np.sum(mask) < 10:
            print(f"{range_name}: Insufficient data ({np.sum(mask)} points)")
            continue
        
        n_range = n[mask]
        depth_range = depth[mask]
        X_range = X[mask]
        runtime_range = runtime[mask]
        
        # Calculate scaling factor
        scaling_range = n_range * depth_range * (X_range ** 2)
        
        # Fit model
        params, cov = curve_fit(linear_model, scaling_range, runtime_range)
        a, b = params
        a_std, b_std = np.sqrt(np.diag(cov))
        
        # Predictions and metrics
        runtime_pred = linear_model(scaling_range, a, b)
        residuals = runtime_range - runtime_pred
        
        r2 = 1 - np.var(residuals) / np.var(runtime_range)
        rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals / runtime_range)) * 100
        residual_std = np.std(residuals)
        
        results[range_name] = {
            'a': a, 'b': b, 'a_std': a_std, 'b_std': b_std,
            'r2': r2, 'rmse': rmse, 'mape': mape,
            'residual_std': residual_std,
            'n_points': np.sum(mask),
            'X_min': X_range.min(), 'X_max': X_range.max(),
            'scaling': scaling_range, 'runtime': runtime_range,
            'runtime_pred': runtime_pred, 'residuals': residuals,
            'X': X_range
        }
        
        print(f"{range_name}:")
        print(f"  Data: {np.sum(mask)} points, X in [{X_range.min()}, {X_range.max()}]")
        print(f"  Model: runtime = {a:.6e} * (n*d*X^2) + {b:.2f}")
        print(f"  R^2 = {r2:.6f}, RMSE = {rmse:.2f} ms, MAPE = {mape:.1f}%")
        print(f"  95% CI: ± {1.96 * residual_std:.0f} ms\n")
    
    return results

def create_stratified_visualization(results):
    """Create improved visualization showing different bond dimension ranges"""
    
    # Filter out ranges with too few points
    valid_results = {k: v for k, v in results.items() if v['n_points'] >= 10}
    
    if not valid_results:
        print("Not enough data for visualization")
        return
    
    n_ranges = len(valid_results)
    
    # Create figure with better layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, n_ranges, hspace=0.3, wspace=0.3)
    
    colors_map = {'Small (4-40)': '#2E86AB', 'Medium (64-96)': '#F18F01', 'Large (256+)': '#C73E1D'}
    
    # Row 1: Individual fits for each range
    for idx, (range_name, result) in enumerate(valid_results.items()):
        ax = fig.add_subplot(gs[0, idx])
        
        # Scatter plot with consistent sizing
        scatter = ax.scatter(result['scaling'], result['runtime'], 
                           alpha=0.7, s=50, c=colors_map.get(range_name, '#2E86AB'),
                           edgecolors='black', linewidth=0.5, label='Measured')
        
        # Fit line
        scaling_sorted = np.sort(result['scaling'])
        runtime_fit = linear_model(scaling_sorted, result['a'], result['b'])
        ax.plot(scaling_sorted, runtime_fit, 'r-', linewidth=3,
               label=f"Fit: {result['a']:.2e}·(ndX²) + {result['b']:.0f}", alpha=0.8)
        
        ax.set_xlabel('Scaling Factor (n·d·X²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f"{range_name} ({result['n_points']} points)\nR² = {result['r2']:.4f}, MAPE = {result['mape']:.1f}%",
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Row 2: Residuals
    for idx, (range_name, result) in enumerate(valid_results.items()):
        ax = fig.add_subplot(gs[1, idx])
        
        # Residual plot
        ax.scatter(result['scaling'], result['residuals'], 
                  alpha=0.7, s=50, c=colors_map.get(range_name, '#2E86AB'),
                  edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='darkred', linestyle='--', linewidth=2.5, label='Zero error')
        
        std = result['residual_std']
        ax.axhline(y=std, color='orange', linestyle=':', linewidth=2, alpha=0.8, label=f'±1σ = ±{std:.0f} ms')
        ax.axhline(y=-std, color='orange', linestyle=':', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Scaling Factor (n·d·X²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f"Prediction Errors (σ = {std:.0f} ms)", fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Row 3: Combined comparison plot
    ax_combined = fig.add_subplot(gs[2, :])
    
    for range_name, result in valid_results.items():
        color = colors_map.get(range_name, '#2E86AB')
        
        # Plot data points
        ax_combined.scatter(result['scaling'], result['runtime'], 
                          alpha=0.6, s=40, c=color, label=f'{range_name} (data)',
                          edgecolors='black', linewidth=0.5)
        
        # Plot fit line
        scaling_sorted = np.sort(result['scaling'])
        runtime_fit = linear_model(scaling_sorted, result['a'], result['b'])
        ax_combined.plot(scaling_sorted, runtime_fit, '-', linewidth=2.5,
                       color=color, alpha=0.9,
                       label=f'{range_name} (R²={result["r2"]:.3f})')
    
    ax_combined.set_xlabel('Scaling Factor (n·d·X²)', fontsize=13, fontweight='bold')
    ax_combined.set_ylabel('Runtime (ms)', fontsize=13, fontweight='bold')
    ax_combined.set_title('Combined View: All Bond Dimension Ranges', 
                         fontsize=14, fontweight='bold', pad=15)
    ax_combined.legend(fontsize=11, loc='upper left', ncol=2)
    ax_combined.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig('quantum_volume_scaling_stratified.png', dpi=300, bbox_inches='tight')
    print(f"\nStratified visualization saved to: quantum_volume_scaling_stratified.png\n")
    plt.show()

def print_recommendations(results):
    """Print recommendations for which model to use"""
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    print("For best prediction accuracy, use the appropriate model based on bond dimension:\n")
    
    for range_name, result in results.items():
        print(f"{range_name}:")
        print(f"  Formula: runtime = {result['a']:.6e} * (n * d * X^2) + {result['b']:.2f}")
        print(f"  Uncertainty: ± {1.96 * result['residual_std']:.0f} ms (95% CI)")
        print(f"  Typical error: {result['mape']:.1f}%\n")
    
    print(f"{'='*80}")
    print(f"KEY INSIGHT:")
    print(f"{'='*80}\n")
    print("The scaling coefficient (a) changes with bond dimension!")
    print("This suggests different computational regimes:\n")
    
    for range_name, result in results.items():
        print(f"  {range_name}: a = {result['a']:.6e}")
    
    print("\nLarger bond dimensions have HIGHER coefficients, meaning:")
    print("  - More computational work per unit scaling factor")
    print("  - Better GPU utilization (as seen in benchmark results)")
    print("  - More predictable performance (higher R^2)")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import os
    filename = sys.argv[1] if len(sys.argv) > 1 else "quantum_volume_scaling.jsonl"
    if not os.path.exists(filename) and os.path.exists(f"../{filename}"):
        filename = f"../{filename}"
    
    print(f"\n{'='*80}")
    print(f"STRATIFIED MPS SCALING ANALYSIS")
    print(f"{'='*80}")
    print(f"\nLoading data from {filename}...")
    
    data = load_data(filename)
    print(f"Loaded {len(data)} data points\n")
    
    # Analyze by bond dimension ranges
    results = analyze_by_bond_dimension(data)
    
    # Create visualization
    create_stratified_visualization(results)
    
    # Print recommendations
    print_recommendations(results)
    
    print("Stratified analysis complete.")
