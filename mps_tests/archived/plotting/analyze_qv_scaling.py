"""
Quantum Volume MPS Scaling Analysis - O(nd X²)

Analyzes MPS runtime scaling and provides high-confidence runtime predictions.

Usage: python analyze_qv_scaling.py [data_file.jsonl]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import sys

# ============================================================================
# Utility Functions
# ============================================================================

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

def r2_score(y_true, y_pred):
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mape(y_true, y_pred):
    """Calculate mean absolute percentage error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_scaling(data):
    """Perform complete O(nd X²) scaling analysis"""
    
    # Extract data
    n = np.array([d['num_qubits'] for d in data])
    depth = np.array([d['depth'] for d in data])
    X = np.array([d['bond_dimension'] for d in data])
    runtime = np.array([d['run_time_ms'] for d in data])
    
    # Calculate scaling factor: nd X²
    scaling = n * depth * (X ** 2)
    
    # Fit model
    params, cov = curve_fit(linear_model, scaling, runtime)
    a, b = params
    a_std, b_std = np.sqrt(np.diag(cov))
    
    # Predictions and metrics
    runtime_pred = linear_model(scaling, a, b)
    residuals = runtime - runtime_pred
    
    r2 = r2_score(runtime, runtime_pred)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    mape_val = mape(runtime, runtime_pred)
    
    # Calculate prediction intervals (95% confidence)
    residual_std = np.std(residuals)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"MPS SCALING ANALYSIS: O(nd X²)")
    print(f"{'='*80}")
    print(f"\nModel: runtime = a·(n·d·X²) + b")
    print(f"\nFitted Parameters:")
    print(f"  a = {a:.6e} ± {a_std:.6e} ms/unit")
    print(f"  b = {b:.2f} ± {b_std:.2f} ms")
    print(f"\nModel Performance:")
    print(f"  R² = {r2:.6f}")
    print(f"  RMSE = {rmse:.2f} ms")
    print(f"  MAE = {mae:.2f} ms")
    print(f"  MAPE = {mape_val:.2f}%")
    print(f"\nPrediction Uncertainty:")
    print(f"  Residual Std Dev = {residual_std:.2f} ms")
    print(f"  95% Confidence Interval = ± {1.96 * residual_std:.2f} ms")
    print(f"\nData Summary:")
    print(f"  Points: {len(data)}")
    print(f"  Qubits: {n.min()}-{n.max()}")
    print(f"  Depth: {depth.min()}-{depth.max()}")
    print(f"  Bond dim: {X.min()}-{X.max()}")
    print(f"  Runtime: {runtime.min():.0f}-{runtime.max():.0f} ms")
    print(f"  Scaling: {scaling.min():.0f}-{scaling.max():.0f}")
    print(f"{'='*80}\n")
    
    return {
        'n': n, 'depth': depth, 'X': X, 'runtime': runtime,
        'scaling': scaling, 'runtime_pred': runtime_pred,
        'residuals': residuals, 'a': a, 'b': b, 'r2': r2,
        'rmse': rmse, 'mae': mae, 'mape': mape_val,
        'residual_std': residual_std
    }

def cross_validate(data, n_splits=5):
    """5-fold cross-validation"""
    
    n = np.array([d['num_qubits'] for d in data])
    depth = np.array([d['depth'] for d in data])
    X = np.array([d['bond_dimension'] for d in data])
    runtime = np.array([d['run_time_ms'] for d in data])
    scaling = n * depth * (X ** 2)
    
    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    fold_size = len(data) // n_splits
    
    print(f"{'='*80}")
    print(f"CROSS-VALIDATION ({n_splits}-Fold)")
    print(f"{'='*80}")
    
    r2_scores = []
    for i in range(n_splits):
        # Split data
        test_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        
        # Fit and predict
        params, _ = curve_fit(linear_model, scaling[train_idx], runtime[train_idx])
        pred = linear_model(scaling[test_idx], *params)
        
        # Metrics
        r2 = r2_score(runtime[test_idx], pred)
        rmse = np.sqrt(np.mean((runtime[test_idx] - pred) ** 2))
        mape_val = mape(runtime[test_idx], pred)
        
        r2_scores.append(r2)
        print(f"Fold {i+1}: R² = {r2:.6f}, RMSE = {rmse:.2f} ms, MAPE = {mape_val:.2f}%")
    
    print(f"{'-'*80}")
    print(f"Mean R²: {np.mean(r2_scores):.6f} ± {np.std(r2_scores):.6f}")
    print(f"{'='*80}\n")

def predict_runtime(n, d, X, a, b, residual_std, confidence=0.95):
    """
    Predict runtime with confidence interval
    
    Args:
        n: number of qubits
        d: circuit depth
        X: bond dimension
        a, b: model parameters
        residual_std: standard deviation of residuals
        confidence: confidence level (default 0.95 for 95%)
    
    Returns:
        dict with prediction, lower_bound, upper_bound
    """
    scaling = n * d * (X ** 2)
    prediction = a * scaling + b
    
    # Calculate confidence interval
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * residual_std
    
    return {
        'prediction': prediction,
        'lower_bound': max(0, prediction - margin),
        'upper_bound': prediction + margin,
        'confidence': confidence
    }

def print_predictions(results):
    """Print example predictions with confidence intervals"""
    
    a, b = results['a'], results['b']
    residual_std = results['residual_std']
    
    print(f"{'='*80}")
    print(f"RUNTIME PREDICTION CALCULATOR")
    print(f"{'='*80}")
    print(f"\nFormula: runtime(n, d, X) = {a:.6e} × (n × d × X²) + {b:.2f} ms")
    print(f"\n95% Confidence Interval: ± {1.96 * residual_std:.0f} ms")
    print(f"\nExample Predictions:")
    print(f"\n{'n':<4} {'d':<4} {'X':<4} {'Prediction':<15} {'95% CI Range':<30} {'Time (sec)':<12}")
    print(f"{'-'*85}")
    
    examples = [
        (20, 16, 32),
        (30, 24, 48),
        (40, 32, 64),
        (48, 40, 80),
        (56, 40, 80),
        (64, 48, 96),
    ]
    
    for n_ex, d_ex, X_ex in examples:
        result = predict_runtime(n_ex, d_ex, X_ex, a, b, residual_std)
        pred = result['prediction']
        lower = result['lower_bound']
        upper = result['upper_bound']
        
        print(f"{n_ex:<4} {d_ex:<4} {X_ex:<4} {pred:>10,.0f} ms    "
              f"[{lower:>8,.0f} - {upper:>8,.0f}] ms    {pred/1000:>6.1f} sec")
    
    print(f"\n{'='*80}")
    print(f"HOW TO USE THIS FOR YOUR PREDICTIONS:")
    print(f"{'='*80}")
    print(f"\n1. Calculate scaling factor: S = n × d × X²")
    print(f"2. Predict runtime: T = {a:.6e} × S + {b:.2f}")
    print(f"3. Add uncertainty: T ± {1.96 * residual_std:.0f} ms (95% confidence)")
    print(f"\nInterpretation:")
    print(f"   - The prediction is the MOST LIKELY runtime")
    print(f"   - 95% of actual runtimes fall within the confidence interval")
    print(f"   - For conservative estimates, use the UPPER BOUND")
    print(f"   - MAPE = {results['mape']:.1f}% means typical error is ~{results['mape']:.0f}%")
    print(f"{'='*80}\n")

def create_visualization(results):
    """Create comprehensive visualization with explanations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 1. Main fit
    ax = axes[0, 0]
    ax.scatter(results['scaling'], results['runtime'], alpha=0.5, s=20, label='Measured', color='steelblue')
    ax.plot(results['scaling'], results['runtime_pred'], 'r-', linewidth=2.5,
            label=f"Fit: {results['a']:.2e}·(ndX²) + {results['b']:.0f}")
    ax.set_xlabel('Scaling Factor (n·d·X²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f"① Main Scaling Relationship: O(ndX²)\nR² = {results['r2']:.4f} (higher is better)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add explanation text
    ax.text(0.98, 0.05, 
            'This shows the linear relationship\nbetween scaling factor and runtime.\n'
            'Points close to the line = good fit.',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals
    ax = axes[0, 1]
    scatter = ax.scatter(results['scaling'], results['residuals'], alpha=0.5, s=20,
                        c=results['X'], cmap='viridis')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Add ±1 std dev bands
    std = results['residual_std']
    ax.axhline(y=std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1σ = ±{std:.0f} ms')
    ax.axhline(y=-std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=2*std, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'±2σ = ±{2*std:.0f} ms')
    ax.axhline(y=-2*std, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Scaling Factor (n·d·X²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted) ms', fontsize=12, fontweight='bold')
    ax.set_title(f'② Residual Analysis: Prediction Errors\nStd Dev = {std:.0f} ms',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Bond Dimension')
    
    # Add explanation
    ax.text(0.02, 0.95, 
            'Shows prediction errors.\n'
            'Random scatter around 0 = good model.\n'
            'Patterns = systematic bias.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 3. Predicted vs Actual
    ax = axes[1, 0]
    scatter = ax.scatter(results['runtime'], results['runtime_pred'], alpha=0.5, s=20,
                        c=results['X'], cmap='plasma')
    min_val = min(results['runtime'].min(), results['runtime_pred'].min())
    max_val = max(results['runtime'].max(), results['runtime_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5,
            label='Perfect prediction', zorder=10)
    ax.set_xlabel('Actual Runtime (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Runtime (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'③ Prediction Accuracy Check\nMAPE = {results["mape"]:.1f}%',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Bond Dimension')
    
    # Add explanation
    ax.text(0.02, 0.98, 
            'Points on diagonal = perfect predictions.\n'
            f'MAPE = {results["mape"]:.1f}% = typical error.',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 4. Error distribution
    ax = axes[1, 1]
    rel_errors = (results['residuals'] / results['runtime']) * 100
    n_bins = 50
    counts, bins, patches = ax.hist(rel_errors, bins=n_bins, alpha=0.7, edgecolor='black', color='steelblue')
    
    # Overlay normal distribution
    mu, sigma = np.mean(rel_errors), np.std(rel_errors)
    x = np.linspace(rel_errors.min(), rel_errors.max(), 100)
    ax2 = ax.twinx()
    ax2.plot(x, stats.norm.pdf(x, mu, sigma) * len(rel_errors) * (bins[1] - bins[0]),
             'r-', linewidth=2.5, label=f'Normal(μ={mu:.1f}%, σ={sigma:.1f}%)')
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.legend(fontsize=9, loc='upper right')
    
    ax.axvline(x=0, color='darkred', linestyle='--', linewidth=2.5, label='Zero error')
    ax.axvline(x=mu, color='orange', linestyle=':', linewidth=2, label=f'Mean = {mu:.1f}%')
    ax.set_xlabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'④ Error Distribution for Confidence Intervals\nMean = {mu:.1f}%, Std = {sigma:.1f}%',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add explanation
    ax.text(0.98, 0.05, 
            f'Bell curve = errors are normally distributed.\n'
            f'95% of errors within ±{1.96*sigma:.0f}%.\n'
            f'Use this for confidence intervals!',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    output = 'quantum_volume_scaling_analysis.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output}\n")
    
    plt.show()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import os
    # Default to parent directory if file not found in current directory
    filename = sys.argv[1] if len(sys.argv) > 1 else "quantum_volume_scaling.jsonl"
    if not os.path.exists(filename) and os.path.exists(f"../{filename}"):
        filename = f"../{filename}"
    
    print(f"\n{'='*80}")
    print(f"QUANTUM VOLUME MPS RUNTIME PREDICTION SYSTEM")
    print(f"{'='*80}")
    print(f"\nLoading data from {filename}...")
    data = load_data(filename)
    
    if not data:
        print("Error: No valid data found")
        sys.exit(1)
    
    print(f"Loaded {len(data)} data points\n")
    
    # Main analysis
    results = analyze_scaling(data)
    
    # Cross-validation
    cross_validate(data)
    
    # Predictions with confidence intervals
    print_predictions(results)
    
    # Visualization
    create_visualization(results)
    
    print("Analysis complete.")
    print("\nTIP: Use the prediction formula above to estimate runtime for any (n, d, X) configuration!")
