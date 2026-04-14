"""
Quantum Volume MPS Scaling Analysis V2

Analyzes MPS runtime scaling using the user-specified formula:
Scale = 1e-4 * (6 * n_su4 * n + n_gates + 2**n + shots) * X**2

Usage: python analyze_qv_scaling_v2.py [data_file.jsonl]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
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
            # Handle inline comments
            if '#' in line:
                line = line[:line.find('#')].strip()
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

def user_scaling_formula(d):
    """
    Calculate scaling factor based on user provided formula:
    1e-4 * (6 * num_su4s * n + num_gates + 2**n + shots) * X**2
    """
    n = d['num_qubits']
    num_su4s = d.get('num_su4s', 0)
    num_gates = d.get('num_gates', 0)
    shots = d.get('shots', 1000)
    X = d['bond_dimension']
    
    # 2**num_measurements: assumes all qubits measured -> 2**n
    prob_comp = 2**n
    
    term = (6 * num_su4s * n) + num_gates + prob_comp + shots
    
    return 1e-4 * term * (X**2)

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
    """Perform scaling analysis with new formula"""
    
    # Extract data and calculate scaling factor
    scaling_factors = []
    runtimes = []
    bonds = []
    qubits = []
    
    for d in data:
        s = user_scaling_formula(d)
        scaling_factors.append(s)
        runtimes.append(d['run_time_ms'])
        bonds.append(d['bond_dimension'])
        qubits.append(d['num_qubits'])
        
    scaling = np.array(scaling_factors)
    runtime = np.array(runtimes)
    X = np.array(bonds)
    
    # Fit model: T = a * Scale + b
    params, cov = curve_fit(linear_model, scaling, runtime)
    a, b = params
    a_std, b_std = np.sqrt(np.diag(cov))
    
    # Predictions and metrics
    runtime_pred = linear_model(scaling, a, b)
    residuals = runtime - runtime_pred
    
    r2 = r2_score(runtime, runtime_pred)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape_val = mape(runtime, runtime_pred)
    
    # Calculate prediction intervals (95% confidence)
    residual_std = np.std(residuals)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"MPS SCALING ANALYSIS V2: User Formula")
    print(f"{'='*80}")
    print(f"Formula: Scale = 1e-4 * (6*n*su4s + ng + 2^n + shots) * X^2")
    print(f"Model: Runtime = a * Scale + b")
    
    # Stats
    print(f"\nData Range:")
    print(f"  Qubits (n): {min(qubits)} - {max(qubits)}")
    print(f"  BondDim (X): {min(bonds)} - {max(bonds)}")
    print(f"  Scaling Factor: {min(scaling):.2e} - {max(scaling):.2e}")
    print(f"  Runtime: {min(runtime):.0f} - {max(runtime):.0f} ms")
    
    # Log-Log Fit
    log_x = np.log10(scaling)
    log_y = np.log10(runtime)
    params_log, _ = curve_fit(linear_model, log_x, log_y)
    slope_log, intercept_log = params_log
    r2_log = r2_score(log_y, linear_model(log_x, *params_log))
    
    print(f"\nFitted Parameters (Linear Fit):")
    print(f"  a = {a:.6e} ± {a_std:.6e}")
    print(f"  b = {b:.2f} ± {b_std:.2f} ms")
    print(f"\nModel Performance:")
    print(f"  Linear R² = {r2:.6f}")
    print(f"  Log-Log R² = {r2_log:.6f} (Slope: {slope_log:.2f})")
    print(f"  RMSE = {rmse:.2f} ms")
    print(f"  MAPE = {mape_val:.2f}%")
    print(f"{'='*80}\n")
    
    return {
        'scaling': scaling, 'runtime': runtime, 'X': X,
        'runtime_pred': runtime_pred, 'a': a, 'b': b, 'r2': r2,
        'mape': mape_val, 'r2_log': r2_log
    }

def create_visualization(results):
    """Create visualization (Linear and Log-Log)"""
    
    # 1. Linear Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(results['scaling'], results['runtime'], alpha=0.5, s=40, label='Measured Data', c=results['X'], cmap='viridis', edgecolors='none')
    plt.colorbar(label='Bond Dimension (X)')
    
    sort_idx = np.argsort(results['scaling'])
    plt.plot(results['scaling'][sort_idx], results['runtime_pred'][sort_idx], 'r-', linewidth=2,
             label=f"Fit: T = {results['a']:.2e} S + {results['b']:.0f}")
    
    plt.xlabel('Scaling Factor S', fontsize=11, fontweight='bold')
    plt.ylabel('Runtime (ms)', fontsize=11, fontweight='bold')
    plt.title(f"MPS scaling (Linear Scale)\nR² = {results['r2']:.4f}", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quantum_volume_scaling_v2_linear.png', dpi=300)
    print(f"Linear plot saved to: quantum_volume_scaling_v2_linear.png")
    
    # 2. Log-Log Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(results['scaling'], results['runtime'], alpha=0.5, s=40, label='Measured Data', c=results['X'], cmap='viridis', edgecolors='none')
    plt.colorbar(label='Bond Dimension (X)')
    
    # Log fit line
    x_range = np.linspace(min(results['scaling']), max(results['scaling']), 100)
    # Re-calculate log fit for line plotting
    log_x = np.log10(results['scaling'])
    log_y = np.log10(results['runtime'])
    p, _ = curve_fit(linear_model, log_x, log_y)
    m, c = p
    # y = 10^(mx + c) = 10^c * x^m
    y_fit = (10**c) * (x_range**m)
    
    plt.plot(x_range, y_fit, 'r--', linewidth=2, label=f"Power Law Fit: T $\propto$ S^{{{m:.2f}}}")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Scaling Factor S (Log Scale)', fontsize=11, fontweight='bold')
    plt.ylabel('Runtime (ms) (Log Scale)', fontsize=11, fontweight='bold')
    plt.title(f"MPS scaling (Log-Log Scale)\nLog-Log R² = {results['r2_log']:.4f}", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('quantum_volume_scaling_v2_log.png', dpi=300)
    print(f"Log-Log plot saved to: quantum_volume_scaling_v2_log.png\n")

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
    print(f"QUANTUM VOLUME MPS RUNTIME ANALYSIS V2")
    print(f"{'='*80}")
    print(f"\nLoading data from {filename}...")
    data = load_data(filename)
    
    if not data:
        print("Error: No valid data found")
        sys.exit(1)
    
    print(f"Loaded {len(data)} data points\n")
    
    # Main analysis
    results = analyze_scaling(data)
    
    # Visualization
    create_visualization(results)

