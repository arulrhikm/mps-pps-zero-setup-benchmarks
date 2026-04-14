"""
Simple Runtime Prediction Calculator for Quantum Volume Circuits

Based on O(nd X²) scaling model with 95% confidence intervals.

Usage: python predict_runtime.py <n_qubits> <depth> <bond_dimension>
"""

import sys
from scipy import stats

# Model parameters (from analysis)
A = 5.380081e-02  # ms per unit scaling
B = -801.96       # ms offset
RESIDUAL_STD = 22345  # ms standard deviation of residuals

def predict_runtime(n, d, X, confidence=0.95):
    """
    Predict MPS runtime for quantum volume circuit
    
    Args:
        n: number of qubits
        d: circuit depth
        X: bond dimension
        confidence: confidence level (default 0.95)
    
    Returns:
        dict with prediction and confidence interval
    """
    # Calculate scaling factor
    scaling = n * d * (X ** 2)
    
    # Predict runtime
    prediction = A * scaling + B
    
    # Calculate confidence interval
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * RESIDUAL_STD
    
    return {
        'n': n,
        'd': d,
        'X': X,
        'scaling': scaling,
        'prediction_ms': max(0, prediction),
        'prediction_sec': max(0, prediction) / 1000,
        'lower_bound_ms': max(0, prediction - margin),
        'upper_bound_ms': prediction + margin,
        'lower_bound_sec': max(0, prediction - margin) / 1000,
        'upper_bound_sec': (prediction + margin) / 1000,
        'confidence': confidence,
        'margin_ms': margin
    }

def print_prediction(result):
    """Pretty print prediction results"""
    print(f"\n{'='*70}")
    print(f"QUANTUM VOLUME CIRCUIT RUNTIME PREDICTION")
    print(f"{'='*70}")
    print(f"\nCircuit Parameters:")
    print(f"  Qubits (n):        {result['n']}")
    print(f"  Depth (d):         {result['d']}")
    print(f"  Bond Dimension (X): {result['X']}")
    print(f"  Scaling Factor:    {result['scaling']:,} (n×d×X²)")
    
    print(f"\nPredicted Runtime:")
    print(f"  Point Estimate:    {result['prediction_ms']:>10,.0f} ms  ({result['prediction_sec']:>8.1f} sec)")
    
    print(f"\n{int(result['confidence']*100)}% Confidence Interval:")
    print(f"  Lower Bound:       {result['lower_bound_ms']:>10,.0f} ms  ({result['lower_bound_sec']:>8.1f} sec)")
    print(f"  Upper Bound:       {result['upper_bound_ms']:>10,.0f} ms  ({result['upper_bound_sec']:>8.1f} sec)")
    print(f"  Margin of Error:   ± {result['margin_ms']:>8,.0f} ms")
    
    print(f"\n{'='*70}")
    print(f"INTERPRETATION:")
    print(f"{'='*70}")
    print(f"• Most likely runtime: {result['prediction_sec']:.1f} seconds")
    print(f"• {int(result['confidence']*100)}% confident actual runtime is between:")
    print(f"  {result['lower_bound_sec']:.1f} - {result['upper_bound_sec']:.1f} seconds")
    print(f"• For conservative planning, use: {result['upper_bound_sec']:.1f} seconds")
    print(f"• For optimistic planning, use: {result['lower_bound_sec']:.1f} seconds")
    print(f"{'='*70}\n")

def batch_predictions():
    """Show predictions for common configurations"""
    print("\n" + "="*70)
    print("COMMON CONFIGURATION PREDICTIONS")
    print("="*70)
    print(f"\n{'n':<4} {'d':<4} {'X':<4} {'Prediction':<15} {'95% CI Range':<35} {'Time':<12}")
    print("-"*70)
    
    configs = [
        (20, 16, 32),
        (24, 20, 40),
        (30, 24, 48),
        (32, 28, 56),
        (40, 32, 64),
        (48, 36, 72),
        (48, 40, 80),
        (56, 40, 80),
        (64, 48, 88),
        (64, 48, 96),
    ]
    
    for n, d, X in configs:
        r = predict_runtime(n, d, X)
        print(f"{n:<4} {d:<4} {X:<4} {r['prediction_ms']:>10,.0f} ms    "
              f"[{r['lower_bound_ms']:>8,.0f} - {r['upper_bound_ms']:>8,.0f}] ms    "
              f"{r['prediction_sec']:>6.1f} sec")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Single prediction from command line
        try:
            n = int(sys.argv[1])
            d = int(sys.argv[2])
            X = int(sys.argv[3])
            
            result = predict_runtime(n, d, X)
            print_prediction(result)
            
        except ValueError:
            print("Error: Arguments must be integers")
            print("Usage: python predict_runtime.py <n_qubits> <depth> <bond_dimension>")
            sys.exit(1)
    
    elif len(sys.argv) == 1:
        # Show batch predictions
        print("\n" + "="*70)
        print("QUANTUM VOLUME MPS RUNTIME PREDICTION CALCULATOR")
        print("="*70)
        print("\nModel: runtime = 5.38×10⁻² × (n×d×X²) - 802 ms")
        print("Uncertainty: ± 42,585 ms (95% confidence)")
        
        batch_predictions()
        
        print("\n💡 TIP: For a specific prediction, run:")
        print("   python predict_runtime.py <n_qubits> <depth> <bond_dimension>")
        print("\nExample:")
        print("   python predict_runtime.py 40 32 64\n")
    
    else:
        print("Usage: python predict_runtime.py <n_qubits> <depth> <bond_dimension>")
        print("   or: python predict_runtime.py  (for batch predictions)")
        sys.exit(1)
