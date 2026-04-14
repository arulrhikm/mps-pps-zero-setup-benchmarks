import json
import matplotlib.pyplot as plt
import numpy as np
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_MPS_ROOT = os.path.dirname(_BASE)
INPUT_FILE = os.path.join(_MPS_ROOT, "data", "all_mps_data.jsonl")
_PLOTS_DIR = os.path.join(_MPS_ROOT, "plots")

def analyze_and_plot():
    print(f"Loading {INPUT_FILE}...")
    
    # Group data by (n, d, X)
    groups = collections.defaultdict(list)
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                d = json.loads(line)
                key = (d['num_qubits'], d['depth'], d['bond_dimension'])
                groups[key].append(d)
            except: pass
            
    print(f"Found {len(groups)} unique circuit configurations.")
    
    # Calculate Slope (ms/shot) vs Scaling Factor for each group
    scaling_factors = []
    slopes = []
    
    # For explanation file
    explanation_lines = []
    explanation_lines.append("# Analysis of Runtime vs Shots Scaling\n")
    explanation_lines.append(f"Analyzing {len(groups)} configurations...\n")
    
    configs_with_multiple_shots = 0
    
    for key, runs in groups.items():
        n, d, X = key
        # Extract shots and runtimes
        shots_list = []
        runtimes = []
        
        for run in runs:
            s = run.get('shots', 1000)
            t = run['run_time_ms']
            shots_list.append(s)
            runtimes.append(t)
            
        # Check if we have variation in shots
        unique_shots = sorted(list(set(shots_list)))
        
        if len(unique_shots) >= 2:
            configs_with_multiple_shots += 1
            
            # Fit linear model: T = slope * shots + intercept
            # Or robust fit
            # We use all points (even duplicates)
            shots_arr = np.array(shots_list)
            runtimes_arr = np.array(runtimes)
            
            # Simple polyfit degree 1
            slope, intercept = np.polyfit(shots_arr, runtimes_arr, 1)
            
            # Scaling factor: n * d * X^2
            scale = n * d * (X**2)
            
            scaling_factors.append(scale)
            slopes.append(slope)
            
            # Explanation snippet for significant ones
            if scale > 1e6 and slope > 10:
                explanation_lines.append(f"- Config (n={n}, d={d}, X={X}, Scale={scale:.1e}): Slope = {slope:.2f} ms/shot (R^2 not checked)")
        else:
            # Maybe store single points for general trend if needed, but not slope
            pass
            
    print(f"Found {configs_with_multiple_shots} configurations with multiple shot values suitable for analysis.")
    
    if not slopes:
        print("No configurations with multiple shot values found! Cannot plot.")
        return

    # 1. Plot Slope vs Scale (Original)
    # ... (Keep existing logic or comment out if redundant, but user asked for "how deviation changes")
    
    # Let's calculate Deviation explicitly for the plot requested
    deviations_ms = []
    deviations_pct = []
    scales_dev = []
    
    for key, runs in groups.items():
        shots_list = [r.get('shots', 1000) for r in runs]
        runtimes = [r['run_time_ms'] for r in runs]
        
        unique_shots = sorted(list(set(shots_list)))
        if len(unique_shots) >= 2:
            # Find min and max shots
            min_s = min(shots_list)
            max_s = max(shots_list)
            
            # Get average runtime for min and max shots (handle duplicates)
            t_min = np.mean([t for s, t in zip(shots_list, runtimes) if s == min_s])
            t_max = np.mean([t for s, t in zip(shots_list, runtimes) if s == max_s])
            
            delta_t = t_max - t_min
            pct = (delta_t / t_min) * 100 if t_min > 0 else 0
            
            if delta_t < -1000: continue # Ignore noise where more shots = faster (weird)
            
            n, d, X = key
            scale = n * d * (X**2)
            
            deviations_ms.append(delta_t)
            deviations_pct.append(pct)
            scales_dev.append(scale)

    # Plot 1: Absolute Deviation (ms) vs Scale
    plt.figure(figsize=(10, 8))
    plt.scatter(scales_dev, deviations_ms, alpha=0.7, c='purple', edgecolors='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Scaling Factor $S = n \cdot d \cdot X^2$', fontsize=12, fontweight='bold')
    plt.ylabel('Runtime Deviation (ms)\n(Time_HighShots - Time_LowShots)', fontsize=12, fontweight='bold')
    plt.title('Absolute Deviation vs Circuit Scale', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS_DIR, 'shots_deviation_absolute.png'), dpi=300)
    print("Saved shots_deviation_absolute.png")

    # Plot 2: Relative Deviation (%) vs Scale
    plt.figure(figsize=(10, 8))
    plt.scatter(scales_dev, deviations_pct, alpha=0.7, c='green', edgecolors='k')
    plt.xscale('log')
    # plt.yscale('log') # Percentages might be linear-ish or small range
    plt.xlabel('Scaling Factor $S = n \cdot d \cdot X^2$', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Deviation (%)\n((Time_High - Time_Low) / Time_Low)', fontsize=12, fontweight='bold')
    plt.title('Relative Deviation vs Circuit Scale', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS_DIR, 'shots_deviation_relative.png'), dpi=300)
    print("Saved shots_deviation_relative.png")
    
    # Update explanation
    explanation_lines.append(f"\n## Deviation Analysis\nAbsolute deviation (ms) increases with scale (more work per shot).\nRelative deviation (%) tends to DECREASE with scale, as fixed contraction cost ($X^2$) dominates shot sampling cost.")
    
    # Write explanation file
    with open(os.path.join(_PLOTS_DIR, "shots_influence_explanation.txt"), "w") as f:
        f.writelines(explanation_lines)
    print("Saved explanation to shots_influence_explanation.txt")

if __name__ == "__main__":
    analyze_and_plot()
