#!/usr/bin/env python3
"""
Master script to generate all experiment plots
Runs all three plotting scripts in sequence
"""

import subprocess
import sys

def run_script(script_name):
    """Run a Python script and report status"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all plotting scripts"""
    scripts = [
        'plot_experiment1_qubit_scaling.py',
        'plot_experiment2_bond_scaling.py',
        'plot_experiment3_depth_scaling.py'
    ]
    
    print("="*60)
    print("MPS Experiments - Plotting All Results")
    print("="*60)
    print("\nThis script will generate plots for:")
    print("  1. Experiment 1: Qubit Scaling (Cubic O(n³))")
    print("  2. Experiment 2: Bond Dimension Scaling (Cubic O(χ³))")
    print("  3. Experiment 3: Depth Scaling (Linear O(d))")
    print()
    
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {script}")
    
    all_success = all(results.values())
    if all_success:
        print("\n✓ All plots generated successfully!")
        print("\nGenerated files:")
        print("  - experiment1_qubit_scaling.png")
        print("  - experiment2_bond_scaling.png")
        print("  - experiment2_bond_scaling_loglog.png")
        print("  - experiment3_depth_scaling.png")
        print("  - experiment3_depth_scaling_individual.png")
    else:
        print("\n✗ Some plots failed to generate. Check errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
