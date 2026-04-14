import json
import numpy as np
import sys
import os

def load_data(filename):
    data = {} # Key: (n, d, X) -> Val: {shots: runtime}
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return data
        
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                d = json.loads(line)
                key = (d['num_qubits'], d['depth'], d['bond_dimension'])
                if key not in data: data[key] = {}
                data[key][d['shots']] = d['run_time_ms']
            except: pass
    return data

    first_write = True
    for key, shots_data in data.items():
        pass # just to iterate
    
    with open("shots_report.txt", "a" if label=="CPU" else "w") as f:
        f.write(f"\nAnalyzing {label} ({filename})...\n")
        
        diffs = []
        
        f.write(f"{'Config (n,d,X)':<20} | {'100 shots':<10} | {'2000 shots':<10} | {'% Diff':<10}\n")
        f.write("-" * 60 + "\n")
        
        for key, shots_data in data.items():
            if 100 in shots_data and 2000 in shots_data:
                t_min = shots_data[100]
                t_max = shots_data[2000]
                pct_diff = ((t_max - t_min) / t_min) * 100
                diffs.append(pct_diff)
                f.write(f"{str(key):<20} | {t_min:<10.0f} | {t_max:<10.0f} | {pct_diff:+.1f}%\n")
                
        if diffs:
            avg_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            f.write("-" * 60 + "\n")
            f.write(f"Average Increase (100 -> 2000 shots): {avg_diff:.1f}%\n")
            f.write(f"Max Increase: {max_diff:.1f}%\n")
            f.write(f"Significant (>1%)? {'YES' if avg_diff > 1 else 'NO'}\n")
        else:
            f.write("No matching data (100 & 2000 shots) found.\n")

_BASE = os.path.dirname(os.path.abspath(__file__))
_MPS_ROOT = os.path.dirname(_BASE)
_DATA_DIR = os.path.join(_MPS_ROOT, "data")

if __name__ == "__main__":
    analyze(os.path.join(_DATA_DIR, "gpu", "experiment4_sampling_scaling_gpu.jsonl"), "GPU")
    analyze(os.path.join(_DATA_DIR, "cpu", "experiment4_sampling_scaling_cpu.jsonl"), "CPU")
