"""
Quantum Volume statevector benchmark — CPU (BlueQubit).

Output: ../data/quantum_volume_runs_cpu.jsonl
Plot:   python qv_plot_combined.py  (from ../plotting/)
"""
import json
import bluequbit
from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
import os

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)
output_file = os.path.join(DATA_DIR, "quantum_volume_runs_cpu.jsonl")
num_trials = 10

# Load already-completed runs to resume from there
completed_runs = set()
if os.path.exists(output_file):
    print(f"Loading existing runs from {output_file}...")
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            run_data = json.loads(line)
            # Track (num_qubits, depth, trial) tuples
            completed_runs.add((run_data['num_qubits'], run_data['depth'], run_data['trial']))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# trial, num_qubits, depth, num_gates, job_id, queue_time_ms, run_time_ms\n")

# Collect trial data for summary stats
trial_data = {}

for num_qubits in range(16, 35):
    for depth in range(10, 50, 10):
        trial_data[(num_qubits, depth)] = []
        for trial in range(num_trials):
            # Skip if already completed
            if (num_qubits, depth, trial) in completed_runs:
                print(f"Skipping (qubits={num_qubits}, depth={depth}, trial={trial}) - already completed")
                continue
            
            print(f"Running (qubits={num_qubits}, depth={depth}, trial={trial})...")
            qc = quantum_volume(num_qubits, depth, seed=42 + trial)
            num_gates = qc.decompose().size()  # Unitary block only (correct calculation)
            job = bq.run(qc, device="cpu")
            run_time_per_gate = job.run_time_ms / num_gates if num_gates > 0 else 0
            
            # Write to file immediately after each run
            with open(output_file, "a", encoding="utf-8") as f:
                run_data = {
                    'trial': trial,
                    'num_qubits': num_qubits,
                    'depth': depth,
                    'num_gates': num_gates,
                    'job_id': job.job_id,
                    'queue_time_ms': job.queue_time_ms,
                    'run_time_ms': job.run_time_ms
                }
                f.write(json.dumps(run_data) + '\n')
            
            trial_data[(num_qubits, depth)].append({
                'num_qubits': num_qubits,
                'depth': depth,
                'trial': trial,
                'job_id': job.job_id,
                'run_time_ms': job.run_time_ms,
                'queue_time_ms': job.queue_time_ms,
                'run_time_per_gate': run_time_per_gate
            })

print(f"Finished sweep; data appended to:\n  {output_file}")
print("To plot:  cd plotting && python qv_plot_combined.py")