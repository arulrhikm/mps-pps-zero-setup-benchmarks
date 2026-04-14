"""
Quantum Volume statevector benchmark — GPU (BlueQubit).

Output: ../data/quantum_volume_runs_gpu.jsonl
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
output_file = os.path.join(DATA_DIR, "quantum_volume_runs_gpu.jsonl")
num_trials = 5

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
    for depth in [30, 60, 90, 120, 150, 300, 450, 600, 750]:
        # Cold-start run to warm up the system for this circuit setting (only for 33+ qubits)
        # Skip cold-start if any trial for this (num_qubits, depth) is already completed.
        need_cold_start = num_qubits >= 33 and not any((num_qubits, depth, t) in completed_runs for t in range(num_trials))
        if need_cold_start:
            print(f"Running cold-start trial for (qubits={num_qubits}, depth={depth})...")
            qc = quantum_volume(num_qubits, depth, seed=42)
            num_gates = qc.size()
            job = bq.run(qc, device="gpu")
            print(f"Cold-start complete: {job.run_time_ms}ms")
        else:
            if num_qubits >= 33:
                print(f"Skipping cold-start for (qubits={num_qubits}, depth={depth}) - existing trial found")
        
        trial_data[(num_qubits, depth)] = []
        for trial in range(num_trials):
            # Skip if already completed
            if (num_qubits, depth, trial) in completed_runs:
                print(f"Skipping (qubits={num_qubits}, depth={depth}, trial={trial}) - already completed")
                continue
            
            print(f"Running (qubits={num_qubits}, depth={depth}, trial={trial})...")
            qc = quantum_volume(num_qubits, depth, seed=42 + trial)
            num_gates = qc.size()
            job = bq.run(qc, device="gpu")
            run_time_per_gate = job.run_time_ms / num_gates if num_gates > 0 else 0
            
            # Write to file immediately after each run
            with open(output_file, 'a') as f:
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