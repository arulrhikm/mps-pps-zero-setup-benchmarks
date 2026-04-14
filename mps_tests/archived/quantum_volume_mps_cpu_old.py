import json
import bluequbit
from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
import os

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

import numpy as np

# Open output file for writing individual run data
output_file = 'quantum_volume_runs_mps_cpu.jsonl'
num_trials = 10

# Load already-completed runs to resume from there
completed_runs = set()
if os.path.exists(output_file):
    print(f"Loading existing runs from {output_file}...")
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            run_data = json.loads(line)
            # Track (num_qubits, depth, trial) tuples
            completed_runs.add((run_data['num_qubits'], run_data['depth'], run_data['bond_dimension'], run_data['trial']))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    with open(output_file, 'w') as f:
        f.write('# trial, num_qubits, num_gates, job_id, queue_time_ms, run_time_ms, depth, bond_dimension\n')

# Collect trial data for summary stats
trial_data = {}
depths_bond_dim = [(16, 16)]

for depth, bond_dimension in depths_bond_dim:
    for num_qubits in range(16, 97):
        trial_data[(num_qubits, depth, bond_dimension)] = []
        for trial in range(num_trials):
            # Skip if already completed
            if (num_qubits, depth, bond_dimension, trial) in completed_runs:
                print(f"Skipping (qubits={num_qubits}, depth={depth}, bond_dimension={bond_dimension}, trial={trial}) - already completed")
                continue
            
            print(f"Running (qubits={num_qubits}, depth={depth}, bond_dimension={bond_dimension}, trial={trial})...")
            qc = quantum_volume(num_qubits, depth, seed=42 + trial)
            num_gates = qc.decompose().size()  # Unitary block only (correct calculation)
            options = {
                "mps_bond_dimension": bond_dimension,
            }
            job = bq.run(qc, device="mps.cpu", options=options)
            run_time_per_gate = job.run_time_ms / num_gates if num_gates > 0 else 0
            
            # Write to file immediately after each run
            with open(output_file, 'a') as f:
                run_data = {
                    'trial': trial,
                    'num_qubits': num_qubits,
                    'depth': depth,
                    'bond_dimension': bond_dimension,
                    'num_gates': num_gates,
                    'job_id': job.job_id,
                    'queue_time_ms': job.queue_time_ms,
                    'run_time_ms': job.run_time_ms
                }
                f.write(json.dumps(run_data) + '\n')
            
            trial_data[(num_qubits, depth, bond_dimension)].append({
                'num_qubits': num_qubits,
                'depth': depth,
                'bond_dimension': bond_dimension,
                'trial': trial,
                'job_id': job.job_id,
                'run_time_ms': job.run_time_ms,
                'queue_time_ms': job.queue_time_ms,
                'run_time_per_gate': run_time_per_gate
            })

print(f'Wrote {len(trial_data) * num_trials} individual run records to {output_file}')
print(f'To plot results, run: python plot.py {output_file}')