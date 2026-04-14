import json
import bluequbit
from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
import os

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

import numpy as np

# Experiment 2: Bond Dimension Scaling
# Goal: Scale bond dimension from 4 to 2048 at fixed 40 qubits
# Configuration: Qubits=40 (fixed), Depth=10 (fixed), Bond=[4-120 by 4, then 128-2048 by 32]
# Track CX gates in addition to total gates
# Single circuit per configuration (no averaging)

output_file = 'experiment2_bond_scaling_cpu_updated.jsonl'
num_trials = 1  # Single circuit per configuration

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
            # Track (num_qubits, depth, bond_dimension, trial) tuples
            completed_runs.add((run_data['num_qubits'], run_data['depth'], run_data['bond_dimension'], run_data['trial']))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    with open(output_file, 'w') as f:
        f.write('# Experiment 2: Bond Dimension Scaling (qubits=40, depth=32, bond=4-120)\\n')
        f.write('# trial, num_qubits, num_gates, num_cx_gates, job_id, queue_time_ms, run_time_ms, depth, bond_dimension\\n')

# Fixed parameters
num_qubits = 40
depth = 10

# Bond dimensions: 4-120 by 4, then 128-2048 by 32
bond_dimensions = list(range(4, 124, 4)) + list(range(128, 2049, 32))

print(f"Configuration: {num_qubits} qubits, depth={depth}, bond dimensions: {bond_dimensions}")
print(f"Total configurations to run: {len(bond_dimensions)}")

# Perform a single warmup run to eliminate cold-start overhead
print("\n" + "="*70)
print("WARMUP RUN (discarded)")
print("="*70)
warmup_qc = quantum_volume(num_qubits, depth, seed=999)
warmup_job = bq.run(warmup_qc, device="mps.cpu", options={"mps_bond_dimension": 16})
print(f"Warmup complete: {warmup_job.run_time_ms}ms (discarded)")
print("="*70 + "\n")

# Iterate over all bond dimensions
for bond_dimension in bond_dimensions:
    for trial in range(num_trials):
        # Skip if already completed
        if (num_qubits, depth, bond_dimension, trial) in completed_runs:
            print(f"Skipping (qubits={num_qubits}, depth={depth}, bond_dimension={bond_dimension}, trial={trial}) - already completed")
            continue
        
        print(f"Running (qubits={num_qubits}, depth={depth}, bond_dimension={bond_dimension}, trial={trial})...")
        qc = quantum_volume(num_qubits, depth, seed=42 + trial)
        
        # Count total gates and CX gates
        decomposed_qc = qc.decompose()
        num_gates = decomposed_qc.size()
        num_cx_gates = decomposed_qc.count_ops().get('cx', 0)
        
        options = {
            "mps_bond_dimension": bond_dimension,
        }
        job = bq.run(qc, device="mps.cpu", options=options)
        
        # Write to file immediately after each run
        with open(output_file, 'a') as f:
            run_data = {
                'trial': trial,
                'num_qubits': num_qubits,
                'depth': depth,
                'bond_dimension': bond_dimension,
                'num_gates': num_gates,
                'num_cx_gates': num_cx_gates,
                'job_id': job.job_id,
                'queue_time_ms': job.queue_time_ms,
                'run_time_ms': job.run_time_ms
            }
            f.write(json.dumps(run_data) + '\n')
            f.flush()  # Ensure data is written to disk immediately
            os.fsync(f.fileno())  # Force write to disk
        
        print(f"  → Completed: {num_gates} gates ({num_cx_gates} CX), runtime={job.run_time_ms}ms")

print(f'\nExperiment 2 complete! Results written to {output_file}')
print(f'Configuration: {num_qubits} qubits, depth={depth}')
print(f'Bond dimensions tested: {len(bond_dimensions)} values from {min(bond_dimensions)} to {max(bond_dimensions)}')
