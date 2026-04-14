import json
import bluequbit
from qiskit.circuit.library import quantum_volume
import os
import time

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

# Experiment 4: Sampling Scaling (GPU)
# Goal: Estimate shot scaling parameters A (overhead) and b (time per shot)
# Model: T = A + b * shots
# Configuration:
#   - Qubits: [20, 40, 60]
#   - Depth: [20, 40, 60]
#   - Bond Dimension: [64, 128, 256]
#   - Shots: [100, 500, 1000, 2000]

output_file = 'experiment4_sampling_scaling_gpu.jsonl'
num_trials = 1  # One trial per configuration

# Parameters
qubit_counts = [20, 40, 60]
depths = [20, 40, 60]
bond_dimensions = [64, 128, 256]
shot_counts = [100, 500, 1000, 2000]

# Load already-completed runs to resume from there
completed_runs = set()
if os.path.exists(output_file):
    print(f"Loading existing runs from {output_file}...")
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                run_data = json.loads(line)
                # Store tuple of parameters to identify unique runs
                completed_runs.add((
                    run_data['num_qubits'], 
                    run_data['depth'], 
                    run_data['bond_dimension'], 
                    run_data['shots'],
                    run_data['trial']
                ))
            except json.JSONDecodeError:
                continue
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    print(f"Creating fresh {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f'# Experiment 4: Sampling Scaling (GPU)\n')
        f.write('# trial, num_qubits, depth, bond_dimension, shots, num_gates, num_cx_gates, job_id, queue_time_ms, run_time_ms\n')

print(f"Qubit counts: {qubit_counts}")
print(f"Depths: {depths}")
print(f"Bond dimensions: {bond_dimensions}")
print(f"Shot counts: {shot_counts}")

# Run experiments
total_runs = len(qubit_counts) * len(depths) * len(bond_dimensions) * len(shot_counts) * num_trials
current_run = 0

for num_qubits in qubit_counts:
    for depth in depths:
        for bond_dimension in bond_dimensions:
            # Generate circuit once for this configuration (fixed circuit setting)
            
            for trial in range(num_trials):
                # Use a deterministic seed based on configuration to ensure reproducibility
                seed = 42 + num_qubits + depth + bond_dimension + trial
                qc = quantum_volume(num_qubits, depth, seed=seed)
                qc.measure_all() # Important for sampling
                
                qc_decomposed = qc.decompose()
                num_gates = qc_decomposed.size()
                num_cx_gates = qc_decomposed.count_ops().get('cx', 0)
                
                for shots in shot_counts:
                    if (num_qubits, depth, bond_dimension, shots, trial) in completed_runs:
                        current_run += 1
                        continue

                    print(f"Running ({current_run + 1}/{total_runs}): Qubits={num_qubits}, Depth={depth}, Bond={bond_dimension}, Shots={shots}, Trial={trial}...")
                    
                    options = {"mps_bond_dimension": bond_dimension}
                    
                    try:
                        job = bq.run(qc, device="mps.gpu", options=options, shots=shots)
                        
                        # Write result immediately
                        with open(output_file, 'a') as f:
                            run_data = {
                                'trial': trial,
                                'num_qubits': num_qubits,
                                'depth': depth,
                                'bond_dimension': bond_dimension,
                                'shots': shots,
                                'num_gates': num_gates,
                                'num_cx_gates': num_cx_gates,
                                'job_id': job.job_id,
                                'queue_time_ms': job.queue_time_ms,
                                'run_time_ms': job.run_time_ms
                            }
                            f.write(json.dumps(run_data) + '\n')
                            
                    except Exception as e:
                        print(f"Error running configuration: {e}")
                        
                    current_run += 1

print(f'Experiment 4 (GPU) complete! Results written to {output_file}')
