import json
import bluequbit
from qiskit.circuit.library import quantum_volume
import os

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

# Experiment 3: Depth Scaling (CPU)
# Goal: Verify that runtime scales linearly with depth (runtime/gate is flat)
# Configuration: Qubits=40, Depth [4, 8, 12, 16, 24, 32, 40, 48, 56, 64], Bond=128 (fixed)
# Single circuit per configuration (no averaging)

output_file = 'experiment3_depth_scaling_cpu.jsonl'
num_trials = 1  

# Fixed parameters
bond_dimension = 128
num_qubits = 40
depths = [4, 8, 12, 16, 24, 32, 40, 48, 56, 64]

# Load already-completed runs to resume from there
completed_runs = set()
if os.path.exists(output_file):
    print(f"Loading existing runs from {output_file}...")
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('{') == False:
                continue
            run_data = json.loads(line)
            completed_runs.add((run_data['num_qubits'], run_data['depth'], run_data['trial']))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    print(f"Creating fresh {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f'# Experiment 3: Depth Scaling (qubits={num_qubits}, depth={depths}, bond={bond_dimension})\n')
        f.write('# trial, num_qubits, num_gates, num_cx_gates, job_id, queue_time_ms, run_time_ms, depth, bond_dimension\n')

print(f"Parameters: bond_dimension={bond_dimension}")
print(f"Qubit counts: {num_qubits}")
print(f"Depths: {depths}")

# Run experiments
new_circuits_run = 0
for depth in depths:
    for trial in range(num_trials):
        if (num_qubits, depth, trial) in completed_runs:
            continue
        
        print(f"Running (qubits={num_qubits}, depth={depth}, bond_dimension={bond_dimension}, trial={trial})...")
        qc = quantum_volume(num_qubits, depth, seed=42 + trial)
        qc_decomposed = qc.decompose()
        num_gates = qc_decomposed.size()
        num_cx_gates = qc_decomposed.count_ops().get('cx', 0)
        options = {"mps_bond_dimension": bond_dimension}
        job = bq.run(qc, device="mps.cpu", options=options, shots=1000)
        
        with open(output_file, 'a') as f:
            run_data = {
                'trial': trial,
                'num_qubits': num_qubits,
                'depth': depth,
                'bond_dimension': bond_dimension,
                'num_gates': num_gates,
                'num_cx_gates': num_cx_gates,
                'shots': 1000,
                'job_id': job.job_id,
                'queue_time_ms': job.queue_time_ms,
                'run_time_ms': job.run_time_ms
            }
            f.write(json.dumps(run_data) + '\n')
        
        new_circuits_run += 1

print(f'Experiment 3 (CPU) complete! Results written to {output_file}')
print(f'Total circuits run: {new_circuits_run}')
