import json
import bluequbit
from qiskit.circuit.library import quantum_volume
import os

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

# Experiment 1: Qubit Scaling (GPU)
# Goal: Measure runtime scaling with number of qubits
# Configuration: Qubits 16-96, Depth=16, Bond=256 (fixed)
# Single circuit per configuration (no averaging)

output_file = 'experiment1_qubit_scaling_gpu.jsonl'
num_trials = 1

# Fixed parameters
depth = 16
bond_dimension = 256
min_qubits = 16
max_qubits = 96

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
            completed_runs.add((run_data['num_qubits'], run_data['trial']))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    print(f"Creating fresh {output_file}...")
    with open(output_file, 'w') as f_out:
        f_out.write(f'# Experiment 1: Qubit Scaling ({min_qubits}-{max_qubits} qubits, depth={depth}, bond={bond_dimension})\n')
        f_out.write('# trial, num_qubits, num_gates, num_cx_gates, job_id, queue_time_ms, run_time_ms, depth, bond_dimension\n')

print(f"Parameters: depth={depth}, bond_dimension={bond_dimension}")
print(f"Will run experiments for qubits {min_qubits}-{max_qubits}")

# Run experiments
new_circuits_run = 0
for num_qubits in range(min_qubits, max_qubits + 1, 4):
    for trial in range(num_trials):
        if (num_qubits, trial) in completed_runs:
            continue
        
        print(f"Running (qubits={num_qubits}, depth={depth}, bond_dimension={bond_dimension}, trial={trial})...")
        qc = quantum_volume(num_qubits, depth, seed=42 + trial)
        qc_decomposed = qc.decompose()
        num_gates = qc_decomposed.size()
        num_cx_gates = qc_decomposed.count_ops().get('cx', 0)
        options = {"mps_bond_dimension": bond_dimension}
        job = bq.run(qc, device="mps.gpu", options=options, shots=1000)
        
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

print(f'Experiment 1 (GPU) complete! Results written to {output_file}')
print(f'Total circuits run: {new_circuits_run}')
