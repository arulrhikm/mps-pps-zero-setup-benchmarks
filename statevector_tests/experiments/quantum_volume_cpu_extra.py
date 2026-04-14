"""
Extra CPU depths (60, 90, 120, 150) for QV — complements data/quantum_volume_runs_cpu_updated.jsonl.

Output: ../data/quantum_volume_runs_cpu_extra.jsonl
"""
import json
import bluequbit
from qiskit.circuit.library import quantum_volume
import os

# Route to the dev environment to bypass runtime limits
os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)
output_file = os.path.join(DATA_DIR, "quantum_volume_runs_cpu_extra.jsonl")
num_trials = 5
extra_depths = [60, 90, 120, 150]
qubit_range = range(16, 35)  # 16 to 34 inclusive

# Load already-completed runs to resume
completed_runs = set()
if os.path.exists(output_file):
    print(f"Loading existing runs from {output_file}...")
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            run_data = json.loads(line)
            completed_runs.add((run_data['num_qubits'], run_data['depth'], run_data['trial']))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    with open(output_file, 'w') as f:
        f.write('# trial, num_qubits, depth, num_gates, job_id, queue_time_ms, run_time_ms\n')

total_written = 0

for num_qubits in qubit_range:
    for depth in extra_depths:
        for trial in range(num_trials):
            if (num_qubits, depth, trial) in completed_runs:
                print(f"Skipping (qubits={num_qubits}, depth={depth}, trial={trial}) - already completed")
                continue

            print(f"Running (qubits={num_qubits}, depth={depth}, trial={trial})...")
            qc = quantum_volume(num_qubits, depth, seed=42 + trial)
            num_gates = qc.decompose().size()
            job = bq.run(qc, device="cpu")

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
            print(f"  Done: {job.run_time_ms}ms")
            total_written += 1

print(f"\nFinished! Wrote {total_written} new runs to {output_file}")
