"""
experiment6_qft_gpu_rerun.py
============================
Add trials 1-4 for Experiment 6 (QFT Scaling) on MPS GPU, degree=0 only.
Submits in PER-QUBIT-COUNT BATCHES (5 trials at a time) to avoid
queue pileup, then polls for that batch before moving to the next n.

Config: qubits [4..96] | approximation_degree=0 | bond_dimension=64 | shots=1
"""

import json
import os
import bluequbit
from qiskit.synthesis.qft import synth_qft_full

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "gpu",
    "experiment6_qft_scaling_gpu.jsonl",
)

NUM_TRIALS = 5
QUBIT_COUNTS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96]
APPROX_DEGREE = 0
BOND_DIMENSION = 64

completed_runs = set()
if os.path.exists(OUTPUT_FILE):
    print(f"Loading existing runs from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                r = json.loads(s)
                if r.get("approximation_degree") == APPROX_DEGREE \
                        and r.get("bond_dimension") == BOND_DIMENSION \
                        and "error" not in r:
                    completed_runs.add((r["num_qubits"], r["trial"]))
    print(f"Found {len(completed_runs)} already completed degree-0 runs")
else:
    with open(OUTPUT_FILE, 'w') as f:
        f.write('# num_qubits, trial, approx_degree, bond_dim, num_gates, depth, cx, cp, job_id, run_time_ms\n')

total_submitted = 0
total_skipped = 0

for n in QUBIT_COUNTS:
    for trial in range(NUM_TRIALS):
        if (n, trial) in completed_runs:
            print(f"Skipping (qubits={n}, trial={trial}) - already completed")
            total_skipped += 1
            continue
            
        print(f"Running (qubits={n}, trial={trial})...")
        qc = synth_qft_full(num_qubits=n, approximation_degree=APPROX_DEGREE,
                            do_swaps=True)
        qc_dec = qc.decompose()
        num_gates = qc_dec.size()
        num_cx = qc_dec.count_ops().get("cx", 0)
        num_cp = qc_dec.count_ops().get("cp", 0)
        circuit_depth = qc_dec.depth()

        job_success = False
        for attempt in range(3):
            try:
                job = bq.run(qc, device="mps.gpu",
                             options={"mps_bond_dimension": BOND_DIMENSION},
                             shots=1)
                if not job.ok:
                    print(f"    [error] job not OK. Status: {job.status}. Retrying...")
                    continue
                job_success = True
                break
            except Exception as e:
                print(f"    [error] run failed: {e}. Retrying...")
                continue

        if not job_success:
            print(f"    [fatal error] Failed after 3 attempts. Skipping.")
            continue

        # Write to file
        with open(OUTPUT_FILE, 'a') as f:
            run_data = {
                "trial": trial,
                "num_qubits": n,
                "approximation_degree": APPROX_DEGREE,
                "bond_dimension": BOND_DIMENSION,
                "num_gates": num_gates,
                "circuit_depth": circuit_depth,
                "num_cx": num_cx,
                "num_cp": num_cp,
                "job_id": job.job_id,
                "queue_time_ms": job.queue_time_ms,
                "run_time_ms": job.run_time_ms,
                "source_file": "gpu/experiment6_qft_scaling_gpu.jsonl"
            }
            f.write(json.dumps(run_data) + '\n')
            
        print(f"    Done. runtime={job.run_time_ms} ms")
        total_submitted += 1

print(f"\n{'='*60}")
print(f"Experiment 6 GPU rerun complete!")
print(f"  Submitted: {total_submitted}  Skipped: {total_skipped}")
print(f"  Output: {OUTPUT_FILE}")
print(f"{'='*60}")
