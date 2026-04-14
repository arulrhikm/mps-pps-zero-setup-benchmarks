"""
experiment3_depth_scaling_gpu_rerun.py
======================================
Add trials 1-4 for Experiment 3 (Depth Scaling) on MPS GPU.
Submits in PER-DEPTH BATCHES (5 trials at a time) to avoid
queue pileup, then polls for that batch before moving to the next depth.

Config: n=40 | depths=[4,8,12,16,24,32,40,48,56,64] | bond_dimension=128 | shots=1
"""

import json
import os
import bluequbit
from qiskit.circuit.library import quantum_volume

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "gpu",
    "experiment3_depth_scaling_gpu.jsonl",
)

NUM_TRIALS = 5
NUM_QUBITS = 40
BOND_DIMENSION = 128
DEPTHS = [4, 8, 12, 16, 24, 32, 40, 48, 56, 64]

completed_runs = set()
if os.path.exists(OUTPUT_FILE):
    print(f"Loading existing runs from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                r = json.loads(s)
                if "error" not in r:
                    completed_runs.add((r["num_qubits"], r["depth"], r["trial"]))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    with open(OUTPUT_FILE, 'w') as f:
        f.write('# trial, num_qubits, depth, bond_dimension, num_gates, num_cx_gates, shots, job_id, queue_time_ms, run_time_ms\n')

total_submitted = 0
total_skipped = 0
MAX_RETRIES = 3

for depth in DEPTHS:
    for trial in range(NUM_TRIALS):
        if (NUM_QUBITS, depth, trial) in completed_runs:
            print(f"Skipping (depth={depth}, trial={trial}) - already completed")
            total_skipped += 1
            continue
            
        print(f"Running (depth={depth}, trial={trial})...")
        qc = quantum_volume(NUM_QUBITS, depth, seed=42 + trial)
        qc_dec = qc.decompose()
        num_gates = qc_dec.size()
        num_cx_gates = qc_dec.count_ops().get("cx", 0)

        job_success = False
        for attempt in range(MAX_RETRIES):
            try:
                job = bq.run(qc, device="mps.gpu",
                             options={"mps_bond_dimension": BOND_DIMENSION},
                             shots=1000)
                if not job.ok:
                    print(f"    [error] job not OK. Status: {job.status}. Retrying...")
                    continue
                job_success = True
                break
            except Exception as e:
                print(f"    [error] run failed: {e}. Retrying...")
                continue

        if not job_success:
            print(f"    [fatal error] Failed after {MAX_RETRIES} attempts. Skipping.")
            continue

        mps_build_time = None
        if job.run_results and "mps_build_time" in job.run_results:
            bt = float(job.run_results["mps_build_time"])
            if bt < job.run_time_ms / 100:
                bt *= 1000
            mps_build_time = bt
            
        sampling_time_ms = max(0, job.run_time_ms - mps_build_time) if mps_build_time else None

        # Write to file
        with open(OUTPUT_FILE, 'a') as f:
            run_data = {
                "trial": trial,
                "num_qubits": NUM_QUBITS,
                "depth": depth,
                "bond_dimension": BOND_DIMENSION,
                "num_gates": num_gates,
                "num_cx_gates": num_cx_gates,
                "shots": 1000,
                "job_id": job.job_id,
                "queue_time_ms": job.queue_time_ms,
                "run_time_ms": job.run_time_ms,
                "mps_build_time_ms": mps_build_time,
                "sampling_time_ms": sampling_time_ms
            }
            f.write(json.dumps(run_data) + '\n')
            
        build_str = f"build={mps_build_time:.1f}ms" if mps_build_time else "build=N/A"
        print(f"    Done. runtime={job.run_time_ms} ms {build_str}")
        total_submitted += 1

print(f"\n{'='*60}")
print(f"Experiment 3 GPU rerun complete!")
print(f"  Submitted: {total_submitted}  Skipped: {total_skipped}")
print(f"  Output: {OUTPUT_FILE}")
print(f"{'='*60}")
