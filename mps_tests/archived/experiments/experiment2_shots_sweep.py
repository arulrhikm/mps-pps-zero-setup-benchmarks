"""
experiment2_shots_sweep.py
==========================
Shots sweep for build-vs-sampling decomposition (Section 5.2).
Shows how sampling overhead scales with shot count and where it
starts competing with build time.

Submits in batches: per (device, shots) combination, 5 trials at a time.

Config: n=40 | d=16 | χ=256 | shots={1,10,100,1000} | CPU & GPU | 5 trials
"""

import json
import os
import bluequbit
from qiskit.circuit.library import quantum_volume

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data",
    "experiment2_shots_sweep.jsonl",
)

NUM_TRIALS = 5
NUM_QUBITS = 40
DEPTH = 16
BOND_DIM = 256
SHOTS_LIST = [1, 10, 100, 1000]
DEVICES = ["mps.cpu", "mps.gpu"]

completed_runs = set()
if os.path.exists(OUTPUT_FILE):
    print(f"Loading existing runs from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                r = json.loads(s)
                if "error" not in r:
                    completed_runs.add((r["device"], r["shots"], r["trial"]))
    print(f"Found {len(completed_runs)} already completed runs")
else:
    with open(OUTPUT_FILE, 'w') as f:
        f.write('# trial, device, num_qubits, depth, bond_dimension, shots, num_gates, job_id, queue_time_ms, run_time_ms\n')

for device in DEVICES:
    for shots in SHOTS_LIST:
        for trial in range(NUM_TRIALS):
            if (device, shots, trial) in completed_runs:
                print(f"Skipping (device={device}, shots={shots}, trial={trial}) - already completed")
                continue
                
            print(f"Running (device={device}, shots={shots}, trial={trial})...")
            qc = quantum_volume(NUM_QUBITS, DEPTH, seed=42 + trial)
            qc_dec = qc.decompose()
            num_gates = qc_dec.size()
            num_cx_gates = qc_dec.count_ops().get("cx", 0)

            try:
                job = bq.run(qc, device=device,
                             options={"mps_bond_dimension": BOND_DIM},
                             shots=shots)
            except Exception as e:
                print(f"    [error] run failed: {e}")
                continue

            if not job.ok:
                print(f"    [error] job not OK. Status: {job.status}")
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
                    "device": device,
                    "num_qubits": NUM_QUBITS,
                    "depth": DEPTH,
                    "bond_dimension": BOND_DIM,
                    "shots": shots,
                    "num_gates": num_gates,
                    "num_cx_gates": num_cx_gates,
                    "job_id": job.job_id,
                    "queue_time_ms": job.queue_time_ms,
                    "run_time_ms": job.run_time_ms,
                    "mps_build_time_ms": mps_build_time,
                    "sampling_time_ms": sampling_time_ms
                }
                f.write(json.dumps(run_data) + '\n')
            
            build_str = f"build={mps_build_time:.1f}ms" if mps_build_time else "build=N/A"
            print(f"    Done. runtime={job.run_time_ms} ms {build_str}")

print(f'\nExperiment 2 shots sweep complete! Output saved to {OUTPUT_FILE}')
