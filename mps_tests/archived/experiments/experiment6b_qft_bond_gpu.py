"""
experiment6b_qft_bond_gpu.py
============================
Extension of experiment 6: sweep bond dimension at fixed approximation_degree=0.

Goal:
  Show how MPS bond dimension (χ) affects QFT runtime on GPU.
  Paired with experiment6 (χ=64) this gives three bond-dim traces:
    χ = 64   (from experiment6_qft_scaling_gpu.jsonl)
    χ = 256  (this file)
    χ = 512  (this file)
    χ = 768  (this file)
    χ = 1024 (this file)

Circuit:
  - Exact QFT via qiskit.synthesis.qft.synth_qft_full
  - approximation_degree=0 (no gate omission)
  - bond_dimension varied: 256, 512, 768, 1024

Parameter sweep:
  n (qubits):       4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96
  bond_dimension:   256, 512, 768, 1024
  approximation_degree: 0 (fixed)
  trials:           1 per config

Output:
  data/gpu/experiment6b_qft_bond_gpu.jsonl
"""

import json
import os

import bluequbit
from qiskit.synthesis.qft import synth_qft_full

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "gpu",
    "experiment6b_qft_bond_gpu.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

NUM_TRIALS      = 1
APPROX_DEGREE   = 0   # exact QFT, fixed

QUBIT_COUNTS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96]
BOND_DIMS    = [256, 512, 768, 1024]

CONFIGS = [
    (n, X)
    for n in QUBIT_COUNTS
    for X in BOND_DIMS
]

# ── Resume support ───────────────────────────────────────────────────────────
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                completed.add((r["num_qubits"], r["bond_dimension"], r["trial"]))
            except Exception:
                pass
    print(f"Resuming: {len(completed)} runs already complete.")
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Experiment 6b: QFT bond-dimension sweep (MPS GPU)\n")
        f.write("# n=4–96, bond_dim=256/512/768/1024, approximation_degree=0, 1 trial per config\n")
    print("Starting fresh.")

# ── Print plan ───────────────────────────────────────────────────────────────
total     = len(CONFIGS) * NUM_TRIALS
remaining = sum(
    1 for (n, X) in CONFIGS for t in range(NUM_TRIALS)
    if (n, X, t) not in completed
)
print(f"\nExperiment 6b – QFT bond-dimension sweep on MPS GPU")
print(f"{'─'*55}")
print(f"Approximation degree: {APPROX_DEGREE} (exact QFT, fixed)")
print(f"Total configs:  {len(CONFIGS)}")
print(f"Trials each:    {NUM_TRIALS}")
print(f"Total runs:     {total}  ({remaining} remaining)\n")
print(f"{'n':>4}  {'bond':>5}  {'circuit_depth_est':>18}")
print("─" * 32)
for n, X in CONFIGS:
    depth_est = n * (n - 1) // 2   # exact QFT ~n(n-1)/2 cp gates
    print(f"{n:>4}  {X:>5}  {depth_est:>18,}")
print()

# ── Run ──────────────────────────────────────────────────────────────────────
for n, X in CONFIGS:
    for trial in range(NUM_TRIALS):
        if (n, X, trial) in completed:
            print(f"  ✓ skip  n={n} bond={X} trial={trial}")
            continue

        print(f"  → n={n:3d}  bond={X:4d}  trial={trial}  ", end="", flush=True)
        try:
            qc = synth_qft_full(num_qubits=n, approximation_degree=APPROX_DEGREE,
                                do_swaps=True)

            qc_dec        = qc.decompose()
            num_gates     = qc_dec.size()
            num_cx        = qc_dec.count_ops().get("cx", 0)
            num_cp        = qc_dec.count_ops().get("cp", 0)
            circuit_depth = qc_dec.depth()

            job = bq.run(qc, device="mps.gpu",
                         options={"mps_bond_dimension": X}, shots=1)

            record = {
                "trial":               trial,
                "num_qubits":          n,
                "approximation_degree": APPROX_DEGREE,
                "bond_dimension":      X,
                "num_gates":           num_gates,
                "circuit_depth":       circuit_depth,
                "num_cx":              num_cx,
                "num_cp":              num_cp,
                "job_id":              job.job_id,
                "queue_time_ms":       job.queue_time_ms,
                "run_time_ms":         job.run_time_ms,
                "source_file":         "gpu/experiment6b_qft_bond_gpu.jsonl",
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            print(f"runtime={job.run_time_ms:>8,.0f} ms   "
                  f"depth={circuit_depth:>6,}  gates={num_gates:>7,}")
            completed.add((n, X, trial))

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({
                    "num_qubits":          n,
                    "approximation_degree": APPROX_DEGREE,
                    "bond_dimension":      X,
                    "trial":               trial,
                    "error":               str(e),
                }) + "\n")
                f.flush()

print("\nExperiment 6b GPU (QFT bond sweep) complete.")
print(f"Results: {OUTPUT_FILE}")
