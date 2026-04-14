"""
experiment6_qft_gpu.py
======================
Demo experiment: Quantum Fourier Transform (QFT) scaling on MPS GPU.

Goal:
  Demonstrate that BlueQubit's MPS GPU simulator can execute QFT circuits
  at high qubit counts (up to 96 qubits) – well beyond what a statevector
  simulator could handle (practical limit ~30 qubits). Compare GPU vs CPU
  runtime by pairing with experiment6_qft_cpu.py.

Circuit:
  - QFT via qiskit.synthesis.qft.synth_qft_full
  - No inverse, no measurement (pure unitary circuit for timing purposes)
  - Bond dimension fixed at 64; approximation_degree varied

Parameter sweep:
  n (qubits):           4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96
  approximation_degree: 0, 1, 2, 3
  bond_dimension:       64 (fixed)
  trials:               1 per config

Approximation degree semantics (Qiskit QFT):
  degree=0  →  full QFT, all controlled-phase gates included  (O(n²) gates)
  degree=k  →  omit the k smallest rotations per qubit        (O(n(n-k)) gates)
  Higher degree → fewer gates, shorter depth, faster runtime, coarser approximation.
  This is the natural QFT-specific knob for trading accuracy vs. speed.

QFT characteristics:
  - Depth:  O(n²) at degree 0, decreases with higher degree
  - Gates:  predominantly controlled-phase (cp) gates
  - GPU tensor contractions parallelize the cp gate sweeps well
  - Long-range entanglement structure makes bond dimension the MPS bottleneck

Output:
  data/gpu/experiment6_qft_scaling_gpu.jsonl
"""

import json
import os

import bluequbit
from qiskit.synthesis.qft import synth_qft_full

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "gpu",
    "experiment6_qft_scaling_gpu.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

NUM_TRIALS = 1

# ── Parameter grid ──────────────────────────────────────────────────────────
# Qubit counts cover the full range – including the 96-qubit upper limit.
# approximation_degree varies from 0 (exact QFT) to 3 (coarse – drops
# the 3 smallest controlled-phase rotations per qubit).
QUBIT_COUNTS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96]
APPROX_DEGREES = [0, 1, 2, 3]
BOND_DIMENSION = 64   # fixed; QFT entanglement is modest for these degrees

CONFIGS = [
    (n, apx)
    for n in QUBIT_COUNTS
    for apx in APPROX_DEGREES
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
                completed.add((r["num_qubits"], r["approximation_degree"], r["trial"]))
            except Exception:
                pass
    print(f"Resuming: {len(completed)} runs already complete.")
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Experiment 6: QFT high-qubit-count scaling demo (MPS GPU)\n")
        f.write("# n=4–96, approximation_degree=0/1/2/3, bond_dim=64 (fixed), 1 trial per config\n")
    print("Starting fresh.")

# ── Print plan ───────────────────────────────────────────────────────────────
total     = len(CONFIGS) * NUM_TRIALS
remaining = sum(
    1 for (n, apx) in CONFIGS for t in range(NUM_TRIALS)
    if (n, apx, t) not in completed
)
print(f"\nExperiment 6 – QFT scaling demo on MPS GPU")
print(f"{'─'*60}")
print(f"Bond dimension: {BOND_DIMENSION} (fixed)")
print(f"Total configs:  {len(CONFIGS)}")
print(f"Trials each:    {NUM_TRIALS}")
print(f"Total runs:     {total}  ({remaining} remaining)\n")
print(f"{'n':>4}  {'apx':>4}  {'cp_gates_est':>13}  {'depth_est':>10}")
print("─" * 40)
for n, apx in CONFIGS:
    cp_gates = max(0, n * (n - 1) // 2 - apx * n)  # controlled-phase gates remaining
    depth_est = max(0, n - apx) + n                  # approx sequential depth
    print(f"{n:>4}  {apx:>4}  {cp_gates:>13,}  {depth_est:>10,}")
print()

# ── Run ──────────────────────────────────────────────────────────────────────
for n, apx in CONFIGS:
    for trial in range(NUM_TRIALS):
        if (n, apx, trial) in completed:
            print(f"  ✓ skip  n={n} apx={apx} trial={trial}")
            continue

        print(f"  → n={n:3d}  apx={apx}  trial={trial}  ", end="", flush=True)
        try:
            # Build (approximate) QFT circuit – no measurement, pure unitary for timing
            qc = synth_qft_full(num_qubits=n, approximation_degree=apx, do_swaps=True)

            qc_dec        = qc.decompose()
            num_gates     = qc_dec.size()
            num_cx        = qc_dec.count_ops().get("cx", 0)
            num_cp        = qc_dec.count_ops().get("cp", 0)   # controlled-phase
            circuit_depth = qc_dec.depth()

            job = bq.run(qc, device="mps.gpu",
                         options={"mps_bond_dimension": BOND_DIMENSION}, shots=1)

            record = {
                "trial":               trial,
                "num_qubits":          n,
                "approximation_degree": apx,
                "bond_dimension":      BOND_DIMENSION,
                "num_gates":           num_gates,
                "circuit_depth":       circuit_depth,
                "num_cx":              num_cx,
                "num_cp":              num_cp,
                "job_id":              job.job_id,
                "queue_time_ms":       job.queue_time_ms,
                "run_time_ms":         job.run_time_ms,
                "source_file":         "gpu/experiment6_qft_scaling_gpu.jsonl",
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            print(f"runtime={job.run_time_ms:>8,.0f} ms   "
                  f"depth={circuit_depth:>6,}  gates={num_gates:>7,}")
            completed.add((n, apx, trial))

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({
                    "num_qubits":          n,
                    "approximation_degree": apx,
                    "bond_dimension":      BOND_DIMENSION,
                    "trial":               trial,
                    "error":               str(e),
                }) + "\n")
                f.flush()

print("\nExperiment 6 GPU (QFT) complete.")
print(f"Results: {OUTPUT_FILE}")
