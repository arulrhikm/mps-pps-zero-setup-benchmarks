"""
fig4_pps_gpu.py
===============
Reproduce Fig. 4 — Norm decay ||O_k|| over circuit execution.

Circuit (Eq. 10):
  - 127 qubits, IBM heavy-hex lattice
  - T = 30 Trotter steps  (paper uses T=30 for Fig 4)
  - θ_ZZ = -π/2
  - Each θ_X sampled uniformly from [-π/4, π/4]  (different per step)
  - Observable ⟨Z_62⟩

Sweep:
  δ ∈ {5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5}  (7 values from Fig 4)
  Trotter steps 1..30 (snapshot after each step)
  → 7 × 30 = 210 runs total

The expectation value ⟨Z_62⟩ at each truncation point tracks how much
the observable norm has decayed due to coefficient truncation.
With the full (δ→0) untruncated evolution the norm would stay ≡ 1.
"""

import os
import json
import numpy as np
import bluequbit
from bluequbit.library.helpers.hardware_connectivites import IBM_127_HEAVY_HEX_MAP
from qiskit import QuantumCircuit

# Route to the dev environment to bypass runtime limits
os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")  # production token

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "fig4_pps_gpu.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
num_qubits     = 127
num_edges      = len(IBM_127_HEAVY_HEX_MAP)  # 144
gates_per_step = num_edges + num_qubits       # 271
T              = 30                           # Fig 4 uses T = 30
rzz_angle      = -np.pi / 2
SEED           = 42

# Random θ_X per step
rng = np.random.default_rng(SEED)
rx_angles_per_step = rng.uniform(-np.pi / 4, np.pi / 4, size=T)

# δ values matching Fig 4 legend
deltas = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]

# Snapshot after every Trotter step
snapshot_steps = list(range(1, T + 1))

# Observable ⟨Z_62⟩
pauli_str = "I" * (num_qubits - 1 - 62) + "Z" + "I" * 62
pauli_sum = [(pauli_str, 1.0)]

# ── Resume support ────────────────────────────────────────────────────────────
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                if "error" not in r:
                    completed.add((r["trotter_step"], r["delta"]))
            except Exception:
                pass
    print(f"Resuming: {len(completed)} runs already complete.")
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Fig 4: Norm decay ||O_k|| over circuit execution, GPU\n")
        f.write(f"# num_qubits={num_qubits}, T={T}\n")
        f.write(f"# deltas={deltas}\n")
        f.write(f"# seed={SEED}\n")
    print("Starting fresh.")

# ── Print plan ────────────────────────────────────────────────────────────────
total = len(snapshot_steps) * len(deltas)
remaining = sum(
    1 for s in snapshot_steps for d in deltas
    if (s, d) not in completed
)
print(f"\nTotal runs: {total}  ({remaining} remaining)")
print(f"Deltas: {deltas}")
print(f"Steps:  1..{T}  ({gates_per_step} gates/step, up to {T * gates_per_step} ops)")
print()

# ── Build circuits for each step ──────────────────────────────────────────────
print("Building circuits for each Trotter step...")
circuits = {}
for step in snapshot_steps:
    qc = QuantumCircuit(num_qubits)
    for t in range(step):
        for edge in IBM_127_HEAVY_HEX_MAP:
            qc.rzz(rzz_angle, edge[0], edge[1])
        for i in range(num_qubits):
            qc.rx(rx_angles_per_step[t], i)
    circuits[step] = qc
    print(f"  Built step {step:2d}: {step * gates_per_step} gates")
print()

# ── Run ───────────────────────────────────────────────────────────────────────
for step in snapshot_steps:
    qc = circuits[step]
    num_ops = step * gates_per_step

    for delta in deltas:
        if (step, delta) in completed:
            print(f"  ✓ skip  step={step:2d}  δ={delta:.0e}")
            continue

        print(f"  → step={step:2d}  ops={num_ops:5d}  δ={delta:.0e}  ", end="", flush=True)
        try:
            options = {"pauli_path_truncation_threshold": delta}
            result = bq.run(
                qc,
                device="pauli-path.gpu",
                pauli_sum=pauli_sum,
                options=options,
            )

            ev       = result.expectation_value
            run_time = result.run_time_ms

            record = {
                "trotter_step":      step,
                "num_operations":    num_ops,
                "delta":             delta,
                "expectation_value": ev,
                "run_time_ms":       run_time,
                "job_id":            result.job_id,
                "num_qubits":        num_qubits,
                "T":                 T,
                "rzz_angle":         rzz_angle,
                "observable":        "Z_62",
                "seed":              SEED,
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            completed.add((step, delta))
            print(f"<Z62>={ev:.6f}  time={run_time:.0f}ms")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({
                    "trotter_step":   step,
                    "num_operations": num_ops,
                    "delta":          delta,
                    "error":          str(e),
                }) + "\n")
                f.flush()

print(f"\nResults saved to: {OUTPUT_FILE}")
