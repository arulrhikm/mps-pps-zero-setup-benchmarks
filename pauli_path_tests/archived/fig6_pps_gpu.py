"""
fig6_pps_gpu.py
===============
Reproduce Fig. 6 — Convergence of ⟨Z_62⟩ with randomly sampled θ_X.

Circuit (Eq. 10):
  - 127 qubits, IBM heavy-hex lattice
  - θ_ZZ = -π/2
  - Each θ_X sampled uniformly from [-π/4, π/4] (different per Trotter step)
  - Observable ⟨Z_62⟩

Sweep:
  T ∈ {12, 16, 20, 24}   (Trotter steps / depth)
  δ = [1/2^i for i in range(21)]  → 21 values from 1.0 down to ~10^-6
  → 4 × 21 = 84 runs total

The two rows of Fig 6 correspond to plotting subsets of this data
with different convergence thresholds:
  Top row:    ε_tol = 0.01  (converged when successive O_k agree within 0.01)
  Bottom row: ε_tol = 0.001 (stricter — need finer δ to converge)

Left panels:  O_k ≈ ⟨Z_62⟩ vs δ_k    (expectation value convergence)
Right panels: runtime (s) vs δ_k       (power-law scaling)
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
    "data", "fig6_pps_gpu.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
num_qubits     = 127
num_edges      = len(IBM_127_HEAVY_HEX_MAP)  # 144
gates_per_step = num_edges + num_qubits       # 271
rzz_angle      = -np.pi / 2
SEED           = 42

# Trotter step counts from Fig 6
trotter_steps = [12, 16, 20, 24]

# Need enough random rx angles for the largest T
T_max = max(trotter_steps)
rng = np.random.default_rng(SEED)
rx_angles_per_step = rng.uniform(-np.pi / 4, np.pi / 4, size=T_max)

# δ sweep: 1/2^i from i=0 to i=20  → δ from 1.0 to ~10^-6
deltas = [1 / 2**i for i in range(21)]

# Observable ⟨Z_62⟩
pauli_str = "I" * (num_qubits - 1 - 62) + "Z" + "I" * 62
pauli_sum = [(pauli_str, 1.0)]

# ── Resume support ────────────────────────────────────────────────────────────
# Key: (T, delta_index)
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
                    completed.add((r["trotter_steps"], r["delta_index"]))
            except Exception:
                pass
    print(f"Resuming: {len(completed)} runs already complete.")
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Fig 6: Convergence with random theta_X, GPU\n")
        f.write(f"# num_qubits={num_qubits}, trotter_steps={trotter_steps}\n")
        f.write(f"# deltas=[1/2^i for i in range({len(deltas)})]\n")
        f.write(f"# rx sampled from [-pi/4, pi/4], seed={SEED}\n")
    print("Starting fresh.")

# ── Print plan ────────────────────────────────────────────────────────────────
total = len(trotter_steps) * len(deltas)
remaining = sum(
    1 for T in trotter_steps for j in range(len(deltas))
    if (T, j) not in completed
)
print(f"\nTotal runs: {total}  ({remaining} remaining)")
print(f"T values:   {trotter_steps}")
print(f"δ range:    {deltas[0]:.0e} → {deltas[-1]:.2e}  ({len(deltas)} values)")
print()

# ── Build circuits ────────────────────────────────────────────────────────────
print("Building circuits...")
circuits = {}
for T in trotter_steps:
    qc = QuantumCircuit(num_qubits)
    for t in range(T):
        for edge in IBM_127_HEAVY_HEX_MAP:
            qc.rzz(rzz_angle, edge[0], edge[1])
        for i in range(num_qubits):
            qc.rx(rx_angles_per_step[t], i)
    circuits[T] = qc
    print(f"  T={T:2d}: {T * gates_per_step} gates")
print()

# ── Run ───────────────────────────────────────────────────────────────────────
for T in trotter_steps:
    qc = circuits[T]
    num_ops = T * gates_per_step

    print(f"\n{'='*60}")
    print(f"T = {T}  ({num_ops} gates)")
    print(f"{'='*60}")

    for j, delta in enumerate(deltas):
        if (T, j) in completed:
            print(f"  ✓ skip  T={T}  δ=2^-{j}")
            continue

        print(f"  → T={T:2d}  δ=2^-{j}={delta:.2e}  ", end="", flush=True)
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
                "trotter_steps":     T,
                "num_operations":    num_ops,
                "delta_index":       j,
                "delta":             delta,
                "expectation_value": ev,
                "run_time_ms":       run_time,
                "job_id":            result.job_id,
                "num_qubits":        num_qubits,
                "rzz_angle":         rzz_angle,
                "rx_mode":           "random",
                "observable":        "Z_62",
                "seed":              SEED,
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            completed.add((T, j))
            print(f"<Z62>={ev:.6f}  time={run_time:.0f}ms")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({
                    "trotter_steps": T,
                    "delta_index":   j,
                    "delta":         delta,
                    "error":         str(e),
                }) + "\n")
                f.flush()

print(f"\nResults saved to: {OUTPUT_FILE}")
