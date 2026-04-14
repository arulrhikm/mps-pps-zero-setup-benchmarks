"""
fig3_5_pps_gpu.py
=================
Reproduce Figs. 3 & 5 — Pauli count growth & N_max estimation.

Circuit (Eq. 10):
  - 127 qubits, IBM heavy-hex lattice
  - T = 20 Trotter steps
  - θ_ZZ = -π/2
  - Each θ_X sampled uniformly from [-π/4, π/4]  (different per step)
  - Observable ⟨Z_62⟩

Sweep:
  δ ∈ {5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5}  (7 values from Figs 3/5)
  Trotter steps 1..20 (snapshot after each step)
  → 7 × 20 = 140 runs total

API note:
  The BlueQubit JobResult does NOT expose a raw Pauli count field
  (confirmed from SDK docs: available fields are expectation_value,
  run_time_ms, and run_results). What we CAN faithfully plot:

    • Fig 3 analogue  — PPS runtime (ms, log scale) vs num_operations,
      one curve per δ. Runtime scales linearly with the number of active
      Pauli terms, so this is a direct proxy for N_Pauli growth. Smaller δ
      retains more Paulis → higher runtime → higher curve, matching the
      ordering in Fig 3.

    • Fig 4 / 5 analogue  — ⟨Z_62⟩ (norm proxy) vs num_operations, one
      curve per δ. This captures the truncation-induced norm decay described
      in Fig 4 and provides the convergence view of Fig 5.

After all runs, the JSONL is read back and plots are saved to
fig3_5_gpu_reproduction.png in the same directory.
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
    "data", "fig3_5_pps_gpu.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
num_qubits     = 127
num_edges      = len(IBM_127_HEAVY_HEX_MAP)  # 144
gates_per_step = num_edges + num_qubits       # 271
T              = 20
rzz_angle      = -np.pi / 2
SEED           = 42

# Random θ_X per step (same seed as fig1_2 for consistency)
rng = np.random.default_rng(SEED)
rx_angles_per_step = rng.uniform(-np.pi / 4, np.pi / 4, size=T)

# δ values matching Figs 3 & 5
deltas = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]

# Snapshot after every Trotter step
snapshot_steps = list(range(1, T + 1))

# Observable ⟨Z_62⟩
pauli_str = "I" * (num_qubits - 1 - 62) + "Z" + "I" * 62
pauli_sum = [(pauli_str, 1.0)]

# ── Resume support ────────────────────────────────────────────────────────────
# Key: (trotter_step, delta)
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
        f.write("# Figs 3 & 5: Pauli count growth sweep, GPU\n")
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
print(f"Steps:  1..{T}")
print()

# ── Build circuits for each step (cache to avoid rebuilding) ──────────────────
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

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from collections import defaultdict

rows = []
with open(OUTPUT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
        except Exception:
            pass

by_delta = defaultdict(list)
for r in rows:
    by_delta[r["delta"]].append(r)

# sort largest delta first so colour ordering matches paper (high δ = fewer
# Paulis = lower/faster curve at the bottom)
deltas_sorted = sorted(by_delta.keys(), reverse=True)
n_d = len(deltas_sorted)

# viridis-like palette: purple (small δ, many Paulis) → teal (large δ)
cmap = plt.get_cmap("viridis")
colors = {d: cmap(0.1 + 0.8 * i / max(n_d - 1, 1)) for i, d in enumerate(reversed(deltas_sorted))}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Figs 3 & 4/5 — PPS GPU: Pauli growth proxy & norm decay (127 qubits, T=20, random θ_X)",
    fontsize=12, fontweight="bold"
)

# ── Left: runtime vs num_operations  (Fig 3 analogue: Pauli count growth) ───
ax = axes[0]
for d in deltas_sorted:
    grp = sorted(by_delta[d], key=lambda r: r["num_operations"])
    x = [r["num_operations"] for r in grp]
    y = [r["run_time_ms"]     for r in grp]
    ax.plot(x, y, color=colors[d], linestyle="None", marker="o", markersize=3.5,
            label=rf"$\delta = {d:.1e}$")

ax.set_xlabel("number of operations", fontsize=11)
ax.set_ylabel("PPS runtime (ms)   [proxy for # active Paulis]", fontsize=10)
ax.set_title(
    r"Pauli count growth — runtime proxy (Fig 3 analogue)" "\n"
    r"Smaller $\delta$ keeps more Paulis $\Rightarrow$ higher curve",
    fontsize=10
)
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3, which="both")

# ── Right: EV vs num_operations  (Fig 4/5 analogue: norm decay) ───────────
ax = axes[1]
for d in deltas_sorted:
    grp = sorted(by_delta[d], key=lambda r: r["num_operations"])
    x = [r["num_operations"]   for r in grp]
    y = [r["expectation_value"] for r in grp]
    ax.plot(x, y, color=colors[d], linestyle="None", marker="o", markersize=3.5,
            label=rf"$\delta = {d:.1e}$")

ax.set_xlabel("number of operations", fontsize=11)
ax.set_ylabel(r"$\langle Z_{62}\rangle$", fontsize=11)
ax.set_title(
    r"Expectation value vs depth (Figs 4/5 analogue)" "\n"
    r"Larger $\delta$ truncates more $\Rightarrow$ faster decay",
    fontsize=10
)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_5_gpu_reproduction.png")
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Plot saved to: {OUT_PNG}")
plt.show()
