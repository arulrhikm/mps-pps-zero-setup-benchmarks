#!/usr/bin/env python3
"""
PPS Benchmark — Qiskit pauli-prop, Rust-accelerated  (LOCAL CPU)
=================================================================
127-qubit TFI Trotter circuit on the IBM Eagle heavy-hex topology.
Sweeps the coefficient truncation threshold δ (atol) and
measures ⟨Z₆₂⟩ = Tr[ρ U† Z₆₂ U] with ρ = |0⟩⟨0|^⊗n.

Benchmarks LOCAL CPU performance of pauli-prop (Qiskit) against the
BlueQubit REMOTE GPU pauli-path results (pps_gpu_benchmark.ipynb) and
the Julia PauliPropagation.jl LOCAL CPU benchmark (pps_benchmark.jl).

KEY DESIGN DECISIONS FOR FAIRNESS:
  • Topology:  We import IBM_127_HEAVY_HEX_MAP from the same bluequbit
              source the GPU benchmark uses.  If that import fails we
              fall back to a hardcoded copy.  Edge ordering is preserved
              exactly to match the GPU benchmark's gate application order.
  • Observable: Z on qubit 62 (0-based), via SparsePauliOp little-endian.
  • Circuit:    Same loop as the GPU benchmark:
                  for step in range(N): rzz_all_edges; rx_all_qubits
  • Timing:     Python's time.perf_counter() around
              propagate_through_rotation_gates() — high-resolution monotonic
              timer, excludes circuit build and gate conversion.  Note: the
              GPU benchmark runs on a remote GPU and uses internal per-gate-
              layer timing.  Neither pauli-prop nor PauliPropagation.jl
              expose built-in timing APIs, so perf_counter / @elapsed are
              the best available options.
  • Deltas:     Configurable list `DELTAS` below; each value must have a
              matching `EXPECTED_PAULIS` max_terms (see loop) or you risk
              huge pre-allocation from the fallback cap.

Usage:
    pip install pauli-prop qiskit
    python pps_benchmark_qiskit.py              # full sweep (NUM_TRIALS per δ)
    python pps_benchmark_qiskit.py --resume     # skip completed (delta_index, trial)

Aligned with pps_cpu_benchmark.py: same DELTAS, NUM_TRIALS, circuit, observable.
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from pauli_prop.propagation import (
    circuit_to_rotation_gates,
    propagate_through_rotation_gates,
)

# ── Output file ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pps_qiskit_benchmark.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters (matching the BlueQubit GPU benchmark exactly) ─────────────────
NUM_QUBITS = 127
NUM_TROTTER_STEPS = 20
RZZ_ANGLE = -np.pi / 2
RX_ANGLE = np.pi / 4
OBS_QUBIT = 62  # measure ⟨Z₆₂⟩ (0-based)

# ── IBM Eagle 127-qubit heavy-hex topology ────────────────────────────────────
# Try to import from the canonical bluequbit source (same as GPU benchmark).
# Fall back to a hardcoded copy if the package isn't installed.
try:
    from bluequbit.library.helpers.hardware_connectivites import IBM_127_HEAVY_HEX_MAP
    TOPOLOGY_SOURCE = "bluequbit.library.helpers.hardware_connectivites"
except ImportError:
    TOPOLOGY_SOURCE = "hardcoded (fallback)"
    IBM_127_HEAVY_HEX_MAP = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (0, 14), (14, 18),
        (4, 15), (15, 22), (8, 16), (16, 26), (12, 17), (17, 30), (18, 19),
        (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),
        (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (20, 33),
        (24, 34), (34, 43), (28, 35), (35, 47), (32, 36), (36, 51), (33, 39),
        (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44),
        (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
        (51, 52), (37, 52), (38, 53), (53, 60), (42, 54), (54, 64), (46, 55),
        (55, 68), (50, 56), (56, 72), (57, 58), (58, 59), (59, 60), (60, 61),
        (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68),
        (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (58, 74), (74, 78),
        (62, 75), (75, 82), (66, 76), (76, 86), (70, 77), (77, 90), (73, 85),
        (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85),
        (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91), (80, 92),
        (92, 102), (84, 93), (93, 100), (88, 94), (94, 104), (91, 95),
        (95, 109), (96, 97), (97, 98), (98, 99), (99, 100), (100, 101),
        (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107),
        (107, 108), (108, 109), (96, 110), (110, 118), (100, 111), (111, 122),
        (104, 112), (112, 116), (109, 113), (113, 114), (114, 115),
        (115, 116), (116, 117), (117, 118), (118, 119), (119, 120),
        (120, 121), (121, 122), (122, 123), (123, 124), (124, 125),
        (125, 126), (114, 109),
    ]

# Truncation thresholds to sweep — matches the GPU benchmark results exactly.
DELTAS = [
    1.0e-2,
    5.0e-3,
    1.0e-3,
    5.0e-4,
    1.0e-4,
    5.0e-5,
    2.5e-5,
]

# Trials per δ (for error bars) — match pps_cpu_benchmark.py / pps_benchmark.jl
NUM_TRIALS = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_number(n):
    """Format an integer with comma separators."""
    return f"{n:,}"


def compute_expectation_zero_state(op: SparsePauliOp) -> float:
    """
    Compute Tr[ρ O] where ρ = |0⟩⟨0|^⊗n.

    ρ = (I+Z)^⊗n / 2^n, so only Pauli strings composed entirely of I and Z
    contribute. Each such string has trace 2^n with |0⟩⟨0|^⊗n, so the
    expectation value is the sum of all coefficients whose Pauli label
    contains only I and Z.
    """
    labels = op.paulis.to_labels()
    coeffs = np.real(op.coeffs)
    total = 0.0
    for label, coeff in zip(labels, coeffs):
        # A Pauli label like "IIZIIZ" — check it only contains I and Z
        if all(c in ("I", "Z") for c in label):
            total += coeff
    return total


# ── Resume support ────────────────────────────────────────────────────────────

def load_completed(filepath):
    """Successful runs as (delta_index, trial). Legacy rows without trial → trial 0."""
    completed = set()
    if not os.path.exists(filepath):
        return completed
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                if "delta_index" in r and "error" not in r:
                    t = int(r.get("trial", 0))
                    completed.add((int(r["delta_index"]), t))
            except Exception:
                pass
    return completed


resume = "--resume" in sys.argv
completed = load_completed(OUTPUT_FILE) if resume else set()

if not resume:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# PPS Qiskit Benchmark (pauli-prop) — <Z_62> sweep\n")
        f.write(f"# num_qubits={NUM_QUBITS}, trotter_steps={NUM_TROTTER_STEPS}\n")
        f.write(f"# rx_angle={RX_ANGLE:.6f} (pi/4), rzz_angle={RZZ_ANGLE:.6f} (-pi/2)\n")
        f.write(f"# topology_source={TOPOLOGY_SOURCE}\n")
        f.write(f"# topology_edges={len(IBM_127_HEAVY_HEX_MAP)}\n")
        f.write(f"# deltas={DELTAS}\n")
        f.write(f"# started={datetime.now().isoformat()}\n")
    print("Starting fresh.")
else:
    print(f"Resuming: {len(completed)} (delta_index, trial) pairs already complete.")

# ── Build circuit ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  Building circuit: {NUM_QUBITS} qubits, {NUM_TROTTER_STEPS} Trotter steps")
print(f"  Topology: IBM Eagle heavy-hex ({len(IBM_127_HEAVY_HEX_MAP)} edges)")
print(f"  Topology source: {TOPOLOGY_SOURCE}")
print(f"  Observable: Z_{OBS_QUBIT} (0-based)")
print(f"{'─'*65}")

# Circuit construction — matches the GPU benchmark exactly:
#   for step in range(NUM_TROTTER_STEPS):
#       for edge in IBM_127_HEAVY_HEX_MAP: qc.rzz(rzz_angle, edge[0], edge[1])
#       for i in range(num_qubits):        qc.rx(rx_angle, i)
qc = QuantumCircuit(NUM_QUBITS)
for _ in range(NUM_TROTTER_STEPS):
    for q0, q1 in IBM_127_HEAVY_HEX_MAP:
        qc.rzz(RZZ_ANGLE, q0, q1)
    for i in range(NUM_QUBITS):
        qc.rx(RX_ANGLE, i)

print(f"  Circuit depth: {qc.depth()}, gate count: {qc.size()}")

# ── Pre-convert circuit to rotation gates (done once, excluded from timing) ──
print("  Converting circuit to rotation gates (one-time cost)…")
t0 = time.time()
rot_gates = circuit_to_rotation_gates(qc)
t_convert = time.time() - t0
print(f"  Conversion done in {t_convert:.1f}s")
print(f"  Total rotation gates: {len(rot_gates.gates)}")

# ── Observable ────────────────────────────────────────────────────────────────
# Z on qubit 62.  SparsePauliOp uses little-endian convention:
# qubit 0 = rightmost character.
pauli_label = "I" * (NUM_QUBITS - 1 - OBS_QUBIT) + "Z" + "I" * OBS_QUBIT
observable = SparsePauliOp.from_list([(pauli_label, 1.0)])
print(f"  Observable: Z_{OBS_QUBIT}")

# ── Run sweep ─────────────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"  PPS-Qiskit: {NUM_TRIALS} trials per δ (same circuit as pps_cpu_benchmark.py)")
print(f"{'='*78}")
print(f"  {'idx':<4}  {'tr':<3}  {'δ':<12}  {'⟨Z₆₂⟩':<14}  {'# Paulis':<16}  "
      f"{'Time (s)':<10}  {'Trunc ‖·‖₁':<12}  St")
print(f"{'─'*78}")

for j, delta in enumerate(DELTAS):
    for trial in range(NUM_TRIALS):
        if (j, trial) in completed:
            print(f"  {j:<4}  {trial:<3}  {delta:<12.2e}  {'—':>14}  {'—':>16}  {'—':>10}  {'—':>12}  skip")
            continue

        try:
            # max_terms is required (cannot be None) and pauli-prop PRE-ALLOCATES
            # memory for this many terms upfront. We must balance:
            #   - Large enough that atol is the effective truncation (not max_terms)
            #   - Small enough that pre-allocation doesn't OOM
            EXPECTED_PAULIS = {
                1.0e-2: 800_000,
                5.0e-3: 1_200_000,
                1.0e-3: 1_800_000,
                5.0e-4: 2_400_000,
                1.0e-4: 3_000_000,
                5.0e-5: 10_000_000,
                2.5e-5: 40_000_000,
                1.0e-5: 200_000_000,
                9.0e-6: 250_000_000,
                8.0e-6: 300_000_000,
                7.0e-6: 400_000_000,
                6.0e-6: 500_000_000,
                5.0e-6: 700_000_000,
                4.5e-6: 850_000_000,
            }
            max_terms = EXPECTED_PAULIS.get(delta, 50_000_000)

            t_start = time.perf_counter()

            evolved_op, trunc_norm = propagate_through_rotation_gates(
                operator=observable,
                rot_gates=rot_gates,
                max_terms=max_terms,
                atol=delta,
                frame="h",            # Heisenberg: U† O U
            )

            t_elapsed = time.perf_counter() - t_start

            ev = compute_expectation_zero_state(evolved_op)
            np_count = evolved_op.size

            record = {
                "delta_index": j,
                "delta": delta,
                "trial": trial,
                "num_trials": NUM_TRIALS,
                "atol": delta,
                "max_terms": max_terms,
                "expectation_value": ev,
                "num_paulis": np_count,
                "truncated_norm": trunc_norm,
                "run_time_s": t_elapsed,
                "num_qubits": NUM_QUBITS,
                "num_trotter_steps": NUM_TROTTER_STEPS,
                "rzz_angle": RZZ_ANGLE,
                "rx_angle": RX_ANGLE,
                "observable": f"Z_{OBS_QUBIT}",
                "topology_source": TOPOLOGY_SOURCE,
                "package": "pauli-prop (Qiskit)",
                "timestamp": datetime.now().isoformat(),
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            completed.add((j, trial))

            print(f"  {j:<4}  {trial:<3}  {delta:<12.2e}  {ev:>14.6f}  {format_number(np_count):>16}  "
                  f"{t_elapsed:>10.1f}  {trunc_norm:>12.4f}  ✓")
            if np_count >= int(0.95 * max_terms):
                print(f"        ⚠ num_paulis is ≥95% of max_terms — δ may not be the active "
                      f"truncation; raise EXPECTED_PAULIS[{delta!r}] for fair comparison to BQ.")

        except Exception as e:
            print(f"  {j:<4}  {trial:<3}  {delta:<12.2e}  {'ERROR':>14}  {'':>16}  {'':>10}  {'':>12}  ✗")
            print(f"        └─ {e}")
            traceback.print_exc()

            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({
                    "delta_index": j,
                    "delta": delta,
                    "trial": trial,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")
                f.flush()

print(f"{'='*78}")
print(f"\nResults saved to: {OUTPUT_FILE}")

# ── Summary: mean ± std over trials per δ ─────────────────────────────────────
rows = []
with open(OUTPUT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
            if "error" not in r and "expectation_value" in r:
                rows.append(r)
        except json.JSONDecodeError:
            pass

print(f"\n{'='*80}")
print(f"{'SUMMARY (mean ± std over trials)':^80}")
print(f"{'='*80}")
print(f"{'idx':<5} {'δ':<12} {'mean ⟨Z⟩':<14} {'std ⟨Z⟩':<12} {'mean t(s)':<12} {'std t(s)':<12}")
print(f"{'─'*80}")
for j, delta in enumerate(DELTAS):
    sub = [r for r in rows if r.get("delta_index") == j]
    if not sub:
        print(f"{j:<5} {delta:<12.2e}  (no data)")
        continue
    evs = [r["expectation_value"] for r in sub]
    ts = [r["run_time_s"] for r in sub]
    print(
        f"{j:<5} {delta:<12.2e} {np.mean(evs):<14.6f} {np.std(evs, ddof=0):<12.6f} "
        f"{np.mean(ts):<12.4f} {np.std(ts, ddof=0):<12.4f}"
    )
print(f"{'='*80}")

# ── Plotting (optional): mean over trials ─────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No valid results to plot")

    delta_log = []
    ev_m, rt_m, np_m = [], [], []
    for j in range(len(DELTAS)):
        sub = [r for r in rows if r.get("delta_index") == j]
        if not sub:
            continue
        delta_log.append(-np.log10(DELTAS[j]))
        evs = [r["expectation_value"] for r in sub]
        ts = [r["run_time_s"] for r in sub]
        nps = [r["num_paulis"] for r in sub]
        ev_m.append(np.mean(evs))
        rt_m.append(np.mean(ts))
        np_m.append(np.mean(nps))

    COLOR = "#1B3FA0"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        rf"pauli-prop (Qiskit) — $\theta_X=\pi/4$, N={NUM_TRIALS} trials per δ",
        fontsize=13,
        fontweight="bold",
    )

    ax1.plot(
        delta_log, ev_m, color=COLOR, marker="o",
        linestyle="-", linewidth=1.6, markersize=7,
    )
    ax1.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
    ax1.set_ylabel(r"$\langle Z_{62} \rangle$", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Expectation (mean)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        delta_log, rt_m, color=COLOR, marker="o",
        linestyle="-", linewidth=1.6, markersize=7,
    )
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
    ax2.set_ylabel("Runtime (s)", fontsize=12)
    ax2.set_title("Runtime (mean)")
    ax2.grid(True, alpha=0.3, which="both")

    ax3.plot(
        delta_log, np_m, color=COLOR, marker="o",
        linestyle="-", linewidth=1.6, markersize=7,
    )
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
    ax3.set_ylabel("# Pauli strings", fontsize=12)
    ax3.set_title("# Paulis (mean)")
    ax3.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "pps_qiskit_benchmark.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to: {out_png}")

except ImportError:
    print("\n(Plotting skipped — pip install matplotlib for figures)")
except Exception as e:
    print(f"\n(Plotting failed: {e})")