#!/usr/bin/env python3
"""
Quantum Rings MPS-Style Bond Scaling Benchmark
==============================================
Mirror of `mps_tests/experiments/bond_scaling_gpu.py`, but executed on the
Quantum Rings simulator with CUSTOM performance mode and threshold tied to the
bond dimension being tested.

Config: n=40  |  depth=10  |  threshold ∈ {32,64,128,256,384,512,768,1024,1280,1536}
Trials: 5     |  shots=1 by default (override with --shots if your QR version
               requires shots > 1)

Timing note:
    Quantum Rings does not expose a simulator-internal runtime field in the
    same shape as BlueQubit MPS. `run_time_ms` is therefore measured with
    `time.perf_counter()` around `backend.run(...) -> job.result()`.

Credentials:
    Set `QR_TOKEN` and `QR_NAME` in your environment before running.

Usage:
    python experiments/quantum_ring_mps_benchmark.py
    python experiments/quantum_ring_mps_benchmark.py --shots 10 --trials 3
"""

import argparse
import json
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "quantum_ring_mps_results.jsonl")

NUM_QUBITS = 40
DEPTH = 10
BOND_DIMS = [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536]
NUM_TRIALS = 5
SHOTS = 1
PRECISION = "double"
MODE = "sync"
PERFORMANCE = "CUSTOM"


def get_credentials():
    token = os.environ.get("QR_TOKEN", "").strip()
    name = os.environ.get("QR_NAME", "").strip()
    if not token or not name:
        print("ERROR: Quantum Rings credentials not set.")
        print("  PowerShell:")
        print("    $env:QR_TOKEN='rings-xxx.your_key_here'")
        print("    $env:QR_NAME='you@email.com'")
        sys.exit(1)
    return token, name


def safe_job_id(job) -> str:
    try:
        jid = job.job_id
        return str(jid() if callable(jid) else jid)
    except Exception:
        return "unknown"


def run_benchmark(output_file: str, depth: int, shots: int, num_trials: int):
    try:
        from QuantumRingsLib import QuantumRingsProvider
        from quantumrings.toolkit.qiskit import QrBackendV2
        from qiskit import transpile
        from qiskit.circuit.library import quantum_volume
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("  pip install QuantumRingsLib quantumrings-toolkit-qiskit qiskit")
        sys.exit(1)

    token, name = get_credentials()
    provider = QuantumRingsProvider(token=token, name=name)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    completed_runs = set()
    if os.path.exists(output_file):
        print(f"Loading existing runs from {output_file}...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    r = json.loads(s)
                    if (
                        "error" not in r
                        and r.get("depth") == depth
                        and r.get("shots") == shots
                    ):
                        completed_runs.add((r["bond_dimension"], r["trial"]))
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(completed_runs)} completed runs")
    total = len(BOND_DIMS) * num_trials
    remaining = total - len(completed_runs)

    print(f"\n{'=' * 70}")
    print("Quantum Rings MPS-Style Bond Scaling Benchmark")
    print(
        f"n={NUM_QUBITS}  |  depth={depth}  |  shots={shots}"
        f"  |  trials={num_trials}"
    )
    print(
        f"threshold range: {BOND_DIMS[0]}-{BOND_DIMS[-1]}"
        f"  ({len(BOND_DIMS)} values)"
    )
    print(
        f"precision={PRECISION}  |  mode={MODE}  |  performance={PERFORMANCE}"
    )
    print(f"Total: {total} configs, {remaining} remaining")
    print(f"{'=' * 70}\n")

    qr_backend = QrBackendV2(provider, num_qubits=NUM_QUBITS)

    for bond_dim in BOND_DIMS:
        for trial in range(num_trials):
            if (bond_dim, trial) in completed_runs:
                print(f"  Skipping (threshold={bond_dim}, trial={trial}) — already done")
                continue

            print(f"Running (threshold={bond_dim}, trial={trial})...", end=" ", flush=True)
            qc = quantum_volume(NUM_QUBITS, depth, seed=42 + trial)
            qc_dec = qc.decompose()
            num_gates = qc_dec.size()
            num_cx_gates = qc_dec.count_ops().get("cx", 0)

            if qc.num_clbits == 0:
                qc.measure_all()

            try:
                qc_transpiled = transpile(
                    qc,
                    qr_backend,
                    initial_layout=list(range(NUM_QUBITS)),
                )

                t_submit = time.perf_counter()
                job = qr_backend.run(
                    qc_transpiled,
                    shots=shots,
                    mode=MODE,
                    precision=PRECISION,
                    performance=PERFORMANCE,
                    threshold=bond_dim,
                )
                job.result()
                t_done = time.perf_counter()

                run_data = {
                    "trial": trial,
                    "num_qubits": NUM_QUBITS,
                    "depth": depth,
                    "bond_dimension": bond_dim,
                    "threshold": bond_dim,
                    "shots": shots,
                    "num_gates": num_gates,
                    "num_cx_gates": num_cx_gates,
                    "job_id": safe_job_id(job),
                    "run_time_ms": round((t_done - t_submit) * 1000.0, 2),
                    "timing_source": "wall_clock",
                    "backend": "scarlet_quantum_rings",
                    "precision": PRECISION,
                    "mode": MODE,
                    "performance": PERFORMANCE,
                }
                print(f"{run_data['run_time_ms']:.0f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
                run_data = {
                    "trial": trial,
                    "num_qubits": NUM_QUBITS,
                    "depth": depth,
                    "bond_dimension": bond_dim,
                    "threshold": bond_dim,
                    "shots": shots,
                    "num_gates": num_gates,
                    "num_cx_gates": num_cx_gates,
                    "backend": "scarlet_quantum_rings",
                    "precision": PRECISION,
                    "mode": MODE,
                    "performance": PERFORMANCE,
                    "error": str(e),
                }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_data) + "\n")

            time.sleep(0.3)

    print(f"\nDone. Results: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantum Rings MPS-style bond scaling benchmark"
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    run_benchmark(
        args.output,
        depth=args.depth,
        shots=args.shots,
        num_trials=args.trials,
    )
