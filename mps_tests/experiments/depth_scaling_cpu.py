#!/usr/bin/env python3
"""
MPS Depth Scaling (CPU)
=====================================
Runtime vs circuit depth at fixed qubits and bond dimension.
Verifies that runtime scales linearly with depth (runtime/gate is flat).

Config: n=40  |  depth ∈ {4,8,12,16,24,32,40,48,56,64}  |  χ=128  |  shots=1000
Device: mps.cpu

Usage:
    python experiments/depth_scaling_cpu.py
    # Default output: data/depth_scaling_cpu.jsonl
"""

import argparse
import json
import os

import bluequbit
from qiskit.circuit.library import quantum_volume

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "depth_scaling_cpu256.jsonl")

NUM_QUBITS = 40
BOND_DIMENSION = 256
DEPTHS = [4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
NUM_TRIALS = 5
SHOTS = 1


def get_phase_times_ms(result):
    run_results = getattr(result, "run_results", {}) or {}
    build_time_s = run_results.get("mps_build_time")
    if build_time_s is None:
        return None, None
    build_time_ms = float(build_time_s) * 1000.0
    sampling_time_ms = max(0.0, float(result.run_time_ms) - build_time_ms)
    return build_time_ms, sampling_time_ms


def is_completed_run(row):
    job_id = str(row.get("job_id", "")).strip().lower()
    run_time_ms = row.get("run_time_ms")
    return (
        "error" not in row
        and isinstance(run_time_ms, (int, float))
        and run_time_ms > 0
        and job_id not in {"", "error", "unknown", "none"}
    )


def run_benchmark(output_file: str):
    bq = bluequbit.init()

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
                    if is_completed_run(r):
                        completed_runs.add((r["depth"], r["trial"]))
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(completed_runs)} completed runs")
    total = len(DEPTHS) * NUM_TRIALS
    remaining = total - len(completed_runs)

    print(f"\n{'='*60}")
    print(f"MPS CPU Depth Scaling Benchmark")
    print(f"n={NUM_QUBITS}  |  χ={BOND_DIMENSION}  |  shots={SHOTS}  |  trials={NUM_TRIALS}")
    print(f"Depths: {DEPTHS}")
    print(f"Total: {total} configs, {remaining} remaining")
    print(f"{'='*60}\n")

    for depth in DEPTHS:
        for trial in range(NUM_TRIALS):
            if (depth, trial) in completed_runs:
                print(f"  Skipping (depth={depth}, trial={trial}) — already done")
                continue

            print(f"Running (depth={depth}, trial={trial})...", end=" ", flush=True)
            qc = quantum_volume(NUM_QUBITS, depth, seed=42 + trial)
            qc_dec = qc.decompose()
            num_gates = qc_dec.size()
            num_cx_gates = qc_dec.count_ops().get("cx", 0)

            try:
                job = bq.run(qc, device="mps.cpu",
                             options={"mps_bond_dimension": BOND_DIMENSION},
                             shots=SHOTS)
                build_time_ms, sampling_time_ms = get_phase_times_ms(job)
                run_data = {
                    "trial": trial,
                    "num_qubits": NUM_QUBITS,
                    "depth": depth,
                    "bond_dimension": BOND_DIMENSION,
                    "shots": SHOTS,
                    "num_gates": num_gates,
                    "num_cx_gates": num_cx_gates,
                    "job_id": job.job_id,
                    "queue_time_ms": job.queue_time_ms,
                    "run_time_ms": job.run_time_ms,
                }
                if build_time_ms is not None:
                    run_data["mps_build_time_ms"] = build_time_ms
                    run_data["sampling_time_ms"] = sampling_time_ms
                print(f"{job.run_time_ms:.0f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
                run_data = {
                    "trial": trial,
                    "num_qubits": NUM_QUBITS,
                    "depth": depth,
                    "bond_dimension": BOND_DIMENSION,
                    "shots": SHOTS,
                    "num_gates": num_gates,
                    "num_cx_gates": num_cx_gates,
                    "error": str(e),
                }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_data) + "\n")

    print(f"\nDone. Results: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPS CPU Depth Scaling")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    run_benchmark(args.output)
