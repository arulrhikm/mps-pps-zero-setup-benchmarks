#!/usr/bin/env python3
"""
MPS QFT scaling (CPU) — high-qubit QFT runtime vs n.

Circuit: full / approximate QFT via ``qiskit.synthesis.qft.synth_qft_full``
(no measurement). Fixed χ; sweep n and approximation_degree.

Config: n ∈ {4,8,...,96}  |  approximation_degree ∈ {0,1,2,3}  |  χ=64  |  shots=1
Device: mps.cpu

Usage:
    python experiments/qft_scaling_cpu.py
    # Default output: data/qft_scaling_cpu.jsonl
"""

import argparse
import json
import os
import time
from collections import deque

import bluequbit
from bluequbit.job_metadata_constants import JOB_TERMINAL_STATES, QUEUED_CPU_JOBS_LIMIT
from qiskit.synthesis.qft import synth_qft_full

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "qft_scaling_cpu.jsonl")

QUBIT_COUNTS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96]
APPROX_DEGREES = [0, 1, 2, 3]
BOND_DIMENSION = 64
NUM_TRIALS = 1
SHOTS = 1
MAX_IN_FLIGHT = QUEUED_CPU_JOBS_LIMIT
POLL_INTERVAL_S = 5.0


def get_phase_times_ms(result):
    run_results = getattr(result, "run_results", {}) or {}
    build_time_s = run_results.get("mps_build_time")
    if build_time_s is None:
        return None, None
    build_time_ms = float(build_time_s) * 1000.0
    sampling_time_ms = max(0.0, float(result.run_time_ms) - build_time_ms)
    return build_time_ms, sampling_time_ms


def is_completed_run(row, *, bond_dimension):
    job_id = str(row.get("job_id", "")).strip().lower()
    run_time_ms = row.get("run_time_ms")
    return (
        "error" not in row
        and row.get("bond_dimension") == bond_dimension
        and isinstance(run_time_ms, (int, float))
        and run_time_ms > 0
        and job_id not in {"", "error", "unknown", "none"}
    )


def write_jsonl_row(output_file: str, row: dict):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def build_success_row(meta: dict, job) -> dict:
    build_time_ms, sampling_time_ms = get_phase_times_ms(job)
    row = {
        "trial": meta["trial"],
        "num_qubits": meta["num_qubits"],
        "approximation_degree": meta["approximation_degree"],
        "bond_dimension": meta["bond_dimension"],
        "num_gates": meta["num_gates"],
        "circuit_depth": meta["circuit_depth"],
        "num_cx": meta["num_cx"],
        "num_cp": meta["num_cp"],
        "shots": SHOTS,
        "job_id": job.job_id,
        "queue_time_ms": job.queue_time_ms,
        "run_time_ms": job.run_time_ms,
    }
    if build_time_ms is not None:
        row["mps_build_time_ms"] = build_time_ms
        row["sampling_time_ms"] = sampling_time_ms
    return row


def build_error_row(meta: dict, error_message: str, job=None) -> dict:
    row = {
        "trial": meta["trial"],
        "num_qubits": meta["num_qubits"],
        "approximation_degree": meta["approximation_degree"],
        "bond_dimension": meta["bond_dimension"],
        "num_gates": meta.get("num_gates"),
        "circuit_depth": meta.get("circuit_depth"),
        "num_cx": meta.get("num_cx"),
        "num_cp": meta.get("num_cp"),
        "shots": SHOTS,
        "error": error_message,
    }
    if job is not None:
        row["job_id"] = job.job_id
        row["run_status"] = job.run_status
        if job.queue_time_ms is not None:
            row["queue_time_ms"] = job.queue_time_ms
        if job.run_time_ms is not None:
            row["run_time_ms"] = job.run_time_ms
    return row


def poll_pending_jobs(bq, pending_jobs, output_file: str, poll_interval_s: float):
    while pending_jobs:
        try:
            refreshed = bq.get(list(pending_jobs.keys()))
        except Exception as e:
            print(f"Polling warning: {e}")
            time.sleep(poll_interval_s)
            continue

        if not isinstance(refreshed, list):
            refreshed = [refreshed]

        num_finished = 0
        for job in refreshed:
            if job.run_status not in JOB_TERMINAL_STATES:
                continue

            meta = pending_jobs.pop(job.job_id)
            if job.run_status == "COMPLETED" and is_completed_run(
                {
                    "bond_dimension": meta["bond_dimension"],
                    "job_id": job.job_id,
                    "run_time_ms": job.run_time_ms,
                },
                bond_dimension=meta["bond_dimension"],
            ):
                write_jsonl_row(output_file, build_success_row(meta, job))
                print(
                    f"Completed (n={meta['num_qubits']}, apx={meta['approximation_degree']}, "
                    f"trial={meta['trial']}) {job.run_time_ms:.0f} ms"
                )
            else:
                error_message = job.error_message or (
                    f"Job {job.job_id} finished with status: {job.run_status}."
                )
                write_jsonl_row(output_file, build_error_row(meta, error_message, job))
                print(
                    f"FAILED (n={meta['num_qubits']}, apx={meta['approximation_degree']}, "
                    f"trial={meta['trial']}) {job.run_status}: {error_message}"
                )
            num_finished += 1

        if num_finished > 0:
            return

        time.sleep(poll_interval_s)


def run_benchmark(
    output_file: str,
    bond_dimension: int,
    qubit_counts: list,
    approx_degrees: list,
    num_trials: int,
    max_in_flight: int,
    poll_interval_s: float,
):
    bq = bluequbit.init(os.environ.get("BLUEQUBIT_API_TOKEN"))

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
                    if is_completed_run(r, bond_dimension=bond_dimension):
                        completed_runs.add(
                            (r["num_qubits"], r["approximation_degree"], r["trial"])
                        )
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Found {len(completed_runs)} completed runs")

    configs = [(n, apx) for n in qubit_counts for apx in approx_degrees]
    total = len(configs) * num_trials
    remaining = total - len(completed_runs)

    print(f"\n{'='*60}")
    print("MPS CPU — QFT scaling")
    print(
        f"χ={bond_dimension}  |  shots={SHOTS}  |  trials={num_trials}  |  "
        f"approx_degrees={approx_degrees}"
    )
    print(f"n: {qubit_counts[0]}…{qubit_counts[-1]} ({len(qubit_counts)} values)")
    print(f"Total: {total} configs, {remaining} remaining")
    print(f"{'='*60}\n")

    config_queue = deque()
    for n, apx in configs:
        for trial in range(num_trials):
            if (n, apx, trial) in completed_runs:
                print(f"  Skipping (n={n}, apx={apx}, trial={trial}) — already done")
                continue

            qc = synth_qft_full(num_qubits=n, approximation_degree=apx, do_swaps=True)
            qc_dec = qc.decompose()
            config_queue.append(
                {
                    "trial": trial,
                    "num_qubits": n,
                    "approximation_degree": apx,
                    "bond_dimension": bond_dimension,
                    "qc": qc,
                    "num_gates": qc_dec.size(),
                    "circuit_depth": qc_dec.depth(),
                    "num_cx": qc_dec.count_ops().get("cx", 0),
                    "num_cp": qc_dec.count_ops().get("cp", 0),
                }
            )

    pending_jobs = {}
    while config_queue or pending_jobs:
        while config_queue and len(pending_jobs) < max_in_flight:
            meta = config_queue.popleft()
            print(
                f"Submitting (n={meta['num_qubits']}, apx={meta['approximation_degree']}, "
                f"trial={meta['trial']})... ",
                end="",
                flush=True,
            )
            try:
                job = bq.run(
                    meta["qc"],
                    device="mps.cpu",
                    options={"mps_bond_dimension": meta["bond_dimension"]},
                    shots=SHOTS,
                    asynchronous=True,
                )
                pending_jobs[job.job_id] = meta
                print(f"submitted {job.job_id}")
            except Exception as e:
                print(f"ERROR: {e}")
                write_jsonl_row(output_file, build_error_row(meta, str(e)))

        if pending_jobs:
            poll_pending_jobs(bq, pending_jobs, output_file, poll_interval_s)

    print(f"\nDone. Results: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPS CPU QFT scaling")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--bond-dimension", type=int, default=BOND_DIMENSION)
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=QUBIT_COUNTS,
        help="Qubit counts (default: 4 8 … 96)",
    )
    parser.add_argument(
        "--approx-degrees",
        type=int,
        nargs="+",
        default=APPROX_DEGREES,
        help="QFT approximation degrees (default: 0 1 2 3)",
    )
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--max-in-flight", type=int, default=MAX_IN_FLIGHT)
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL_S)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    run_benchmark(
        args.output,
        bond_dimension=args.bond_dimension,
        qubit_counts=args.qubits,
        approx_degrees=args.approx_degrees,
        num_trials=args.trials,
        max_in_flight=args.max_in_flight,
        poll_interval_s=args.poll_interval,
    )
