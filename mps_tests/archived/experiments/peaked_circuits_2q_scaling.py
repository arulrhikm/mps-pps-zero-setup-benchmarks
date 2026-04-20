#!/usr/bin/env python3
"""
MPS runtime vs two-qubit gate count — peaked swept circuits (1 shot, 5 trials).

Primary inputs: ``mps_tests/input-peaked-circuits/*.qasm`` (e.g. ``swept_circuit_*_N56_tau*_..._RZZ*_CZ*_...qasm``).
Files are ordered by parsed ``tau`` (Trotter / schedule parameter in the filename), then by name.

Jobs are submitted **asynchronously** with bounded concurrency (same pattern as ``bond_scaling_*.py``):
``mps.cpu`` uses ``QUEUED_CPU_JOBS_LIMIT``; ``mps.gpu`` uses ``MAXIMUM_NUMBER_OF_JOBS_FOR_RUN``.

If ``input-peaked-circuits/`` has no ``.qasm``, falls back to the BlueQubit peaked-circuit API.

Usage::
    python experiments/peaked_circuits_2q_scaling.py --device mps.cpu
    python experiments/peaked_circuits_2q_scaling.py --device mps.gpu
    python experiments/peaked_circuits_2q_scaling.py --device mps.cpu --max-in-flight 3 --poll-interval 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import deque

import bluequbit
from bluequbit.job_metadata_constants import (
    JOB_TERMINAL_STATES,
    MAXIMUM_NUMBER_OF_JOBS_FOR_RUN,
    QUEUED_CPU_JOBS_LIMIT,
)
from qiskit import QuantumCircuit

os.environ.setdefault("BLUEQUBIT_MAIN_ENDPOINT", "https://dev.app.bluequbit.io/api/v1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPS_TESTS_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(MPS_TESTS_ROOT, "data")
INPUT_PEAKED_DIR = os.path.join(MPS_TESTS_ROOT, "input-peaked-circuits")

BOND_DIMENSION = 256
SHOTS = 1
NUM_TRIALS = 5
POLL_INTERVAL_S = 5.0

# API fallback when input-peaked-circuits/ has no .qasm
API_DIFFICULTIES = [1, 2, 3, 5, 8, 10, 15, 20]
API_CIRCUIT_IDS = [0, 1, 2, 3, 4]


def default_max_in_flight(device: str) -> int:
    if device.strip().lower() == "mps.gpu":
        return MAXIMUM_NUMBER_OF_JOBS_FOR_RUN
    return QUEUED_CPU_JOBS_LIMIT


def default_output_path(device: str) -> str:
    tag = device.replace(".", "_")
    return os.path.join(DATA_DIR, f"peaked_2q_scaling_{tag}.jsonl")


def get_phase_times_ms(result):
    run_results = getattr(result, "run_results", {}) or {}
    build_time_s = run_results.get("mps_build_time")
    if build_time_s is None:
        return None, None
    build_time_ms = float(build_time_s) * 1000.0
    sampling_time_ms = max(0.0, float(result.run_time_ms) - build_time_ms)
    return build_time_ms, sampling_time_ms


def circuit_metrics(qc: QuantumCircuit) -> tuple[int, int, int, int]:
    """Return (num_two_qubit_gates, num_qubits, total_gates, num_cx) after decomposition."""
    d = qc.decompose(reps=10)
    n2 = sum(1 for ins in d.data if len(ins.qubits) == 2)
    nq = d.num_qubits
    ng = d.size()
    cx = d.count_ops().get("cx", 0)
    return n2, nq, ng, cx


def parse_swept_qasm_filename(filename: str) -> dict:
    """
    Extract sweep metadata from ``swept_circuit_*_N{nn}_tau{t}_*_*_RZZ{r}_CZ{c}_*`` style names.
    """
    meta: dict = {}
    if m := re.search(r"_N(\d+)_", filename):
        meta["sweep_num_qubits"] = int(m.group(1))
    if m := re.search(r"_tau(\d+)_", filename):
        meta["sweep_tau"] = int(m.group(1))
    if m := re.search(r"_RZZ(\d+)_CZ(\d+)", filename):
        meta["filename_rzz_count"] = int(m.group(1))
        meta["filename_cz_count"] = int(m.group(2))
    return meta


def discover_specs(bq) -> list[dict]:
    """Build list of run specs: each has circuit_key, loader callable."""
    qasm_paths = glob.glob(os.path.join(INPUT_PEAKED_DIR, "*.qasm"))
    specs: list[dict] = []

    if qasm_paths:
        for path in qasm_paths:
            key = os.path.basename(path)
            parsed = parse_swept_qasm_filename(key)

            def _load_local(p=path):
                with open(p, encoding="utf-8") as f:
                    return QuantumCircuit.from_qasm_str(f.read())

            specs.append(
                {
                    "circuit_key": key,
                    "source": "local_qasm",
                    "path": path,
                    "parsed": parsed,
                    "loader": _load_local,
                }
            )
        specs.sort(
            key=lambda s: (
                s["parsed"].get("sweep_tau", 10**9),
                s["parsed"].get("filename_rzz_count", 0),
                s["circuit_key"],
            )
        )
        return specs

    for d in API_DIFFICULTIES:
        for cid in API_CIRCUIT_IDS:
            key = f"api_d{d}_id{cid}"

            def _load_api(diff=d, c_id=cid):
                pc = bq.get_peaked_circuit(diff, c_id)
                return pc.circuit

            specs.append(
                {
                    "circuit_key": key,
                    "source": "api",
                    "difficulty": d,
                    "circuit_id": cid,
                    "parsed": {},
                    "loader": _load_api,
                }
            )
    return specs


def is_completed_run(row: dict, *, shots: int) -> bool:
    job_id = str(row.get("job_id", "")).strip().lower()
    run_time_ms = row.get("run_time_ms")
    return (
        "error" not in row
        and row.get("circuit_key")
        and isinstance(row.get("trial"), int)
        and row.get("shots") == shots
        and isinstance(run_time_ms, (int, float))
        and run_time_ms > 0
        and job_id not in {"", "error", "unknown", "none"}
    )


def load_completed(output_file: str, *, shots: int) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                r = json.loads(s)
                if is_completed_run(r, shots=shots):
                    done.add((r["circuit_key"], r["trial"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def write_jsonl_row(output_file: str, row: dict) -> None:
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def build_row_base_from_spec(
    spec: dict,
    trial: int,
    n2: int,
    nq: int,
    ng: int,
    ncx: int,
) -> dict:
    row_base = {
        "circuit_key": spec["circuit_key"],
        "source": spec["source"],
        "trial": trial,
        "num_qubits": nq,
        "bond_dimension": BOND_DIMENSION,
        "shots": SHOTS,
        "num_two_qubit_gates": n2,
        "num_gates": ng,
        "num_cx_gates": ncx,
    }
    if spec["source"] == "api":
        row_base["difficulty"] = spec["difficulty"]
        row_base["circuit_id"] = spec["circuit_id"]
    if spec["source"] == "local_qasm":
        row_base["qasm_path"] = spec["path"]
        for k, v in spec.get("parsed", {}).items():
            row_base[k] = v
    return row_base


def build_success_row(meta: dict, job) -> dict:
    build_time_ms, sampling_time_ms = get_phase_times_ms(job)
    row = {
        **meta["row_base"],
        "job_id": job.job_id,
        "queue_time_ms": job.queue_time_ms,
        "run_time_ms": job.run_time_ms,
    }
    if build_time_ms is not None:
        row["mps_build_time_ms"] = build_time_ms
        row["sampling_time_ms"] = sampling_time_ms
    return row


def build_error_row(meta: dict, error_message: str, job=None) -> dict:
    row = {**meta["row_base"], "error": error_message}
    if job is not None:
        row["job_id"] = job.job_id
        row["run_status"] = getattr(job, "run_status", None)
        if job.queue_time_ms is not None:
            row["queue_time_ms"] = job.queue_time_ms
        if job.run_time_ms is not None:
            row["run_time_ms"] = job.run_time_ms
    return row


def poll_pending_jobs(
    bq,
    pending_jobs: dict,
    output_file: str,
    poll_interval_s: float,
    *,
    shots: int,
) -> None:
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
            ck = meta["row_base"]["circuit_key"]
            tr = meta["row_base"]["trial"]
            if job.run_status == "COMPLETED" and is_completed_run(
                {
                    "circuit_key": ck,
                    "trial": tr,
                    "shots": shots,
                    "job_id": job.job_id,
                    "run_time_ms": job.run_time_ms,
                },
                shots=shots,
            ):
                write_jsonl_row(output_file, build_success_row(meta, job))
                print(f"Completed ({ck[:48]}… trial={tr}) {job.run_time_ms:.0f} ms")
            else:
                error_message = job.error_message or (
                    f"Job {job.job_id} finished with status: {job.run_status}."
                )
                write_jsonl_row(output_file, build_error_row(meta, error_message, job))
                print(f"FAILED ({ck[:48]}… trial={tr}) {job.run_status}: {error_message}")
            num_finished += 1

        if num_finished > 0:
            return

        time.sleep(poll_interval_s)


def write_header(f, device: str) -> None:
    f.write(
        f"# MPS peaked-circuit 2-qubit gate scaling | device={device} | shots={SHOTS} | chi={BOND_DIMENSION}\n"
    )


def run_benchmark(
    device: str,
    output_file: str,
    *,
    max_in_flight: int,
    poll_interval_s: float,
) -> None:
    bq = bluequbit.init(os.environ.get("BLUEQUBIT_API_TOKEN"))
    specs = discover_specs(bq)
    if not specs:
        raise SystemExit(
            f"No circuits: add *.qasm under {INPUT_PEAKED_DIR} or use API (check token / network)."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    completed = load_completed(output_file, shots=SHOTS)
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            write_header(f, device)

    total_jobs = len(specs) * NUM_TRIALS
    remaining = total_jobs - len(completed)

    print(f"\n{'=' * 60}")
    print("MPS peaked-circuit scaling vs #2-qubit gates (async batch)")
    print(f"device={device}  |  χ={BOND_DIMENSION}  |  shots={SHOTS}  |  trials={NUM_TRIALS}")
    print(f"max_in_flight={max_in_flight}  |  poll_interval_s={poll_interval_s}")
    print(f"Circuits: {len(specs)} (source={'local QASM' if specs[0]['source'] == 'local_qasm' else 'API'})")
    print(f"Total: {total_jobs} jobs, ~{remaining} remaining")
    print(f"Output: {output_file}")
    print(f"{'=' * 60}\n")

    config_queue: deque = deque()

    for spec in specs:
        key = spec["circuit_key"]
        try:
            qc = spec["loader"]()
        except Exception as e:
            print(f"  SKIP load {key}: {e}")
            row = {
                "circuit_key": key,
                "source": spec["source"],
                "trial": 0,
                "bond_dimension": BOND_DIMENSION,
                "shots": SHOTS,
                "error": f"load_failed: {e}",
            }
            if spec["source"] == "api":
                row["difficulty"] = spec["difficulty"]
                row["circuit_id"] = spec["circuit_id"]
            write_jsonl_row(output_file, row)
            continue

        n2, nq, ng, ncx = circuit_metrics(qc)
        for trial in range(NUM_TRIALS):
            if (key, trial) in completed:
                print(f"  Skipping ({key[:48]}… trial={trial}) — already done")
                continue
            config_queue.append(
                {
                    "qc": qc,
                    "row_base": build_row_base_from_spec(spec, trial, n2, nq, ng, ncx),
                }
            )

    pending_jobs: dict = {}
    while config_queue or pending_jobs:
        while config_queue and len(pending_jobs) < max_in_flight:
            meta = config_queue.popleft()
            rb = meta["row_base"]
            ck_short = rb["circuit_key"][:48]
            print(
                f"Submitting ({ck_short}… trial={rb['trial']})... ",
                end="",
                flush=True,
            )
            try:
                job = bq.run(
                    meta["qc"],
                    device=device,
                    options={"mps_bond_dimension": BOND_DIMENSION},
                    shots=SHOTS,
                    asynchronous=True,
                )
                pending_jobs[job.job_id] = meta
                print(f"submitted {job.job_id}")
            except Exception as e:
                print(f"ERROR: {e}")
                write_jsonl_row(output_file, build_error_row(meta, str(e)))

        if pending_jobs:
            poll_pending_jobs(
                bq,
                pending_jobs,
                output_file,
                poll_interval_s,
                shots=SHOTS,
            )

    print(f"\nDone. Results: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MPS 2-qubit gate scaling (peaked circuits)")
    parser.add_argument("--device", type=str, default="mps.cpu", help="mps.cpu or mps.gpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Concurrent async jobs (default: CPU queue limit or GPU batch limit from SDK)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
        help="Seconds between polls when waiting on jobs",
    )
    args = parser.parse_args()
    out = args.output or default_output_path(args.device)
    cap = args.max_in_flight if args.max_in_flight is not None else default_max_in_flight(args.device)
    os.makedirs(DATA_DIR, exist_ok=True)
    run_benchmark(
        args.device,
        out,
        max_in_flight=cap,
        poll_interval_s=args.poll_interval,
    )


if __name__ == "__main__":
    main()
