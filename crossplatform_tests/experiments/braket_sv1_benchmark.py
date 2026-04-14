#!/usr/bin/env python3
"""
AWS Braket SV1 Cross-Platform Benchmark
=======================================
Runs Quantum Volume circuits on Braket SV1 (on-demand state-vector simulator).

Circuit generation: quantum_volume(num_qubits, depth, seed=42 + trial)
Matches the BlueQubit script exactly.

Config: n ∈ {16..34}, depths ∈ {30, 60}, 5 trials.

TIMING NOTE
-----------
run_time_ms is extracted from task_metadata.executionDuration — the time
SV1 spent actually simulating the circuit, reported by the Braket service.
This excludes Python overhead, network latency, S3 upload/download, and
job queue wait time, all of which are recorded separately.

    executionDuration  : simulator compute time only  ← used for paper
    total_wall_ms      : full round-trip for reference
    queue_time_ms      : endedAt - createdAt - executionDuration (approx)

GATE COUNT NOTE
---------------
num_gates = qc.decompose().size() on the original Qiskit circuit, before
any translation to Braket. This is identical to the BlueQubit baseline and
serves as a circuit complexity proxy. It does NOT reflect the number of
instructions actually submitted to SV1 (which is higher due to triple
decomposition during translation), but it is the consistent cross-platform
metric used throughout this benchmark suite.

Prerequisites:
    pip install amazon-braket-sdk qiskit numpy pandas
    AWS CLI configured with Braket permissions + S3 bucket

Usage:
    # Test locally (free; wall-clock only — no executionDuration available)
    python experiments/braket_sv1_benchmark.py --local --qubits 16,18,20

    # Full run on SV1
    python experiments/braket_sv1_benchmark.py \\
        --s3-bucket amazon-braket-YOURACCOUNTID \\
        --s3-prefix qv-benchmark

    # Default output: data/braket_sv1_results.jsonl (under crossplatform_tests/)

    # Resume after interruption (same command; appends to data/ file)
"""

import argparse
import json
import time
import os
import sys
import numpy as np
from datetime import datetime

from qiskit.circuit.library import quantum_volume

from braket.circuits import Circuit as BraketCircuit
from braket.devices import LocalSimulator

# Output defaults: ../data/ (same layout as pauli_path_tests / statevector_tests)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUTPUT_BRAKET = os.path.join(DATA_DIR, "braket_sv1_results.jsonl")


# ---------------------------------------------------------------------------
# Gate translation
# ---------------------------------------------------------------------------

def qiskit_to_braket(qc) -> BraketCircuit:
    """
    Convert a Qiskit QuantumCircuit to a Braket Circuit.

    Decomposes to a known basis set first, then maps gate-by-gate.
    Raises ValueError if any gate cannot be translated, so callers
    always know the executed circuit matches the intended circuit.
    """
    qc_dec = qc.decompose().decompose().decompose()
    bc = BraketCircuit()
    unknown = []

    for inst in qc_dec.data:
        gate   = inst.operation
        qubits = [qc_dec.qubits.index(q) for q in inst.qubits]
        name   = gate.name.lower()
        params = [float(p) for p in gate.params]

        if name in ('cx', 'cnot'):
            bc.cnot(qubits[0], qubits[1])
        elif name == 'cz':
            bc.cz(qubits[0], qubits[1])
        elif name == 'h':
            bc.h(qubits[0])
        elif name == 'x':
            bc.x(qubits[0])
        elif name == 'y':
            bc.y(qubits[0])
        elif name == 'z':
            bc.z(qubits[0])
        elif name == 's':
            bc.s(qubits[0])
        elif name == 'sdg':
            bc.si(qubits[0])
        elif name == 't':
            bc.t(qubits[0])
        elif name == 'tdg':
            bc.ti(qubits[0])
        elif name == 'rx':
            bc.rx(qubits[0], params[0])
        elif name == 'ry':
            bc.ry(qubits[0], params[0])
        elif name == 'rz':
            bc.rz(qubits[0], params[0])
        elif name in ('u1', 'p'):
            bc.phaseshift(qubits[0], params[0])
        elif name == 'u2':
            # u2(φ,λ) = rz(λ) ry(π/2) rz(φ)
            bc.rz(qubits[0], params[1])
            bc.ry(qubits[0], np.pi / 2)
            bc.rz(qubits[0], params[0])
        elif name in ('u3', 'u'):
            # u3(θ,φ,λ) = rz(λ) ry(θ) rz(φ)
            bc.rz(qubits[0], params[2])
            bc.ry(qubits[0], params[0])
            bc.rz(qubits[0], params[1])
        elif name == 'swap':
            bc.swap(qubits[0], qubits[1])
        elif name in ('id', 'i', 'barrier', 'measure'):
            pass  # identity / non-unitary: skip
        else:
            unknown.append(name)

    if unknown:
        raise ValueError(
            f"Untranslatable gates encountered: {set(unknown)}. "
            "The Braket circuit does not match the intended circuit — aborting this run."
        )

    return bc


# ---------------------------------------------------------------------------
# Timing extraction
# ---------------------------------------------------------------------------

def extract_braket_timing(task, t_submit, t_done, use_local):
    """
    Return a dict with all timing fields.

    For SV1 tasks, executionDuration from task_metadata is the authoritative
    simulator compute time and is what we report as run_time_ms.

    For local simulator, executionDuration is not available; we fall back to
    wall-clock time and flag the record so downstream analysis can exclude it
    or treat it separately.
    """
    total_wall_ms = (t_done - t_submit) * 1000

    if use_local:
        return {
            'run_time_ms':   round(total_wall_ms, 2),
            'total_wall_ms': round(total_wall_ms, 2),
            'queue_time_ms': 0.0,
            'timing_source': 'wall_clock_local',  # flag: not comparable to SV1
        }

    meta = task.metadata()

    # executionDuration is a float in seconds reported by the Braket service.
    # It covers only simulator compute — no queue wait, no S3 transfer.
    exec_duration_s = meta.get('executionDuration')
    if exec_duration_s is None:
        # Unexpected: fall back to wall-clock and flag it
        return {
            'run_time_ms':   round(total_wall_ms, 2),
            'total_wall_ms': round(total_wall_ms, 2),
            'queue_time_ms': None,
            'timing_source': 'wall_clock_fallback',
        }

    run_time_ms = exec_duration_s * 1000

    # Approximate queue time from lifecycle timestamps.
    # createdAt and endedAt are ISO-8601 strings.
    try:
        created_at = datetime.fromisoformat(
            meta['createdAt'].replace('Z', '+00:00')
        )
        ended_at = datetime.fromisoformat(
            meta['endedAt'].replace('Z', '+00:00')
        )
        lifecycle_ms = (ended_at - created_at).total_seconds() * 1000
        queue_time_ms = round(max(lifecycle_ms - run_time_ms, 0), 2)
    except (KeyError, ValueError):
        queue_time_ms = None

    return {
        'run_time_ms':   round(run_time_ms, 2),    # ← simulator compute only
        'total_wall_ms': round(total_wall_ms, 2),  # ← full client round-trip
        'queue_time_ms': queue_time_ms,             # ← approx service-side wait
        'timing_source': 'executionDuration',       # ← authoritative Braket field
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

DEPTHS = [30, 60]


def run_benchmark(
    s3_bucket: str,
    s3_prefix: str,
    qubit_range: list,
    depths: list,
    num_trials: int,
    shots: int,
    output_file: str,
    use_local: bool,
):
    if use_local:
        print("Using LOCAL simulator (free, for testing)")
        print("  WARNING: local timing uses wall-clock — not comparable to SV1 results")
        device    = LocalSimulator()
        s3_folder = None
    else:
        from braket.aws import AwsDevice
        print("Using AWS Braket SV1 ($4.50/hr)")
        device    = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        s3_folder = (s3_bucket, s3_prefix)
        print(f"  S3: s3://{s3_bucket}/{s3_prefix}/")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)

    # Resume: load completed (num_qubits, depth, trial) tuples.
    # Only successful runs (run_time_ms > 0) are treated as done;
    # failed runs are eligible for retry.
    completed_runs = set()
    if os.path.exists(output_file):
        print(f"Loading existing runs from {output_file}...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get('run_time_ms', -1) > 0:
                        completed_runs.add((d['num_qubits'], d['depth'], d['trial']))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Found {len(completed_runs)} completed successful runs")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write('# Braket SV1 cross-platform benchmark\n')
            f.write('# run_time_ms = executionDuration from task_metadata (simulator compute only)\n')
            f.write('# total_wall_ms = client-side perf_counter round-trip\n')
            f.write('# queue_time_ms = approx service-side wait (endedAt - createdAt - executionDuration)\n')
            f.write('# num_gates = qc.decompose().size() on original Qiskit circuit (circuit complexity proxy, matches BlueQubit baseline)\n')

    total_configs = len(qubit_range) * len(depths) * num_trials
    remaining     = total_configs - len(completed_runs)

    print(f"\n{'='*70}")
    print(f"Braket SV1 cross-platform benchmark")
    print(f"Qubits: {qubit_range[0]}–{qubit_range[-1]}  |  Depths: {depths}")
    print(f"Trials: {num_trials}  |  Shots: {shots}")
    print(f"Circuit: quantum_volume(n, d, seed=42+trial)")
    print(f"Total: {total_configs} configs, {remaining} remaining")
    print(f"{'='*70}\n")

    total_written = 0

    for num_qubits in qubit_range:
        for depth in depths:
            for trial in range(num_trials):
                if (num_qubits, depth, trial) in completed_runs:
                    continue

                print(
                    f"Running (qubits={num_qubits}, depth={depth}, trial={trial})...",
                    end=" ", flush=True
                )

                qc = quantum_volume(num_qubits, depth, seed=42 + trial)

                # Circuit complexity proxy — identical to BlueQubit baseline.
                # Computed on the original Qiskit circuit before any translation.
                num_gates = qc.decompose().size()

                try:
                    braket_circuit = qiskit_to_braket(qc)

                    t_submit = time.perf_counter()
                    if use_local:
                        task = device.run(braket_circuit, shots=shots)
                    else:
                        task = device.run(
                            braket_circuit,
                            s3_destination_folder=s3_folder,
                            shots=shots,
                        )
                    result = task.result()
                    t_done = time.perf_counter()

                    timing = extract_braket_timing(task, t_submit, t_done, use_local)

                    run_data = {
                        'trial':      trial,
                        'num_qubits': num_qubits,
                        'depth':      depth,
                        'num_gates':  num_gates,
                        'job_id':     getattr(task, 'id', 'local'),
                        **timing,
                        'backend':    'braket_sv1' if not use_local else 'braket_local',
                    }

                    with open(output_file, 'a') as f:
                        f.write(json.dumps(run_data) + '\n')

                    print(
                        f"{timing['run_time_ms']:.0f}ms "
                        f"(wall={timing['total_wall_ms']:.0f}ms, "
                        f"src={timing['timing_source']})"
                    )
                    total_written += 1

                except ValueError as e:
                    # Translation failure — circuit mismatch, do not record as timing data
                    print(f"TRANSLATION ERROR: {e}")
                    run_data = {
                        'trial':      trial,
                        'num_qubits': num_qubits,
                        'depth':      depth,
                        'num_gates':  num_gates,
                        'job_id':     'error',
                        'run_time_ms':   -1,
                        'total_wall_ms': -1,
                        'queue_time_ms': None,
                        'timing_source': 'none',
                        'backend':       'braket_sv1',
                        'error':         str(e),
                    }
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(run_data) + '\n')

                except Exception as e:
                    print(f"FAILED: {e}")
                    run_data = {
                        'trial':      trial,
                        'num_qubits': num_qubits,
                        'depth':      depth,
                        'num_gates':  num_gates,
                        'job_id':     'error',
                        'run_time_ms':   -1,
                        'total_wall_ms': -1,
                        'queue_time_ms': None,
                        'timing_source': 'none',
                        'backend':       'braket_sv1',
                        'error':         str(e),
                    }
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(run_data) + '\n')

                time.sleep(0.3)

    print(f"\nFinished! Wrote {total_written} new runs to {output_file}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(output_file: str):
    import pandas as pd

    rows = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(rows)
    df = df[df['run_time_ms'] > 0]

    # Warn if wall-clock records slipped through
    wall_clock_rows = df[df['timing_source'] != 'executionDuration']
    if not wall_clock_rows.empty:
        print(
            f"\nWARNING: {len(wall_clock_rows)} rows use wall-clock timing "
            f"(timing_source != executionDuration) — these are NOT comparable "
            f"to SV1 executionDuration rows and should be excluded from analysis figures."
        )

    sv1_df = df[df['timing_source'] == 'executionDuration']

    print(f"\n{'='*70}")
    print("SUMMARY — run_time_ms = executionDuration (simulator compute only)")
    print("          num_gates   = qc.decompose().size() [circuit complexity proxy]")
    print(f"{'='*70}")

    for (n, d), g in sv1_df.groupby(['num_qubits', 'depth']):
        med         = g['run_time_ms'].median()
        iqr         = g['run_time_ms'].quantile(0.75) - g['run_time_ms'].quantile(0.25)
        ng          = g['num_gates'].median()
        ms_per_gate = med / ng if ng and ng > 0 else float('nan')
        print(f"n={n:2d} d={d:3d}: {med:10.1f}ms ± {iqr:7.1f} | {ms_per_gate:.4f} ms/gate")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Braket SV1 cross-platform benchmark")
    parser.add_argument("--s3-bucket", type=str, default="amazon-braket-default")
    parser.add_argument("--s3-prefix", type=str, default="qv-benchmark")
    parser.add_argument("--n-runs",    type=int, default=5)
    parser.add_argument("--shots",     type=int, default=100)
    parser.add_argument("--local",     action="store_true")
    parser.add_argument("--output",    type=str, default=DEFAULT_OUTPUT_BRAKET)
    parser.add_argument("--qubits",    type=str, default=None,
                        help="Comma-sep qubit counts (default: 16-34)")
    parser.add_argument("--depths",    type=str, default=None,
                        help="Comma-sep depths (default: 30,60)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    qubits = list(range(16, 35)) if not args.qubits else [int(x) for x in args.qubits.split(",")]
    depths = DEPTHS if not args.depths else [int(x) for x in args.depths.split(",")]

    run_benchmark(
        args.s3_bucket, args.s3_prefix,
        qubits, depths,
        args.n_runs, args.shots,
        args.output, args.local,
    )

    if os.path.exists(args.output):
        print_summary(args.output)