#!/usr/bin/env python3
"""
Quantum Rings SDK Cross-Platform Benchmark
===========================================
Circuit generation: quantum_volume(num_qubits, depth, seed=42 + trial)
Matches the BlueQubit script exactly.

Config: n ∈ {16..34}, depths ∈ {30, 60}, 5 trials.

TIMING NOTE
-----------
The Quantum Rings Result object does not expose a simulator-internal
execution time field. run_time_ms is therefore wall-clock time measured
with time.perf_counter() around job.run() → job.result(), which includes
job submission latency but excludes Python-side circuit build and transpile
time. This is the finest-grained timing available via the QrBackendV2 API.

    run_time_ms  : perf_counter wall-clock, submit → result  ← used for paper
    timing_source: always 'wall_clock' for this backend

GATE COUNT NOTE
---------------
num_gates = qc.decompose().size() on the original Qiskit circuit, before
measure_all() or transpilation. This is identical to the BlueQubit baseline
and serves as a circuit complexity proxy. It does NOT reflect the number of
instructions actually executed by the Quantum Rings simulator (which may
differ after transpilation), but it is the consistent cross-platform metric
used throughout this benchmark suite.

=== SETUP ===
    pip install QuantumRingsLib quantumrings-toolkit-qiskit qiskit numpy pandas

=== USAGE ===
    export QR_TOKEN="rings-xxx.your_key_here"
    export QR_NAME="you@email.com"

    python experiments/quantum_rings_benchmark.py --n-runs 1 --qubits 16,20 --depths 30

    # Default output: data/quantum_rings_results.jsonl (under crossplatform_tests/)
"""

import argparse
import json
import time
import os
import sys
import traceback

# Output defaults: ../data/ (same layout as pauli_path_tests / statevector_tests)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUTPUT_QR = os.path.join(DATA_DIR, "quantum_rings_results.jsonl")

# ============================================================
# CREDENTIALS — always load from environment variables.
# Never hardcode tokens in source files.
# ============================================================

def get_credentials():
    token = os.environ.get("QR_TOKEN", "").strip()
    name  = os.environ.get("QR_NAME",  "").strip()
    if not token or not name:
        print("ERROR: Quantum Rings credentials not set.")
        print("  export QR_TOKEN='rings-xxx.your_key_here'")
        print("  export QR_NAME='you@email.com'")
        sys.exit(1)
    return token, name


def safe_job_id(job) -> str:
    try:
        jid = job.job_id
        return str(jid() if callable(jid) else jid)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

DEPTHS = [30, 60]
SHOTS  = 1000


def run_benchmark(
    qubit_range: list,
    depths: list,
    num_trials: int,
    shots: int,
    output_file: str,
):
    try:
        from QuantumRingsLib import QuantumRingsProvider
        from quantumrings.toolkit.qiskit import QrBackendV2
    except ImportError as e:
        print(f"ERROR: Could not import Quantum Rings SDK: {e}")
        print("  pip install QuantumRingsLib quantumrings-toolkit-qiskit")
        sys.exit(1)

    from qiskit.circuit.library import quantum_volume
    from qiskit import transpile

    token, name = get_credentials()
    provider    = QuantumRingsProvider(token=token, name=name)

    account    = provider.active_account()
    max_qubits = int(account.get("max_qubits", 0))
    print(f"Quantum Rings account loaded (max_qubits={max_qubits})")

    if max_qubits == 0:
        print("ERROR: max_qubits=0. License key may be invalid.")
        sys.exit(1)

    qubit_range = [n for n in qubit_range if n <= max_qubits]
    if not qubit_range:
        print(f"ERROR: All qubit counts exceed max_qubits={max_qubits}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)

    # Resume: only skip runs that previously succeeded (run_time_ms > 0)
    completed_runs = set()
    if os.path.exists(output_file):
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
        print(f"Resuming: {len(completed_runs)} successful runs already done")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write('# Quantum Rings cross-platform benchmark\n')
            f.write('# run_time_ms = wall-clock perf_counter from job.run() to job.result()\n')
            f.write('# timing_source: always wall_clock (QR Result API exposes no internal simulator timing)\n')
            f.write('# num_gates = qc.decompose().size() on original Qiskit circuit (circuit complexity proxy, matches BlueQubit baseline)\n')

    total_configs = len(qubit_range) * len(depths) * num_trials
    remaining     = total_configs - len(completed_runs)

    print(f"\n{'='*70}")
    print(f"Quantum Rings cross-platform benchmark")
    print(f"Qubits: {qubit_range[0]}-{qubit_range[-1]}  |  Depths: {depths}")
    print(f"Trials: {num_trials}  |  Shots: {shots}")
    print(f"Circuit: quantum_volume(n, d, seed=42+trial)")
    print(f"Total: {total_configs} configs, {remaining} remaining")
    print(f"{'='*70}\n")

    total_written = 0

    for num_qubits in qubit_range:
        qr_backend = QrBackendV2(provider, num_qubits=num_qubits)
        print(f"--- n={num_qubits} ---")

        for depth in depths:
            for trial in range(num_trials):
                if (num_qubits, depth, trial) in completed_runs:
                    continue

                print(f"  (depth={depth}, trial={trial})...", end=" ", flush=True)

                qc = quantum_volume(num_qubits, depth, seed=42 + trial)

                # Circuit complexity proxy — identical to BlueQubit baseline.
                # Computed on the original Qiskit circuit before measure_all()
                # or transpilation, so it is not inflated by either step.
                num_gates = qc.decompose().size()

                if qc.num_clbits == 0:
                    qc.measure_all()

                try:
                    qc_transpiled = transpile(
                        qc, qr_backend,
                        initial_layout=list(range(num_qubits)),
                    )

                    t_submit = time.perf_counter()
                    job      = qr_backend.run(qc_transpiled, shots=shots)
                    result   = job.result()
                    t_done   = time.perf_counter()

                    run_time_ms = round((t_done - t_submit) * 1000, 2)

                    run_data = {
                        'trial':         trial,
                        'num_qubits':    num_qubits,
                        'depth':         depth,
                        'num_gates':     num_gates,
                        'job_id':        safe_job_id(job),
                        'run_time_ms':   run_time_ms,
                        'timing_source': 'wall_clock',
                        'backend':       'quantum_rings',
                    }

                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(run_data) + '\n')

                    print(f"{run_time_ms:.0f}ms")
                    total_written += 1

                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()
                    run_data = {
                        'trial':         trial,
                        'num_qubits':    num_qubits,
                        'depth':         depth,
                        'num_gates':     num_gates,
                        'job_id':        'error',
                        'run_time_ms':   -1,
                        'timing_source': 'none',
                        'backend':       'quantum_rings',
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

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    df = df[df['run_time_ms'] > 0]

    if df.empty:
        print("All runs failed.")
        return

    print(f"\n{'='*70}")
    print("SUMMARY — run_time_ms = wall-clock (submit → result)")
    print("          num_gates   = qc.decompose().size() [circuit complexity proxy]")
    print(f"{'='*70}")

    for (n, d), g in df.groupby(['num_qubits', 'depth']):
        med         = g['run_time_ms'].median()
        iqr         = g['run_time_ms'].quantile(0.75) - g['run_time_ms'].quantile(0.25)
        ng          = g['num_gates'].median()
        ms_per_gate = med / ng if ng and ng > 0 else float('nan')
        print(f"n={n:2d} d={d:3d}: {med:10.1f}ms ± {iqr:7.1f} | {ms_per_gate:.4f} ms/gate")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Rings cross-platform benchmark")
    parser.add_argument("--n-runs",  type=int, default=5)
    parser.add_argument("--shots",   type=int, default=SHOTS)
    parser.add_argument("--output",  type=str, default=DEFAULT_OUTPUT_QR)
    parser.add_argument("--qubits",  type=str, default=None,
                        help="Comma-sep qubit counts (default: 16-34)")
    parser.add_argument("--depths",  type=str, default=None,
                        help="Comma-sep depths (default: 30,60)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    qubits = list(range(16, 35)) if not args.qubits else [int(x) for x in args.qubits.split(",")]
    depths = DEPTHS if not args.depths else [int(x) for x in args.depths.split(",")]

    run_benchmark(qubits, depths, args.n_runs, args.shots, args.output)

    if os.path.exists(args.output):
        print_summary(args.output)