#!/usr/bin/env python3
"""
Single-circuit peaked overlap vs bond dimension (synchronous, resumable).

Required arguments:
  - --device {mps.cpu,mps.gpu}
  - one of:
      --rzz-gates <int>   (recommended)
      --input-circuit <path under mps_tests/input-peaked-circuits>

Default metric behavior uses the notebook's "MPS marginal attack" strategy:
  - Run with pauli_sum over local-Z observables.
  - Convert each local <Z_i> to bit i via sign (<0 -> 1, >=0 -> 0).
  - Compare inferred bitstring vs target bitstring for overlap.

This script uses one MPS sample/run per bond dimension by default (shots=1).

Results are written to:
  mps_tests/data/peaked-circuits-results/{CPU|GPU}_peaked_overlap_bond_dimension.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import bluequbit
from qiskit import QuantumCircuit

os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

SCRIPT_DIR = Path(__file__).resolve().parent
MPS_TESTS_ROOT = SCRIPT_DIR.parent
INPUT_PEAKED_DIR = MPS_TESTS_ROOT / "input-peaked-circuits"
RESULTS_DIR = MPS_TESTS_ROOT / "data" / "peaked-circuits-results"

BOND_DIMS_CPU = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
BOND_DIMS_GPU = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
NUM_TRIALS = 5
SHOTS = 1


def parse_target_from_filename(filename: str) -> str | None:
    m = re.search(r"target=([01]+)", filename)
    return m.group(1) if m else None


def parse_swept_qasm_filename(filename: str) -> dict:
    meta: dict = {}
    if m := re.search(r"_N(\d+)_", filename):
        meta["sweep_num_qubits"] = int(m.group(1))
    if m := re.search(r"_tau(\d+)_", filename):
        meta["sweep_tau"] = int(m.group(1))
    if m := re.search(r"_RZZ(\d+)_CZ(\d+)", filename):
        meta["filename_rzz_count"] = int(m.group(1))
        meta["filename_cz_count"] = int(m.group(2))
    return meta


def circuit_metrics(qc: QuantumCircuit) -> tuple[int, int, int, int]:
    d = qc.decompose(reps=10)
    n2 = sum(1 for ins in d.data if len(ins.qubits) == 2)
    return n2, d.num_qubits, d.size(), d.count_ops().get("cx", 0)


def normalize_bits(bits: str, expected_bits: int) -> str:
    b = bits.replace(" ", "").strip()
    if len(b) < expected_bits:
        return b.zfill(expected_bits)
    if len(b) > expected_bits:
        return b[-expected_bits:]
    return b


def overlap_percent(bits: str, target: str, expected_bits: int) -> float:
    b = normalize_bits(bits, expected_bits)
    t = normalize_bits(target, expected_bits)
    matches = sum(1 for x, y in zip(b, t) if x == y)
    return 100.0 * matches / expected_bits


def result_to_local_z_expectations(run_obj, bq) -> list[float]:
    """Extract local-Z expectation values (pauli_sum result) from run object."""
    if hasattr(run_obj, "expectation_value"):
        ev = getattr(run_obj, "expectation_value")
        if isinstance(ev, list):
            return [float(x) for x in ev]

    if hasattr(run_obj, "result") and callable(run_obj.result):
        res = run_obj.result()
        if hasattr(res, "expectation_value"):
            ev = getattr(res, "expectation_value")
            if isinstance(ev, list):
                return [float(x) for x in ev]

    job_id = getattr(run_obj, "job_id", None)
    if job_id:
        refreshed = bq.get(job_id)
        if isinstance(refreshed, list):
            refreshed = refreshed[0]
        if hasattr(refreshed, "expectation_value"):
            ev = getattr(refreshed, "expectation_value")
            if isinstance(ev, list):
                return [float(x) for x in ev]
        if hasattr(refreshed, "result") and callable(refreshed.result):
            res = refreshed.result()
            if hasattr(res, "expectation_value"):
                ev = getattr(res, "expectation_value")
                if isinstance(ev, list):
                    return [float(x) for x in ev]

    raise RuntimeError("Could not extract local-Z expectations from BlueQubit run object")


def get_phase_times_ms(run_obj):
    run_results = getattr(run_obj, "run_results", {}) or {}
    build_s = run_results.get("mps_build_time")
    if build_s is None:
        return None, None
    build_ms = float(build_s) * 1000.0
    run_ms = float(getattr(run_obj, "run_time_ms", 0.0) or 0.0)
    return build_ms, max(0.0, run_ms - build_ms)


def z_to_bit(z_val: float) -> str:
    return "1" if z_val < 0 else "0"


def metrics_from_local_z(zs: list[float], target_bits: str, expected_bits: int) -> dict:
    if len(zs) != expected_bits:
        raise ValueError(
            f"Expected {expected_bits} local-Z values, got {len(zs)}. "
            "Cannot compute marginal-attack overlap with mismatched bit length."
        )
    key = "".join(z_to_bit(float(z)) for z in zs)
    key_norm = normalize_bits(key, expected_bits)
    target_norm = normalize_bits(target_bits, expected_bits)
    ov = overlap_percent(key_norm, target_norm, expected_bits)
    matched = sum(1 for x, y in zip(key_norm, target_norm) if x == y)
    return {
        "measured_bitstring": key_norm,
        "target_bitstring": target_norm,
        "matched_bits": matched,
        "total_bits": expected_bits,
        "overlap_percent": ov,
        # Keep legacy field for plot compatibility.
        "dominant_overlap_percent": ov,
        "weighted_overlap_percent": ov,
        "target_hit_rate_percent": 100.0 if key_norm == target_norm else 0.0,
        "top_count": 1,
        "num_unique_bitstrings": 1,
        "total_counts": 1,
        "marginal_attack_key": key_norm,
        "local_z_expectations": [float(z) for z in zs],
    }


def is_completed(row: dict, circuit_key: str, shots: int) -> bool:
    return (
        row.get("circuit_key") == circuit_key
        and isinstance(row.get("trial"), int)
        and isinstance(row.get("bond_dimension"), int)
        and row.get("shots") == shots
        and "error" not in row
        and isinstance(row.get("run_time_ms"), (int, float))
        and row.get("run_time_ms", 0) > 0
        and isinstance(row.get("dominant_overlap_percent"), (int, float))
    )


def load_completed(output_file: Path, circuit_key: str, shots: int) -> dict[tuple[int, int], dict]:
    done: dict[tuple[int, int], dict] = {}
    if not output_file.exists():
        return done
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            if is_completed(r, circuit_key=circuit_key, shots=shots):
                done[(int(r["bond_dimension"]), int(r["trial"]))] = r
    return done


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", name)


def default_output_for(device: str) -> Path:
    prefix = "GPU" if device == "mps.gpu" else "CPU"
    return RESULTS_DIR / f"{prefix}_peaked_overlap_bond_dimension.jsonl"


def resolve_input_circuit(input_circuit_arg: str) -> Path:
    p = Path(input_circuit_arg)
    if not p.is_absolute():
        p = INPUT_PEAKED_DIR / p
    p = p.resolve()
    try:
        p.relative_to(INPUT_PEAKED_DIR.resolve())
    except ValueError as e:
        raise SystemExit(f"--input-circuit must be inside {INPUT_PEAKED_DIR}") from e
    if not p.exists() or not p.is_file():
        raise SystemExit(f"Input circuit not found: {p}")
    return p


def resolve_by_rzz_gates(rzz_gates: int) -> Path:
    paths = sorted(
        set(INPUT_PEAKED_DIR.glob("*.qasm")) | set(INPUT_PEAKED_DIR.glob("*.qasm_*"))
    )
    matched = []
    token = f"_RZZ{int(rzz_gates)}_"
    for p in paths:
        if token in p.name:
            matched.append(p)

    if not matched:
        raise SystemExit(
            f"No circuit found in {INPUT_PEAKED_DIR} with RZZ={rzz_gates} in filename."
        )
    if len(matched) > 1:
        names = "\n".join(f"  - {p.name}" for p in matched)
        raise SystemExit(
            f"Multiple circuits matched RZZ={rzz_gates}; please disambiguate with --input-circuit:\n{names}"
        )
    return matched[0].resolve()


def run_benchmark(device: str, input_circuit: Path, output_file: Path, bond_dims: list[int], shots: int, trials: int) -> None:
    bq = bluequbit.init(os.environ.get("BLUEQUBIT_API_TOKEN"))

    circuit_key = input_circuit.name
    target_bits = parse_target_from_filename(circuit_key)
    if not target_bits:
        raise SystemExit(f"Could not parse target=<bits> from filename: {circuit_key}")
    parsed = parse_swept_qasm_filename(circuit_key)

    with input_circuit.open(encoding="utf-8") as f:
        qc = QuantumCircuit.from_qasm_str(f.read())
    n2, nq, ng, ncx = circuit_metrics(qc)
    pauli_observable = [
        "I" * (nq - qubit_no - 1) + "Z" + "I" * qubit_no
        for qubit_no in range(nq)
    ]
    expected_bits = int(parsed.get("sweep_num_qubits", nq))
    if expected_bits <= 0:
        expected_bits = nq

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_file, circuit_key=circuit_key, shots=shots)
    if not output_file.exists():
        with output_file.open("w", encoding="utf-8") as f:
            f.write(
                f"# Single-circuit peaked overlap | device={device} | shots={shots} | trials={trials} | circuit={circuit_key}\n"
            )

    print(f"\n{'=' * 80}")
    print("Single-circuit peaked overlap vs bond dimension")
    print(f"device={device} | shots={shots} | trials={trials}")
    print(f"input={input_circuit}")
    print(f"output={output_file}")
    print(f"n={nq} (name N={parsed.get('sweep_num_qubits', nq)}), two_qubit={n2}")
    print(f"resume rows={len(completed)}")
    print(f"{'=' * 80}\n")

    for bond_dim in bond_dims:
        dominant_vals = []
        for t in range(trials):
            key = (bond_dim, t)
            if key in completed:
                dominant_vals.append(float(completed[key]["dominant_overlap_percent"]))
                print(f"Skipping (chi={bond_dim}, trial={t}) — already done")
                continue

            print(f"Submitting (chi={bond_dim}, trial={t})... ", end="", flush=True)
            row_base = {
                "circuit_key": circuit_key,
                "source": "local_qasm",
                "trial": t,
                "bond_dimension": int(bond_dim),
                "shots": int(shots),
                "num_qubits": nq,
                "num_two_qubit_gates": n2,
                "num_gates": ng,
                "num_cx_gates": ncx,
                "qasm_path": str(input_circuit),
                **parsed,
            }
            try:
                run_obj = bq.run(
                    qc,
                    device=device,
                    options={"mps_bond_dimension": int(bond_dim)},
                    pauli_sum=pauli_observable,
                    shots=int(shots),
                    asynchronous=False,
                )
                local_zs = result_to_local_z_expectations(run_obj, bq)
                metrics = metrics_from_local_z(local_zs, target_bits, expected_bits)
                build_ms, sampling_ms = get_phase_times_ms(run_obj)
                row = {
                    **row_base,
                    **metrics,
                    "job_id": getattr(run_obj, "job_id", None),
                    "queue_time_ms": getattr(run_obj, "queue_time_ms", None),
                    "run_time_ms": getattr(run_obj, "run_time_ms", None),
                }
                if build_ms is not None:
                    row["mps_build_time_ms"] = build_ms
                    row["sampling_time_ms"] = sampling_ms
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                dominant_vals.append(float(row["dominant_overlap_percent"]))
                print(
                    f"done | overlap={row['dominant_overlap_percent']:.2f}% "
                    f"matched={row['matched_bits']}/{row['total_bits']}"
                )
            except Exception as e:
                err = {**row_base, "error": str(e)}
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(err) + "\n")
                print(f"ERROR: {e}")

        if dominant_vals:
            mean_dom = sum(dominant_vals) / len(dominant_vals)
            print(f"chi={bond_dim}: dominant mean={mean_dom:.2f}% over {len(dominant_vals)} trial(s)")
            if len(dominant_vals) >= trials and all(v >= 100.0 for v in dominant_vals):
                print(f"chi={bond_dim}: 100% dominant overlap on all trials -> early stop")
                break

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-circuit peaked overlap vs bond dimension")
    parser.add_argument("--device", required=True, choices=["mps.cpu", "mps.gpu"], help="Execution backend")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--rzz-gates",
        type=int,
        help="Select circuit by RZZ gate count in filename (e.g., 619, 835, 1320)",
    )
    selector.add_argument(
        "--input-circuit",
        help="Path to a single circuit under mps_tests/input-peaked-circuits",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output JSONL path")
    parser.add_argument(
        "--bond-dims",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit bond dimensions; otherwise CPU uses up to 512 and GPU up to 1536",
    )
    parser.add_argument("--shots", type=int, default=SHOTS, help="Shots per trial (default: 1)")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS, help="Trials per bond dimension (default: 5)")
    args = parser.parse_args()

    if args.rzz_gates is not None:
        input_circuit = resolve_by_rzz_gates(int(args.rzz_gates))
    else:
        input_circuit = resolve_input_circuit(args.input_circuit)
    output_file = Path(args.output).resolve() if args.output else default_output_for(args.device)

    default_bond_dims = BOND_DIMS_GPU if args.device == "mps.gpu" else BOND_DIMS_CPU
    bond_dims = [int(x) for x in (args.bond_dims if args.bond_dims is not None else default_bond_dims)]

    run_benchmark(
        device=args.device,
        input_circuit=input_circuit,
        output_file=output_file,
        bond_dims=bond_dims,
        shots=int(args.shots),
        trials=int(args.trials),
    )


if __name__ == "__main__":
    main()

