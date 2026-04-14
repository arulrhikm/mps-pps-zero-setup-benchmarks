"""
add_su4_counts.py
-----------------
Augments all_mps_data.jsonl with a `num_su4s` field for each record.

Strategy:
  - Reconstruct the exact QuantumVolume circuit used in each experiment using
    the same seed convention: seed = 42 + trial  (trial is always 0 in the data)
  - Count 1-qubit and 2-qubit instructions directly from the raw (undecomposed)
    QV circuit. QV circuits are natively expressed as layers of SU(4) unitaries
    (plus single-qubit gates), so this gives the true raw counts with ZERO
    optimization passes.
    Sanity check: num_su4s * 3 == num_cx_gates  (each SU(4) ~ 3 CX gates).
  - Adds three new fields per record:
      num_su4s          – 2-qubit gate count
      num_su2s          – 1-qubit gate count
      num_su4s_and_su2s – sum of both
  - Write results to all_mps_data_with_su4.jsonl (does NOT overwrite original).

Usage:
  python add_su4_counts.py
"""

import json
import sys
from pathlib import Path

from qiskit.circuit.library import quantum_volume

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
INPUT_FILE  = DATA_DIR / "all_mps_data.jsonl"
OUTPUT_FILE = DATA_DIR / "all_mps_data_with_su4.jsonl"


def count_ops_raw(qc) -> tuple[int, int]:
    """Count 1-qubit (SU(2)) and 2-qubit (SU(4)) instructions in the raw QV
    circuit with zero optimization passes.

    Returns:
        (num_su2s, num_su4s)
    """
    su2 = su4 = 0
    for inst in qc.data:
        nq = inst.operation.num_qubits
        if nq == 1:
            su2 += 1
        elif nq == 2:
            su4 += 1
    return su2, su4


# ── Cache: avoid rebuilding the same (num_qubits, depth, trial) circuit twice ─
_circuit_cache: dict[tuple, tuple[int, int]] = {}  # key -> (num_su2s, num_su4s)


def get_op_counts(num_qubits: int, depth: int, trial: int) -> tuple[int, int]:
    """Return (num_su2s, num_su4s) for the given circuit configuration."""
    key = (num_qubits, depth, trial)
    if key not in _circuit_cache:
        seed = 42 + trial
        qc = quantum_volume(num_qubits, depth, seed=seed)
        _circuit_cache[key] = count_ops_raw(qc)
    return _circuit_cache[key]


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    augmented = 0
    skipped_comment = 0

    with INPUT_FILE.open("r") as fin, OUTPUT_FILE.open("w") as fout:
        for lineno, raw_line in enumerate(fin, start=1):
            line = raw_line.rstrip("\n")

            stripped = line.strip()
            # Pass through blank lines and comment lines unchanged
            if not stripped or stripped.startswith("#"):
                fout.write(line + "\n")
                skipped_comment += 1
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"WARNING line {lineno}: JSON parse error ({exc}), passing through.", file=sys.stderr)
                fout.write(line + "\n")
                continue

            num_qubits = record["num_qubits"]
            depth      = record["depth"]
            trial      = record.get("trial", 0)

            try:
                num_su2s, num_su4s = get_op_counts(num_qubits, depth, trial)
            except Exception as exc:
                print(
                    f"WARNING line {lineno}: failed to compute op counts for "
                    f"(n={num_qubits}, d={depth}, trial={trial}): {exc}",
                    file=sys.stderr,
                )
                num_su2s = num_su4s = None

            record["num_su4s"]          = num_su4s
            record["num_su2s"]          = num_su2s
            record["num_su4s_and_su2s"] = (num_su4s + num_su2s) if num_su4s is not None else None
            fout.write(json.dumps(record) + "\n")

            augmented += 1
            if augmented % 100 == 0:
                print(f"  Processed {augmented} records so far …", flush=True)

    print(f"\nDone!")
    print(f"  Records augmented    : {augmented}")
    print(f"  Comment/blank lines  : {skipped_comment}")
    print(f"  Unique circuits built: {len(_circuit_cache)}")
    print(f"  New fields added     : num_su4s, num_su2s, num_su4s_and_su2s")
    print(f"  Output written to    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
