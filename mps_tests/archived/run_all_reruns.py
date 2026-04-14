"""
run_all_reruns.py
=================
Convenience wrapper to execute all MPS "Phase 2" rerun scripts:
  - experiment1 qubit scaling (CPU + GPU)
  - experiment3 depth scaling (CPU + GPU)
  - experiment6 QFT scaling (CPU + GPU)

By default, runs scripts sequentially. Each underlying script already has
resume logic and will append only missing (config, trial) pairs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUITE_SCRIPTS = [
    # Qubit scaling
    Path("mps_tests/experiments/cpu/experiment1_qubit_scaling_cpu_rerun.py"),
    Path("mps_tests/experiments/gpu/experiment1_qubit_scaling_gpu_rerun.py"),
    # Depth scaling
    Path("mps_tests/experiments/cpu/experiment3_depth_scaling_cpu_rerun.py"),
    Path("mps_tests/experiments/gpu/experiment3_depth_scaling_gpu_rerun.py"),
    # QFT scaling (exact QFT only, degree=0)
    Path("mps_tests/experiments/cpu/experiment6_qft_cpu_rerun.py"),
    Path("mps_tests/experiments/gpu/experiment6_qft_gpu_rerun.py"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for running the scripts (default: current).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going even if a script fails.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    failures = 0
    for idx, rel_script in enumerate(SUITE_SCRIPTS, start=1):
        script_path = (repo_root / rel_script).resolve()
        cmd = [args.python, str(script_path)]

        print(f"[{idx}/{len(SUITE_SCRIPTS)}] Running: {script_path}")
        print("  Command:", " ".join(cmd))

        if args.dry_run:
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failures += 1
            print(f"  ERROR: script failed with exit code {proc.returncode}")
            if not args.continue_on_error:
                print("Stopping due to --continue-on-error not set.")
                sys.exit(proc.returncode)

    if failures:
        print(f"Completed with {failures} failure(s).")
        sys.exit(1)
    print("All rerun scripts completed successfully.")


if __name__ == "__main__":
    main()

