#!/usr/bin/env python3
"""Run all BlueQubit benchmark plot scripts (non-interactive)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
os.environ.setdefault("MPLBACKEND", "Agg")


def run(label: str, argv: list[str], cwd: Path | None = None) -> bool:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUTF8", "1")
    r = subprocess.run(argv, cwd=cwd or ROOT, env=env)
    if r.returncode != 0:
        print(f"[FAIL] {label} (exit {r.returncode})")
        return False
    print(f"[ok] {label}")
    return True


def main() -> int:
    ok = True
    mps_plot = ROOT / "mps_tests" / "plotting"
    for name in (
        "plot_bond_scaling.py",
        "plot_bond_scaling_with_quantum_rings.py",
        "plot_build_vs_sampling.py",
        "plot_depth_scaling.py",
        "plot_qubit_scaling.py",
        "plot_qft_scaling.py",
        "plot_shots_scaling.py",
    ):
        ok &= run(f"mps_tests/plotting/{name}", [PY, str(mps_plot / name)])

    cp = ROOT / "crossplatform_tests" / "plotting"
    ok &= run("crossplatform/plot_crossplatform.py", [PY, str(cp / "plot_crossplatform.py")])
    ok &= run(
        "crossplatform/plot_quantum_rings_mps_overlay.py",
        [PY, str(cp / "plot_quantum_rings_mps_overlay.py")],
    )

    sv = ROOT / "statevector_tests" / "plotting"
    ok &= run("statevector/qv_plot_combined.py", [PY, str(sv / "qv_plot_combined.py")])

    pps = ROOT / "pauli_path_tests" / "plotting"
    for name in (
        "pps_gpu_speedup.py",
        "pps_runtime_comparison.py",
        "pps_expectation_convergence.py",
        "pps_accuracy_comparison.py",
    ):
        ok &= run(f"pauli_path_tests/plotting/{name}", [PY, str(pps / name)])

    arch_plot = ROOT / "mps_tests" / "archived" / "plotting"
    arch_cwd = arch_plot
    ok &= run(
        "archived/plot_experiment1_qubit_scaling.py",
        [PY, "plot_experiment1_qubit_scaling.py"],
        cwd=arch_cwd,
    )
    ok &= run(
        "archived/plot_experiment2_bond_scaling.py (CPU burned)",
        [PY, "plot_experiment2_bond_scaling.py", "../experiment2_bond_scaling_cpu_burned.jsonl"],
        cwd=arch_cwd,
    )
    ok &= run(
        "archived/plot_experiment3_depth_scaling.py",
        [PY, "plot_experiment3_depth_scaling.py"],
        cwd=arch_cwd,
    )
    ok &= run(
        "archived/plot_experiment2_shots_sweep.py",
        [PY, "plot_experiment2_shots_sweep.py"],
        cwd=arch_cwd,
    )
    for name in (
        "plot_figure6_qft_scaling.py",
        "plot_experiment6_qft_scaling.py",
        "plot_experiment6_bond_scaling.py",
    ):
        ok &= run(f"archived/{name}", [PY, str(arch_plot / name)], cwd=arch_plot)

    ok &= run(
        "archived/analyze_qv_scaling_4panel.py",
        [PY, "analyze_qv_scaling_4panel.py"],
        cwd=arch_cwd,
    )
    data_qv = "../data/quantum_volume_scaling.jsonl"
    for name in ("analyze_qv_scaling.py", "analyze_qv_scaling_v2.py", "analyze_qv_scaling_stratified.py"):
        ok &= run(
            f"archived/{name}",
            [PY, name, data_qv],
            cwd=arch_cwd,
        )

    pauli_arch = ROOT / "pauli_path_tests" / "archived"
    ok &= run(
        "pauli_path_tests/archived/plot_all_figs.py",
        [PY, str(pauli_arch / "plot_all_figs.py")],
        cwd=pauli_arch,
    )

    skipped = """
Skipped (missing local data or heavy benchmarks):
  mps_tests/archived/plotting/compare_experiment2_cpu_gpu.py — needs ../cpu|gpu/*updated*.jsonl
  mps_tests/archived/plotting/plot_cpu_vs_gpu.py — needs mps_tests/cpu|gpu/*.jsonl
  mps_tests/archived/plotting/plot_sampling_scaling.py — needs mps_tests/cpu|gpu experiment4 files
  pauli_path_tests/archived/fig*_pps_gpu.py — BlueQubit API benchmark drivers, not static replots
"""
    print(skipped)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
