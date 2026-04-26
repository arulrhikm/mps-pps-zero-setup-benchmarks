#!/usr/bin/env python3
"""
Generate one runtime-vs-bond plot per RZZ circuit family.

Requested layout:
  - bottom x-axis: bond dimension
  - top x-axis: mean overlap_percent at matching bond dimensions
  - y-axis: run_time_ms

By default reads:
  mps_tests/data/peaked-circuits-results/CPU_peaked_overlap_bond_dimension.jsonl
  mps_tests/data/peaked-circuits-results/GPU_peaked_overlap_bond_dimension.jsonl

Outputs (default):
  mps_tests/plots/fig_peaked_runtime_rzz_1271.png
  mps_tests/plots/fig_peaked_runtime_rzz_1320.png
  mps_tests/plots/fig_peaked_runtime_rzz_1438.png
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPS_TESTS_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(MPS_TESTS_DIR, "data", "peaked-circuits-results")
PLOTS_DIR = os.path.join(MPS_TESTS_DIR, "plots")

CPU_FILE = os.path.join(RESULTS_DIR, "CPU_peaked_overlap_bond_dimension.jsonl")
GPU_FILE = os.path.join(RESULTS_DIR, "GPU_peaked_overlap_bond_dimension.jsonl")
TARGET_RZZS = (1271, 1320, 1438)
COMBINED_RZZS = (1320, 1438)

CPU_COLOR = "#1f77b4"
GPU_COLOR = "#d62728"


def _extract_rzz(row: dict) -> int | None:
    rzz = row.get("filename_rzz_count")
    if isinstance(rzz, int):
        return rzz
    key = str(row.get("circuit_key", ""))
    m = re.search(r"_RZZ(\d+)_", key)
    if m:
        return int(m.group(1))
    return None


def _load_device(path: str) -> dict[int, dict[int, dict[str, list[float]]]]:
    """
    Returns:
      rzz -> bond_dimension -> {"runtime_ms": [...], "overlap_percent": [...]}
    """
    data: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"runtime_ms": [], "overlap_percent": []})
    )
    if not os.path.exists(path):
        return data

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            rzz = _extract_rzz(row)
            bd = row.get("bond_dimension")
            rt = row.get("run_time_ms")
            ov = row.get("overlap_percent", row.get("dominant_overlap_percent"))
            if rzz is None or not isinstance(bd, int):
                continue
            if not isinstance(rt, (int, float)) or rt <= 0:
                continue
            if not isinstance(ov, (int, float)):
                continue
            data[rzz][bd]["runtime_ms"].append(float(rt))
            data[rzz][bd]["overlap_percent"].append(float(ov))
    return data


def _means_by_bond(by_bond: dict[int, dict[str, list[float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not by_bond:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    bds = np.array(sorted(by_bond.keys()), dtype=float)
    runtime = np.array([np.mean(by_bond[int(b)]["runtime_ms"]) for b in bds], dtype=float)
    overlap = np.array([np.mean(by_bond[int(b)]["overlap_percent"]) for b in bds], dtype=float)
    return bds, runtime, overlap


def _top_labels(
    xticks: list[int],
    cpu_bond: dict[int, dict[str, list[float]]],
    gpu_bond: dict[int, dict[str, list[float]]],
) -> list[str]:
    labels: list[str] = []
    for bd in xticks:
        vals: list[float] = []
        if bd in cpu_bond and cpu_bond[bd]["overlap_percent"]:
            vals.extend(cpu_bond[bd]["overlap_percent"])
        if bd in gpu_bond and gpu_bond[bd]["overlap_percent"]:
            vals.extend(gpu_bond[bd]["overlap_percent"])
        labels.append(f"{np.mean(vals):.1f}%" if vals else "")
    return labels


def _plot_one(rzz: int, cpu: dict, gpu: dict, out_path: str) -> None:
    cpu_bond = cpu.get(rzz, {})
    gpu_bond = gpu.get(rzz, {})
    b_cpu, rt_cpu, _ = _means_by_bond(cpu_bond)
    b_gpu, rt_gpu, _ = _means_by_bond(gpu_bond)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    if b_cpu.size > 0:
        ax.plot(
            b_cpu,
            rt_cpu,
            color=CPU_COLOR,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label="CPU mean runtime",
        )
    if b_gpu.size > 0:
        ax.plot(
            b_gpu,
            rt_gpu,
            color=GPU_COLOR,
            marker="s",
            linewidth=2.2,
            markersize=5,
            label="GPU mean runtime",
        )

    all_ticks = sorted({*(int(x) for x in b_cpu.tolist()), *(int(x) for x in b_gpu.tolist())})
    if not all_ticks:
        all_ticks = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Bond Dimension ($\\chi$)")
    ax.set_ylabel("Run Time (ms)")
    ax.set_title(f"Peaked Circuit Runtime vs Bond Dimension (RZZ={rzz})")
    ax.grid(True, which="both", alpha=0.28, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.95)

    ax_top = ax.twiny()
    ax_top.set_xscale("log", base=2)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(all_ticks)
    ax_top.set_xticklabels(_top_labels(all_ticks, cpu_bond, gpu_bond), fontsize=8)
    ax_top.set_xlabel("Mean Overlap (%)", labelpad=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_combined(cpu: dict, gpu: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    cpu_colors = {1271: "#9ecae1", 1320: "#6baed6", 1438: "#2171b5"}
    gpu_colors = {1271: "#fcae91", 1320: "#fb6a4a", 1438: "#cb181d"}

    for rzz in COMBINED_RZZS:
        cpu_bond = cpu.get(rzz, {})
        gpu_bond = gpu.get(rzz, {})
        b_cpu, rt_cpu, _ = _means_by_bond(cpu_bond)
        b_gpu, rt_gpu, _ = _means_by_bond(gpu_bond)

        if b_cpu.size > 0:
            ax.plot(
                b_cpu,
                rt_cpu,
                color=cpu_colors[rzz],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                linestyle="-",
                label=f"CPU rzz={rzz}",
            )
        if b_gpu.size > 0:
            ax.plot(
                b_gpu,
                rt_gpu,
                color=gpu_colors[rzz],
                marker="s",
                linewidth=2.0,
                markersize=4.5,
                linestyle="--",
                label=f"GPU rzz={rzz}",
            )

    xticks = sorted(
        {
            bd
            for rzz in COMBINED_RZZS
            for bd in list(cpu.get(rzz, {}).keys()) + list(gpu.get(rzz, {}).keys())
        }
    )
    if not xticks:
        xticks = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Bond Dimension ($\\chi$)")
    ax.set_ylabel("Run Time (ms)")
    ax.set_title("Peaked Circuit Runtime: CPU vs GPU")
    ax.grid(True, which="both", alpha=0.28, linestyle="--")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95, fontsize=8.8)

    # Top axis: mean overlap across available CPU/GPU rows per bond dimension.
    overlap_by_bd: dict[int, list[float]] = defaultdict(list)
    for rzz in COMBINED_RZZS:
        for bd, vals in cpu.get(rzz, {}).items():
            overlap_by_bd[int(bd)].extend(vals.get("overlap_percent", []))
        for bd, vals in gpu.get(rzz, {}).items():
            overlap_by_bd[int(bd)].extend(vals.get("overlap_percent", []))

    ax_top = ax.twiny()
    ax_top.set_xscale("log", base=2)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels(
        [f"{np.mean(overlap_by_bd[bd]):.1f}%" if overlap_by_bd.get(bd) else "" for bd in xticks],
        fontsize=8,
    )
    ax_top.set_xlabel("Mean Overlap (%)", labelpad=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cpu = _load_device(CPU_FILE)
    gpu = _load_device(GPU_FILE)

    for rzz in TARGET_RZZS:
        out = os.path.join(PLOTS_DIR, f"fig_peaked_runtime_rzz_{rzz}.png")
        _plot_one(rzz, cpu, gpu, out)
        print(f"Saved {out}")

    combined_out = os.path.join(PLOTS_DIR, "fig_peaked_runtime_rzz_combined.png")
    _plot_combined(cpu, gpu, combined_out)
    print(f"Saved {combined_out}")


if __name__ == "__main__":
    main()

