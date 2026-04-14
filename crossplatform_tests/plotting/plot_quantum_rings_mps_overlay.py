#!/usr/bin/env python3
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSSPLATFORM_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(CROSSPLATFORM_ROOT, "data")
PLOT_DIR = os.path.join(CROSSPLATFORM_ROOT, "plots")
MPS_ARCHIVED_DIR = os.path.join(
    os.path.dirname(CROSSPLATFORM_ROOT), "mps_tests", "archived", "data"
)

QR_FILE = os.path.join(DATA_DIR, "quantum_ring_mps_results.jsonl")
MPS_FILE = os.path.join(MPS_ARCHIVED_DIR, "bond_scaling_gpu_old.jsonl")

OUTPUT_PNG = os.path.join(PLOT_DIR, "quantum_rings_vs_standard_mps_bond_scaling.png")
OUTPUT_PDF = os.path.join(PLOT_DIR, "quantum_rings_vs_standard_mps_bond_scaling.pdf")


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print(f"WARNING: missing {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "error" in row or row.get("job_id") == "error":
                continue
            rows.append(row)
    return rows


def aggregate_runtime(rows):
    by_chi = defaultdict(list)
    for row in rows:
        chi = row.get("bond_dimension")
        runtime = row.get("run_time_ms")
        if chi is None or runtime is None or runtime <= 0:
            continue
        by_chi[int(chi)].append(float(runtime))

    agg = {}
    for chi in sorted(by_chi):
        vals = np.array(by_chi[chi], dtype=float)
        agg[chi] = {
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "n_trials": len(vals),
        }
    return agg


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    qr_rows = [r for r in load_jsonl(QR_FILE) if r.get("depth") == 10]
    mps_rows = [r for r in load_jsonl(MPS_FILE) if r.get("depth") == 10]

    qr = aggregate_runtime(qr_rows)
    mps = aggregate_runtime(mps_rows)

    if not qr:
        raise SystemExit("No Quantum Rings bond-scaling data found.")
    if not mps:
        raise SystemExit("No archived MPS bond-scaling data found.")

    plt.figure(figsize=(9, 6))

    # Archived standard MPS baseline: single-trial old data over wide chi range.
    mps_chi = np.array(sorted(mps.keys()), dtype=float)
    mps_med = np.array([mps[c]["median"] for c in mps_chi], dtype=float)
    plt.plot(
        mps_chi,
        mps_med,
        color="#1f77b4",
        marker="o",
        linestyle="None",
        markersize=4.5,
        alpha=0.85,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="Standard MPS GPU old (shots=1000)",
    )

    # Quantum Rings: median run time per bond dimension (markers only; no error bars).
    qr_chi = np.array(sorted(qr.keys()), dtype=float)
    qr_med = np.array([qr[c]["median"] for c in qr_chi], dtype=float)

    plt.plot(
        qr_chi,
        qr_med,
        color="#d62728",
        marker="s",
        linestyle="None",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="Quantum Rings CUSTOM threshold (median)",
    )

    for chi in qr_chi:
        n = qr[int(chi)]["n_trials"]
        plt.annotate(
            f"n={n}",
            xy=(chi, qr[int(chi)]["median"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#d62728",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bond Dimension / Threshold", fontsize=12)
    plt.ylabel("Runtime (ms)", fontsize=12)
    plt.title(
        "Quantum Rings vs Standard MPS Bond Scaling\n"
        "n=40, depth=10",
        fontsize=13,
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(fontsize=10)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()

    plt.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
