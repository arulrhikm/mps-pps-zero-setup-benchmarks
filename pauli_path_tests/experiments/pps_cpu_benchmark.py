"""
PPS CPU benchmark — BlueQubit pauli-path API.
Aligned with pps_benchmark_qiskit.py and pps_benchmark.jl:
  same circuit, deltas, and num_trials (for error bars).

Resume:
  • By default, if data/pps_cpu_benchmark.jsonl exists, completed (delta_index, trial)
    pairs are loaded and only missing runs are executed (append mode).
  • Pass --fresh to truncate the file and start a new sweep from scratch.
  • --resume is accepted as a no-op (for older scripts); append behavior is the default.
"""
import os
import sys
import json
import bluequbit
import matplotlib.pyplot as plt
import numpy as np

from bluequbit.library.helpers.hardware_connectivites import IBM_127_HEAVY_HEX_MAP
from qiskit import QuantumCircuit

# Route to the dev environment to bypass runtime limits
# os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")  # production token

# ── Output file ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "data", "pps_cpu_benchmark.jsonl",
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Parameters (must match Qiskit / Julia benchmarks) ─────────────────────────
num_qubits = 127
num_trotter_steps = 20
rzz_angle = -np.pi / 2
rx_angle = np.pi / 4

# Same δ list as pps_benchmark_qiskit.py / pps_benchmark.jl
deltas = [
    1.0e-2,
    5.0e-3,
    1.0e-3,
    5.0e-4,
    1.0e-4,
    5.0e-5,
    2.5e-5,
]

NUM_TRIALS = 5

# Observable <Z_62> — BQ big-endian: qubit 0 = rightmost character.
pauli_str = "I" * (num_qubits - 1 - 62) + "Z" + "I" * 62
pauli_sum = [(pauli_str, 1.0)]


def load_completed(path):
    """
    Successful runs keyed by (delta_index, trial).

    • Requires expectation_value (failed / error rows are NOT treated as complete).
    • Legacy rows without 'trial' count as trial 0.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                if "error" in r:
                    continue
                if "expectation_value" not in r:
                    continue
                if "delta_index" not in r:
                    continue
                t = int(r.get("trial", 0))
                done.add((int(r["delta_index"]), t))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
    return done


def write_header(f):
    f.write("# PPS CPU Benchmark (pauli-path) — <Z_62> sweep\n")
    f.write(f"# num_qubits={num_qubits}, num_trotter_steps={num_trotter_steps}\n")
    f.write(f"# rx_angle={rx_angle:.6f} (pi/4), rzz_angle={rzz_angle:.6f} (-pi/2)\n")
    f.write(f"# deltas={deltas}\n")
    f.write(f"# num_trials={NUM_TRIALS} (per delta, for error bars)\n")


fresh = "--fresh" in sys.argv

if fresh:
    completed = set()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        write_header(f)
    print("Starting fresh (--fresh): output file truncated.")
elif os.path.exists(OUTPUT_FILE):
    completed = load_completed(OUTPUT_FILE)
    print(f"Resuming: {len(completed)} successful (delta_index, trial) pairs already complete.")
else:
    completed = set()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        write_header(f)
    print("Created new output file.")

# ── Build circuit ─────────────────────────────────────────────────────────────
print(f"\nBuilding circuit: {num_qubits} qubits, {num_trotter_steps} Trotter steps  (device=pauli-path)")
qc = QuantumCircuit(num_qubits)
for _ in range(num_trotter_steps):
    for edge in IBM_127_HEAVY_HEX_MAP:
        qc.rzz(rzz_angle, edge[0], edge[1])
    for i in range(num_qubits):
        qc.rx(rx_angle, i)
print(f"Circuit depth: {qc.depth()}, gate count: {qc.size()}")

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PPS-CPU: theta_X = pi/4, {NUM_TRIALS} trials per delta")
print(f"{'='*60}")

for j, delta in enumerate(deltas):
    for trial in range(NUM_TRIALS):
        if (j, trial) in completed:
            print(f"  skip  delta_index={j}  delta={delta:.2e}  trial={trial}")
            continue

        print(f"  -> delta_index={j}  delta={delta:.2e}  trial={trial}/{NUM_TRIALS - 1}  ", end="", flush=True)
        try:
            options = {"pauli_path_truncation_threshold": delta}
            result = bq.run(
                qc,
                device="pauli-path",
                pauli_sum=pauli_sum,
                options=options,
            )

            ev = result.expectation_value
            run_time = result.run_time_ms

            record = {
                "delta_index": j,
                "delta": delta,
                "trial": trial,
                "num_trials": NUM_TRIALS,
                "expectation_value": ev,
                "run_time_ms": run_time,
                "job_id": result.job_id,
                "num_qubits": num_qubits,
                "num_trotter_steps": num_trotter_steps,
                "rzz_angle": rzz_angle,
                "rx_angle": rx_angle,
                "observable": "Z_62",
                "backend": "pauli-path",
            }
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            completed.add((j, trial))
            print(f"<Z62>={ev:.6f}  time={run_time:.0f}ms")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "delta_index": j,
                            "delta": delta,
                            "trial": trial,
                            "error": str(e),
                        }
                    )
                    + "\n"
                )
                f.flush()

print(f"\nResults saved to: {OUTPUT_FILE}")

# ── Quick aggregate (mean ± std) for sanity / plotting ────────────────────────
rows = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                if "error" not in r and "expectation_value" in r:
                    rows.append(r)
            except json.JSONDecodeError:
                pass

if rows:
    print(f"\n{'='*72}")
    print(f"{'delta_index':<12} {'delta':<12} {'mean <Z>':<14} {'std <Z>':<12} {'mean t (ms)':<14} {'std t (ms)':<12}")
    print("-" * 72)
    for j in range(len(deltas)):
        sub = [r for r in rows if r.get("delta_index") == j]
        if len(sub) < 1:
            continue
        evs = [r["expectation_value"] for r in sub]
        ts = [r["run_time_ms"] for r in sub]
        print(
            f"{j:<12} {deltas[j]:<12.2e} {np.mean(evs):<14.6f} {np.std(evs, ddof=0):<12.6f} "
            f"{np.mean(ts):<14.1f} {np.std(ts, ddof=0):<12.1f}"
        )

# ── Plotting (means over trials) ──────────────────────────────────────────────
try:
    if not rows:
        raise ValueError("no data")

    delta_log = [-np.log10(deltas[j]) for j in range(len(deltas))]
    means_ev = []
    means_rt = []
    for j in range(len(deltas)):
        sub = [r for r in rows if r.get("delta_index") == j]
        if not sub:
            means_ev.append(np.nan)
            means_rt.append(np.nan)
            continue
        evs = [r["expectation_value"] for r in sub]
        ts = [r["run_time_ms"] / 1000.0 for r in sub]
        means_ev.append(np.mean(evs))
        means_rt.append(np.mean(ts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        rf"PPS CPU — $\theta_X=\pi/4$, {num_qubits}q, {num_trotter_steps} steps "
        f"(N={NUM_TRIALS} trials per δ)",
        fontsize=13,
        fontweight="bold",
    )
    COLOR = "#1B3FA0"
    ax1.plot(
        delta_log,
        means_ev,
        color=COLOR,
        marker="o",
        linestyle="-",
        linewidth=1.6,
        markersize=7,
    )
    ax1.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
    ax1.set_ylabel(r"$\langle Z_{62} \rangle$", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Expectation (mean over trials)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        delta_log,
        means_rt,
        color=COLOR,
        marker="o",
        linestyle="-",
        linewidth=1.6,
        markersize=7,
    )
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$-\log_{10}(\delta)$", fontsize=11)
    ax2.set_ylabel("Runtime (s)", fontsize=12)
    ax2.set_title("Runtime (mean over trials)")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(OUTPUT_FILE), "pps_cpu_benchmark.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {out_png}")
    plt.show()
except Exception as e:
    print(f"(Plot skipped: {e})")
