import os
import json
import bluequbit
import matplotlib.pyplot as plt
import numpy as np

from bluequbit.library.helpers.hardware_connectivites import IBM_127_HEAVY_HEX_MAP
from qiskit import QuantumCircuit

# Route to the dev environment to bypass runtime limits
os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"

bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")  # production token

# ── Output file ───────────────────────────────────────────────────────────────
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "fig8_pps_gpu.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
num_qubits        = 127
num_trotter_steps = 20
rzz_angle         = -np.pi / 2
rx_angles         = [0.3, 0.4, 0.6, 0.7]
deltas            = [1 / 2**i for i in range(13)]

# Observable <Z_62> as a list of (pauli_string, coefficient) — native BQ format.
# BQ uses big-endian ordering: qubit 0 = rightmost character.
# Qubit 62 from the right → position (num_qubits - 1 - 62) = 64 from the left.
pauli_str = "I" * (num_qubits - 1 - 62) + "Z" + "I" * 62
pauli_sum = [(pauli_str, 1.0)]

# ── Resume support ────────────────────────────────────────────────────────────
# Key: (rx_angle, delta_index)
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                completed.add((r["rx_angle"], r["delta_index"]))
            except Exception:
                pass
    print(f"Resuming: {len(completed)} runs already complete.")
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Fig. 8 PPS GPU — <Z_62> expectation value sweep\n")
        f.write(f"# num_qubits={num_qubits}, num_trotter_steps={num_trotter_steps}\n")
        f.write(f"# rx_angles={rx_angles}\n")
        f.write(f"# deltas=[1/2^i for i in range({len(deltas)})]\n")
    print("Starting fresh.")

# ── Run ───────────────────────────────────────────────────────────────────────
results = {}  # for plotting after all runs

for rx_angle in rx_angles:
    print(f"\n{'='*60}")
    print(f"Running theta_X = {rx_angle}")
    print(f"{'='*60}")

    # Build circuit once per angle
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_trotter_steps):
        for edge in IBM_127_HEAVY_HEX_MAP:
            qc.rzz(rzz_angle, edge[0], edge[1])
        for i in range(num_qubits):
            qc.rx(rx_angle, i)

    exp_vals = []
    run_times = []

    for j, delta in enumerate(deltas):
        if (rx_angle, j) in completed:
            print(f"  ✓ skip  rx={rx_angle} delta=2^-{j}")
            # placeholder so plot indices stay aligned
            exp_vals.append(None)
            run_times.append(None)
            continue

        print(f"  → rx={rx_angle}  delta=2^-{j}={delta:.6f}  ", end="", flush=True)
        try:
            options = {"pauli_path_truncation_threshold": delta}
            result = bq.run(
                qc,
                device="pauli-path.gpu",
                pauli_sum=pauli_sum,
                options=options,
            )

            ev       = result.expectation_value
            run_time = result.run_time_ms  # ms

            exp_vals.append(ev)
            run_times.append(run_time)

            record = {
                "rx_angle":    rx_angle,
                "delta_index": j,
                "delta":       delta,
                "expectation_value": ev,
                "run_time_ms": run_time,
                "job_id":      result.job_id,
                "num_qubits":  num_qubits,
                "num_trotter_steps": num_trotter_steps,
                "rzz_angle":   rzz_angle,
                "observable":  "Z_62",
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            completed.add((rx_angle, j))
            print(f"<Z62>={ev:.6f}  time={run_time:.0f}ms")

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({
                    "rx_angle":    rx_angle,
                    "delta_index": j,
                    "delta":       delta,
                    "error":       str(e),
                }) + "\n")
                f.flush()
            exp_vals.append(None)
            run_times.append(None)

    results[rx_angle] = {"exp_vals": exp_vals, "run_times": run_times}

print(f"\nResults saved to: {OUTPUT_FILE}")

# ── Plotting ──────────────────────────────────────────────────────────────────
colors  = {0.3: "#E89B2D", 0.4: "#1B3FA0", 0.6: "#C93030", 0.7: "#2A9D8F"}
markers = {0.3: "o",       0.4: "s",       0.6: "D",       0.7: "^"}
labels  = {a: rf"$\theta_X = {a}$" for a in rx_angles}
delta_log = -np.log10(deltas)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("IBM Reproduction — PPS GPU", fontsize=14, fontweight="bold")

for rx_angle in rx_angles:
    ev = [v for v in results[rx_angle]["exp_vals"] if v is not None]
    dl = [delta_log[j] for j, v in enumerate(results[rx_angle]["exp_vals"]) if v is not None]

    if not ev:
        continue

    ax1.plot(dl, ev, color=colors[rx_angle], marker=markers[rx_angle],
             markersize=6, linewidth=1.5, label=labels[rx_angle])
    last_vals = ev[-5:]
    mid    = np.mean(last_vals)
    spread = max(np.std(last_vals) * 2, 0.02)
    ax1.axhspan(mid - spread, mid + spread, color=colors[rx_angle], alpha=0.12)

garnet_delta_log = -np.log10(2e-3)
ax1.axvline(garnet_delta_log, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax1.set_xlabel(r"$\delta_k$", fontsize=12)
ax1.set_ylabel(r"$\mathcal{O}_k \approx \langle Z_{62} \rangle$", fontsize=12)
ax1.set_xlim(0, delta_log[-1] + 0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=10)

xtick_positions = list(range(0, int(delta_log[-1]) + 1))
xtick_labels    = [rf"$10^{{-{i}}}$" for i in xtick_positions]
ax1.set_xticks(xtick_positions)
ax1.set_xticklabels(xtick_labels)

for rx_angle in rx_angles:
    rt = [v / 1000 for v in results[rx_angle]["run_times"] if v is not None]
    dl = [delta_log[j] for j, v in enumerate(results[rx_angle]["run_times"]) if v is not None]
    if not rt:
        continue
    ax2.plot(dl, rt, color=colors[rx_angle], marker=markers[rx_angle],
             markersize=6, linewidth=1.5, label=labels[rx_angle])

ax2.axvline(garnet_delta_log, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.set_yscale("log")
ax2.set_xlabel(r"$\delta_k$", fontsize=12)
ax2.set_ylabel("Runtime (s)", fontsize=12)
ax2.set_xlim(0, delta_log[-1] + 0.3)
ax2.legend(fontsize=10)
ax2.set_xticks(xtick_positions)
ax2.set_xticklabels(xtick_labels)

plt.tight_layout()
plt.savefig("fig8_gpu_reproduction.png", dpi=200, bbox_inches="tight")
print("Plot saved to: fig8_gpu_reproduction.png")
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'SUMMARY TABLE':^80}")
print(f"{'='*80}")
print(f"{'delta':<12}", end="")
for a in rx_angles:
    print(f"| theta={a} <Z62>    time(ms) ", end="")
print()
print("-" * 80)

for j, delta in enumerate(deltas):
    print(f"2^-{j:<2} = {delta:<8.6f}", end="")
    for a in rx_angles:
        ev = results[a]["exp_vals"][j]
        rt = results[a]["run_times"][j]
        ev_str = f"{ev:>8.4f}" if ev is not None else "    skip"
        rt_str = f"{rt:>9.0f}" if rt is not None else "     skip"
        print(f"| {ev_str}  {rt_str}ms ", end="")
    print()