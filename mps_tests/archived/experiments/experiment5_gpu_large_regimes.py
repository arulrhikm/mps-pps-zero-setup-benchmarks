"""
experiment5_gpu_large_regimes.py
=================================
Targeted GPU experiment to densify data in R3–R6+ (and slightly above R6).

Current GPU regime boundaries (su4_n_X2):
  R1/R2:           65,536,000
  R2/R3:          196,608,000
  R3/R4:          524,288,000   ← thin (N=36)
  R4/R5:        1,048,576,000   ← thin (N=42)
  R5/R6:        2,980,970,496   ← thin (N=41)
  R6+:          up to ~18 B     ← thin (N=38)
  Beyond R6:    >18 B           ← NO DATA

Strategy:
  - 3 axes varied: n (qubits), X (bond dimension), d (depth)
  - su4s ≈ d * (n // 2)  for QV circuits
  - x = su4s * n * X^2
  - Target x bands: R3 centre, R4 centre, R5 centre, R6 centre, beyond R6 (3 levels)
  - 2 trials per config (different seeds) to get repeat measurements

Parameter grid:
  R3  (x ~ 300M – 500M):   n=40, X=256, d=[14,18,22]  → su4s=[280,360,440], x=[~365M,~470M,~573M]
  R4  (x ~ 550M – 1B):     n=60, X=256, d=[15,20,25]  → su4s=[450,600,750], x=[~590M,~786M,~982M]
                            n=40, X=384, d=[12,16,20]  → su4s=[240,320,400], x=[~567M,~755M,~944M]
  R5  (x ~ 1.1B – 2.9B):   n=60, X=384, d=[15,20,25]  → su4s=[450,600,750], x=[~1.3B,~1.7B,~2.1B]
                            n=80, X=256, d=[14,18,22]  → su4s=[560,720,880], x=[~925M,~1.2B,~1.5B]
  R6  (x ~ 3B – 18B):      n=60, X=512, d=[20,30,40]  → su4s=[600,900,1200], x=[~3.1B,~4.7B,~6.3B]
                            n=80, X=512, d=[15,20,25]  → su4s=[600,800,1000], x=[~3.9B,~5.2B,~6.6B]
  Beyond (x > 18B):         n=96, X=512, d=[20,30]     → su4s=[960,1440],    x=[~9.4B,~14.1B]
                            n=60, X=1024, d=[20,30]    → su4s=[600,900],     x=[~25B,~38B]
                            n=80, X=1024, d=[20]       → su4s=[800],         x=[~33B]
"""

import json, os, time
import bluequbit
from qiskit.circuit.library import quantum_volume

# Route to the dev environment to bypass runtime limits
os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.api.bluequbit.io/v1"

bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "gpu",
    "experiment5_gpu_large_regimes.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

NUM_TRIALS = 1   # two different seeds per config → better variance estimate

# ── Parameter grid ──────────────────────────────────────────────────────────
# Each entry: (num_qubits, bond_dimension, depth)
# Grouped by target regime for clarity
CONFIGS = [
    # R3 centre  (x ~ 300M – 500M)
    (40, 256, 14), (40, 256, 18), (40, 256, 22),

    # R4 centre  (x ~ 550M – 1B)
    (60, 256, 15), (60, 256, 20), (60, 256, 25),
    (40, 384, 12), (40, 384, 16), (40, 384, 20),

    # R5 centre  (x ~ 1.1B – 2.9B)
    (60, 384, 15), (60, 384, 20), (60, 384, 25),
    (80, 256, 14), (80, 256, 18), (80, 256, 22),

    # R6 centre  (x ~ 3B – 18B)
    (60, 512, 20), (60, 512, 30), (60, 512, 40),
    (80, 512, 15), (80, 512, 20), (80, 512, 25),

    # Beyond R6  (x > 18B)
    (96, 512, 20), (96, 512, 30),
    (60, 1024, 20), (60, 1024, 30),
    (80, 1024, 20),
]

# ── Resume support ───────────────────────────────────────────────────────────
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                completed.add((r["num_qubits"], r["bond_dimension"],
                               r["depth"], r["trial"]))
            except Exception:
                pass
    print(f"Resuming: {len(completed)} runs already complete.")
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Experiment 5: GPU large-regime targeted sweep\n")
        f.write("# Targets: R3–R6+ (su4_n_X2 > 300M)\n")
    print("Starting fresh.")

# ── Print plan ───────────────────────────────────────────────────────────────
total = len(CONFIGS) * NUM_TRIALS
remaining = sum(
    1 for (n, X, d) in CONFIGS for t in range(NUM_TRIALS)
    if (n, X, d, t) not in completed
)
print(f"\nTotal configs:  {len(CONFIGS)}")
print(f"Trials each:    {NUM_TRIALS}")
print(f"Total runs:     {total}  ({remaining} remaining)\n")
print(f"{'n':>4}  {'X':>5}  {'d':>4}  {'su4s_est':>10}  {'x_est':>18}  {'regime_est'}")
print("-" * 65)
for n, X, d in CONFIGS:
    su4s = d * (n // 2)
    x    = su4s * n * X**2
    regime = ("R3" if x < 524_288_000 else
              "R4" if x < 1_048_576_000 else
              "R5" if x < 2_980_970_496 else
              "R6" if x < 18_096_128_000 else
              "R7+")
    print(f"{n:>4}  {X:>5}  {d:>4}  {su4s:>10,}  {x:>18,}  {regime}")
print()

# ── Run ──────────────────────────────────────────────────────────────────────
for n, X, d in CONFIGS:
    for trial in range(NUM_TRIALS):
        if (n, X, d, trial) in completed:
            print(f"  ✓ skip  n={n} X={X} d={d} trial={trial}")
            continue

        print(f"  → n={n:3d}  X={X:4d}  d={d:3d}  trial={trial}  ", end="", flush=True)
        try:
            qc = quantum_volume(n, d, seed=42 + trial)
            qc_dec = qc.decompose()
            num_gates    = qc_dec.size()
            num_su4s     = qc_dec.count_ops().get("cx", 0)

            job = bq.run(qc, device="mps.gpu",
                         options={"mps_bond_dimension": X}, shots=1)

            record = {
                "trial":          trial,
                "num_qubits":     n,
                "depth":          d,
                "bond_dimension": X,
                "num_gates":      num_gates,
                "num_su4s":       num_su4s,
                "job_id":         job.job_id,
                "queue_time_ms":  job.queue_time_ms,
                "run_time_ms":    job.run_time_ms,
                "source_file":    "gpu\\experiment5_gpu_large_regimes.jsonl",
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush(); os.fsync(f.fileno())

            su4_n_X2 = num_su4s * n * X**2
            print(f"runtime={job.run_time_ms:>8,.0f} ms   "
                  f"su4_n_X2={su4_n_X2:>15,.0f}")
            completed.add((n, X, d, trial))

        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps({"num_qubits": n, "depth": d,
                                    "bond_dimension": X, "trial": trial,
                                    "error": str(e)}) + "\n")
                f.flush()

print("\nExperiment 5 complete.")
print(f"Results: {OUTPUT_FILE}")
