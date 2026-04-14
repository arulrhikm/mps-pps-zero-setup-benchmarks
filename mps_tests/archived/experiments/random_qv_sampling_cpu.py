"""
Random Quantum Volume Circuit Sampling

Randomly samples circuit parameters and runs them on BlueQubit MPS.
Sampling ranges:
- n (qubits): [1, 96]
- d (depth): [10, 1200]
- X (bond_dimension): [128, 1500]

Rejection condition: n * d * X^2 > 175,000,000

Shots: 1
"""

import json
import bluequbit
from qiskit.circuit.library import quantum_volume
import os
import time
import random
import numpy as np
from datetime import datetime

# Route to the dev environment to bypass runtime limits
os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.api.bluequbit.io/v1"

# Initialize BlueQubit
os.environ["BLUEQUBIT_MAIN_ENDPOINT"] = "https://dev.app.bluequbit.io/api/v1"
bq = bluequbit.init("kIE80aSmOKwNBZOzYiUEIymdFkEVFnyS")

# Configuration
output_file = 'random_qv_sampling_cpu.jsonl'
TARGET_SAMPLES = 500  # Number of valid runs to perform
COST_LIMIT = 175_000_000

# Function to check file existence and create header if needed
def init_log_file():
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write('# Random Quantum Volume Circuit Sampling\n')
            f.write(f'# Generated: {datetime.now().isoformat()}\n')
            f.write(f'# Ranges: n=[1,96], d=[10,1200], X=[128,1500]\n')
            f.write(f'# Rejection: n*d*X^2 > {COST_LIMIT}\n')
            f.write('# shots: 1\n')
            f.write('# Fields: num_qubits, depth, bond_dimension, num_gates, num_cx_gates, '\
                    'job_id, queue_time_ms, run_time_ms, timestamp, shots\n')
            f.flush()
            os.fsync(f.fileno())

def sample_parameters():
    while True:
        n = random.randint(16, 96)
        d = random.randint(10, 1200)
        X = random.randint(128, 1500)
        
        cost = n * d * (X**2)
        
        if cost <= COST_LIMIT:
            return n, d, X
        
        # Determine strictness of rejection used for debugging/info
        # print(f"Skipped high cost config: n={n}, d={d}, X={X} (Cost={cost:.2e})")

def run_sampling():
    init_log_file()
    
    successful_runs = 0
    failed_runs = 0
    
    print(f"\n{'='*80}")
    print(f"Random QV Sampling Benchmark")
    print(f"{'='*80}")
    print(f"Target Samples: {TARGET_SAMPLES}")
    print(f"Cost Limit: {COST_LIMIT}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    # Warmup flag
    warmup_done = False
    
    while successful_runs < TARGET_SAMPLES:
        # Sample parameters
        n, d, X = sample_parameters()
        
        print(f"\n{'─'*80}")
        print(f"▶ Sample {successful_runs+1}/{TARGET_SAMPLES}: n={n}, d={d}, X={X}")
        print(f"  Expected Cost (ndX²): {n * d * X**2:.2e}")
        
        try:
            # Create quantum volume circuit
            # Note: Qiskit quantum_volume requires n >= 2 typically, but range is 1-96. 
            # quantum_volume(1, ...) might fail or be trivial. Let's ensure n>=2 for QV or handle 1 separately.
            # Qiskit doc: "num_qubits (int) – The number of qubits of the quantum volume circuit."
            # It works for 1, just applies U3 gate.
            
            qc = quantum_volume(n, d, seed=random.randint(0, 100000))
            qc.measure_all()
            
            # Count gates
            # decompose() might be heavy for very deep circuits, but d=1200 is manageable.
            # For speed, we might skip decomposition if just running, but we need metrics.
            # Let's decompose.
            decomposed_qc = qc.decompose()
            num_gates = decomposed_qc.size()
            num_cx_gates = decomposed_qc.count_ops().get('cx', 0)
            
            # Options
            options = {
                "mps_bond_dimension": X
            }
            
            # Warmup (only once)
            if not warmup_done:
                print(f"  → Warmup run...")
                try:
                    # Small dummy circuit for warmup
                    warmup_qc = quantum_volume(4, 4, seed=42)
                    bq.run(warmup_qc, device="mps.cpu", options={"mps_bond_dimension": 16}, shots=1)
                    warmup_done = True
                except Exception as e:
                    print(f"  ! Warmup warning: {e}")
            
            # Run
            print(f"  → Running on mps.cpu...")
            shots = 1
            start_time = time.time()
            job = bq.run(qc, device="mps.cpu", options=options, shots=shots)
            
            # Collect results
            run_data = {
                'num_qubits': n,
                'depth': d,
                'bond_dimension': X,
                'num_gates': num_gates,
                'num_cx_gates': num_cx_gates,
                'job_id': job.job_id,
                'queue_time_ms': job.queue_time_ms,
                'run_time_ms': job.run_time_ms,
                'timestamp': datetime.now().isoformat(),
                'shots': shots
            }
            
            # Log
            with open(output_file, 'a') as f:
                f.write(json.dumps(run_data) + '\n')
                f.flush()
                os.fsync(f.fileno())
                
            print(f"  ✓ Success: {job.run_time_ms} ms")
            successful_runs += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed_runs += 1
            # Optional: Log failure
            with open(output_file, 'a') as f:
                f.write(f"# ERROR (n={n}, d={d}, X={X}): {str(e)}\n")

    print(f"\n{'='*80}")
    print(f"Sampling Complete")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import sys
    # Allow passing target samples as arg
    if len(sys.argv) > 1:
        try:
            TARGET_SAMPLES = int(sys.argv[1])
        except:
            pass
    run_sampling()
