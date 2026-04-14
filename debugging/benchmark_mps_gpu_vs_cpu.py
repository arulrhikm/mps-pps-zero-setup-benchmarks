"""
MPS GPU vs CPU Benchmarking Script

This script benchmarks the performance of mps.gpu vs mps.cpu across different parameters:
- Number of qubits: 1 to 96
- Circuit depth: 1 to 64
- Bond dimension: 1 to 64

The script runs quantum volume circuits and measures runtime for both devices.
Results are saved to a JSONL file for later analysis.
"""

import json
import bluequbit
from qiskit.circuit.library import quantum_volume
import os
import time
from datetime import datetime

# Initialize BlueQubit
bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

# Configuration
output_file = 'mps_gpu_vs_cpu_benchmark.jsonl'

# Parameter ranges (noting max settings: 96 qubits, 64 depth, 64 bond dim)
qubit_values = [4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
depth_values = [4, 8, 16, 24, 32, 40, 48, 56, 64]
bond_dim_values = [4, 8, 16, 24, 32, 40, 48, 56, 64]

# For quick testing, you can use smaller ranges:
# qubit_values = [4, 8, 16, 24, 32]
# depth_values = [4, 8, 16]
# bond_dim_values = [4, 8, 16]

# Load already-completed runs to resume from there
completed_runs = set()
if os.path.exists(output_file):
    print(f"Loading existing runs from {output_file}...")
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            try:
                run_data = json.loads(line)
                # Track (num_qubits, depth, bond_dimension, device) tuples
                completed_runs.add((
                    run_data['num_qubits'], 
                    run_data['depth'], 
                    run_data['bond_dimension'],
                    run_data['device']
                ))
            except json.JSONDecodeError:
                continue
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    with open(output_file, 'w') as f:
        f.write('# MPS GPU vs CPU Benchmark\n')
        f.write(f'# Generated: {datetime.now().isoformat()}\n')
        f.write('# Fields: num_qubits, depth, bond_dimension, device, num_gates, num_cx_gates, '
                'job_id, queue_time_ms, run_time_ms, timestamp\n')
        f.flush()
        os.fsync(f.fileno())

print(f"\n{'='*80}")
print(f"MPS GPU vs CPU Benchmark")
print(f"{'='*80}")
print(f"Qubit values: {qubit_values}")
print(f"Depth values: {depth_values}")
print(f"Bond dimension values: {bond_dim_values}")
print(f"Total configurations per device: {len(qubit_values) * len(depth_values) * len(bond_dim_values)}")
print(f"Total runs (GPU + CPU): {2 * len(qubit_values) * len(depth_values) * len(bond_dim_values)}")
print(f"{'='*80}\n")

# Track statistics
total_runs = 0
skipped_runs = 0
failed_runs = 0
gpu_warmup_done = False
cpu_warmup_done = False

# Iterate through all parameter combinations
for num_qubits in qubit_values:
    for depth in depth_values:
        for bond_dimension in bond_dim_values:
            # Test both GPU and CPU
            for device in ["mps.gpu", "mps.cpu"]:
                # Skip if already completed
                if (num_qubits, depth, bond_dimension, device) in completed_runs:
                    print(f"⊘ Skipping (qubits={num_qubits}, depth={depth}, bond={bond_dimension}, device={device}) - already completed")
                    skipped_runs += 1
                    continue
                
                print(f"\n{'─'*80}")
                print(f"▶ Running: qubits={num_qubits}, depth={depth}, bond_dim={bond_dimension}, device={device}")
                
                try:
                    # Create quantum volume circuit
                    qc = quantum_volume(num_qubits, depth, seed=42)
                    
                    # Count gates
                    decomposed_qc = qc.decompose()
                    num_gates = decomposed_qc.size()
                    num_cx_gates = decomposed_qc.count_ops().get('cx', 0)
                    
                    # Set options
                    options = {
                        "mps_bond_dimension": bond_dimension,
                    }
                    
                    # Warmup run for the first circuit on each device
                    if device == "mps.gpu" and not gpu_warmup_done:
                        print(f"  → GPU warmup run (discarding)...")
                        job_warmup = bq.run(qc, device=device, options=options)
                        print(f"     Warmup: {job_warmup.run_time_ms}ms (discarded)")
                        gpu_warmup_done = True
                    elif device == "mps.cpu" and not cpu_warmup_done:
                        print(f"  → CPU warmup run (discarding)...")
                        job_warmup = bq.run(qc, device=device, options=options)
                        print(f"     Warmup: {job_warmup.run_time_ms}ms (discarded)")
                        cpu_warmup_done = True
                    
                    # Run the actual measurement
                    print(f"  → Measurement run...")
                    start_time = time.time()
                    job = bq.run(qc, device=device, options=options)
                    end_time = time.time()
                    
                    # Prepare result data
                    run_data = {
                        'num_qubits': num_qubits,
                        'depth': depth,
                        'bond_dimension': bond_dimension,
                        'device': device,
                        'num_gates': num_gates,
                        'num_cx_gates': num_cx_gates,
                        'job_id': job.job_id,
                        'queue_time_ms': job.queue_time_ms,
                        'run_time_ms': job.run_time_ms,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Write to file immediately
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(run_data) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Display results
                    print(f"  ✓ Completed: {num_gates} gates ({num_cx_gates} CX)")
                    print(f"    Queue time: {job.queue_time_ms}ms")
                    print(f"    Run time: {job.run_time_ms}ms")
                    print(f"    Time per gate: {job.run_time_ms / num_gates:.3f}ms")
                    
                    total_runs += 1
                    
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    failed_runs += 1
                    # Log the failure
                    with open(output_file, 'a') as f:
                        error_data = {
                            'num_qubits': num_qubits,
                            'depth': depth,
                            'bond_dimension': bond_dimension,
                            'device': device,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        f.write(f"# ERROR: {json.dumps(error_data)}\n")
                        f.flush()
                        os.fsync(f.fileno())

# Final summary
print(f"\n{'='*80}")
print(f"Benchmark Complete!")
print(f"{'='*80}")
print(f"Total runs completed: {total_runs}")
print(f"Skipped (already done): {skipped_runs}")
print(f"Failed runs: {failed_runs}")
print(f"Results saved to: {output_file}")
print(f"{'='*80}\n")

print("To analyze results, you can use pandas:")
print("  import pandas as pd")
print("  import json")
print(f"  data = [json.loads(line) for line in open('{output_file}') if not line.startswith('#')]")
print("  df = pd.DataFrame(data)")
print("  # Compare GPU vs CPU")
print("  gpu_df = df[df['device'] == 'mps.gpu']")
print("  cpu_df = df[df['device'] == 'mps.cpu']")
