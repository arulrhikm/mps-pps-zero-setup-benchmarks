"""
Quantum Volume Circuit Scaling Analysis

This script runs quantum volume circuits on CPU MPS and analyzes the scaling
behavior with respect to:
- n: number of qubits
- d: depth
- X: bond dimension

The goal is to fit the runtime to O(nd X²) scaling.
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
output_file = 'quantum_volume_scaling.jsonl'

# Parameter ranges for quantum volume circuits
qubit_values = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
depth_values = [4, 8, 12, 16, 20, 24, 28, 32]
bond_dim_values = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

# Additional large-scale configurations (25 more challenging circuits)
# These explore dramatically higher nd X² values to test scaling limits
# Standard max: n=40, d=32, X=40 → nd X² = 2,048,000
# These configs range from 5x to 25x larger scaling factors
large_scale_configs = [
    # Very high bond dimension (X=64-80) - tests X² scaling
    (40, 32, 64),   # nd X² = 5,242,880 (2.6x)
    (40, 36, 64),   # nd X² = 5,898,240 (2.9x)
    (48, 32, 64),   # nd X² = 6,291,456 (3.1x)
    (40, 40, 68),   # nd X² = 7,398,400 (3.6x)
    (48, 36, 68),   # nd X² = 7,983,744 (3.9x)
    
    # High qubits + high bond dimension
    (56, 32, 64),   # nd X² = 7,340,032 (3.6x)
    (64, 32, 64),   # nd X² = 8,388,608 (4.1x)
    (56, 36, 68),   # nd X² = 9,314,688 (4.5x)
    (64, 36, 68),   # nd X² = 10,645,504 (5.2x)
    (72, 32, 68),   # nd X² = 10,616,832 (5.2x)
    
    # Very high bond dimension (X=72-80)
    (40, 40, 72),   # nd X² = 8,294,400 (4.0x)
    (48, 40, 72),   # nd X² = 9,953,280 (4.9x)
    (56, 40, 72),   # nd X² = 11,612,160 (5.7x)
    (40, 48, 76),   # nd X² = 11,110,400 (5.4x)
    (48, 48, 76),   # nd X² = 13,332,480 (6.5x)
    
    # Extreme bond dimension (X=80-96)
    (40, 40, 80),   # nd X² = 10,240,000 (5.0x)
    (48, 40, 80),   # nd X² = 12,288,000 (6.0x)
    (56, 40, 80),   # nd X² = 14,336,000 (7.0x)
    (64, 40, 80),   # nd X² = 16,384,000 (8.0x)
    (40, 48, 84),   # nd X² = 13,547,520 (6.6x)
    
    # Maximum scaling tests (X=88-96)
    (48, 48, 88),   # nd X² = 17,915,904 (8.7x)
    (56, 48, 88),   # nd X² = 20,901,888 (10.2x)
    (40, 56, 92),   # nd X² = 19,046,400 (9.3x)
    (48, 56, 96),   # nd X² = 24,772,608 (12.1x)
    (56, 56, 96),   # nd X² = 28,901,376 (14.1x)
]

# For quick testing, you can use smaller ranges:
# qubit_values = [4, 8, 12, 16]
# depth_values = [4, 8, 12]
# bond_dim_values = [4, 8, 12]
# large_scale_configs = []

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
                # Track (num_qubits, depth, bond_dimension) tuples
                completed_runs.add((
                    run_data['num_qubits'], 
                    run_data['depth'], 
                    run_data['bond_dimension']
                ))
            except json.JSONDecodeError:
                continue
    print(f"Found {len(completed_runs)} already completed runs")
else:
    # Create new file with header
    with open(output_file, 'w') as f:
        f.write('# Quantum Volume Circuit Scaling Analysis (CPU MPS)\n')
        f.write(f'# Generated: {datetime.now().isoformat()}\n')
        f.write('# Target scaling: O(nd X^3) where n=qubits, d=depth, X=bond_dimension\n')
        f.write('# Fields: num_qubits, depth, bond_dimension, num_gates, num_cx_gates, '\
                'job_id, queue_time_ms, run_time_ms, timestamp\n')
        f.flush()
        os.fsync(f.fileno())

print(f"\n{'='*80}")
print(f"Quantum Volume Circuit Scaling Analysis")
print(f"{'='*80}")
print(f"Qubit values: {qubit_values}")
print(f"Depth values: {depth_values}")
print(f"Bond dimension values: {bond_dim_values}")
print(f"Standard configurations: {len(qubit_values) * len(depth_values) * len(bond_dim_values)}")
print(f"Large-scale configurations: {len(large_scale_configs)}")
print(f"Total configurations: {len(qubit_values) * len(depth_values) * len(bond_dim_values) + len(large_scale_configs)}")
print(f"Device: mps.cpu")
print(f"{'='*80}\n")

# Track statistics
total_runs = 0
skipped_runs = 0
failed_runs = 0
warmup_done = False

# Iterate through all parameter combinations
for num_qubits in qubit_values:
    for depth in depth_values:
        for bond_dimension in bond_dim_values:
            # Skip if already completed
            if (num_qubits, depth, bond_dimension) in completed_runs:
                print(f"⊘ Skipping (qubits={num_qubits}, depth={depth}, bond={bond_dimension}) - already completed")
                skipped_runs += 1
                continue
            
            print(f"\n{'─'*80}")
            print(f"▶ Running: qubits={num_qubits}, depth={depth}, bond_dim={bond_dimension}")
            
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
                
                # Warmup run for the first circuit
                if not warmup_done:
                    print(f"  → Warmup run (discarding)...")
                    job_warmup = bq.run(qc, device="mps.cpu", options=options, shots=1)
                    print(f"     Warmup: {job_warmup.run_time_ms}ms (discarded)")
                    warmup_done = True
                
                # Run the actual measurement
                print(f"  → Measurement run...")
                start_time = time.time()
                job = bq.run(qc, device="mps.cpu", options=options, shots=1)
                end_time = time.time()
                
                # Prepare result data
                run_data = {
                    'num_qubits': num_qubits,
                    'depth': depth,
                    'bond_dimension': bond_dimension,
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
                
                # Calculate expected scaling factor (nd X^3)
                scaling_factor = num_qubits * depth * (bond_dimension ** 3)
                print(f"    Scaling factor (nd X³): {scaling_factor}")
                print(f"    Time / scaling: {job.run_time_ms / scaling_factor:.6f}ms")
                
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
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    f.write(f"# ERROR: {json.dumps(error_data)}\n")
                    f.flush()
                    os.fsync(f.fileno())

# Run large-scale configurations
print(f"\n{'='*80}")
print(f"Running Large-Scale Configurations")
print(f"{'='*80}\n")

for num_qubits, depth, bond_dimension in large_scale_configs:
    # Skip if already completed
    if (num_qubits, depth, bond_dimension) in completed_runs:
        print(f"⊘ Skipping (qubits={num_qubits}, depth={depth}, bond={bond_dimension}) - already completed")
        skipped_runs += 1
        continue
    
    print(f"\n{'─'*80}")
    print(f"▶ Running LARGE-SCALE: qubits={num_qubits}, depth={depth}, bond_dim={bond_dimension}")
    
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
        
        # Run the actual measurement (no warmup needed, already done)
        print(f"  → Measurement run...")
        start_time = time.time()
        job = bq.run(qc, device="mps.cpu", options=options, shots=1)
        end_time = time.time()
        
        # Prepare result data
        run_data = {
            'num_qubits': num_qubits,
            'depth': depth,
            'bond_dimension': bond_dimension,
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
        
        # Calculate expected scaling factor (nd X²)
        scaling_factor_x2 = num_qubits * depth * (bond_dimension ** 2)
        scaling_factor_x3 = num_qubits * depth * (bond_dimension ** 3)
        print(f"    Scaling factor (nd X²): {scaling_factor_x2}")
        print(f"    Scaling factor (nd X³): {scaling_factor_x3}")
        print(f"    Time / X² scaling: {job.run_time_ms / scaling_factor_x2:.6f}ms")
        
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

print("To analyze results and fit to O(nd X³) scaling, run:")
print(f"  python plot_quantum_volume_scaling.py {output_file}")
