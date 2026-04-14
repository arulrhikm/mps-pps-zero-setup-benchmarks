"""
Simple MPS GPU vs CPU Benchmark
================================
Minimal, transparent code to compare mps.gpu vs mps.cpu performance.
No complex logic - just run circuits and measure time.
"""

import json
import bluequbit
from qiskit.circuit.library import quantum_volume
from datetime import datetime

# Initialize BlueQubit
bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")

# Simple test configurations
# Format: (num_qubits, depth, bond_dimension)
test_configs = [
    # Original small bond dimension tests
    (8, 8, 8),
    (16, 16, 16),
    (24, 16, 16),
    (32, 16, 16),
    (40, 16, 16),
    (48, 16, 16),
    (56, 16, 16),
    (64, 16, 16),
    
    # Larger bond dimensions (default is 256, testing up to 1024)
    # These should show better GPU performance
    (16, 16, 256),
    (24, 16, 256),
    (32, 16, 256),
    (40, 16, 256),
    (48, 16, 256),
    
    (16, 16, 512),
    (24, 16, 512),
    (32, 16, 512),
    (40, 16, 512),
    
    (16, 16, 1024),
    (24, 16, 1024),
    (32, 16, 1024),
]

# Load existing results to avoid re-running tests
output_file = 'simple_benchmark_results.json'
results = []
completed_configs = set()

import os
if os.path.exists(output_file):
    print(f"Loading existing results from {output_file}...")
    with open(output_file, 'r') as f:
        data = json.load(f)
        results = data.get('results', [])
        # Track which configs we've already tested
        for r in results:
            completed_configs.add((r['num_qubits'], r['depth'], r['bond_dimension']))
    print(f"Found {len(results)} existing results\n")
else:
    print("No existing results found, starting fresh\n")

print("="*80)
print("MPS GPU vs CPU Benchmark - Simple Version")
print("="*80)
print(f"Total configurations: {len(test_configs)}")
print(f"Already completed: {len(completed_configs)}")
print(f"To run: {len(test_configs) - len(completed_configs)}")
print(f"Devices tested: mps.gpu, mps.cpu")
print("="*80)

# One-time warmup to eliminate cold-start for both devices
if len(test_configs) - len(completed_configs) > 0:
    print("\n🔥 Running one-time warmup to eliminate cold-start effects...")
    warmup_qc = quantum_volume(8, 8, seed=42)
    print("  Warming up GPU...")
    _ = bq.run(warmup_qc, device="mps.gpu", options={"mps_bond_dimension": 8})
    print("  Warming up CPU...")
    _ = bq.run(warmup_qc, device="mps.cpu", options={"mps_bond_dimension": 8})
    print("  ✓ Warmup complete!\n")

for num_qubits, depth, bond_dim in test_configs:
    # Skip if already tested
    if (num_qubits, depth, bond_dim) in completed_configs:
        print(f"\n✓ Skipping: {num_qubits} qubits, depth={depth}, bond_dim={bond_dim} (already tested)")
        continue
    
    print(f"\nTesting: {num_qubits} qubits, depth={depth}, bond_dim={bond_dim}")
    
    # Create the same circuit for both devices
    qc = quantum_volume(num_qubits, depth, seed=42)
    num_gates = qc.decompose().size()
    
    # Test GPU
    print(f"  Running on mps.gpu...")
    job_gpu = bq.run(qc, device="mps.gpu", options={"mps_bond_dimension": bond_dim})
    gpu_time = job_gpu.run_time_ms
    print(f"    GPU time: {gpu_time} ms")
    
    # Test CPU
    print(f"  Running on mps.cpu...")
    job_cpu = bq.run(qc, device="mps.cpu", options={"mps_bond_dimension": bond_dim})
    cpu_time = job_cpu.run_time_ms
    print(f"    CPU time: {cpu_time} ms")
    
    # Calculate speedup (positive = GPU faster, negative = GPU slower)
    speedup = cpu_time / gpu_time
    
    # Store result
    result = {
        'num_qubits': num_qubits,
        'depth': depth,
        'bond_dimension': bond_dim,
        'num_gates': num_gates,
        'gpu_time_ms': gpu_time,
        'cpu_time_ms': cpu_time,
        'speedup': speedup,
        'gpu_faster': speedup > 1.0
    }
    results.append(result)
    
    print(f"    Speedup: {speedup:.2f}x ({'GPU FASTER' if speedup > 1 else 'CPU FASTER'})")

# Save results (output_file already defined above)
with open(output_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results
    }, f, indent=2)

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

gpu_faster_count = sum(1 for r in results if r['gpu_faster'])
cpu_faster_count = len(results) - gpu_faster_count

print(f"\nTotal tests: {len(results)}")
print(f"GPU faster: {gpu_faster_count} times")
print(f"CPU faster: {cpu_faster_count} times")

avg_speedup = sum(r['speedup'] for r in results) / len(results)
print(f"\nAverage speedup: {avg_speedup:.2f}x")
if avg_speedup > 1:
    print("→ GPU is on average FASTER than CPU")
else:
    print("→ CPU is on average FASTER than GPU")

print("\nDetailed Results:")
print("-"*80)
print(f"{'Qubits':<8} {'Depth':<8} {'Bond':<8} {'GPU (ms)':<12} {'CPU (ms)':<12} {'Speedup':<10} {'Winner'}")
print("-"*80)
for r in results:
    winner = "GPU" if r['gpu_faster'] else "CPU"
    print(f"{r['num_qubits']:<8} {r['depth']:<8} {r['bond_dimension']:<8} "
          f"{r['gpu_time_ms']:<12} {r['cpu_time_ms']:<12} "
          f"{r['speedup']:<10.2f} {winner}")

print("="*80)
print(f"Results saved to: {output_file}")
print("="*80)
