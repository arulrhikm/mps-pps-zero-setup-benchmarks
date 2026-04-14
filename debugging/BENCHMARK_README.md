# Simple GPU vs CPU Benchmark

**Purpose**: Provide clear, irrefutable evidence comparing `mps.gpu` vs `mps.cpu` performance.

## Quick Start

```bash
# 1. Run the benchmark (takes ~5-10 minutes)
python simple_gpu_vs_cpu_benchmark.py

# 2. Generate plots and report
python visualize_simple_benchmark.py
```

## What It Does

The benchmark:
1. Creates identical quantum volume circuits
2. Runs each circuit on **both** `mps.gpu` and `mps.cpu`
3. Measures runtime for each device
4. Calculates speedup (CPU time / GPU time)
   - **Speedup > 1.0** = GPU is faster
   - **Speedup < 1.0** = CPU is faster

## Test Configurations

| Qubits | Depth | Bond Dimension |
|--------|-------|----------------|
| 8      | 8     | 8              |
| 16     | 16    | 16             |
| 24     | 16    | 16             |
| 32     | 16    | 16             |
| 40     | 16    | 16             |
| 48     | 16    | 16             |
| 56     | 16    | 16             |
| 64     | 16    | 16             |

## Output Files

After running both scripts, you'll have:

1. **`simple_benchmark_results.json`** - Raw data in JSON format
2. **`simple_benchmark_plot.png`** - Visual comparison (2 plots)
3. **`benchmark_report.txt`** - Text summary with conclusion

## Sharing with Engineers

Share all three output files. The code is intentionally simple:
- No complex logic
- No caching or optimization tricks
- Direct comparison: same circuit, both devices
- Clear speedup calculation: `cpu_time / gpu_time`

## Code Transparency

The benchmark code is ~100 lines with no hidden complexity:

```python
# For each configuration:
qc = quantum_volume(num_qubits, depth, seed=42)

# Run on GPU
job_gpu = bq.run(qc, device="mps.gpu", options={"mps_bond_dimension": bond_dim})
gpu_time = job_gpu.run_time_ms

# Run on CPU  
job_cpu = bq.run(qc, device="mps.cpu", options={"mps_bond_dimension": bond_dim})
cpu_time = job_cpu.run_time_ms

# Calculate speedup
speedup = cpu_time / gpu_time  # > 1.0 means GPU faster
```

That's it. No tricks, no complex analysis.

## Interpreting Results

- **Speedup = 2.0x**: GPU is 2x faster than CPU
- **Speedup = 0.5x**: GPU is 2x slower than CPU (CPU is faster)
- **Speedup = 1.0x**: GPU and CPU have same performance

## Expected Runtime

- ~30 seconds per configuration (both GPU + CPU)
- 8 configurations total
- **Total time: ~5-10 minutes**

## Customizing Tests

To add more configurations, edit `simple_gpu_vs_cpu_benchmark.py`:

```python
test_configs = [
    (num_qubits, depth, bond_dimension),
    # Add more here...
]
```
