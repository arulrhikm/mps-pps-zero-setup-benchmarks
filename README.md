# Zero-Setup Quantum Simulator Benchmarks

This repository contains the code and experiment configurations for the paper:

**Benchmarking Zero-Setup Quantum Circuit Simulators**  
Arul Rhik Mazumder, Hayk Tepanyan

It reproduces the benchmark workflows and plots for:
- state-vector (SV) runtime scaling
- matrix product state (MPS) scaling (bond dimension, depth, qubits, shots)
- Pauli-path simulation (PPS) runtime/accuracy scaling

## Reproducibility Scope

The repository provides:
- experiment runners
- input circuit generation/loading logic
- plotting scripts
- JSONL result formats used in the paper

## Quick Reproduction

1. Set `BLUEQUBIT_API_TOKEN`
2. Run experiment scripts under `benchmarking/.../experiments/`
3. Generate figures with scripts under `benchmarking/.../plotting/`
4. Compare produced figures against `paper/plots` references

## Citation

If you use this artifact, please cite:

```bibtex
% add your BibTeX entry here
