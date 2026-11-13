# Zero-Setup Quantum Simulator Benchmarks

Code, configurations and recorded data for the paper

**Benchmarking Zero-Setup Quantum Circuit Simulators**
Arul Rhik Mazumder, Hovnatan Karapetyan, Mohammed Zuhair Mullath, Rudy H. Tanin, Hrant Gharibyan, Hayk Tepanyan
[arXiv:2607.09882](https://arxiv.org/abs/2607.09882)

The paper compares BlueQubit (CPU and GPU backends), AWS Braket SV1, Quantum Rings, PPS-Qiskit and
PauliPropagation.jl on state-vector, matrix product state (MPS) and Pauli path simulation (PPS) workloads.

## Layout

| Path | Contents |
|---|---|
| `statevector_tests/` | Quantum-volume state-vector runtime scaling, CPU and GPU |
| `mps_tests/` | MPS scaling in bond dimension, depth, qubits and shots, QFT, and the peaked-circuit overlap runs (inputs in `input-peaked-circuits/`) |
| `pauli_path_tests/` | PPS runtime and accuracy on the 127-qubit kicked Ising circuit: BlueQubit CPU/GPU, PPS-Qiskit (`pps_benchmark_qiskit.py`) and PauliPropagation.jl (`pps_benchmark.jl`) |
| `crossplatform_tests/` | AWS Braket SV1 and Quantum Rings runs |
| `tools/regenerate_all_plots.py` | Rebuilds every figure from the recorded data |

Each test directory has `experiments/` (scripts that run the benchmarks), `data/` (recorded results, one JSON
object per line), `plotting/` (figure scripts) and `plots/` (the figures).

## Reproducing

Regenerate the figures from the recorded data (no account needed):

```
python tools/regenerate_all_plots.py
```

Rerunning the experiments needs accounts on the platforms being measured. Set `BLUEQUBIT_API_TOKEN` for the
BlueQubit runs, `QR_TOKEN` and `QR_NAME` for Quantum Rings, and the usual AWS credentials for Braket
(`crossplatform_tests/experiments/*ENV.txt` show the variables). The Python scripts use `bluequbit`, `qiskit`,
`numpy`, `scipy`, `pandas` and `matplotlib`, plus `amazon-braket-sdk` and the Quantum Rings toolkit for the
cross-platform runs. The PauliPropagation.jl baseline needs Julia.

## Citing

```bibtex
@misc{mazumder2026zerosetup,
  author        = {Mazumder, Arul Rhik and Karapetyan, Hovnatan and Mullath, Mohammed Zuhair and
                   Tanin, Rudy H. and Gharibyan, Hrant and Tepanyan, Hayk},
  title         = {Benchmarking Zero-Setup Quantum Circuit Simulators},
  year          = {2026},
  eprint        = {2607.09882},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```
