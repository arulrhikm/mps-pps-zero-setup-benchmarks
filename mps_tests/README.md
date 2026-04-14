# MPS Tests

Comprehensive testing, analysis, and runtime prediction for Matrix Product State (MPS) quantum circuit simulation.

## Directory Structure

```
mps_tests/
├── data/                              # Raw experimental data
│   ├── all_mps_data.jsonl             # Consolidated GPU + CPU data
│   ├── quantum_volume_scaling.jsonl   # QV scaling experiment data
│   ├── cpu/                           # CPU experiment results (.jsonl)
│   └── gpu/                           # GPU experiment results (.jsonl)
│
├── experiments/                       # Scripts that RUN experiments
│   ├── cpu/                           # CPU experiment runners
│   └── gpu/                           # GPU experiment runners
│
├── analysis/                          # Scripts that ANALYZE data
│   ├── plot_predictor_fit.py          # Robust multivariate regime predictor
│   ├── predict_runtime.py             # Simple runtime predictor
│   ├── predict_runtime_regimes.py     # Regime-based linear predictor
│   ├── analyze_shots_impact.py        # Shots impact analysis
│   ├── analyze_shots_scaling.py       # Shots deviation scaling
│   ├── consolidate_mps_data.py        # Merge JSONL files
│   ├── rebuild_all_mps_data.py        # Rebuild all_mps_data.jsonl
│   └── quantum_volume_scaling.py      # QV data collection + analysis
│
├── plots/                             # Generated outputs (PNGs, text)
│   ├── predictor_fit_plot.png         # Latest predictor visualization
│   ├── predictor_pseudocode.txt       # Generated pseudocode
│   └── ...                            # All experiment plots
│
├── plotting/                          # Standalone plotting scripts
│   ├── analyze_qv_scaling*.py         # QV scaling visualizations
│   ├── plot_experiment*.py            # Individual experiment plots
│   ├── compare_experiment2_cpu_gpu.py # CPU vs GPU comparison
│   └── ...
│
├── archive/                           # Old/superseded data and scripts
├── PREDICTION_GUIDE.md                # Guide to the prediction models
└── README.md                          # This file
```

## Quick Start

### 1. Run the Predictor

```bash
python mps_tests/analysis/plot_predictor_fit.py
```

This runs the robust multivariate regime predictor, outputting:
- `plots/predictor_fit_plot.png` — fit visualization
- `plots/predictor_pseudocode.txt` — deployable prediction functions

**Results:** GPU 7.1% weighted median error, CPU 19.0%

### 2. Predict a Runtime

```bash
python mps_tests/analysis/predict_runtime_regimes.py 30 20 64
```

### 3. Rebuild Data

```bash
python mps_tests/analysis/rebuild_all_mps_data.py
```

## Prediction Model

**Robust Multivariate Regime Predictor** with ridge regression and c1≥0 constraint:

```
runtime_ms = (c1*cx_gates*n + c2*gates + c3*shots) * X² + c4
```

- 10 regimes per hardware type (GPU/CPU), selected by S = n·d·X²
- Ridge regularization (α=1e-3) prevents overfitting
- Scaled-formula fallback for sparse regimes (N < 15)
- `max(0, ...)` prevents negative predictions

See [PREDICTION_GUIDE.md](PREDICTION_GUIDE.md) for details.

## Experiments

| # | Experiment | Varies | Fixed |
|---|-----------|--------|-------|
| 1 | Qubit scaling | n: 16→96 | d, X |
| 2 | Bond scaling | X | n, d |
| 3 | Depth scaling | d | n, X |
| 4 | Sampling scaling | shots | n, d, X |
| 5 | Peaked 2Q scaling | #2-qubit gates (per circuit); swept suite also tags `tau` in filename | χ, shots=1, 5 trials, `input-peaked-circuits/*.qasm` or API |

**Peaked 2Q gate scaling** (`experiments/peaked_circuits_2q_scaling.py`): put peaked QASM under `mps_tests/input-peaked-circuits/` (e.g. swept `*_tau*_RZZ*_CZ*_*.qasm`, ordered by `tau`). If that folder is empty, uses `get_peaked_circuit(difficulty, id)`. Plot: `plotting/plot_peaked_2q_scaling.py` → `plots/fig_peaked_2q_gate_scaling.png` (CPU and GPU on one figure; log y by default).
