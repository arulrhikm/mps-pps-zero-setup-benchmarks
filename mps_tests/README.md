# MPS benchmarks

Matrix product state (MPS) runtime scaling on BlueQubit's CPU and GPU backends.

| Experiment | Script (`experiments/`) | Data (`data/`) | Figure (`plots/`) |
|---|---|---|---|
| Bond dimension | `bond_scaling_{cpu,gpu}.py` | `bond_scaling_*.jsonl` | `fig2_bond_dimension_scaling*.png` |
| Build vs. sampling time | `shots_scaling_{cpu,gpu}.py` | `shots_scaling_*.jsonl` | `fig3_build_vs_sampling.png`, `fig_shots_build_sampling.png` |
| Qubit count | `qubit_scaling_{cpu,gpu}.py` | `qubit_scaling_*.jsonl` | `fig4_qubit_scaling.png` |
| Depth | `depth_scaling_{cpu,gpu}.py` | `depth_scaling_*.jsonl` | `fig5_depth_scaling.png` |
| QFT | `qft_scaling_{cpu,gpu}.py` | `qft_scaling_*.jsonl` | `fig6_qft_scaling.png` |
| Peaked circuits | `peaked_overlap_bond_dimension.py` (inputs in `input-peaked-circuits/`) | `peaked-circuits-results/` | `fig_peaked_*.png` |

The plotting scripts are in `plotting/`, and `python tools/regenerate_all_plots.py` from the repository root
rebuilds every figure from the recorded data. The experiment scripts need `BLUEQUBIT_API_TOKEN`.

`archived/` holds earlier versions of these experiments and their data, kept for the record.
