@echo off
setlocal
cd /d "%~dp0..\..\.."
if not exist "mps_tests\experiments\peaked_overlap_bond_dimension.py" (
  echo ERROR: Run this from the benchmarking repo ^(repo root not found^).
  exit /b 1
)
python mps_tests\experiments\peaked_overlap_bond_dimension.py --device mps.cpu --rzz-gates 1320
