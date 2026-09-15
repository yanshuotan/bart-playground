# BART fixed-100 analysis

This directory is the current analysis entry point.

## Contents

- `Abalone_fixed100_final.ipynb`: the only retained dataset-specific notebook.
- `run_all_analyses.py`: generates the standard diagnostics for any dataset in
  `../store/`.
- `run_all_datasets.ps1`: runs the standard analysis for the selected datasets.
- `analysis_outputs/`: generated tables, figures, and logs.

The former `core/` and `additional_diagnostics/` notebooks are archived at:

`../../backup/diagnosis_2026-09-15_before_reorganization/analysis_notebooks/`

## Usage

Run one dataset from the repository root:

```powershell
.venv\Scripts\python.exe diagnosis\analysis\run_all_analyses.py fixed100_Concrete
```

Run the dataset list configured in the PowerShell driver:

```powershell
powershell -ExecutionPolicy Bypass -File diagnosis\analysis\run_all_datasets.ps1
```

By default, the script reads from `diagnosis/store/` and writes generated
results to `diagnosis/analysis/analysis_outputs/`.
