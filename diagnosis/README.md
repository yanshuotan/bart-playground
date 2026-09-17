# Diagnosis experiments and analyses

This directory holds fixed-100 test-point BART experiments, their stored results,
and analyses. The main comparison uses `default`, `default_pt`, `mtmh`, and
`mtmh_pt` short chains, with `default_long` as a long-chain reference where
available.

## Directory guide

| Path | Purpose |
| --- | --- |
| `store/` | Results for Abalone, CalHousing, CCPP, Concrete, Friedman, FriedmanSparseDir (p20/p100/p200), and SeoulBike. Each dataset has metric CSVs and metadata. |
| `run_fixed100.py` | Single entry point for the fixed datasets, selected run IDs, and the p20/p200 sparse variants. |
| `experiment_fixed100.py` | Runs chains and writes fixed-100 results. `fixed100_support.py` provides data loading, ladder search, and sparse data generation. |
| `analysis/` | Analysis scripts, the Abalone notebook, figures and tables in `analysis_outputs/`, and timing results in `timing_outputs/`. |

New runs save predictions, per-draw RMSE/noise traces, PT swap summaries, test
splits, and metadata. The analysis scripts recompute predictive RMSE and CRPS
from predictions and test targets.

## Run experiments

Run commands from the repository root. Results go to `diagnosis/store/` by
default. `--store-dir` selects another location relative to `diagnosis/`.

```bash
python diagnosis/run_fixed100.py fixed --datasets abalone --skip-long
python diagnosis/run_fixed100.py fixed --datasets abalone --run-ids 3 4 --ladder original --skip-short
python diagnosis/run_fixed100.py sparse --variants friedman_p20_k5 friedman_p200_k5
```

For `fixed`, `--skip-long` runs only short methods and `--skip-short` runs only
the long reference. `--run-ids` selects specific run numbers. The sparse command
uses the same output format and allows chain count, draws, and ladder settings
to be changed through command-line options. Use `--help` after either command
to see all settings.

## Analyze stored results

```bash
python diagnosis/analysis/run_all_analyses.py fixed100_Abalone
python diagnosis/analysis/benchmark_pt_parallelization.py --dry-run
```

`run_all_analyses.py` reads a named directory in `store/` and writes diagnostic
tables and figures to `analysis/analysis_outputs/<dataset>/`. Use `--only` to
select analysis groups. `analysis/run_all_datasets.ps1` runs this script for
its configured dataset list. `analysis/Abalone_fixed100_final.ipynb` contains
the Abalone notebook analysis.

`benchmark_pt_parallelization.py` measures Abalone PT execution time using the
stored split and temperatures; its CSV summaries go to `analysis/timing_outputs/`.
`--dry-run` checks inputs without fitting models.
