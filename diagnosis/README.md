# Diagnosis experiments and analyses

This directory contains the fixed-100 BART experiments, stored chains, paper
analysis, timing measurements, and older exploratory work. The main comparison
uses `default`, `default_pt`, `mtmh`, and `mtmh_pt`, with `default_long` as the
long-chain empirical reference.

## Directory guide

| Path | Purpose |
| --- | --- |
| `store/` | Stored predictions, traces, splits, temperatures, and metadata for every dataset and run. |
| `paper/` | Standalone diagnosis and comparison scripts plus paper-facing tables, figures, and summaries. |
| `timing/` | PT timing benchmark, per-run CSV files, and the timing summary used by the paper analysis. |
| `exploratory/` | Earlier general analysis scripts, the notebook, and exploratory results. |
| `run_fixed100.py` | Entry point for running selected fixed datasets and run IDs. |
| `experiment_fixed100.py` | Chain execution and fixed-100 result writing. |
| `fixed100_support.py` | Dataset loading, ladder search, and sparse-data utilities. |

## Run experiments

Run commands from the repository root. Results go to `diagnosis/store/` by
default. `--store-dir` selects another location relative to `diagnosis/`.

```bash
python diagnosis/run_fixed100.py fixed --datasets abalone --skip-long
python diagnosis/run_fixed100.py fixed --datasets abalone --run-ids 3 4 --ladder original --skip-short
python diagnosis/run_fixed100.py sparse --variants friedman_p20_k5 friedman_p200_k5
```

For `fixed`, `--skip-long` runs only short methods and `--skip-short` runs only
the long reference. `--run-ids` selects specific run numbers. Use `--help` for
all settings.

## Paper analysis

```bash
python diagnosis/paper/diagnosis.py
python diagnosis/paper/comparison.py
```

The scripts analyze five paired runs of Abalone, Concrete, and Friedman. They
write CSV tables to `paper/tables/` and figures to `paper/figures/`. The rolling
R-hat window is 1,000 draws and the step is 100.

## Exploratory analysis

```bash
python diagnosis/exploratory/run_all_analyses.py fixed100_Abalone
```

Results go to `diagnosis/exploratory/results/<dataset>/`. The notebook and the
older focused plotting scripts are retained in the same directory.

## Timing

```bash
python diagnosis/timing/benchmark_pt_parallelization.py --dry-run
```

The benchmark reads stored splits and temperature ladders. Its CSV files and
summary are written directly to `diagnosis/timing/`.
