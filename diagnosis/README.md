# Diagnosis experiments and analyses

This directory contains the fixed-100 BART experiments, stored chains, paper
analysis, timing measurements, and older exploratory work. The main comparison
uses `default`, `default_pt`, `mtmh`, and `mtmh_pt`, with `default_long` as the
long-chain empirical reference.

## Directory guide

| Path | Purpose |
| --- | --- |
| `store/` | Stored predictions, traces, splits, temperatures, and metadata for every dataset and run. |
| `store_default_ext/` | Extended run-0 Default chains, 30,000 stored draws per setting. |
| `store_scaling/` | The sample-size and noise sweep. |
| `timing/` | PT timing benchmark, per-run CSV files, and the timing summary used by the paper analysis. |
| `paper/` | Standalone diagnosis and comparison scripts plus paper-facing tables, figures, and summaries. |
| `exploratory/` | Earlier general analysis scripts, the notebook, and exploratory results. |
| `run_fixed100.py` | Entry point for running selected fixed datasets and run IDs. |
| `experiment_fixed100.py` | Temperature-ladder search, chain execution, and fixed-100 result writing. |
| `fixed100_support.py` | Dataset loading and sparse-data utilities. |

## Run experiments

Run commands from the repository root. Results go to `diagnosis/store/` by
default. `--store-dir` selects another location relative to `diagnosis/`.

```bash
python diagnosis/run_fixed100.py fixed --datasets abalone --skip-long
python diagnosis/run_fixed100.py fixed --datasets abalone --run-ids 3 4 --n-chains 1 --n-jobs 1 --skip-short
python diagnosis/run_fixed100.py sparse --variants friedman_p20_k5 friedman_p200_k5
```

For `fixed`, `--skip-long` runs only short methods and `--skip-short` runs only
the long reference. `--run-ids` selects specific run numbers. `--long-tail-ndpost N`
additionally keeps the closing `N` long-chain iterations at stride 1; the tail is
written only after the final chunk, so a run that stops early leaves none. Use
`--help` for all settings.

## Paper analysis

```bash
python diagnosis/paper/section3_default.py
python diagnosis/paper/section6_methods.py
```

The scripts analyze five paired runs of every registered setting. They write CSV
tables to `paper/tables/` and figures to `paper/figures/`. The rolling R-hat
window is 1,000 draws and the step is 100.

## Which directory each analysis reads

| Directory | Read by | Holds |
|---|---|---|
| `store/` | every section script | the main runs |
| `store_default_ext/` | `section6_equal_time` | the extended Default runs |
| `store_scaling/` | `section3_scaling` | the sample-size and noise sweep |
| `timing/` | `section3_long_budget` `section6_*` `section7_costs` | run-0 wall-clock summaries |

Each script takes the matching path as its `--store-root` (or `--extended-store`,
`--timing-dir`) default, so running one without arguments reads the directory
named above. No analysis script contains an absolute cluster path.

### `store/`

One directory per setting, each with `preds/`, `rmses/`, `sigmas/`, `metadata/`,
`indices/`, and for the tempered arms `swap_temperatures/` and
`swap_accept_rates/`. File names are `<tag>__run<NNN>__<method>__<kind>.csv`.

Coverage is five runs of all four short methods and `default_long` for every
setting, with two exceptions worth remembering:

- `default_long_tail` exists only for `fixed100_Abalone` and
  `fixed100_SeoulBike`, which were rebuilt with `--long-tail-ndpost 10000`.
- `fixed100_Superconduct_subsample5000` has the four short methods only; its
  long phase was lost when the cluster home quota filled.

`openml_cache/` and `uci_cache/` hold downloaded source data, not runs.

### `store_default_ext/`

The predictions here are `.npz`, not `.csv`: the cluster wrote ~476 MB of CSV per
setting and the local copy is the same array compressed to ~80 MB, shaped
`(chain, test point, draw)` = `(4, 100, 30000)`. `provenance.json` records each
source file's size and modification time so the conversion can be checked against
the cluster copy. `sigmas/` stayed CSV.

### Cluster directories that were not inputs

The cluster accumulated staging copies that no analysis read. They were compared
file by file against this tree, found to add nothing but logs, and removed:
`store_top5` and `store_long1e7` (per-job and consolidated copies of
`default_long`, already in `store/`), the CSV originals behind the local `.npz`
extended store, `stale_ladder_pilot_1000` and `stale_before_feature_fix`
(superseded by the swap-interval revert), and `backup/`. Where a cluster copy
differed it was the older version: Abalone and SeoulBike were rebuilt to add
their tails, so the cluster still held their pre-rebuild long chains.

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

## Adding a setting

`run_fixed100.py` writes into the directory given by `--store-dir`. Analyses then
need the setting registered in `fixed100_support.py` and, if it is to appear in a
cost or equal-time table, a `timing/<tag>_run000_summary.csv`.
