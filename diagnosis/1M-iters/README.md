# 1M-iteration BART diagnostics

This folder contains long-chain `DefaultBART` diagnostics intended for
1,000,000 posterior iterations.

## Purpose

The runner is designed to study long single-model chains while keeping memory
usage manageable. It runs `DefaultBART` only, splitting a long chain into
chunks and storing a thinned trace.

Typical notebook settings are:

- `ndpost = 1000000`
- `nskip = 0`
- `n_trees = 100`
- `n_runs = 2`
- `n_chains = 4`
- `store_every = 100`

With `store_every=100`, the final stored trace contains every 100th posterior
state rather than all one million states.

## Files

- `experiment.py`: shared runner for chunked long-chain `DefaultBART`
  experiments.
- `real1_Abalone.ipynb`: Abalone long-chain run.
- `real4_Concrete.ipynb`: Concrete long-chain run.
- `store/`: generated CSV outputs from completed runs.

## How the runner works

`run_chain` first fits `DefaultBART` for the first chunk. It then repeatedly
calls `sampler.continue_run(...)` from the last state until the requested
`ndpost` is reached.

After each chunk, `_thin_trace(...)` keeps only every `store_every`-th state.
The model trace is reduced to the last state between chunks to avoid retaining
the full chain in memory.

## Output layout

`experiment.py` writes categorized CSV files under `store/`, including:

- `sigmas/`
- `rmses/`
- `preds/` when `store_preds=True`
- `metadata/`
- `subsample_X_test/`
- `subsample_y_test/`

Each file name includes the notebook tag, run id, model name, and data name.

## Notes

- This directory is for long-chain baseline diagnostics only.
- It does not run XGBoost initialization, MTMH, or parallel tempering.
- Increase `chunk_size` for fewer continuation calls, or decrease it if memory
  pressure becomes an issue.
