# 10M-iters: Concrete Long-Chain Baseline Experiment

This folder contains the code and notebook used for the earlier **10M-iteration Concrete baseline experiment**. The purpose of this experiment is to run long default BART chains on the UCI Concrete dataset and use the resulting posterior predictive samples as a long-chain reference for later short-chain / PT / MTMH comparisons.

## Files

### `experiment_10M.py`

This is the disk-streaming long-chain runner. It is adapted from the original `diagnosis/1M-iters/experiment.py` logic, but modified so that long traces do not stay fully in RAM.

Main responsibilities:

- run `DefaultBART` chains in chunks;
- continue each chain with `sampler.continue_run(...)`;
- thin the trace using `store_every`;
- compute and save only the quantities needed for diagnostics:
  - posterior predictions on test points;
  - `sigmas`;
  - RMSEs;
- write intermediate per-chain rows to `_stream_tmp`;
- assemble final notebook-compatible CSV files after each run.

Important detail: `continue_run(...)` returns one extra initial state, so `_thin_trace(...)` removes the duplicate initial state when needed.

### `run_concrete_10m_2runs_server.py`

This is the server entry-point script for the long Concrete run.

It:

- loads the UCI Concrete Compressive Strength dataset;
- calls `run_parallel_experiments(...)` from `experiment_10M.py`;
- runs the long-chain default BART experiment;
- starts a background memory logger;
- writes outputs to the selected `store_dir`.

Default configuration:

```bash
ndpost=10000000
n_runs=2
n_chains=4
n_jobs=4
store_every=1000
chunk_size=10000
n_trees=100
```

The intended server command was:

```bash
nohup python diagnosis/10M-iters/run_concrete_10m_2runs_server.py \
  --ndpost 10000000 \
  --n-runs 2 \
  --n-chains 4 \
  --n-jobs 4 \
  --store-every 1000 \
  --chunk-size 10000 \
  --store-dir store \
  > concrete_10m_2runs.log 2>&1 &
```

### `real4_Concrete_10M.ipynb`

This notebook is for analysis and plotting after the long-chain results have been produced.

It loads the stored CSV files and generates:

- PCA scatter plots of posterior predictive samples;
- PC1 histogram + KDE plots;
- sigma traces;
- RMSE traces;
- PC1 traces;
- optional projections of short-chain PT/MTMH samples onto the long-chain PCA axes.

The notebook expects results under a store directory such as:

```text
diagnosis/10M-iters/store/
```

or a manually edited local path after downloading the results.

## Output structure

The experiment writes results in a notebook-compatible structure:

```text
store/
├── preds/
├── sigmas/
├── rmses/
├── subsample_X_test/
├── subsample_y_test/
├── metadata/
├── memory_log.csv
└── _stream_tmp/        # temporary during running; removed after successful assembly
```

Typical final files include:

```text
real4_Concrete__run000__default__preds.csv
real4_Concrete__run000__default__sigmas.csv
real4_Concrete__run000__default__rmses.csv
real4_Concrete__run001__default__preds.csv
real4_Concrete__run001__default__sigmas.csv
real4_Concrete__run001__default__rmses.csv
```

The saved array shapes are written into the first header line of each CSV as `original_shape=...`.

For this experiment, with `store_every=1000`, each chain stores:

```text
10000000 / 1000 = 10000 posterior states
```

So the expected shapes are approximately:

```text
preds:  (4, 100, 10000)
sigmas: (4, 10000, 1)
rmses:  (4, 10000)
```

## Memory monitoring

The runner writes:

```text
store/memory_log.csv
```

with columns:

```text
elapsed_sec,rss_gb_total,n_python_processes
```

This was used to monitor RAM growth during the long run.

A useful monitoring command on the server is:

```bash
watch -n 60 'free -h; echo; ps -o pid,%cpu,%mem,rss,etime,cmd -C python; echo; du -sh diagnosis/10M-iters/store; echo; df -h /root'
```

## Notes from the completed run

The previous Concrete 10M / 2-run experiment completed successfully on a 128GB RAM server using 4 parallel chains. The final output size was small, around the hundred-MB scale, because only thinned numeric summaries were saved.

The main bottleneck was RAM, not disk. The disk-streaming code keeps output files small, but the BART sampler workers can still grow substantially in memory during long chains.

## How to reproduce

From the repository root:

```bash
cd /root/bart-playground
conda activate bartts
```

Run:

```bash
nohup python diagnosis/10M-iters/run_concrete_10m_2runs_server.py \
  --ndpost 10000000 \
  --n-runs 2 \
  --n-chains 4 \
  --n-jobs 4 \
  --store-every 1000 \
  --chunk-size 10000 \
  --store-dir store \
  > concrete_10m_2runs.log 2>&1 &
```

Monitor:

```bash
tail -f concrete_10m_2runs.log
```

and in another terminal:

```bash
watch -n 60 'free -h; echo; ps -o pid,%cpu,%mem,rss,etime,cmd -C python; echo; du -sh diagnosis/10M-iters/store; echo; df -h /root'
```

## Recommended workflow

Use the server only for the long computation. After the run finishes, download the `store/` folder and use `real4_Concrete_10M.ipynb` locally or on the server to generate plots.

Before shutting down the server, save:

- `store/`
- `concrete_10m_2runs.log`
- `memory_log.csv`
- the exact code files in this folder
- the environment requirements, if needed
