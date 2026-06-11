# fixed100_testpoints

This folder is a new diagnosis benchmark. It does not replace `diagnosis/1M-iters` or `diagnosis/pt+mtmh_2000`.

## Purpose

Compare short-chain methods against a long-chain default BART reference using exactly the same 100 fixed prediction/test points and the same run-specific training subsample.

## What changed relative to the original folders

1. The original `train_test_split` inside each runner is removed. Instead, `make_fixed100_splits` selects 100 fixed test points once per dataset and reuses them for every run and method.
2. Each run samples a training subset from the remaining data pool using a deterministic run seed. Default `train_fraction=0.75` is chosen to stay close to sklearn's original train/test default.
3. The short-chain logic is copied from `pt+mtmh_2000`: `default`, `default_pt`, `mtmh`, and `mtmh_pt`, each with 2000 posterior iterations by default.
4. The long-chain logic is copied from the streaming `1M-iters` runner: `default_long` uses chunked `continue_run`, saves numeric summaries to disk, and keeps only the last sampler state in RAM.
5. Abalone uses `long_ndpost=1_000_000` and `long_store_every=100`. Concrete uses `long_ndpost=10_000_000` and `long_store_every=1000`.
6. Datasets, runs, and phases run sequentially. Only the 4 chains within the current phase are parallelized, avoiding 16 parallel workers.

## Output structure

Outputs are written under `diagnosis/fixed100_testpoints/store` by default:

```text
store/
  memory_log.csv
  fixed100_Abalone/
    preds/
    pred_samples/
    coverage/
    sigmas/
    rmses/
    leaves/
    depths/
    trace_features/
    trace_feature_columns/
    accepted_moves_logmh/
    subsample_rmse/
    subsample_crps/
    swap_accept_rates/
    subsample_X_test/
    subsample_y_test/
    indices/
    metadata/
  fixed100_Concrete/
    ... same ...
```

Method names are:

```text
default
default_pt
mtmh
mtmh_pt
default_long
```

## Run command

```bash
cd /root/bart-playground
conda activate bartts

nohup python diagnosis/fixed100_testpoints/run_fixed100.py \
  --datasets abalone concrete \
  --n-runs 2 \
  --n-chains 4 \
  --n-jobs 4 \
  --store-dir store \
  > fixed100_2runs.log 2>&1 &
```

For 3 runs, use `--n-runs 3` and redirect to `fixed100_3runs.log`.

## Monitor

```bash
watch -n 60 'free -h; echo; ps -o pid,%cpu,%mem,rss,etime,cmd -C python; echo; du -sh diagnosis/fixed100_testpoints/store; echo; df -h /root'
```
