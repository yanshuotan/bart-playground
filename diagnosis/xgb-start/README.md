# XGBoost-start BART diagnostics

This folder contains small diagnostics for starting BART chains from tree
structures learned by XGBoost.

## Purpose

The experiment uses XGBoost as a structure warm start for BART:

1. Preprocess `X` and `y` with `DefaultBART`'s preprocessor.
2. Fit an `XGBRegressor` on the transformed training data.
3. Convert each XGBoost JSON tree into a `bart_playground.params.Tree`.
4. Pass those trees through `init_trees`.
5. Let the BART sampler jointly resample all initialized leaf values before
   MCMC starts.

The XGBoost tree structures are reused, but the XGBoost leaf values are not
treated as final BART leaf values. They are replaced by a joint BART posterior
leaf-value draw in `bart_playground/samplers.py`.

## Current scope

`experiment.py` currently runs only:

- `default`: `DefaultBART`
- `mtmh`: `MultiBART`

The previous `default_pt` and `mtmh_pt` branches using `ParallelTemperingBART`
are commented out in place, not deleted, so they can be restored later if
needed.

## Files

- `experiment.py`: shared runner for XGBoost-start experiments, parallelized
  over runs and chains with `joblib`.
- `test.ipynb`: minimal smoke-test notebook using Abalone. It builds XGBoost
  initialization trees and runs one `DefaultBART` chain and one `MultiBART`
  chain directly, without using `experiment.py`.
- `real1_Abalone_mtmh.ipynb`: Abalone experiment using the shared runner.
- `real4_Concrete_pt_mtmh.ipynb`: Concrete experiment notebook kept from the
  PT/MTMH workflow; current runner output is default/mtmh only.
- `syn1_Friedman_pt_mtmh.ipynb`: Friedman synthetic experiment notebook kept
  from the earlier PT/MTMH workflow.
- `store/`: generated CSV outputs from experiment runs.

## Output layout

`experiment.py` writes categorized CSV files under `store/`, including:

- `sigmas/`
- `rmses/`
- `leaves/`
- `depths/`
- `accepted_moves_logmh/`
- `subsample_rmse/`
- `subsample_crps/`
- `preds/` and `coverage/` when `store_preds=True`
- `metadata/`
- `subsample_X_test/`
- `subsample_y_test/`

Each file name includes the notebook tag, run id, model name, and data name.

## Notes

- The XGBoost conversion code lives in `bart_playground/xgb_init.py`.
- Empty leaves are repaired during XGBoost-to-BART conversion so the resulting
  BART trees are valid under the current training data.
- The sampler-level leaf resampling is intentionally joint over all initialized
  trees, not one tree at a time.
