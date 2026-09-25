# Paper analysis

This directory contains the paper-facing diagnosis and comparison analyses for
`fixed100_Abalone`, `fixed100_Concrete`, and `fixed100_Friedman`. Each dataset
uses runs `000` through `004`.

The two scripts are self-contained and do not import scripts from
`diagnosis/exploratory`.

## Run from the repository root

```powershell
.\.venv\Scripts\python.exe diagnosis\paper\diagnosis.py
.\.venv\Scripts\python.exe diagnosis\paper\comparison.py
```

On Vanda, submit the two analyses as separate jobs:

```bash
qsub diagnosis/paper/run_diagnosis.pbs
qsub diagnosis/paper/run_comparison.pbs
```

Both Python scripts accept command-line options. Use `--help` to list them.

## Outputs

- `diagnosis.py` writes Default-only tables to `tables/`, figures to
  `figures/diagnosis/`, and `diagnosis_summary.md`.
- `comparison.py` writes four-method tables to `tables/`, figures to
  `figures/comparison/`, `comparison_summary.md`, and the combined
  `summary.md`.
- Timing values are read from `diagnosis/timing/summary.md`; timing experiments
  are not rerun by the paper job.

## Analysis conventions

- Cross-chain pointwise rank-split R-hat uses rolling windows of 1,000 draws
  with a step of 100.
- Within-chain segment rank-split R-hat treats four adjacent 1,000-draw
  segments as dependent pseudo-chains, also advanced in steps of 100. It
  measures local stability and is not a formal convergence certificate.
- The worst direction is the leading generalized eigenvector of between-chain
  and pooled within-chain covariance in the original 100-test-point prediction
  space, calculated after the same 3,000-draw short-chain burn-in used for the
  post-burn comparisons.
- `long_burn=30` counts stored long-chain draws. Since the long chains were
  saved after downsampling, its effective burn-in is
  `30 × long_store_every` original iterations. The stored metadata gives
  `long_store_every=100` for Abalone and `1000` for Concrete and Friedman.
- The original-space between/within statistic is the mean squared distance
  between chain centroids divided by the mean within-chain squared radius.
- Raw energy distance is calculated for all 16 short-chain/long-chain pairs.
  Each chain is independently sampled without replacement to 1,000 post-burn
  draws using reproducible random seeds; the run-level value is the mean over
  the 16 pairs, and their standard deviation records chain-pair heterogeneity.
  The cross-dataset version divides the pair mean by
  `SD(y_train) × sqrt(number of test points)`.
- Relative RMSE divides method RMSE by the RMSE of the training-mean predictor.
  Relative CRPS divides method CRPS by the CRPS of the empirical training-target
  distribution. Both baselines use only the training portion of each run, and
  values below one improve on that baseline.
- Raw RMSE, CRPS, and energy distance remain in the CSV outputs.
- Only measured parallel PT timing rows are included in the paper tables.
