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
- Energy distance compares deterministic pooled subsamples of 500 short-run
  and 500 long-run predictive draws. The long Default run is an empirical
  reference.
- RMSE and CRPS use post-burn predictive draws. Only measured parallel PT
  timing rows are included in the paper tables.
