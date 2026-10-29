# Paper analysis

One script per part of the paper, each named for the section it feeds. Every
configured dataset uses runs `000` through `004`.

| Script | Produces | Where it lands |
|---|---|---|
| `section3_default.py` | Default-only diagnosis figure and tables | Figure 1 |
| `section3_long_budget.py` | two-budget R-hat table, long-chain projection figures | `tab:rhat-budgets`, Figure 2, appendix pages |
| `section3_scaling.py` | sample-size and noise sweep | §3.5 figure and tables |
| `section6_methods.py` | four-method factorial tables, rolling and projection figures | Figures 4--5, §6 tables |
| `section6_equal_time.py` | equal-wall-time curves and the pooled-estimator baseline | §6 wall-time figure and tables |
| `section7_costs.py` | pooling, decomposition, coverage, ESS; scalar-quantity diagnostics | §7 tables and figure |
| `section7_feature_use.py` | split-use shares, inclusion probabilities and their chain agreement (`feature_use_*.csv`) | `tab:feature-use` |
| `long_tail_tables.py` | long and contiguous-tail Default records scored as methods (`*_tail_as_method_*.csv`) | `long`/`tail` columns of §7 tables, tail rows of the appendix tables |
| `long_tail_note.py` | contiguous-tail comparison | supporting figures and CSVs for the outline's tail analysis |
| `core.py` | shared loading, pointwise R-hat, between/within directions | imported, not run |

Three scripts take a subcommand as their first argument. Omitting it runs the
first one, so existing invocations still work.

## Run from the repository root

```powershell
.\.venv\Scripts\python.exe diagnosis\paper\section3_default.py
.\.venv\Scripts\python.exe diagnosis\paper\section3_long_budget.py table
.\.venv\Scripts\python.exe diagnosis\paper\section3_long_budget.py figures
.\.venv\Scripts\python.exe diagnosis\paper\section3_scaling.py
.\.venv\Scripts\python.exe diagnosis\paper\section6_methods.py
.\.venv\Scripts\python.exe diagnosis\paper\section6_equal_time.py curves
.\.venv\Scripts\python.exe diagnosis\paper\section6_equal_time.py baselines
.\.venv\Scripts\python.exe diagnosis\paper\section7_costs.py costs --long
.\.venv\Scripts\python.exe diagnosis\paper\section7_costs.py scalar
.\.venv\Scripts\python.exe diagnosis\paper\long_tail_tables.py
.\.venv\Scripts\python.exe diagnosis\paper\section7_feature_use.py
```


Every script accepts `--help`.

## Caching

`section6_methods.py`, `section7_costs.py costs`, `section7_costs.py scalar`,
`long_tail_tables.py` and `section7_feature_use.py` keep one result per (dataset, run, method) under `diagnosis/paper/cache/<script>/`
(ignored by git). An entry is reused while the analysis settings, the script's
`CACHE_VERSION` and the size and mtime of every store file it reads (that
method's preds and sigmas, the run's indices and test targets) are unchanged.
Adding a dataset or replacing one method's draws therefore recomputes only those
entries, and the tables are rebuilt from the cache in seconds. A figure is
redrawn only when one of its inputs was recomputed or the file is missing.

Bump `CACHE_VERSION` in a script whenever its per-method calculation changes;
`--no-cache` recomputes everything and rewrites the entries. The
`section3_long_budget.py table` cache (`tables/budget_comparison_cache.csv`)
checks the same input signature per run.

## Notes on individual parts

`section3_long_budget.py figures` reads the table that `... table` writes, so
run the table first after any change to the stored draws.

`section3_scaling.py --figure-only` redraws from the stored table without
rescoring the draws.

`section7_costs.py costs --long --reuse-ess` reuses ESS from a cache entry with
matching analysis settings and input-file signatures, and recalculates variance, MSE, agreement
and wall-time summaries. Without the flag it recomputes ESS as well. Both the
finite-mixture variance and MSE identities are asserted on every run.

`section7_costs.py scalar` diagnoses the latent prediction averaged over the 100
fixed inputs and the outcome-scale noise variance, across runs 0--4 and all four
arms plus long Default. It writes run-level and chain-level CSVs, a summary, and
full run-0 traces with the warm-up boundary. Single-chain R-hat there compares
temporal halves; it does not measure independent-chain agreement.

`long_tail_tables.py` scores three Default records with the section 6 and 7
definitions: production `default`, the stride-1,000 `default_long` record and
the contiguous `default_long_tail` record of the final 10,000 iterations. Each
keeps 7,000 draws per chain after the 3,000-draw warm-up. Standard-BART settings
go to `long_tail_as_method_*`, DART settings to `dart_tail_as_method_*`; the
latent-f containment columns are filled on the four generated designs only.

`section7_feature_use.py` reads `vars_histogram`, the per-draw split counts per
predictor summed over all trees, for the four methods and both long-Default
records. Split-use shares and inclusion indicators follow the package's
`feature_inclusion_frequency(normalize='per_draw')` and
`feature_inclusion_probability`. DART selection weights are the conditional
posterior means `(a/d + c_v)/(a + sum c)` with `a = 1`, computed from the same
counts.

`section6_equal_time.py baselines` uses run 0 and production wall time only.
Each single-chain estimator gets the full budget and its four stored chains are
repeated realizations for scoring; four **sequential** pooled Default chains
divide that budget equally.

## Effective sample size

The appendix reports single-chain and joint four-chain bulk ESS. The former
averages separate chain estimates and divides by `S`; the latter uses all
chains jointly and divides by `C*S`. Electronic outputs also retain tail ESS.
The main text uses a few relative bulk ESS values to describe dependence in
the terminal-window versus long-span comparison.

## Metric conventions

- Primary agreement summaries are the median and 90th percentile of pointwise
  rank-normalized split/folded R-hat. Temporal segment diagnostics and geometric
  separation summaries support these results in the appendix.
- Raw prediction variance uses `core.prediction_variance_terms` throughout:
  population denominators `S` and `C` give pointwise pooled variance `W + B`.
  Sweeps, terminal windows and finite-mixture consequences use this definition.
  Projection covariance matrices and the transformed R-hat variance terms have
  their separately stated definitions.
- Method RMSE is the square root of mean chain MSE within each run. Pooled
  RMSE scores the combined posterior-mean estimate. Setting-level tables average
  the five run scores. Electronic tables retain mean chain-wise RMSE separately.
- Relative RMSE uses the training-mean reference score; relative CRPS uses the
  training climatology. CRPS scores observation-predictive draws on outcome scale.

## Shared code

`core.py` holds the pieces every section needs, including the cross-fitted
between/within eigenproblem behind all the projection figures. That function was
previously duplicated verbatim in the section 3 and section 6 scripts, so the
maths behind Figure 1 and Figure 5 could drift apart silently. There is now one
copy.
