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
| `long_tail_note.py` | contiguous-tail comparison | the standalone note |
| `core.py` | shared loading, pointwise R-hat, between/within directions | imported, not run |
| `sync_outline_tables.py` | rewrites three outline tables from the CSVs | edits `paper/outline.tex` |

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
```


Every script accepts `--help`.

## Notes on individual parts

`section3_long_budget.py figures` reads the table that `... table` writes, so
run the table first after any change to the stored draws.

`section3_scaling.py --figure-only` redraws from the stored table without
rescoring the draws.

`section7_costs.py costs --long --reuse-ess` reuses existing ESS counts when the
retained draw count is unchanged and recalculates the variance, MSE, agreement
and wall-time summaries. Without the flag it recomputes ESS as well. Both the
finite-mixture variance and MSE identities are asserted on every run.

`section7_costs.py scalar` diagnoses the latent prediction averaged over the 100
fixed inputs and the outcome-scale noise variance, across runs 0--4 and all four
arms plus long Default. It writes run-level and chain-level CSVs, a summary, and
full run-0 traces with the warm-up boundary. Single-chain R-hat there compares
temporal halves; it does not measure independent-chain agreement.

`section6_equal_time.py baselines` uses run 0 and production wall time only.
Each single-chain estimator gets the full budget and its four stored chains are
repeated realizations for scoring; four **sequential** pooled Default chains
divide that budget equally.

## Effective sample size

Every ESS in the paper is the rank-normalized bulk ESS, reported two ways with
its denominator stated: one chain measured alone and averaged over the four,
divided by `S`; and all four chains passed to one call, divided by `C*S`. The
four-chain figure carries the between-chain variance in its own denominator, so
it falls when the chains disagree. It is a diagnostic, not the information
content of the pooled predictor.

## Shared code

`core.py` holds the pieces every section needs, including the cross-fitted
between/within eigenproblem behind all the projection figures. That function was
previously duplicated verbatim in the section 3 and section 6 scripts, so the
maths behind Figure 1 and Figure 5 could drift apart silently. There is now one
copy.
