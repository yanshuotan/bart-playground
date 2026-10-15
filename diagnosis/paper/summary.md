# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 1 datasets.

## Main result

MTMH+PT has the lowest five-run mean for all three comparison diagnostics in all 1 settings.
Predictive changes are reported relative to training-only baselines.
The measured MTMH+PT cost is 10.38x to 14.72x the Default runtime per chain (11 of 1 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- | --- |
| Concrete | 1.138 -> 1.019 | 1.213 -> 1.049 | 0.184 -> 0.020 | 0.286 -> 0.272 | 0.270 -> 0.255 | 12.59x |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [s.md](s.md).

## Interpretation

The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.
