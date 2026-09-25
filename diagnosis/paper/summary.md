# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of three datasets.

## Main result

MTMH+PT has the lowest five-run mean for all five reported mixing diagnostics in all three datasets.
Predictive changes are reported relative to training-only baselines.
The measured MTMH+PT cost is 10.08x to 15.95x the Default runtime per chain.

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Abalone | 1.949 -> 1.311 | 1.245 -> 1.072 | 1.201 -> 1.068 | 0.103 -> 0.020 | 0.020 -> 0.004 | 0.649 -> 0.644 | 0.686 -> 0.674 | 10.08x |
| Concrete | 2.283 -> 1.549 | 1.389 -> 1.105 | 1.255 -> 1.091 | 0.308 -> 0.038 | 0.044 -> 0.008 | 0.294 -> 0.280 | 0.296 -> 0.270 | 15.95x |
| Friedman | 2.315 -> 1.529 | 1.470 -> 1.120 | 1.271 -> 1.106 | 0.347 -> 0.043 | 0.037 -> 0.005 | 0.212 -> 0.210 | 0.232 -> 0.224 | 15.18x |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, original-space separation, and scaled energy-distance diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains, and the long Default run is an empirical reference for energy distance.
