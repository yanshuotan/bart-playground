# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 9 datasets.

## Main result

Predictive changes are reported relative to training-only baselines.
The measured MTMH+PT cost is 10.08x to 15.95x the Default runtime per chain (3 of 9 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Abalone | 1.949 -> 1.311 | 1.245 -> 1.072 | 1.201 -> 1.068 | 0.103 -> 0.020 | 0.019 -> 0.004 | 0.649 -> 0.644 | 0.686 -> 0.674 | 10.08x |
| CCPP | 2.700 -> 2.073 | 1.761 -> 1.438 | 1.384 -> 1.244 | 0.801 -> 0.450 | 0.072 -> 0.050 | 0.193 -> 0.190 | 0.222 -> 0.211 | not measured |
| CalHousing_subsample5000 | 2.510 -> 2.236 | 1.496 -> 1.257 | 1.284 -> 1.183 | 0.471 -> 0.194 | 0.083 -> 0.038 | 0.388 -> 0.370 | 0.411 -> 0.379 | not measured |
| Concrete | 2.283 -> 1.549 | 1.389 -> 1.105 | 1.255 -> 1.091 | 0.306 -> 0.038 | 0.044 -> 0.008 | 0.294 -> 0.280 | 0.296 -> 0.270 | 15.95x |
| Friedman | 2.315 -> 1.529 | 1.470 -> 1.120 | 1.271 -> 1.106 | 0.343 -> 0.043 | 0.037 -> 0.005 | 0.212 -> 0.210 | 0.232 -> 0.224 | 15.18x |
| FriedmanSparseDir_p100 | 2.518 -> 1.375 | 1.474 -> 1.095 | 1.327 -> 1.091 | 0.213 -> 0.019 | 0.021 -> 0.002 | 0.226 -> 0.221 | 0.248 -> 0.237 | not measured |
| FriedmanSparseDir_p20 | 2.520 -> 1.370 | 1.476 -> 1.096 | 1.320 -> 1.090 | 0.255 -> 0.018 | 0.024 -> 0.002 | 0.226 -> 0.221 | 0.249 -> 0.236 | not measured |
| FriedmanSparseDir_p200 | 2.453 -> 1.256 | 1.491 -> 1.078 | 1.342 -> 1.076 | 0.216 -> 0.012 | 0.023 -> 0.002 | 0.225 -> 0.221 | 0.245 -> 0.237 | not measured |
| SeoulBike | 2.594 -> 2.403 | 1.635 -> 1.409 | 1.312 -> 1.222 | 0.869 -> 0.541 | 0.240 -> 0.186 | 0.505 -> 0.499 | 0.504 -> 0.477 | not measured |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, original-space separation, and scaled energy-distance diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains, and the long Default run is an empirical reference for energy distance.
