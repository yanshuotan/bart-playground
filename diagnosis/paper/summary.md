# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 11 datasets.

## Main result

Predictive changes are reported relative to training-only baselines.
The measured MTMH+PT cost is 10.08x to 15.95x the Default runtime per chain (3 of 11 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Abalone | 1.949 -> 1.311 | 1.245 -> 1.072 | 1.201 -> 1.068 | 0.102 -> 0.020 | 0.649 -> 0.644 | 0.686 -> 0.674 | 10.08x |
| Airfoil | 2.464 -> 1.959 | 1.588 -> 1.173 | 1.337 -> 1.127 | 0.582 -> 0.127 | 0.498 -> 0.483 | 0.533 -> 0.489 | not measured |
| CCPP | 2.700 -> 2.073 | 1.761 -> 1.438 | 1.384 -> 1.244 | 0.805 -> 0.450 | 0.193 -> 0.190 | 0.222 -> 0.211 | not measured |
| CPUAct | 2.670 -> 2.472 | 1.867 -> 1.376 | 1.427 -> 1.208 | 1.091 -> 0.544 | 0.137 -> 0.136 | 0.215 -> 0.202 | not measured |
| CalHousing_subsample5000 | 2.510 -> 2.236 | 1.496 -> 1.257 | 1.284 -> 1.183 | 0.472 -> 0.195 | 0.388 -> 0.370 | 0.411 -> 0.379 | not measured |
| Concrete | 2.283 -> 1.549 | 1.389 -> 1.105 | 1.255 -> 1.091 | 0.307 -> 0.038 | 0.294 -> 0.280 | 0.296 -> 0.270 | 15.95x |
| Friedman | 2.315 -> 1.529 | 1.470 -> 1.120 | 1.271 -> 1.106 | 0.345 -> 0.042 | 0.212 -> 0.210 | 0.232 -> 0.224 | 15.18x |
| FriedmanSparseDir_p100 | 2.518 -> 1.375 | 1.474 -> 1.095 | 1.327 -> 1.091 | 0.212 -> 0.019 | 0.226 -> 0.221 | 0.248 -> 0.237 | not measured |
| FriedmanSparseDir_p20 | 2.520 -> 1.370 | 1.476 -> 1.096 | 1.320 -> 1.090 | 0.255 -> 0.018 | 0.226 -> 0.221 | 0.249 -> 0.236 | not measured |
| FriedmanSparseDir_p200 | 2.453 -> 1.256 | 1.491 -> 1.078 | 1.342 -> 1.076 | 0.216 -> 0.012 | 0.225 -> 0.221 | 0.245 -> 0.237 | not measured |
| SeoulBike | 2.594 -> 2.403 | 1.635 -> 1.409 | 1.312 -> 1.222 | 0.871 -> 0.542 | 0.505 -> 0.499 | 0.504 -> 0.477 | not measured |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.
