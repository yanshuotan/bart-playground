# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 11 datasets.

## Main result

Predictive changes are reported relative to training-only baselines.
The measured MTMH+PT cost is 10.08x to 15.95x the Default runtime per chain (3 of 11 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Abalone | 1.370 -> 1.081 | 1.289 -> 1.076 | 1.225 -> 1.072 | 0.139 -> 0.027 | 0.659 -> 0.651 | 0.655 -> 0.647 | 10.08x |
| Airfoil | 2.555 -> 1.453 | 1.588 -> 1.173 | 1.337 -> 1.127 | 0.582 -> 0.127 | 0.498 -> 0.483 | 0.496 -> 0.469 | not measured |
| CCPP | 2.630 -> 2.291 | 1.761 -> 1.438 | 1.384 -> 1.244 | 0.805 -> 0.450 | 0.193 -> 0.190 | 0.189 -> 0.185 | not measured |
| CalHousing_subsample5000 | 2.326 -> 1.917 | 1.496 -> 1.257 | 1.284 -> 1.183 | 0.472 -> 0.195 | 0.388 -> 0.370 | 0.381 -> 0.362 | not measured |
| Concrete | 2.103 -> 1.094 | 1.389 -> 1.105 | 1.255 -> 1.091 | 0.307 -> 0.038 | 0.294 -> 0.280 | 0.279 -> 0.261 | 15.95x |
| CPUAct | 2.875 -> 2.210 | 1.867 -> 1.376 | 1.427 -> 1.208 | 1.091 -> 0.544 | 0.137 -> 0.136 | 0.189 -> 0.185 | not measured |
| Friedman | 2.174 -> 1.124 | 1.470 -> 1.120 | 1.271 -> 1.106 | 0.345 -> 0.042 | 0.212 -> 0.210 | 0.210 -> 0.208 | 15.18x |
| FriedmanSparseDir_p20 | 1.977 -> 1.042 | 1.476 -> 1.096 | 1.320 -> 1.090 | 0.255 -> 0.018 | 0.226 -> 0.221 | 0.225 -> 0.219 | not measured |
| FriedmanSparseDir_p100 | 1.890 -> 1.056 | 1.474 -> 1.095 | 1.327 -> 1.091 | 0.212 -> 0.019 | 0.226 -> 0.221 | 0.225 -> 0.220 | not measured |
| FriedmanSparseDir_p200 | 1.751 -> 1.047 | 1.491 -> 1.078 | 1.342 -> 1.076 | 0.216 -> 0.012 | 0.225 -> 0.221 | 0.223 -> 0.220 | not measured |
| SeoulBike | 2.688 -> 2.306 | 1.757 -> 1.352 | 1.354 -> 1.188 | 1.102 -> 0.413 | 0.351 -> 0.371 | 0.319 -> 0.314 | not measured |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.
