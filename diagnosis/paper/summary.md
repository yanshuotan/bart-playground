# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 11 datasets.

## Main result

Across the 8 standard-BART datasets, MTMH+PT has the smallest five-run mean in 15 of 24 diagnostic cells. DART dimensions form a separate controlled extension.
Predictive changes are reported relative to training-only baselines.
The measured MTMH+PT cost is 10.38x to 14.72x the Default runtime per chain (11 of 11 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- | --- |
| Abalone | 1.097 -> 1.018 | 1.185 -> 1.047 | 0.139 -> 0.027 | 0.659 -> 0.651 | 0.655 -> 0.647 | 10.38x |
| Airfoil | 1.298 -> 1.080 | 1.292 -> 1.105 | 0.582 -> 0.127 | 0.498 -> 0.483 | 0.496 -> 0.469 | 13.24x |
| CCPP | 1.439 -> 1.238 | 1.354 -> 1.218 | 0.805 -> 0.450 | 0.193 -> 0.190 | 0.189 -> 0.185 | 12.51x |
| CalHousing_subsample5000 | 1.251 -> 1.111 | 1.258 -> 1.148 | 0.472 -> 0.195 | 0.388 -> 0.370 | 0.381 -> 0.362 | 11.69x |
| Concrete | 1.173 -> 1.032 | 1.222 -> 1.066 | 0.307 -> 0.038 | 0.294 -> 0.280 | 0.279 -> 0.261 | 12.59x |
| CPUAct | 1.518 -> 1.208 | 1.409 -> 1.188 | 1.091 -> 0.544 | 0.137 -> 0.136 | 0.189 -> 0.185 | 14.72x |
| Friedman | 1.233 -> 1.037 | 1.244 -> 1.078 | 0.345 -> 0.042 | 0.212 -> 0.210 | 0.210 -> 0.208 | 13.01x |
| FriedmanSparseDir_p20 | 1.199 -> 1.019 | 1.274 -> 1.060 | 0.255 -> 0.018 | 0.226 -> 0.221 | 0.225 -> 0.219 | 13.21x |
| FriedmanSparseDir_p100 | 1.173 -> 1.019 | 1.280 -> 1.058 | 0.212 -> 0.019 | 0.226 -> 0.221 | 0.225 -> 0.220 | 13.41x |
| FriedmanSparseDir_p200 | 1.184 -> 1.014 | 1.304 -> 1.049 | 0.216 -> 0.012 | 0.225 -> 0.221 | 0.223 -> 0.220 | 13.73x |
| SeoulBike | 1.466 -> 1.219 | 1.318 -> 1.182 | 1.102 -> 0.413 | 0.351 -> 0.371 | 0.319 -> 0.314 | 13.60x |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.
