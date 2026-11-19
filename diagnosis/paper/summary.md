# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 20 datasets.

## Main result

Across the 17 standard-BART datasets, MTMH+PT has the smallest five-run mean in 25 of 34 diagnostic cells. DART dimensions form a separate controlled extension.
Predictive changes use root-average-chain-MSE RMSE and observation-predictive CRPS, relative to training-derived references.
The measured MTMH+PT cost is 6.15x to 9.38x the Default runtime per chain (20 of 20 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | cross R-hat | cross R-hat q90 | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- |
| Abalone | 1.097 -> 1.013 | 1.248 -> 1.035 | 0.659 -> 0.652 | 0.655 -> 0.648 | 7.29x |
| Airfoil | 1.298 -> 1.056 | 1.675 -> 1.128 | 0.499 -> 0.483 | 0.496 -> 0.470 | 7.22x |
| CCPP | 1.439 -> 1.213 | 1.812 -> 1.430 | 0.193 -> 0.190 | 0.189 -> 0.185 | 7.90x |
| CalHousing_subsample10000 | 1.388 -> 1.181 | 1.734 -> 1.374 | 0.347 -> 0.348 | 0.343 -> 0.341 | 7.85x |
| Concrete | 1.173 -> 1.032 | 1.394 -> 1.065 | 0.294 -> 0.280 | 0.279 -> 0.261 | 6.56x |
| CPUAct | 1.518 -> 1.183 | 1.927 -> 1.390 | 0.137 -> 0.134 | 0.189 -> 0.183 | 9.38x |
| Friedman | 1.233 -> 1.028 | 1.482 -> 1.058 | 0.212 -> 0.209 | 0.210 -> 0.208 | 7.22x |
| FriedmanSparseDir_p20 | 1.199 -> 1.018 | 1.399 -> 1.035 | 0.226 -> 0.221 | 0.225 -> 0.220 | 7.13x |
| FriedmanSparseDir_p100 | 1.173 -> 1.021 | 1.353 -> 1.042 | 0.226 -> 0.221 | 0.225 -> 0.220 | 7.12x |
| FriedmanSparseDir_p200 | 1.184 -> 1.015 | 1.359 -> 1.029 | 0.225 -> 0.221 | 0.223 -> 0.220 | 7.38x |
| SeoulBike | 1.466 -> 1.278 | 1.913 -> 1.570 | 0.351 -> 0.373 | 0.319 -> 0.315 | 8.39x |
| Superconduct_subsample10000 | 1.562 -> 1.217 | 2.087 -> 1.446 | 0.325 -> 0.299 | 0.325 -> 0.294 | 8.86x |
| Protein_subsample10000 | 1.229 -> 1.131 | 1.469 -> 1.254 | 0.721 -> 0.714 | 0.731 -> 0.721 | 7.54x |
| SpaceGA | 1.245 -> 1.053 | 1.523 -> 1.114 | 0.529 -> 0.524 | 0.484 -> 0.474 | 7.37x |
| QSARFishToxicity | 1.013 -> 1.003 | 1.032 -> 1.007 | 0.628 -> 0.630 | 0.624 -> 0.625 | 6.15x |
| Cars | 1.029 -> 1.004 | 1.087 -> 1.009 | 0.239 -> 0.238 | 0.242 -> 0.241 | 7.71x |
| CPS88Wages_subsample10000 | 1.093 -> 1.013 | 1.203 -> 1.028 | 0.727 -> 0.727 | 0.758 -> 0.758 | 7.70x |
| BlackFriday_subsample10000 | 1.042 -> 1.018 | 1.206 -> 1.081 | 0.741 -> 0.742 | 0.742 -> 0.743 | 8.08x |
| Fiat500 | 1.027 -> 1.005 | 1.054 -> 1.009 | 0.380 -> 0.380 | 0.378 -> 0.378 | 6.59x |
| Kin8nm | 1.604 -> 1.231 | 1.995 -> 1.421 | 0.472 -> 0.427 | 0.466 -> 0.423 | 8.75x |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across pointwise cross-chain and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost.
