# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 12 datasets.

## Main result

Across the 9 standard-BART datasets, MTMH+PT has the smallest five-run mean in 12 of 18 diagnostic cells. DART dimensions form a separate controlled extension.
Predictive changes use root-average-chain-MSE RMSE and observation-predictive CRPS, relative to training-derived references.
The measured MTMH+PT cost is 6.56x to 9.38x the Default runtime per chain (12 of 12 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | cross R-hat | cross R-hat q90 | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- |
| Abalone | 1.097 -> 1.013 | 1.248 -> 1.035 | 0.659 -> 0.652 | 0.655 -> 0.648 | 7.29x |
| Airfoil | 1.298 -> 1.056 | 1.675 -> 1.128 | 0.499 -> 0.483 | 0.496 -> 0.470 | 7.22x |
| CCPP | 1.439 -> 1.213 | 1.812 -> 1.430 | 0.193 -> 0.190 | 0.189 -> 0.185 | 7.90x |
| CalHousing_subsample5000 | 1.251 -> 1.072 | 1.531 -> 1.170 | 0.388 -> 0.367 | 0.381 -> 0.359 | 7.42x |
| Concrete | 1.173 -> 1.032 | 1.394 -> 1.065 | 0.294 -> 0.280 | 0.279 -> 0.261 | 6.56x |
| CPUAct | 1.518 -> 1.183 | 1.927 -> 1.390 | 0.137 -> 0.134 | 0.189 -> 0.183 | 9.38x |
| Friedman | 1.233 -> 1.028 | 1.482 -> 1.058 | 0.212 -> 0.209 | 0.210 -> 0.208 | 7.22x |
| FriedmanSparseDir_p20 | 1.199 -> 1.018 | 1.399 -> 1.035 | 0.226 -> 0.221 | 0.225 -> 0.220 | 7.13x |
| FriedmanSparseDir_p100 | 1.173 -> 1.021 | 1.353 -> 1.042 | 0.226 -> 0.221 | 0.225 -> 0.220 | 7.12x |
| FriedmanSparseDir_p200 | 1.184 -> 1.015 | 1.359 -> 1.029 | 0.225 -> 0.221 | 0.223 -> 0.220 | 7.38x |
| Superconduct_subsample5000 | 1.441 -> 1.142 | 1.978 -> 1.342 | 0.359 -> 0.341 | 0.354 -> 0.325 | 7.82x |
| SeoulBike | 1.466 -> 1.278 | 1.913 -> 1.570 | 0.351 -> 0.373 | 0.319 -> 0.315 | 8.39x |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.
