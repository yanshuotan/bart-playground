# Paper analysis outputs

This directory contains standalone paper-facing analyses for five paired runs of 12 datasets.

## Main result

Across the 9 standard-BART datasets, MTMH+PT has the smallest five-run mean in 11 of 18 diagnostic cells. DART dimensions form a separate controlled extension.
Predictive changes use root-average-chain-MSE RMSE and observation-predictive CRPS, relative to training-derived references.
The measured MTMH+PT cost is 10.38x to 14.72x the Default runtime per chain (11 of 12 datasets have timing runs).

The table reports five-run means as `Default -> MTMH+PT`.

| dataset | cross R-hat | cross R-hat q90 | relative RMSE | relative CRPS | time / Default |
| --- | --- | --- | --- | --- | --- |
| Abalone | 1.097 -> 1.019 | 1.248 -> 1.052 | 0.659 -> 0.652 | 0.655 -> 0.647 | 10.38x |
| Airfoil | 1.298 -> 1.067 | 1.675 -> 1.137 | 0.499 -> 0.478 | 0.496 -> 0.464 | 13.24x |
| CCPP | 1.439 -> 1.179 | 1.812 -> 1.357 | 0.193 -> 0.189 | 0.189 -> 0.184 | 12.51x |
| CalHousing_subsample5000 | 1.251 -> 1.101 | 1.531 -> 1.224 | 0.388 -> 0.366 | 0.381 -> 0.359 | 11.69x |
| Concrete | 1.173 -> 1.029 | 1.394 -> 1.064 | 0.294 -> 0.282 | 0.279 -> 0.262 | 12.59x |
| CPUAct | 1.518 -> 1.211 | 1.927 -> 1.410 | 0.137 -> 0.133 | 0.189 -> 0.182 | 14.72x |
| Friedman | 1.233 -> 1.027 | 1.482 -> 1.055 | 0.212 -> 0.210 | 0.210 -> 0.208 | 13.01x |
| FriedmanSparseDir_p20 | 1.199 -> 1.018 | 1.399 -> 1.036 | 0.226 -> 0.221 | 0.225 -> 0.220 | 13.21x |
| FriedmanSparseDir_p100 | 1.173 -> 1.014 | 1.353 -> 1.029 | 0.226 -> 0.220 | 0.225 -> 0.219 | 13.41x |
| FriedmanSparseDir_p200 | 1.184 -> 1.014 | 1.359 -> 1.026 | 0.225 -> 0.221 | 0.223 -> 0.220 | 13.73x |
| SeoulBike | 1.466 -> 1.271 | 1.913 -> 1.534 | 0.351 -> 0.379 | 0.319 -> 0.318 | 13.60x |
| Superconduct_subsample5000 | 1.441 -> 1.140 | 1.978 -> 1.308 | 0.359 -> 0.343 | 0.354 -> 0.326 | not measured |

## Default diagnosis

See [diagnosis_summary.md](diagnosis_summary.md).

## Four-method comparison

See [comparison_summary.md](comparison_summary.md).

## Interpretation

The results support improved mixing through agreement across pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.
