# Long-chain mixing diagnosis

Are the stored `default_long` chains themselves mixed, and are their draws enough?
The short-chain diagnostics in `diagnosis/paper` use them as the reference, so this
asks the prior question. Windows, segments and burn-in all count *stored* draws;
multiply by `store_every` for original iterations.

## Settings

- `long_burn`: 30
- `window`: 1000
- `step`: 100
- `segment_length`: 1000
- `n_segments`: 4
- `prefix_fractions`: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- `ridge_fraction`: 1e-08
- `rhat_threshold`: 1.01
- `ess_per_chain_target`: 100.0

`mixed` is max pointwise R-hat < 1.01 **and** worst-direction R-hat < 1.01 **and**
min bulk ESS ≥ 100 per chain. It is a screening rule, not a proof.

## Per run

| dataset | run | stored_draws_per_chain | iterations_per_chain | rhat_max | worst_projected_rhat | ess_bulk_min | ess_tail_min | between_within_ratio | mixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 0 | 9970 | 997000 | 1.0287 | 1.1107 | 112 | 343 | 0.0115 | False |
| fixed100_Abalone | 1 | 9970 | 997000 | 1.0103 | 1.0323 | 653 | 1428 | 0.0040 | False |
| fixed100_Abalone | 2 | 9970 | 997000 | 1.0103 | 1.0194 | 200 | 647 | 0.0029 | False |
| fixed100_Abalone | 3 | 9970 | 997000 | 1.0058 | 1.0192 | 1053 | 2292 | 0.0024 | False |
| fixed100_Abalone | 4 | 9970 | 997000 | 1.0074 | 1.0255 | 673 | 430 | 0.0029 | False |
| fixed100_CCPP | 0 | 9970 | 997000 | 1.5929 | 2.1933 | 7 | 12 | 0.7402 | False |
| fixed100_CCPP | 1 | 9970 | 997000 | 1.8132 | 1.9593 | 6 | 22 | 0.7335 | False |
| fixed100_CalHousing_subsample5000 | 0 | 9970 | 997000 | 1.2908 | 1.8688 | 11 | 29 | 0.2318 | False |
| fixed100_CalHousing_subsample5000 | 1 | 9970 | 997000 | 1.3234 | 2.0123 | 10 | 19 | 0.3124 | False |
| fixed100_CalHousing_subsample5000 | 2 | 9970 | 997000 | 1.1746 | 1.7240 | 16 | 43 | 0.1516 | False |
| fixed100_CalHousing_subsample5000 | 3 | 9970 | 997000 | 1.2878 | 1.8907 | 10 | 13 | 0.2311 | False |
| fixed100_CalHousing_subsample5000 | 4 | 9970 | 997000 | 1.3558 | 1.8180 | 9 | 27 | 0.3136 | False |
| fixed100_Concrete | 0 | 9970 | 9970000 | 1.0045 | 1.0224 | 959 | 1825 | 0.0030 | False |
| fixed100_Concrete | 1 | 9970 | 9970000 | 1.0076 | 1.0376 | 629 | 2260 | 0.0053 | False |
| fixed100_Concrete | 2 | 9970 | 9970000 | 1.0199 | 1.1073 | 187 | 431 | 0.0128 | False |
| fixed100_Concrete | 3 | 9970 | 9970000 | 1.0728 | 1.1881 | 35 | 64 | 0.0264 | False |
| fixed100_Concrete | 4 | 9970 | 9970000 | 1.0054 | 1.0227 | 984 | 4316 | 0.0032 | False |
| fixed100_Friedman | 0 | 9970 | 9970000 | 1.0112 | 1.0382 | 842 | 2641 | 0.0054 | False |
| fixed100_Friedman | 1 | 9970 | 9970000 | 1.0049 | 1.0213 | 699 | 1816 | 0.0026 | False |
| fixed100_Friedman | 2 | 9970 | 9970000 | 1.0084 | 1.0297 | 918 | 3361 | 0.0032 | False |
| fixed100_Friedman | 3 | 9970 | 9970000 | 1.0044 | 1.0286 | 893 | 3163 | 0.0029 | False |
| fixed100_Friedman | 4 | 9970 | 9970000 | 1.0060 | 1.0219 | 649 | 1749 | 0.0025 | False |
| fixed100_FriedmanSparseDir_p100 | 0 | 9970 | 997000 | 1.0106 | 1.0528 | 463 | 1942 | 0.0056 | False |
| fixed100_FriedmanSparseDir_p100 | 1 | 9970 | 997000 | 1.0157 | 1.0614 | 389 | 1552 | 0.0084 | False |
| fixed100_FriedmanSparseDir_p100 | 2 | 9970 | 997000 | 1.0462 | 1.0911 | 66 | 118 | 0.0092 | False |
| fixed100_FriedmanSparseDir_p100 | 3 | 9970 | 997000 | 1.0107 | 1.0474 | 439 | 1662 | 0.0057 | False |
| fixed100_FriedmanSparseDir_p100 | 4 | 9970 | 997000 | 1.0162 | 1.0604 | 306 | 490 | 0.0073 | False |
| fixed100_SeoulBike | 0 | 9970 | 9970000 | 2.0770 | 2.3753 | 5 | 11 | 1.7725 | False |
| fixed100_SeoulBike | 1 | 9970 | 9970000 | 1.8526 | 2.1895 | 6 | 13 | 1.9345 | False |
| fixed100_SeoulBike | 2 | 9970 | 9970000 | 1.7345 | 2.5269 | 6 | 11 | 1.4319 | False |
| fixed100_SeoulBike | 3 | 9970 | 9970000 | 2.1630 | 2.7473 | 5 | 13 | 2.2148 | False |
| fixed100_SeoulBike | 4 | 9970 | 9970000 | 2.4566 | 2.2440 | 5 | 12 | 2.0770 | False |

## Per dataset (mean over runs)

| dataset | runs | stored_draws | iterations | rhat_max | worst_rhat_max | ess_bulk_min | mixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 5 | 9970 | 997000 | 1.0287 | 1.1107 | 112 | False |
| fixed100_CCPP | 2 | 9970 | 997000 | 1.8132 | 2.1933 | 6 | False |
| fixed100_CalHousing_subsample5000 | 5 | 9970 | 997000 | 1.3558 | 2.0123 | 9 | False |
| fixed100_Concrete | 5 | 9970 | 9970000 | 1.0728 | 1.1881 | 35 | False |
| fixed100_Friedman | 5 | 9970 | 9970000 | 1.0112 | 1.0382 | 649 | False |
| fixed100_FriedmanSparseDir_p100 | 5 | 9970 | 997000 | 1.0462 | 1.0911 | 66 | False |
| fixed100_SeoulBike | 5 | 9970 | 9970000 | 2.4566 | 2.7473 | 5 | False |

Figures: one per run in `figures/`, panels (a)-(f) as described in the script docstring.
