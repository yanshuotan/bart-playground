# Long-chain mixing diagnosis

Are the stored `default_long` chains themselves mixed, and are their draws enough?
The short-chain diagnostics in `diagnosis/paper` use them as the reference, so this
asks the prior question. Windows, segments and burn-in all count *stored* draws;
multiply by `store_every` for original iterations. Every ESS is computed on one
chain alone; only R-hat, the separation ratio and the centroid distance look across
chains, which is what the four chains are for.

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
single-chain bulk ESS ≥ 100 for every chain and test point.
It is a screening rule, not a proof.

## Per run

| dataset | run | stored_draws_per_chain | iterations_per_chain | rhat_max | worst_projected_rhat | ess_bulk_min | ess_bulk_spread | ess_growth_exponent_min | ess_tail_min | between_within_ratio | mixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 0 | 9970 | 9970000 | 1.0024 | 1.0067 | 462 | 1.9x | 0.38 | 815 | 0.0008 | True |
| fixed100_Abalone | 1 | 9970 | 9970000 | 1.0006 | 1.0043 | 959 | 1.8x | 1.12 | 370 | 0.0006 | True |
| fixed100_Abalone | 2 | 9970 | 9970000 | 1.0019 | 1.0039 | 306 | 1.6x | 0.64 | 745 | 0.0008 | True |
| fixed100_Abalone | 3 | 9970 | 9970000 | 1.0004 | 1.0028 | 2243 | 1.4x | 0.90 | 1779 | 0.0004 | True |
| fixed100_Abalone | 4 | 9970 | 9970000 | 1.0006 | 1.0033 | 1431 | 1.2x | 1.17 | 1166 | 0.0004 | True |
| fixed100_Airfoil | 0 | 9970 | 9970000 | 1.2283 | 1.7932 | 2 | 1.9x | 0.03 | 14 | 0.2058 | False |
| fixed100_Airfoil | 1 | 9970 | 9970000 | 1.3494 | 1.8568 | 2 | 2.2x | 0.03 | 14 | 0.1909 | False |
| fixed100_Airfoil | 2 | 9970 | 9970000 | 1.1760 | 1.7188 | 2 | 1.9x | -0.04 | 14 | 0.2001 | False |
| fixed100_Airfoil | 3 | 9970 | 9970000 | 1.2099 | 1.8722 | 2 | 2.1x | -0.07 | 13 | 0.1712 | False |
| fixed100_Airfoil | 4 | 9970 | 9970000 | 1.2220 | 1.8107 | 2 | 2.2x | 0.11 | 14 | 0.1175 | False |
| fixed100_CCPP | 0 | 9970 | 9970000 | 1.4587 | 1.9184 | 2 | 1.1x | -0.04 | 11 | 0.6115 | False |
| fixed100_CCPP | 1 | 9970 | 9970000 | 1.6391 | 2.0343 | 1 | 1.5x | -0.07 | 13 | 0.6167 | False |
| fixed100_CCPP | 2 | 9970 | 9970000 | 1.7191 | 1.9925 | 2 | 1.2x | -0.08 | 12 | 0.6268 | False |
| fixed100_CCPP | 3 | 9970 | 9970000 | 1.6652 | 1.7011 | 1 | 2.4x | -0.13 | 12 | 0.5220 | False |
| fixed100_CCPP | 4 | 9970 | 9970000 | 1.6824 | 2.2806 | 2 | 1.2x | -0.10 | 12 | 0.5387 | False |
| fixed100_CalHousing_subsample5000 | 0 | 9970 | 9970000 | 1.1275 | 1.3433 | 2 | 2.7x | -0.16 | 16 | 0.0625 | False |
| fixed100_CalHousing_subsample5000 | 1 | 9970 | 9970000 | 1.1023 | 1.3802 | 3 | 3.3x | 0.30 | 17 | 0.0569 | False |
| fixed100_CalHousing_subsample5000 | 2 | 9970 | 9970000 | 1.1355 | 1.4071 | 3 | 3.1x | -0.03 | 19 | 0.0735 | False |
| fixed100_CalHousing_subsample5000 | 3 | 9970 | 9970000 | 1.2104 | 1.3498 | 2 | 2.5x | -0.03 | 16 | 0.0549 | False |
| fixed100_CalHousing_subsample5000 | 4 | 9970 | 9970000 | 1.1869 | 1.5131 | 3 | 2.1x | -0.10 | 15 | 0.1148 | False |
| fixed100_Concrete | 0 | 9970 | 9970000 | 1.0045 | 1.0224 | 183 | 1.6x | 0.88 | 267 | 0.0030 | False |
| fixed100_Concrete | 1 | 9970 | 9970000 | 1.0076 | 1.0376 | 145 | 1.4x | 1.16 | 533 | 0.0053 | False |
| fixed100_Concrete | 2 | 9970 | 9970000 | 1.0199 | 1.1073 | 20 | 2.9x | 0.54 | 41 | 0.0128 | False |
| fixed100_Concrete | 3 | 9970 | 9970000 | 1.0728 | 1.1881 | 3 | 12.3x | 0.11 | 28 | 0.0264 | False |
| fixed100_Concrete | 4 | 9970 | 9970000 | 1.0054 | 1.0227 | 159 | 2.2x | 1.02 | 728 | 0.0032 | False |
| fixed100_Friedman | 0 | 9970 | 9970000 | 1.0112 | 1.0382 | 55 | 4.0x | 0.96 | 389 | 0.0054 | False |
| fixed100_Friedman | 1 | 9970 | 9970000 | 1.0049 | 1.0213 | 93 | 2.1x | 1.23 | 418 | 0.0026 | False |
| fixed100_Friedman | 2 | 9970 | 9970000 | 1.0084 | 1.0297 | 122 | 2.2x | 0.96 | 506 | 0.0032 | False |
| fixed100_Friedman | 3 | 9970 | 9970000 | 1.0044 | 1.0286 | 156 | 1.8x | 0.96 | 425 | 0.0029 | False |
| fixed100_Friedman | 4 | 9970 | 9970000 | 1.0060 | 1.0219 | 93 | 2.6x | 0.63 | 181 | 0.0025 | False |
| fixed100_FriedmanSparseDir_p100 | 0 | 9970 | 9970000 | 1.0011 | 1.0048 | 1116 | 1.4x | 1.13 | 1168 | 0.0006 | True |
| fixed100_FriedmanSparseDir_p100 | 1 | 9970 | 9970000 | 1.0024 | 1.0081 | 691 | 1.4x | 1.08 | 1705 | 0.0009 | True |
| fixed100_FriedmanSparseDir_p100 | 2 | 9970 | 9970000 | 1.0077 | 1.0155 | 78 | 2.1x | 0.46 | 73 | 0.0017 | False |
| fixed100_FriedmanSparseDir_p100 | 3 | 9970 | 9970000 | 1.0010 | 1.0054 | 1162 | 1.1x | 1.08 | 2971 | 0.0008 | True |
| fixed100_FriedmanSparseDir_p100 | 4 | 9970 | 9970000 | 1.0017 | 1.0081 | 627 | 1.4x | 0.91 | 938 | 0.0010 | True |
| fixed100_FriedmanSparseDir_p20 | 0 | 9970 | 9970000 | 1.0010 | 1.0058 | 1065 | 1.2x | 0.95 | 3129 | 0.0006 | True |
| fixed100_FriedmanSparseDir_p20 | 1 | 9970 | 9970000 | 1.0018 | 1.0103 | 806 | 1.4x | 0.95 | 1553 | 0.0010 | False |
| fixed100_FriedmanSparseDir_p20 | 2 | 9970 | 9970000 | 1.0013 | 1.0061 | 1204 | 1.2x | 1.07 | 2463 | 0.0007 | True |
| fixed100_FriedmanSparseDir_p20 | 3 | 9970 | 9970000 | 1.0010 | 1.0058 | 1024 | 1.2x | 1.01 | 2858 | 0.0007 | True |
| fixed100_FriedmanSparseDir_p20 | 4 | 9970 | 9970000 | 1.0012 | 1.0069 | 582 | 1.6x | 1.10 | 1912 | 0.0009 | True |
| fixed100_FriedmanSparseDir_p200 | 0 | 9970 | 9970000 | 1.0015 | 1.0080 | 1013 | 1.2x | 0.99 | 674 | 0.0009 | True |
| fixed100_FriedmanSparseDir_p200 | 1 | 9970 | 9970000 | 1.0012 | 1.0063 | 719 | 1.4x | 1.03 | 1972 | 0.0008 | True |
| fixed100_FriedmanSparseDir_p200 | 2 | 9970 | 9970000 | 1.0096 | 1.0120 | 57 | 2.4x | -0.12 | 35 | 0.0012 | False |
| fixed100_FriedmanSparseDir_p200 | 3 | 9970 | 9970000 | 1.0012 | 1.0065 | 995 | 1.3x | 1.25 | 2722 | 0.0008 | True |
| fixed100_FriedmanSparseDir_p200 | 4 | 9970 | 9970000 | 1.0013 | 1.0083 | 589 | 1.1x | 0.95 | 1541 | 0.0009 | True |
| fixed100_SeoulBike | 0 | 9970 | 9970000 | 2.0770 | 2.3753 | 1 | 1.4x | -0.04 | 11 | 1.7725 | False |
| fixed100_SeoulBike | 1 | 9970 | 9970000 | 1.8526 | 2.1895 | 1 | 1.9x | -0.02 | 11 | 1.9345 | False |
| fixed100_SeoulBike | 2 | 9970 | 9970000 | 1.7345 | 2.5269 | 1 | 1.1x | -0.04 | 12 | 1.4319 | False |
| fixed100_SeoulBike | 3 | 9970 | 9970000 | 2.1630 | 2.7473 | 2 | 1.4x | 0.01 | 11 | 2.2148 | False |
| fixed100_SeoulBike | 4 | 9970 | 9970000 | 2.4566 | 2.2440 | 1 | 1.2x | -0.04 | 10 | 2.0770 | False |

## Per dataset (mean over runs)

| dataset | runs | stored_draws | iterations | rhat_max | worst_rhat_max | ess_bulk_min | ess_growth_exponent_min | mixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 5 | 9970 | 9970000 | 1.0024 | 1.0067 | 306 | 0.38 | True |
| fixed100_Airfoil | 5 | 9970 | 9970000 | 1.3494 | 1.8722 | 2 | -0.07 | False |
| fixed100_CCPP | 5 | 9970 | 9970000 | 1.7191 | 2.2806 | 1 | -0.13 | False |
| fixed100_CalHousing_subsample5000 | 5 | 9970 | 9970000 | 1.2104 | 1.5131 | 2 | -0.16 | False |
| fixed100_Concrete | 5 | 9970 | 9970000 | 1.0728 | 1.1881 | 3 | 0.11 | False |
| fixed100_Friedman | 5 | 9970 | 9970000 | 1.0112 | 1.0382 | 55 | 0.63 | False |
| fixed100_FriedmanSparseDir_p100 | 5 | 9970 | 9970000 | 1.0077 | 1.0155 | 78 | 0.46 | False |
| fixed100_FriedmanSparseDir_p20 | 5 | 9970 | 9970000 | 1.0018 | 1.0103 | 582 | 0.95 | False |
| fixed100_FriedmanSparseDir_p200 | 5 | 9970 | 9970000 | 1.0096 | 1.0120 | 57 | -0.12 | False |
| fixed100_SeoulBike | 5 | 9970 | 9970000 | 2.4566 | 2.7473 | 1 | -0.04 | False |

Figures: one per run in `figures/`, panels (a)-(f) as described in the script docstring.
