# Long-chain mixing diagnosis

Are the stored `default_long` chains themselves mixed, and are their draws enough?
The short-chain diagnostics in `diagnosis/paper` use them as the reference, so this
asks the prior question. Windows, segments and burn-in all count *stored* draws;
multiply by `store_every` for original iterations. Every ESS is computed on one
chain alone and then averaged; only R-hat, the separation ratio and the centroid
distance look across chains, which is what the four chains are for.

## Settings

- `long_burn`: 30
- `window`: 1000
- `step`: 100
- `segment_length`: 1000
- `n_segments`: 4
- `prefix_fractions`: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- `ridge_fraction`: 1e-08
- `rhat_threshold`: 1.01
- `separation_blocks`: 4

`separation_index` divides the across-chain separation by the same statistic computed
on 4 consecutive blocks of one chain, times 4,
so that perfect mixing gives 1 whatever the autocorrelation. Above one, independent
chains sit further apart than a single chain's own drift accounts for. No threshold is
applied anywhere in this file; 1.01 appears on the figures as a reference line only.

## Per run

| dataset | run | iterations_per_chain | separation_index | between_within_ratio | separation_within_null | rhat_max | worst_projected_rhat | ess_bulk_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 0 | 9970000 | 0.83 | 0.00031 | 0.00150 | 1.0024 | 1.0067 | 5313 |
| fixed100_Abalone | 1 | 9970000 | 1.11 | 0.00021 | 0.00076 | 1.0006 | 1.0043 | 5738 |
| fixed100_Abalone | 2 | 9970000 | 1.02 | 0.00031 | 0.00122 | 1.0019 | 1.0039 | 6231 |
| fixed100_Abalone | 3 | 9970000 | 1.09 | 0.00014 | 0.00053 | 1.0004 | 1.0028 | 6802 |
| fixed100_Abalone | 4 | 9970000 | 0.98 | 0.00016 | 0.00065 | 1.0006 | 1.0033 | 6717 |
| fixed100_Airfoil | 0 | 9970000 | 3.07 | 0.07718 | 0.10058 | 1.2283 | 1.7932 | 110 |
| fixed100_Airfoil | 1 | 9970000 | 5.66 | 0.07160 | 0.05064 | 1.3494 | 1.8568 | 633 |
| fixed100_Airfoil | 2 | 9970000 | 2.93 | 0.07502 | 0.10247 | 1.1760 | 1.7188 | 123 |
| fixed100_Airfoil | 3 | 9970000 | 2.48 | 0.06422 | 0.10342 | 1.2099 | 1.8722 | 123 |
| fixed100_Airfoil | 4 | 9970000 | 3.71 | 0.04406 | 0.04748 | 1.2220 | 1.8107 | 280 |
| fixed100_CCPP | 0 | 9970000 | 4.55 | 0.22931 | 0.20144 | 1.4587 | 1.9184 | 61 |
| fixed100_CCPP | 1 | 9970000 | 5.62 | 0.23125 | 0.16467 | 1.6391 | 2.0343 | 47 |
| fixed100_CCPP | 2 | 9970000 | 4.82 | 0.23506 | 0.19500 | 1.7191 | 1.9925 | 50 |
| fixed100_CCPP | 3 | 9970000 | 5.52 | 0.19573 | 0.14177 | 1.6652 | 1.7011 | 63 |
| fixed100_CCPP | 4 | 9970000 | 4.41 | 0.20201 | 0.18332 | 1.6824 | 2.2806 | 59 |
| fixed100_CalHousing_subsample5000 | 0 | 9970000 | 1.80 | 0.02345 | 0.05207 | 1.1275 | 1.3433 | 272 |
| fixed100_CalHousing_subsample5000 | 1 | 9970000 | 1.98 | 0.02134 | 0.04315 | 1.1023 | 1.3802 | 369 |
| fixed100_CalHousing_subsample5000 | 2 | 9970000 | 2.18 | 0.02755 | 0.05065 | 1.1355 | 1.4071 | 333 |
| fixed100_CalHousing_subsample5000 | 3 | 9970000 | 1.54 | 0.02058 | 0.05359 | 1.2104 | 1.3498 | 271 |
| fixed100_CalHousing_subsample5000 | 4 | 9970000 | 2.80 | 0.04305 | 0.06157 | 1.1869 | 1.5131 | 298 |
| fixed100_Concrete | 0 | 9970000 | 1.08 | 0.00112 | 0.00414 | 1.0045 | 1.0224 | 1609 |
| fixed100_Concrete | 1 | 9970000 | 1.38 | 0.00198 | 0.00575 | 1.0076 | 1.0376 | 1325 |
| fixed100_Concrete | 2 | 9970000 | 1.61 | 0.00482 | 0.01198 | 1.0199 | 1.1073 | 1347 |
| fixed100_Concrete | 3 | 9970000 | 1.46 | 0.00990 | 0.02707 | 1.0728 | 1.1881 | 433 |
| fixed100_Concrete | 4 | 9970000 | 1.25 | 0.00122 | 0.00389 | 1.0054 | 1.0227 | 1589 |
| fixed100_Friedman | 0 | 9970000 | 1.46 | 0.00201 | 0.00549 | 1.0112 | 1.0382 | 1015 |
| fixed100_Friedman | 1 | 9970000 | 0.69 | 0.00096 | 0.00560 | 1.0049 | 1.0213 | 1051 |
| fixed100_Friedman | 2 | 9970000 | 0.92 | 0.00119 | 0.00516 | 1.0084 | 1.0297 | 1139 |
| fixed100_Friedman | 3 | 9970000 | 1.08 | 0.00110 | 0.00409 | 1.0044 | 1.0286 | 1077 |
| fixed100_Friedman | 4 | 9970000 | 0.70 | 0.00093 | 0.00533 | 1.0060 | 1.0219 | 1015 |
| fixed100_FriedmanSparseDir_p100 | 0 | 9970000 | 0.77 | 0.00022 | 0.00116 | 1.0011 | 1.0048 | 3128 |
| fixed100_FriedmanSparseDir_p100 | 1 | 9970000 | 1.04 | 0.00034 | 0.00131 | 1.0024 | 1.0081 | 2879 |
| fixed100_FriedmanSparseDir_p100 | 2 | 9970000 | 1.10 | 0.00064 | 0.00231 | 1.0077 | 1.0155 | 2662 |
| fixed100_FriedmanSparseDir_p100 | 3 | 9970000 | 0.95 | 0.00028 | 0.00118 | 1.0010 | 1.0054 | 3067 |
| fixed100_FriedmanSparseDir_p100 | 4 | 9970000 | 1.10 | 0.00037 | 0.00134 | 1.0017 | 1.0081 | 2611 |
| fixed100_FriedmanSparseDir_p20 | 0 | 9970000 | 0.97 | 0.00024 | 0.00100 | 1.0010 | 1.0058 | 3198 |
| fixed100_FriedmanSparseDir_p20 | 1 | 9970000 | 1.08 | 0.00037 | 0.00137 | 1.0018 | 1.0103 | 2937 |
| fixed100_FriedmanSparseDir_p20 | 2 | 9970000 | 1.10 | 0.00028 | 0.00100 | 1.0013 | 1.0061 | 2978 |
| fixed100_FriedmanSparseDir_p20 | 3 | 9970000 | 0.97 | 0.00026 | 0.00108 | 1.0010 | 1.0058 | 3144 |
| fixed100_FriedmanSparseDir_p20 | 4 | 9970000 | 0.89 | 0.00033 | 0.00146 | 1.0012 | 1.0069 | 2783 |
| fixed100_FriedmanSparseDir_p200 | 0 | 9970000 | 1.19 | 0.00034 | 0.00115 | 1.0015 | 1.0080 | 3144 |
| fixed100_FriedmanSparseDir_p200 | 1 | 9970000 | 0.85 | 0.00028 | 0.00134 | 1.0012 | 1.0063 | 2879 |
| fixed100_FriedmanSparseDir_p200 | 2 | 9970000 | 0.59 | 0.00044 | 0.00300 | 1.0096 | 1.0120 | 2619 |
| fixed100_FriedmanSparseDir_p200 | 3 | 9970000 | 0.99 | 0.00030 | 0.00122 | 1.0012 | 1.0065 | 3118 |
| fixed100_FriedmanSparseDir_p200 | 4 | 9970000 | 0.93 | 0.00034 | 0.00146 | 1.0013 | 1.0083 | 2679 |
| fixed100_SeoulBike | 0 | 9970000 | 9.32 | 0.66469 | 0.28533 | 2.0770 | 2.3753 | 138 |
| fixed100_SeoulBike | 1 | 9970000 | 12.21 | 0.72543 | 0.23771 | 1.8526 | 2.1895 | 107 |
| fixed100_SeoulBike | 2 | 9970000 | 6.52 | 0.53694 | 0.32950 | 1.7345 | 2.5269 | 109 |
| fixed100_SeoulBike | 3 | 9970000 | 13.41 | 0.83054 | 0.24780 | 2.1630 | 2.7473 | 87 |
| fixed100_SeoulBike | 4 | 9970000 | 10.11 | 0.77886 | 0.30802 | 2.4566 | 2.2440 | 96 |

## Per dataset, ordered by separation index (range over runs)

| dataset | runs | iterations | rhat_max | worst_rhat_max | ess_bulk_mean | separation_index |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p200 | 5 | 9970000 | 1.0096 | 1.0120 | 2619 | 0.59-1.19 |
| fixed100_Friedman | 5 | 9970000 | 1.0112 | 1.0382 | 1015 | 0.69-1.46 |
| fixed100_FriedmanSparseDir_p100 | 5 | 9970000 | 1.0077 | 1.0155 | 2611 | 0.77-1.10 |
| fixed100_Abalone | 5 | 9970000 | 1.0024 | 1.0067 | 5313 | 0.83-1.11 |
| fixed100_FriedmanSparseDir_p20 | 5 | 9970000 | 1.0018 | 1.0103 | 2783 | 0.89-1.10 |
| fixed100_Concrete | 5 | 9970000 | 1.0728 | 1.1881 | 433 | 1.08-1.61 |
| fixed100_CalHousing_subsample5000 | 5 | 9970000 | 1.2104 | 1.5131 | 271 | 1.54-2.80 |
| fixed100_Airfoil | 5 | 9970000 | 1.3494 | 1.8722 | 110 | 2.48-5.66 |
| fixed100_CCPP | 5 | 9970000 | 1.7191 | 2.2806 | 47 | 4.41-5.62 |
| fixed100_SeoulBike | 5 | 9970000 | 2.4566 | 2.7473 | 87 | 6.52-13.41 |

Figures: one per run in `figures/`, panels (a)-(f) as described in the script docstring.
