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
| fixed100_Abalone | 0 | 9970 | 9970000 | 1.0024 | 1.0067 | 2680 | 3810 | 0.0008 | True |
| fixed100_Abalone | 1 | 9970 | 9970000 | 1.0006 | 1.0043 | 6793 | 5311 | 0.0006 | True |
| fixed100_Abalone | 2 | 9970 | 9970000 | 1.0019 | 1.0039 | 1694 | 4959 | 0.0008 | True |
| fixed100_Abalone | 3 | 9970 | 9970000 | 1.0004 | 1.0028 | 13089 | 11176 | 0.0004 | True |
| fixed100_Abalone | 4 | 9970 | 9970000 | 1.0006 | 1.0033 | 6532 | 4998 | 0.0004 | True |
| fixed100_CCPP | 0 | 9970 | 9970000 | 1.4587 | 1.9184 | 8 | 14 | 0.6115 | False |
| fixed100_CCPP | 1 | 9970 | 9970000 | 1.6391 | 2.0343 | 7 | 17 | 0.6167 | False |
| fixed100_CCPP | 2 | 9970 | 9970000 | 1.7191 | 1.9925 | 6 | 15 | 0.6268 | False |
| fixed100_CCPP | 3 | 9970 | 9970000 | 1.6652 | 1.7011 | 6 | 19 | 0.5220 | False |
| fixed100_CCPP | 4 | 9970 | 9970000 | 1.6824 | 2.2806 | 6 | 13 | 0.5387 | False |
| fixed100_CalHousing_subsample5000 | 0 | 9970 | 9970000 | 1.1275 | 1.3433 | 21 | 52 | 0.0625 | False |
| fixed100_CalHousing_subsample5000 | 1 | 9970 | 9970000 | 1.1023 | 1.3802 | 24 | 110 | 0.0569 | False |
| fixed100_CalHousing_subsample5000 | 2 | 9970 | 9970000 | 1.1355 | 1.4071 | 20 | 72 | 0.0735 | False |
| fixed100_CalHousing_subsample5000 | 3 | 9970 | 9970000 | 1.2104 | 1.3498 | 13 | 45 | 0.0549 | False |
| fixed100_CalHousing_subsample5000 | 4 | 9970 | 9970000 | 1.1869 | 1.5131 | 15 | 32 | 0.1148 | False |
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
| fixed100_FriedmanSparseDir_p100 | 0 | 9970 | 9970000 | 1.0011 | 1.0048 | 5401 | 7152 | 0.0006 | True |
| fixed100_FriedmanSparseDir_p100 | 1 | 9970 | 9970000 | 1.0024 | 1.0081 | 3008 | 10027 | 0.0009 | True |
| fixed100_FriedmanSparseDir_p100 | 2 | 9970 | 9970000 | 1.0077 | 1.0155 | 364 | 462 | 0.0017 | False |
| fixed100_FriedmanSparseDir_p100 | 3 | 9970 | 9970000 | 1.0010 | 1.0054 | 4965 | 13592 | 0.0008 | True |
| fixed100_FriedmanSparseDir_p100 | 4 | 9970 | 9970000 | 1.0017 | 1.0081 | 2762 | 5672 | 0.0010 | True |
| fixed100_FriedmanSparseDir_p20 | 0 | 9970 | 9970000 | 1.0010 | 1.0058 | 4781 | 12802 | 0.0006 | True |
| fixed100_FriedmanSparseDir_p20 | 1 | 9970 | 9970000 | 1.0018 | 1.0103 | 4070 | 7474 | 0.0010 | False |
| fixed100_FriedmanSparseDir_p20 | 2 | 9970 | 9970000 | 1.0013 | 1.0061 | 5329 | 10015 | 0.0007 | True |
| fixed100_FriedmanSparseDir_p20 | 3 | 9970 | 9970000 | 1.0010 | 1.0058 | 4601 | 12416 | 0.0007 | True |
| fixed100_FriedmanSparseDir_p20 | 4 | 9970 | 9970000 | 1.0012 | 1.0069 | 2899 | 8760 | 0.0009 | True |
| fixed100_FriedmanSparseDir_p200 | 0 | 9970 | 9970000 | 1.0015 | 1.0080 | 4642 | 7759 | 0.0009 | True |
| fixed100_FriedmanSparseDir_p200 | 1 | 9970 | 9970000 | 1.0012 | 1.0063 | 3356 | 9077 | 0.0008 | True |
| fixed100_FriedmanSparseDir_p200 | 2 | 9970 | 9970000 | 1.0096 | 1.0120 | 325 | 272 | 0.0012 | False |
| fixed100_FriedmanSparseDir_p200 | 3 | 9970 | 9970000 | 1.0012 | 1.0065 | 4939 | 13892 | 0.0008 | True |
| fixed100_FriedmanSparseDir_p200 | 4 | 9970 | 9970000 | 1.0013 | 1.0083 | 2548 | 7224 | 0.0009 | True |
| fixed100_SeoulBike | 0 | 9970 | 9970000 | 2.0770 | 2.3753 | 5 | 11 | 1.7725 | False |
| fixed100_SeoulBike | 1 | 9970 | 9970000 | 1.8526 | 2.1895 | 6 | 13 | 1.9345 | False |
| fixed100_SeoulBike | 2 | 9970 | 9970000 | 1.7345 | 2.5269 | 6 | 11 | 1.4319 | False |
| fixed100_SeoulBike | 3 | 9970 | 9970000 | 2.1630 | 2.7473 | 5 | 13 | 2.2148 | False |
| fixed100_SeoulBike | 4 | 9970 | 9970000 | 2.4566 | 2.2440 | 5 | 12 | 2.0770 | False |

## Per dataset (mean over runs)

| dataset | runs | stored_draws | iterations | rhat_max | worst_rhat_max | ess_bulk_min | mixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 5 | 9970 | 9970000 | 1.0024 | 1.0067 | 1694 | True |
| fixed100_CCPP | 5 | 9970 | 9970000 | 1.7191 | 2.2806 | 6 | False |
| fixed100_CalHousing_subsample5000 | 5 | 9970 | 9970000 | 1.2104 | 1.5131 | 13 | False |
| fixed100_Concrete | 5 | 9970 | 9970000 | 1.0728 | 1.1881 | 35 | False |
| fixed100_Friedman | 5 | 9970 | 9970000 | 1.0112 | 1.0382 | 649 | False |
| fixed100_FriedmanSparseDir_p100 | 5 | 9970 | 9970000 | 1.0077 | 1.0155 | 364 | False |
| fixed100_FriedmanSparseDir_p20 | 5 | 9970 | 9970000 | 1.0018 | 1.0103 | 2899 | False |
| fixed100_FriedmanSparseDir_p200 | 5 | 9970 | 9970000 | 1.0096 | 1.0120 | 325 | False |
| fixed100_SeoulBike | 5 | 9970 | 9970000 | 2.4566 | 2.7473 | 5 | False |

Figures: one per run in `figures/`, panels (a)-(f) as described in the script docstring.
