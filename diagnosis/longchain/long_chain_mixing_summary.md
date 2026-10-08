# Long-chain mixing diagnosis

Are the stored `default_long` chains themselves mixed, and are their draws enough?
The short-chain diagnostics in `diagnosis/paper` use them as the reference, so this
asks the prior question. Windows, segments and burn-in all count *stored* draws;
multiply by `store_every` for original iterations. Every ESS is computed on one
chain alone and then averaged; only R-hat, the separation ratio and the centroid
distance look across chains, which is what the four chains are for.

## Settings

- `long_burn`: 3000
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
| fixed100_Abalone | 0 | 7000000 | 0.80 | 0.00029 | 0.00147 | 1.0010 | 1.0066 | 3805 |
| fixed100_Abalone | 1 | 7000000 | 1.33 | 0.00033 | 0.00098 | 1.0012 | 1.0071 | 4082 |
| fixed100_Abalone | 2 | 7000000 | 0.67 | 0.00030 | 0.00179 | 1.0046 | 1.0048 | 4385 |
| fixed100_Abalone | 3 | 7000000 | 1.20 | 0.00022 | 0.00072 | 1.0006 | 1.0043 | 4774 |
| fixed100_Abalone | 4 | 7000000 | 1.26 | 0.00026 | 0.00083 | 1.0009 | 1.0051 | 4750 |
| fixed100_Airfoil | 0 | 7000000 | 4.86 | 0.11377 | 0.09357 | 1.3364 | 1.7709 | 99 |
| fixed100_Airfoil | 1 | 7000000 | 5.57 | 0.07947 | 0.05710 | 1.5054 | 2.0064 | 576 |
| fixed100_Airfoil | 2 | 7000000 | 4.74 | 0.12280 | 0.10366 | 1.2554 | 1.8838 | 114 |
| fixed100_Airfoil | 3 | 7000000 | 3.94 | 0.10024 | 0.10189 | 1.3514 | 1.9302 | 112 |
| fixed100_Airfoil | 4 | 7000000 | 4.29 | 0.05737 | 0.05351 | 1.3053 | 1.8785 | 271 |
| fixed100_CCPP | 0 | 7000000 | 8.44 | 0.34732 | 0.16457 | 1.5360 | 1.6965 | 75 |
| fixed100_CCPP | 1 | 7000000 | 7.63 | 0.30043 | 0.15744 | 1.7632 | 1.7071 | 52 |
| fixed100_CCPP | 2 | 7000000 | 7.71 | 0.34403 | 0.17855 | 1.9158 | 2.5859 | 63 |
| fixed100_CCPP | 3 | 7000000 | 9.43 | 0.28754 | 0.12200 | 1.6340 | 2.4911 | 71 |
| fixed100_CCPP | 4 | 7000000 | 7.62 | 0.30194 | 0.15850 | 1.7204 | 2.1299 | 67 |
| fixed100_CPUAct | 0 | 7000000 | 14.52 | 0.69547 | 0.19161 | 1.5716 | 2.7939 | 100 |
| fixed100_CPUAct | 1 | 7000000 | 11.13 | 0.59800 | 0.21489 | 2.2435 | 2.7869 | 106 |
| fixed100_CPUAct | 2 | 7000000 | 14.74 | 0.80907 | 0.21953 | 2.1999 | 2.1171 | 64 |
| fixed100_CPUAct | 3 | 7000000 | 14.89 | 1.01126 | 0.27175 | 2.0363 | 2.1552 | 57 |
| fixed100_CPUAct | 4 | 7000000 | 15.76 | 0.74653 | 0.18951 | 2.0006 | 2.8414 | 86 |
| fixed100_CalHousing_subsample5000 | 0 | 7000000 | 2.53 | 0.03246 | 0.05130 | 1.1549 | 1.4339 | 237 |
| fixed100_CalHousing_subsample5000 | 1 | 7000000 | 2.03 | 0.02625 | 0.05185 | 1.1812 | 1.4462 | 320 |
| fixed100_CalHousing_subsample5000 | 2 | 7000000 | 2.90 | 0.04133 | 0.05697 | 1.1422 | 1.5598 | 289 |
| fixed100_CalHousing_subsample5000 | 3 | 7000000 | 2.78 | 0.03891 | 0.05605 | 1.2961 | 1.6586 | 243 |
| fixed100_CalHousing_subsample5000 | 4 | 7000000 | 4.30 | 0.05773 | 0.05371 | 1.2270 | 1.7168 | 255 |
| fixed100_Concrete | 0 | 7000000 | 0.98 | 0.00137 | 0.00560 | 1.0054 | 1.0227 | 1184 |
| fixed100_Concrete | 1 | 7000000 | 1.52 | 0.00272 | 0.00714 | 1.0132 | 1.0604 | 975 |
| fixed100_Concrete | 2 | 7000000 | 2.41 | 0.00560 | 0.00932 | 1.0252 | 1.1434 | 1184 |
| fixed100_Concrete | 3 | 7000000 | 1.92 | 0.01555 | 0.03236 | 1.0962 | 1.2917 | 336 |
| fixed100_Concrete | 4 | 7000000 | 1.18 | 0.00173 | 0.00587 | 1.0088 | 1.0368 | 1153 |
| fixed100_Friedman | 0 | 7000000 | 1.26 | 0.00240 | 0.00760 | 1.0158 | 1.0518 | 756 |
| fixed100_Friedman | 1 | 7000000 | 0.79 | 0.00153 | 0.00779 | 1.0082 | 1.0312 | 775 |
| fixed100_Friedman | 2 | 7000000 | 0.86 | 0.00142 | 0.00659 | 1.0092 | 1.0390 | 820 |
| fixed100_Friedman | 3 | 7000000 | 0.95 | 0.00145 | 0.00610 | 1.0045 | 1.0345 | 800 |
| fixed100_Friedman | 4 | 7000000 | 0.85 | 0.00152 | 0.00718 | 1.0102 | 1.0371 | 763 |
| fixed100_FriedmanSparseDir_p100 | 0 | 7000000 | 0.67 | 0.00029 | 0.00174 | 1.0015 | 1.0066 | 2203 |
| fixed100_FriedmanSparseDir_p100 | 1 | 7000000 | 1.02 | 0.00050 | 0.00197 | 1.0026 | 1.0142 | 2056 |
| fixed100_FriedmanSparseDir_p100 | 2 | 7000000 | 1.18 | 0.00096 | 0.00326 | 1.0185 | 1.0293 | 1911 |
| fixed100_FriedmanSparseDir_p100 | 3 | 7000000 | 1.11 | 0.00045 | 0.00162 | 1.0017 | 1.0086 | 2156 |
| fixed100_FriedmanSparseDir_p100 | 4 | 7000000 | 1.12 | 0.00057 | 0.00201 | 1.0024 | 1.0121 | 1856 |
| fixed100_FriedmanSparseDir_p20 | 0 | 7000000 | 0.84 | 0.00032 | 0.00150 | 1.0014 | 1.0080 | 2252 |
| fixed100_FriedmanSparseDir_p20 | 1 | 7000000 | 1.02 | 0.00053 | 0.00210 | 1.0033 | 1.0140 | 2073 |
| fixed100_FriedmanSparseDir_p20 | 2 | 7000000 | 0.89 | 0.00039 | 0.00176 | 1.0020 | 1.0081 | 2115 |
| fixed100_FriedmanSparseDir_p20 | 3 | 7000000 | 1.11 | 0.00039 | 0.00141 | 1.0010 | 1.0079 | 2235 |
| fixed100_FriedmanSparseDir_p20 | 4 | 7000000 | 1.14 | 0.00051 | 0.00180 | 1.0020 | 1.0118 | 1970 |
| fixed100_FriedmanSparseDir_p200 | 0 | 7000000 | 1.20 | 0.00047 | 0.00158 | 1.0014 | 1.0100 | 2221 |
| fixed100_FriedmanSparseDir_p200 | 1 | 7000000 | 1.08 | 0.00048 | 0.00176 | 1.0017 | 1.0100 | 2042 |
| fixed100_FriedmanSparseDir_p200 | 2 | 7000000 | 0.80 | 0.00071 | 0.00355 | 1.0092 | 1.0167 | 1808 |
| fixed100_FriedmanSparseDir_p200 | 3 | 7000000 | 1.06 | 0.00044 | 0.00165 | 1.0012 | 1.0093 | 2217 |
| fixed100_FriedmanSparseDir_p200 | 4 | 7000000 | 1.02 | 0.00052 | 0.00203 | 1.0022 | 1.0128 | 1910 |
| fixed100_SeoulBike | 0 | 7000000 | 11.49 | 0.86414 | 0.30082 | 2.0928 | 1.9926 | 128 |
| fixed100_SeoulBike | 1 | 7000000 | 23.07 | 1.03809 | 0.17995 | 1.9671 | 2.6047 | 164 |
| fixed100_SeoulBike | 2 | 7000000 | 15.10 | 0.85004 | 0.22517 | 2.1138 | 2.5663 | 133 |
| fixed100_SeoulBike | 3 | 7000000 | 19.88 | 1.15605 | 0.23262 | 2.6306 | 2.3765 | 84 |
| fixed100_SeoulBike | 4 | 7000000 | 19.78 | 1.11459 | 0.22537 | 2.5418 | 2.5820 | 112 |

## Per dataset, ordered by separation index (range over runs)

| dataset | runs | iterations | rhat_max | worst_rhat_max | ess_bulk_mean | separation_index |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 5 | 7000000 | 1.0046 | 1.0071 | 3805 | 0.67-1.33 |
| fixed100_FriedmanSparseDir_p100 | 5 | 7000000 | 1.0185 | 1.0293 | 1856 | 0.67-1.18 |
| fixed100_Friedman | 5 | 7000000 | 1.0158 | 1.0518 | 756 | 0.79-1.26 |
| fixed100_FriedmanSparseDir_p200 | 5 | 7000000 | 1.0092 | 1.0167 | 1808 | 0.80-1.20 |
| fixed100_FriedmanSparseDir_p20 | 5 | 7000000 | 1.0033 | 1.0140 | 1970 | 0.84-1.14 |
| fixed100_Concrete | 5 | 7000000 | 1.0962 | 1.2917 | 336 | 0.98-2.41 |
| fixed100_CalHousing_subsample5000 | 5 | 7000000 | 1.2961 | 1.7168 | 237 | 2.03-4.30 |
| fixed100_Airfoil | 5 | 7000000 | 1.5054 | 2.0064 | 99 | 3.94-5.57 |
| fixed100_CCPP | 5 | 7000000 | 1.9158 | 2.5859 | 52 | 7.62-9.43 |
| fixed100_CPUAct | 5 | 7000000 | 2.2435 | 2.8414 | 57 | 11.13-15.76 |
| fixed100_SeoulBike | 5 | 7000000 | 2.6306 | 2.6047 | 84 | 11.49-23.07 |

Figures: one per run in `figures/`, panels (a)-(f) as described in the script docstring.
