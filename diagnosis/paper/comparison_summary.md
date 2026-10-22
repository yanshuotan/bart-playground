# Four-method comparison summary

All diagnostic and predictive entries summarize five paired runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `segment_length`: 1000
- `segments_per_block`: 4
- `short_burn`: 3000
- `rmse_convention`: sqrt(mean chain MSE) within each run

`short_burn` is also the start of the worst-direction calculation.
Table cross R-hat uses every retained draw, with the median and q90 taken across test inputs. Table within R-hat splits each retained original chain into equal consecutive segments (four segments of 1,750 draws for 7,000 retained draws), then summarizes across original chains and test inputs. Both use the rank-normalized split-and-folded estimator. Rolling-window R-hat is used only in the temporal figures, not in these table columns.

## fixed100_Abalone

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.0972 (0.0053) | 1.2478 (0.0232) | 1.1850 (0.0120) | 0.1393 (0.0182) | 0.6592 (0.0204) | 0.6546 (0.0142) |
| Default+PT | 1.0404 (0.0112) | 1.0992 (0.0274) | 1.0914 (0.0282) | 0.0574 (0.0122) | 0.6542 (0.0220) | 0.6496 (0.0151) |
| MTMH | 1.0312 (0.0053) | 1.0872 (0.0070) | 1.0844 (0.0089) | 0.0476 (0.0155) | 0.6524 (0.0196) | 0.6482 (0.0128) |
| MTMH+PT | 1.0188 (0.0044) | 1.0520 (0.0164) | 1.0521 (0.0120) | 0.0262 (0.0133) | 0.6524 (0.0204) | 0.6469 (0.0145) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 63.1 | 0.3 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 478.9 | 2.0 | 7.59 |
| fixed100_Abalone | Default+PT | 21 | 21 | 226.5 | 5.3 | 3.59 |
| fixed100_Abalone | MTMH+PT | 21 | 21 | 655.0 | 2.8 | 10.38 |

## fixed100_Airfoil

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.2975 (0.0637) | 1.6747 (0.0532) | 1.2923 (0.0575) | 0.5817 (0.2363) | 0.4989 (0.0388) | 0.4957 (0.0438) |
| Default+PT | 1.0647 (0.0197) | 1.1397 (0.0244) | 1.0931 (0.0209) | 0.0793 (0.0275) | 0.4808 (0.0439) | 0.4648 (0.0443) |
| MTMH | 1.2088 (0.0510) | 1.5567 (0.0578) | 1.2248 (0.0467) | 0.4346 (0.1719) | 0.4908 (0.0400) | 0.4832 (0.0480) |
| MTMH+PT | 1.0674 (0.0284) | 1.1367 (0.0455) | 1.0817 (0.0280) | 0.0915 (0.0463) | 0.4780 (0.0435) | 0.4642 (0.0489) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Airfoil | Default | - | 1 | 45.3 | 0.1 | 1.0 |
| fixed100_Airfoil | MTMH | - | 1 | 301.8 | 0.4 | 6.66 |
| fixed100_Airfoil | Default+PT | 45 | 45 | 262.7 | 5.8 | 5.8 |
| fixed100_Airfoil | MTMH+PT | 45 | 45 | 599.9 | 3.4 | 13.24 |

## fixed100_CCPP

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.4394 (0.0105) | 1.8121 (0.0549) | 1.3538 (0.0204) | 0.8047 (0.0939) | 0.1926 (0.0044) | 0.1890 (0.0044) |
| Default+PT | 1.2122 (0.0601) | 1.4208 (0.1379) | 1.2116 (0.0438) | 0.2958 (0.1191) | 0.1910 (0.0038) | 0.1875 (0.0036) |
| MTMH | 1.3134 (0.0290) | 1.6304 (0.0419) | 1.2408 (0.0068) | 0.5629 (0.0699) | 0.1926 (0.0050) | 0.1876 (0.0057) |
| MTMH+PT | 1.1791 (0.0396) | 1.3574 (0.0806) | 1.1829 (0.0384) | 0.2901 (0.0627) | 0.1893 (0.0033) | 0.1842 (0.0026) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CCPP | Default | - | 1 | 112.5 | 1.2 | 1.0 |
| fixed100_CCPP | MTMH | - | 1 | 998.5 | 5.7 | 8.88 |
| fixed100_CCPP | Default+PT | 25 | 25 | 414.5 | 6.5 | 3.68 |
| fixed100_CCPP | MTMH+PT | 25 | 25 | 1408.0 | 3.5 | 12.51 |

## fixed100_CalHousing_subsample5000

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.2506 (0.0236) | 1.5309 (0.0758) | 1.2578 (0.0116) | 0.4720 (0.0811) | 0.3878 (0.0073) | 0.3814 (0.0050) |
| Default+PT | 1.1276 (0.0577) | 1.2565 (0.1030) | 1.1659 (0.0616) | 0.1933 (0.0854) | 0.3800 (0.0116) | 0.3727 (0.0074) |
| MTMH | 1.1543 (0.0256) | 1.3718 (0.0376) | 1.1911 (0.0132) | 0.2852 (0.0372) | 0.3758 (0.0077) | 0.3682 (0.0069) |
| MTMH+PT | 1.1006 (0.0423) | 1.2244 (0.0695) | 1.1299 (0.0395) | 0.1713 (0.0840) | 0.3665 (0.0090) | 0.3586 (0.0080) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CalHousing_subsample5000 | Default | - | 1 | 77.7 | 0.3 | 1.0 |
| fixed100_CalHousing_subsample5000 | MTMH | - | 1 | 590.6 | 3.2 | 7.6 |
| fixed100_CalHousing_subsample5000 | Default+PT | 28 | 28 | 310.6 | 1.2 | 4.0 |
| fixed100_CalHousing_subsample5000 | MTMH+PT | 28 | 28 | 908.8 | 0.7 | 11.69 |

## fixed100_Concrete

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1733 (0.0268) | 1.3938 (0.0936) | 1.2221 (0.0269) | 0.3066 (0.1116) | 0.2944 (0.0088) | 0.2786 (0.0081) |
| Default+PT | 1.0392 (0.0105) | 1.0827 (0.0247) | 1.0718 (0.0097) | 0.0561 (0.0210) | 0.2807 (0.0175) | 0.2628 (0.0160) |
| MTMH | 1.1084 (0.0377) | 1.2654 (0.0721) | 1.1598 (0.0196) | 0.1823 (0.0925) | 0.2914 (0.0201) | 0.2732 (0.0182) |
| MTMH+PT | 1.0292 (0.0081) | 1.0642 (0.0174) | 1.0578 (0.0088) | 0.0366 (0.0144) | 0.2816 (0.0175) | 0.2623 (0.0166) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 45.7 | 0.3 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 295.5 | 0.9 | 6.47 |
| fixed100_Concrete | Default+PT | 37 | 37 | 230.9 | 10.6 | 5.06 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 574.7 | 2.6 | 12.59 |

## fixed100_CPUAct

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.5176 (0.0527) | 1.9266 (0.1162) | 1.4088 (0.0127) | 1.0914 (0.2695) | 0.1371 (0.0074) | 0.1887 (0.0094) |
| Default+PT | 1.2014 (0.0579) | 1.3792 (0.1158) | 1.1876 (0.0626) | 0.3394 (0.1509) | 0.1341 (0.0063) | 0.1840 (0.0074) |
| MTMH | 1.4279 (0.0240) | 1.8216 (0.0289) | 1.2939 (0.0250) | 1.1184 (0.1884) | 0.1414 (0.0063) | 0.1912 (0.0071) |
| MTMH+PT | 1.2112 (0.0973) | 1.4102 (0.1807) | 1.1892 (0.0638) | 0.3977 (0.1733) | 0.1335 (0.0064) | 0.1824 (0.0075) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CPUAct | Default | - | 1 | 104.4 | 0.5 | 1.0 |
| fixed100_CPUAct | MTMH | - | 1 | 876.6 | 12.9 | 8.4 |
| fixed100_CPUAct | Default+PT | 52 | 52 | 726.4 | 40.1 | 6.96 |
| fixed100_CPUAct | MTMH+PT | 52 | 52 | 1536.7 | 1.8 | 14.72 |

## fixed100_Friedman

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.2330 (0.0181) | 1.4820 (0.0227) | 1.2437 (0.0068) | 0.3445 (0.0399) | 0.2122 (0.0070) | 0.2101 (0.0064) |
| Default+PT | 1.0549 (0.0140) | 1.1114 (0.0271) | 1.0823 (0.0145) | 0.0698 (0.0230) | 0.2076 (0.0077) | 0.2058 (0.0081) |
| MTMH | 1.1082 (0.0122) | 1.2538 (0.0270) | 1.1775 (0.0057) | 0.1460 (0.0273) | 0.2120 (0.0056) | 0.2099 (0.0061) |
| MTMH+PT | 1.0269 (0.0073) | 1.0551 (0.0157) | 1.0667 (0.0166) | 0.0300 (0.0082) | 0.2096 (0.0080) | 0.2079 (0.0088) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 54.0 | 0.5 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 387.0 | 3.5 | 7.16 |
| fixed100_Friedman | Default+PT | 36 | 36 | 285.3 | 4.9 | 5.28 |
| fixed100_Friedman | MTMH+PT | 36 | 36 | 703.0 | 14.5 | 13.01 |

## fixed100_FriedmanSparseDir_p20

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1993 (0.0083) | 1.3988 (0.0220) | 1.2742 (0.0153) | 0.2554 (0.0113) | 0.2260 (0.0023) | 0.2248 (0.0025) |
| Default+PT | 1.0411 (0.0114) | 1.0826 (0.0246) | 1.0879 (0.0157) | 0.0427 (0.0173) | 0.2212 (0.0028) | 0.2201 (0.0030) |
| MTMH | 1.0655 (0.0074) | 1.1387 (0.0119) | 1.1549 (0.0083) | 0.0700 (0.0092) | 0.2229 (0.0013) | 0.2218 (0.0012) |
| MTMH+PT | 1.0177 (0.0024) | 1.0358 (0.0038) | 1.0552 (0.0065) | 0.0160 (0.0026) | 0.2208 (0.0020) | 0.2197 (0.0019) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p20 | Default | - | 1 | 56.2 | 0.4 | 1.0 |
| fixed100_FriedmanSparseDir_p20 | MTMH | - | 1 | 422.9 | 1.3 | 7.53 |
| fixed100_FriedmanSparseDir_p20 | Default+PT | 36 | 36 | 295.4 | 9.4 | 5.26 |
| fixed100_FriedmanSparseDir_p20 | MTMH+PT | 36 | 36 | 741.8 | 4.6 | 13.21 |

## fixed100_FriedmanSparseDir_p100

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1734 (0.0107) | 1.3534 (0.0610) | 1.2800 (0.0102) | 0.2123 (0.0282) | 0.2264 (0.0040) | 0.2251 (0.0039) |
| Default+PT | 1.0365 (0.0089) | 1.0749 (0.0171) | 1.0782 (0.0143) | 0.0386 (0.0088) | 0.2213 (0.0023) | 0.2200 (0.0027) |
| MTMH | 1.0700 (0.0038) | 1.1392 (0.0140) | 1.1622 (0.0083) | 0.0732 (0.0054) | 0.2212 (0.0030) | 0.2201 (0.0033) |
| MTMH+PT | 1.0144 (0.0025) | 1.0287 (0.0038) | 1.0496 (0.0070) | 0.0125 (0.0021) | 0.2204 (0.0030) | 0.2192 (0.0030) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p100 | Default | - | 1 | 59.6 | 2.3 | 1.0 |
| fixed100_FriedmanSparseDir_p100 | MTMH | - | 1 | 439.9 | 9.3 | 7.38 |
| fixed100_FriedmanSparseDir_p100 | Default+PT | 38 | 38 | 369.2 | 94.0 | 6.19 |
| fixed100_FriedmanSparseDir_p100 | MTMH+PT | 38 | 38 | 799.7 | 49.2 | 13.41 |

## fixed100_FriedmanSparseDir_p200

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1841 (0.0157) | 1.3586 (0.0169) | 1.3036 (0.0246) | 0.2160 (0.0317) | 0.2250 (0.0014) | 0.2233 (0.0016) |
| Default+PT | 1.0308 (0.0078) | 1.0620 (0.0153) | 1.0785 (0.0163) | 0.0323 (0.0120) | 0.2210 (0.0031) | 0.2198 (0.0030) |
| MTMH | 1.0676 (0.0066) | 1.1429 (0.0185) | 1.1564 (0.0069) | 0.0738 (0.0154) | 0.2223 (0.0021) | 0.2208 (0.0026) |
| MTMH+PT | 1.0138 (0.0024) | 1.0263 (0.0067) | 1.0491 (0.0067) | 0.0117 (0.0029) | 0.2209 (0.0021) | 0.2198 (0.0019) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p200 | Default | - | 1 | 57.3 | 0.5 | 1.0 |
| fixed100_FriedmanSparseDir_p200 | MTMH | - | 1 | 436.4 | 2.1 | 7.61 |
| fixed100_FriedmanSparseDir_p200 | Default+PT | 43 | 43 | 373.9 | 37.5 | 6.52 |
| fixed100_FriedmanSparseDir_p200 | MTMH+PT | 43 | 43 | 787.3 | 0.9 | 13.73 |

## fixed100_SeoulBike

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.4664 (0.0308) | 1.9129 (0.0889) | 1.3183 (0.0125) | 1.1017 (0.1835) | 0.3511 (0.0277) | 0.3189 (0.0135) |
| Default+PT | 1.2463 (0.0754) | 1.4900 (0.1460) | 1.1802 (0.0383) | 0.5293 (0.1998) | 0.3558 (0.0218) | 0.3129 (0.0087) |
| MTMH | 1.4147 (0.0388) | 1.8420 (0.0407) | 1.2599 (0.0168) | 1.2027 (0.1825) | 0.3685 (0.0235) | 0.3190 (0.0129) |
| MTMH+PT | 1.2712 (0.0628) | 1.5337 (0.1061) | 1.1980 (0.0365) | 0.6131 (0.1755) | 0.3786 (0.0163) | 0.3180 (0.0111) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_SeoulBike | Default | - | 1 | 87.7 | 0.8 | 1.0 |
| fixed100_SeoulBike | MTMH | - | 1 | 688.7 | 19.0 | 7.85 |
| fixed100_SeoulBike | Default+PT | 36 | 36 | 523.6 | 12.7 | 5.97 |
| fixed100_SeoulBike | MTMH+PT | 36 | 36 | 1192.8 | 21.5 | 13.6 |

## fixed100_Superconduct_subsample5000

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.4411 (0.0360) | 1.9784 (0.0556) | 1.2958 (0.0140) | 1.2734 (0.1273) | 0.3592 (0.0203) | 0.3536 (0.0178) |
| Default+PT | 1.1207 (0.0110) | 1.2509 (0.0344) | 1.1315 (0.0243) | 0.2143 (0.0577) | 0.3349 (0.0163) | 0.3246 (0.0132) |
| MTMH | 1.4028 (0.0268) | 1.8705 (0.0863) | 1.2717 (0.0144) | 1.1872 (0.2054) | 0.3752 (0.0147) | 0.3578 (0.0106) |
| MTMH+PT | 1.1404 (0.0337) | 1.3076 (0.0625) | 1.1323 (0.0227) | 0.2991 (0.0759) | 0.3429 (0.0192) | 0.3258 (0.0134) |

## Metric definitions

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Method RMSE is sqrt(mean chain MSE) within each run; setting summaries average run scores.
- Segment R-hat summarizes temporal stability across consecutive segments.
- Raw RMSE and CRPS remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
