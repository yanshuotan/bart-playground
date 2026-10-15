# Four-method comparison summary

All diagnostic and predictive entries summarize five paired runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `segment_length`: 1000
- `segments_per_block`: 4
- `short_burn`: 3000
- `chain_geometry_draws_per_chain`: all post-burn draws
- `table_rhat_draws`: all post-burn draws; within-chain uses equal consecutive segments

`short_burn` is also the start of the worst-direction calculation.
Table cross R-hat uses every retained draw, with the median and q90 taken across test inputs. Table within R-hat splits each retained original chain into equal consecutive segments (four segments of 1,750 draws for 7,000 retained draws), then summarizes across original chains and test inputs. Both use the rank-normalized split-and-folded estimator. Rolling-window R-hat is used only in the temporal figures, not in these table columns.

## fixed100_Abalone

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.3698 (0.1580) | 1.0972 (0.0053) | 1.1850 (0.0120) | 0.1393 (0.0182) | 0.6592 (0.0205) | 0.6546 (0.0142) |
| Default+PT | 1.2163 (0.1107) | 1.0336 (0.0120) | 1.0812 (0.0292) | 0.0473 (0.0125) | 0.6534 (0.0179) | 0.6491 (0.0128) |
| MTMH | 1.2057 (0.1497) | 1.0312 (0.0053) | 1.0844 (0.0089) | 0.0476 (0.0155) | 0.6524 (0.0196) | 0.6482 (0.0128) |
| MTMH+PT | 1.0815 (0.0833) | 1.0178 (0.0060) | 1.0472 (0.0128) | 0.0274 (0.0122) | 0.6515 (0.0194) | 0.6470 (0.0137) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 62.9 | 0.8 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 477.0 | 3.5 | 7.59 |
| fixed100_Abalone | Default+PT | 21 | 21 | 229.0 | 6.8 | 3.64 |
| fixed100_Abalone | MTMH+PT | 21 | 21 | 677.7 | 22.3 | 10.78 |

## fixed100_Airfoil

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.5552 (0.3238) | 1.2975 (0.0637) | 1.2923 (0.0575) | 0.5817 (0.2363) | 0.4981 (0.0385) | 0.4957 (0.0438) |
| Default+PT | 1.5769 (0.2575) | 1.0828 (0.0172) | 1.1006 (0.0142) | 0.1120 (0.0419) | 0.4822 (0.0432) | 0.4702 (0.0464) |
| MTMH | 2.6762 (0.2991) | 1.2088 (0.0510) | 1.2248 (0.0467) | 0.4346 (0.1719) | 0.4903 (0.0399) | 0.4832 (0.0480) |
| MTMH+PT | 1.4525 (0.0635) | 1.0798 (0.0209) | 1.1047 (0.0288) | 0.1271 (0.0476) | 0.4835 (0.0423) | 0.4691 (0.0457) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Airfoil | Default | - | 1 | 45.3 | 0.1 | 1.0 |
| fixed100_Airfoil | MTMH | - | 1 | 301.8 | 0.4 | 6.66 |
| fixed100_Airfoil | Default+PT | 45 | 45 | 262.7 | 5.8 | 5.8 |
| fixed100_Airfoil | MTMH+PT | 45 | 45 | 599.9 | 3.4 | 13.24 |

## fixed100_CCPP

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.6297 (0.4899) | 1.4394 (0.0105) | 1.3538 (0.0204) | 0.8047 (0.0939) | 0.1925 (0.0044) | 0.1890 (0.0044) |
| Default+PT | 2.1337 (0.2865) | 1.2172 (0.0285) | 1.2566 (0.0356) | 0.3380 (0.0966) | 0.1925 (0.0037) | 0.1885 (0.0027) |
| MTMH | 2.6369 (0.1492) | 1.3134 (0.0290) | 1.2408 (0.0068) | 0.5629 (0.0699) | 0.1925 (0.0050) | 0.1876 (0.0057) |
| MTMH+PT | 2.2908 (0.3123) | 1.2379 (0.0262) | 1.2176 (0.0099) | 0.4503 (0.1447) | 0.1901 (0.0058) | 0.1852 (0.0056) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CCPP | Default | - | 1 | 112.5 | 1.2 | 1.0 |
| fixed100_CCPP | MTMH | - | 1 | 998.5 | 5.7 | 8.88 |
| fixed100_CCPP | Default+PT | 25 | 25 | 414.5 | 6.5 | 3.68 |
| fixed100_CCPP | MTMH+PT | 25 | 25 | 1408.0 | 3.5 | 12.51 |

## fixed100_CalHousing_subsample5000

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.3257 (0.2028) | 1.2506 (0.0236) | 1.2578 (0.0116) | 0.4720 (0.0811) | 0.3876 (0.0073) | 0.3814 (0.0050) |
| Default+PT | 1.8858 (0.2174) | 1.1462 (0.0309) | 1.1668 (0.0324) | 0.2515 (0.0654) | 0.3821 (0.0103) | 0.3750 (0.0079) |
| MTMH | 2.0520 (0.1633) | 1.1543 (0.0256) | 1.1911 (0.0132) | 0.2852 (0.0372) | 0.3757 (0.0077) | 0.3682 (0.0069) |
| MTMH+PT | 1.9168 (0.3068) | 1.1108 (0.0238) | 1.1483 (0.0202) | 0.1951 (0.0500) | 0.3700 (0.0108) | 0.3619 (0.0093) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CalHousing_subsample5000 | Default | - | 1 | 77.7 | 0.3 | 1.0 |
| fixed100_CalHousing_subsample5000 | MTMH | - | 1 | 590.6 | 3.2 | 7.6 |
| fixed100_CalHousing_subsample5000 | Default+PT | 28 | 28 | 310.6 | 1.2 | 4.0 |
| fixed100_CalHousing_subsample5000 | MTMH+PT | 28 | 28 | 908.8 | 0.7 | 11.69 |

## fixed100_Concrete

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.1029 (0.5306) | 1.1733 (0.0268) | 1.2221 (0.0269) | 0.3066 (0.1116) | 0.2940 (0.0088) | 0.2786 (0.0081) |
| Default+PT | 1.2865 (0.1018) | 1.0457 (0.0173) | 1.0810 (0.0185) | 0.0653 (0.0291) | 0.2855 (0.0140) | 0.2669 (0.0128) |
| MTMH | 1.5933 (0.1683) | 1.1084 (0.0377) | 1.1598 (0.0196) | 0.1823 (0.0925) | 0.2911 (0.0202) | 0.2732 (0.0182) |
| MTMH+PT | 1.0940 (0.0617) | 1.0323 (0.0110) | 1.0660 (0.0141) | 0.0379 (0.0132) | 0.2804 (0.0188) | 0.2612 (0.0179) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 45.7 | 0.3 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 295.5 | 0.9 | 6.47 |
| fixed100_Concrete | Default+PT | 37 | 37 | 230.9 | 10.6 | 5.06 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 574.7 | 2.6 | 12.59 |

## fixed100_CPUAct

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.8752 (0.1264) | 1.5176 (0.0527) | 1.4088 (0.0127) | 1.0914 (0.2695) | 0.1370 (0.0073) | 0.1887 (0.0094) |
| Default+PT | 2.0019 (0.2749) | 1.1828 (0.0470) | 1.1753 (0.0352) | 0.3167 (0.0999) | 0.1321 (0.0045) | 0.1823 (0.0053) |
| MTMH | 2.6826 (0.3445) | 1.4279 (0.0240) | 1.2939 (0.0250) | 1.1184 (0.1884) | 0.1410 (0.0062) | 0.1912 (0.0071) |
| MTMH+PT | 2.2101 (0.1336) | 1.2076 (0.0244) | 1.1884 (0.0229) | 0.5445 (0.1866) | 0.1363 (0.0075) | 0.1848 (0.0081) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CPUAct | Default | - | 1 | 104.4 | 0.5 | 1.0 |
| fixed100_CPUAct | MTMH | - | 1 | 876.6 | 12.9 | 8.4 |
| fixed100_CPUAct | Default+PT | 52 | 52 | 726.4 | 40.1 | 6.96 |
| fixed100_CPUAct | MTMH+PT | 52 | 52 | 1536.7 | 1.8 | 14.72 |

## fixed100_Friedman

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.1742 (0.0502) | 1.2330 (0.0181) | 1.2437 (0.0068) | 0.3445 (0.0399) | 0.2122 (0.0070) | 0.2101 (0.0064) |
| Default+PT | 1.3282 (0.1437) | 1.0576 (0.0146) | 1.0968 (0.0190) | 0.0746 (0.0221) | 0.2097 (0.0073) | 0.2081 (0.0073) |
| MTMH | 1.6790 (0.1553) | 1.1082 (0.0122) | 1.1775 (0.0057) | 0.1460 (0.0273) | 0.2119 (0.0056) | 0.2099 (0.0061) |
| MTMH+PT | 1.1235 (0.0622) | 1.0369 (0.0109) | 1.0785 (0.0137) | 0.0416 (0.0136) | 0.2095 (0.0066) | 0.2078 (0.0069) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 54.0 | 0.5 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 387.0 | 3.5 | 7.16 |
| fixed100_Friedman | Default+PT | 36 | 36 | 285.3 | 4.9 | 5.28 |
| fixed100_Friedman | MTMH+PT | 36 | 36 | 703.0 | 14.5 | 13.01 |

## fixed100_FriedmanSparseDir_p20

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.9772 (0.2305) | 1.1993 (0.0083) | 1.2742 (0.0153) | 0.2554 (0.0113) | 0.2260 (0.0023) | 0.2248 (0.0025) |
| Default+PT | 1.1845 (0.0349) | 1.0438 (0.0077) | 1.1012 (0.0051) | 0.0463 (0.0126) | 0.2216 (0.0032) | 0.2205 (0.0029) |
| MTMH | 1.3363 (0.1201) | 1.0655 (0.0074) | 1.1549 (0.0083) | 0.0700 (0.0092) | 0.2229 (0.0014) | 0.2218 (0.0012) |
| MTMH+PT | 1.0417 (0.0219) | 1.0192 (0.0045) | 1.0598 (0.0049) | 0.0177 (0.0045) | 0.2207 (0.0019) | 0.2195 (0.0016) |

## fixed100_FriedmanSparseDir_p100

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.8905 (0.2146) | 1.1734 (0.0107) | 1.2800 (0.0102) | 0.2123 (0.0282) | 0.2264 (0.0040) | 0.2251 (0.0039) |
| Default+PT | 1.1321 (0.0642) | 1.0407 (0.0084) | 1.0934 (0.0113) | 0.0396 (0.0079) | 0.2214 (0.0021) | 0.2202 (0.0021) |
| MTMH | 1.3072 (0.1106) | 1.0700 (0.0038) | 1.1622 (0.0083) | 0.0732 (0.0054) | 0.2212 (0.0030) | 0.2201 (0.0033) |
| MTMH+PT | 1.0564 (0.0418) | 1.0194 (0.0027) | 1.0577 (0.0064) | 0.0188 (0.0055) | 0.2209 (0.0020) | 0.2197 (0.0021) |

## fixed100_FriedmanSparseDir_p200

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.7511 (0.2278) | 1.1841 (0.0157) | 1.3036 (0.0246) | 0.2160 (0.0317) | 0.2250 (0.0014) | 0.2233 (0.0016) |
| Default+PT | 1.0961 (0.0276) | 1.0308 (0.0078) | 1.0785 (0.0163) | 0.0323 (0.0120) | 0.2210 (0.0031) | 0.2198 (0.0030) |
| MTMH | 1.2461 (0.0988) | 1.0676 (0.0066) | 1.1564 (0.0069) | 0.0738 (0.0154) | 0.2223 (0.0021) | 0.2208 (0.0026) |
| MTMH+PT | 1.0474 (0.0597) | 1.0138 (0.0024) | 1.0491 (0.0067) | 0.0117 (0.0029) | 0.2209 (0.0021) | 0.2198 (0.0019) |

## fixed100_SeoulBike

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.6879 (0.2918) | 1.4664 (0.0308) | 1.3183 (0.0125) | 1.1017 (0.1835) | 0.3508 (0.0277) | 0.3189 (0.0135) |
| Default+PT | 2.1852 (0.4166) | 1.2151 (0.0489) | 1.1720 (0.0196) | 0.4374 (0.0908) | 0.3452 (0.0199) | 0.3068 (0.0077) |
| MTMH | 2.9862 (0.1697) | 1.4147 (0.0388) | 1.2599 (0.0168) | 1.2027 (0.1825) | 0.3678 (0.0235) | 0.3190 (0.0129) |
| MTMH+PT | 2.3057 (0.5161) | 1.2190 (0.0257) | 1.1815 (0.0194) | 0.4127 (0.0855) | 0.3707 (0.0209) | 0.3143 (0.0085) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_SeoulBike | Default | - | 1 | 87.7 | 0.8 | 1.0 |
| fixed100_SeoulBike | MTMH | - | 1 | 688.7 | 19.0 | 7.85 |
| fixed100_SeoulBike | Default+PT | 36 | 36 | 523.6 | 12.7 | 5.97 |
| fixed100_SeoulBike | MTMH+PT | 36 | 36 | 1192.8 | 21.5 | 13.6 |

## Interpretation guardrails

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Segment R-hat uses temporally dependent pseudo-chains and is a stability diagnostic, not a formal convergence certificate.
- Raw RMSE and CRPS remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
