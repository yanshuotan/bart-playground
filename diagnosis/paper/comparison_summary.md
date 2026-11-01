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

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.0972 (0.0053) | 1.2478 (0.0232) | 1.1850 (0.0120) | 0.1393 (0.0182) | 0.6592 (0.0204) | 0.6546 (0.0142) |
| Default+PT | 1.0287 (0.0072) | 1.0713 (0.0193) | 1.0696 (0.0190) | 0.0464 (0.0199) | 0.6553 (0.0208) | 0.6506 (0.0151) |
| MTMH | 1.0312 (0.0053) | 1.0872 (0.0070) | 1.0844 (0.0089) | 0.0476 (0.0155) | 0.6524 (0.0196) | 0.6482 (0.0128) |
| MTMH+PT | 1.0130 (0.0024) | 1.0349 (0.0081) | 1.0403 (0.0066) | 0.0156 (0.0045) | 0.6524 (0.0196) | 0.6476 (0.0139) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 64.1 | 1.1 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 327.2 | 1.1 | 5.1 |
| fixed100_Abalone | Default+PT | 23 | 23 | 188.6 | 3.9 | 2.94 |
| fixed100_Abalone | MTMH+PT | 23 | 23 | 467.4 | 3.0 | 7.29 |

## fixed100_Airfoil

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.2975 (0.0637) | 1.6747 (0.0532) | 1.2923 (0.0575) | 0.5817 (0.2363) | 0.4989 (0.0388) | 0.4957 (0.0438) |
| Default+PT | 1.0640 (0.0140) | 1.1299 (0.0183) | 1.0885 (0.0191) | 0.0827 (0.0282) | 0.4808 (0.0475) | 0.4671 (0.0517) |
| MTMH | 1.2088 (0.0510) | 1.5567 (0.0578) | 1.2248 (0.0467) | 0.4346 (0.1719) | 0.4908 (0.0400) | 0.4832 (0.0480) |
| MTMH+PT | 1.0557 (0.0178) | 1.1280 (0.0317) | 1.0805 (0.0166) | 0.0828 (0.0291) | 0.4830 (0.0507) | 0.4698 (0.0563) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Airfoil | Default | - | 1 | 45.1 | 0.1 | 1.0 |
| fixed100_Airfoil | MTMH | - | 1 | 181.6 | 0.9 | 4.03 |
| fixed100_Airfoil | Default+PT | 55 | 55 | 163.7 | 13.5 | 3.63 |
| fixed100_Airfoil | MTMH+PT | 55 | 55 | 325.5 | 12.0 | 7.22 |

## fixed100_CCPP

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.4394 (0.0105) | 1.8121 (0.0549) | 1.3538 (0.0204) | 0.8047 (0.0939) | 0.1926 (0.0044) | 0.1890 (0.0044) |
| Default+PT | 1.1972 (0.0407) | 1.3916 (0.0658) | 1.1991 (0.0525) | 0.2935 (0.0266) | 0.1912 (0.0038) | 0.1876 (0.0038) |
| MTMH | 1.3134 (0.0290) | 1.6304 (0.0419) | 1.2408 (0.0068) | 0.5629 (0.0699) | 0.1926 (0.0050) | 0.1876 (0.0057) |
| MTMH+PT | 1.2129 (0.0531) | 1.4296 (0.1064) | 1.2061 (0.0539) | 0.3349 (0.0743) | 0.1900 (0.0039) | 0.1852 (0.0036) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CCPP | Default | - | 1 | 112.9 | 1.4 | 1.0 |
| fixed100_CCPP | MTMH | - | 1 | 712.2 | 3.1 | 6.31 |
| fixed100_CCPP | Default+PT | 26 | 26 | 290.5 | 7.4 | 2.57 |
| fixed100_CCPP | MTMH+PT | 26 | 26 | 891.5 | 4.1 | 7.9 |

## fixed100_CalHousing_subsample5000

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.2506 (0.0236) | 1.5309 (0.0758) | 1.2578 (0.0116) | 0.4720 (0.0811) | 0.3878 (0.0073) | 0.3814 (0.0050) |
| Default+PT | 1.0898 (0.0174) | 1.1916 (0.0302) | 1.1255 (0.0157) | 0.1488 (0.0304) | 0.3724 (0.0076) | 0.3673 (0.0061) |
| MTMH | 1.1543 (0.0256) | 1.3718 (0.0376) | 1.1911 (0.0132) | 0.2852 (0.0372) | 0.3758 (0.0077) | 0.3682 (0.0069) |
| MTMH+PT | 1.0722 (0.0089) | 1.1699 (0.0210) | 1.1043 (0.0059) | 0.1206 (0.0232) | 0.3668 (0.0089) | 0.3590 (0.0101) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CalHousing_subsample5000 | Default | - | 1 | 76.9 | 0.5 | 1.0 |
| fixed100_CalHousing_subsample5000 | MTMH | - | 1 | 389.4 | 2.1 | 5.07 |
| fixed100_CalHousing_subsample5000 | Default+PT | 35 | 35 | 235.2 | 2.6 | 3.06 |
| fixed100_CalHousing_subsample5000 | MTMH+PT | 35 | 35 | 570.4 | 5.2 | 7.42 |

## fixed100_Concrete

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1733 (0.0268) | 1.3938 (0.0936) | 1.2221 (0.0269) | 0.3066 (0.1116) | 0.2944 (0.0088) | 0.2786 (0.0081) |
| Default+PT | 1.0362 (0.0070) | 1.0819 (0.0125) | 1.0739 (0.0064) | 0.0501 (0.0092) | 0.2851 (0.0161) | 0.2668 (0.0141) |
| MTMH | 1.1084 (0.0377) | 1.2654 (0.0721) | 1.1598 (0.0196) | 0.1823 (0.0925) | 0.2914 (0.0201) | 0.2732 (0.0182) |
| MTMH+PT | 1.0317 (0.0104) | 1.0646 (0.0159) | 1.0660 (0.0187) | 0.0405 (0.0149) | 0.2803 (0.0169) | 0.2609 (0.0148) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 45.6 | 0.9 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 174.3 | 0.1 | 3.83 |
| fixed100_Concrete | Default+PT | 37 | 37 | 143.0 | 9.6 | 3.14 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 298.7 | 4.6 | 6.56 |

## fixed100_CPUAct

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.5176 (0.0527) | 1.9266 (0.1162) | 1.4088 (0.0127) | 1.0914 (0.2695) | 0.1371 (0.0074) | 0.1887 (0.0094) |
| Default+PT | 1.1717 (0.0558) | 1.3218 (0.1273) | 1.1669 (0.0343) | 0.2660 (0.1242) | 0.1301 (0.0048) | 0.1800 (0.0064) |
| MTMH | 1.4279 (0.0240) | 1.8216 (0.0289) | 1.2939 (0.0250) | 1.1184 (0.1884) | 0.1414 (0.0063) | 0.1912 (0.0071) |
| MTMH+PT | 1.1827 (0.0544) | 1.3897 (0.1591) | 1.1562 (0.0311) | 0.3520 (0.1740) | 0.1343 (0.0097) | 0.1831 (0.0115) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CPUAct | Default | - | 1 | 104.6 | 0.4 | 1.0 |
| fixed100_CPUAct | MTMH | - | 1 | 612.6 | 12.4 | 5.86 |
| fixed100_CPUAct | Default+PT | 60 | 60 | 359.5 | 19.2 | 3.44 |
| fixed100_CPUAct | MTMH+PT | 60 | 60 | 981.7 | 10.1 | 9.38 |

## fixed100_Friedman

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.2330 (0.0181) | 1.4820 (0.0227) | 1.2437 (0.0068) | 0.3445 (0.0399) | 0.2122 (0.0070) | 0.2101 (0.0064) |
| Default+PT | 1.0518 (0.0029) | 1.1001 (0.0062) | 1.0838 (0.0119) | 0.0609 (0.0079) | 0.2081 (0.0073) | 0.2064 (0.0070) |
| MTMH | 1.1082 (0.0122) | 1.2538 (0.0270) | 1.1775 (0.0057) | 0.1460 (0.0273) | 0.2120 (0.0056) | 0.2099 (0.0061) |
| MTMH+PT | 1.0275 (0.0049) | 1.0576 (0.0095) | 1.0619 (0.0096) | 0.0315 (0.0045) | 0.2094 (0.0086) | 0.2078 (0.0087) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 53.9 | 0.6 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 244.5 | 1.1 | 4.54 |
| fixed100_Friedman | Default+PT | 44 | 44 | 174.6 | 8.2 | 3.24 |
| fixed100_Friedman | MTMH+PT | 44 | 44 | 389.2 | 0.9 | 7.22 |

## fixed100_FriedmanSparseDir_p20

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1993 (0.0083) | 1.3988 (0.0220) | 1.2742 (0.0153) | 0.2554 (0.0113) | 0.2260 (0.0023) | 0.2248 (0.0025) |
| Default+PT | 1.0388 (0.0077) | 1.0787 (0.0077) | 1.0911 (0.0143) | 0.0381 (0.0050) | 0.2225 (0.0029) | 0.2214 (0.0028) |
| MTMH | 1.0655 (0.0074) | 1.1387 (0.0119) | 1.1549 (0.0083) | 0.0700 (0.0092) | 0.2229 (0.0013) | 0.2218 (0.0012) |
| MTMH+PT | 1.0175 (0.0040) | 1.0347 (0.0069) | 1.0585 (0.0082) | 0.0160 (0.0038) | 0.2210 (0.0026) | 0.2197 (0.0023) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p20 | Default | - | 1 | 57.1 | 1.1 | 1.0 |
| fixed100_FriedmanSparseDir_p20 | MTMH | - | 1 | 252.9 | 0.9 | 4.43 |
| fixed100_FriedmanSparseDir_p20 | Default+PT | 47 | 47 | 181.4 | 3.2 | 3.18 |
| fixed100_FriedmanSparseDir_p20 | MTMH+PT | 47 | 47 | 406.9 | 2.2 | 7.13 |

## fixed100_FriedmanSparseDir_p100

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1734 (0.0107) | 1.3534 (0.0610) | 1.2800 (0.0102) | 0.2123 (0.0282) | 0.2264 (0.0040) | 0.2251 (0.0039) |
| Default+PT | 1.0410 (0.0092) | 1.0857 (0.0222) | 1.0920 (0.0180) | 0.0423 (0.0108) | 0.2215 (0.0027) | 0.2203 (0.0022) |
| MTMH | 1.0700 (0.0038) | 1.1392 (0.0140) | 1.1622 (0.0083) | 0.0732 (0.0054) | 0.2212 (0.0030) | 0.2201 (0.0033) |
| MTMH+PT | 1.0206 (0.0039) | 1.0422 (0.0116) | 1.0621 (0.0075) | 0.0205 (0.0077) | 0.2210 (0.0029) | 0.2197 (0.0030) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p100 | Default | - | 1 | 56.9 | 0.8 | 1.0 |
| fixed100_FriedmanSparseDir_p100 | MTMH | - | 1 | 255.7 | 1.0 | 4.49 |
| fixed100_FriedmanSparseDir_p100 | Default+PT | 41 | 41 | 181.1 | 10.1 | 3.18 |
| fixed100_FriedmanSparseDir_p100 | MTMH+PT | 41 | 41 | 404.9 | 5.3 | 7.12 |

## fixed100_FriedmanSparseDir_p200

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.1841 (0.0157) | 1.3586 (0.0169) | 1.3036 (0.0246) | 0.2160 (0.0317) | 0.2250 (0.0014) | 0.2233 (0.0016) |
| Default+PT | 1.0331 (0.0070) | 1.0662 (0.0158) | 1.0770 (0.0141) | 0.0349 (0.0069) | 0.2220 (0.0036) | 0.2209 (0.0037) |
| MTMH | 1.0676 (0.0066) | 1.1429 (0.0185) | 1.1564 (0.0069) | 0.0738 (0.0154) | 0.2223 (0.0021) | 0.2208 (0.0026) |
| MTMH+PT | 1.0152 (0.0027) | 1.0292 (0.0055) | 1.0496 (0.0078) | 0.0130 (0.0024) | 0.2207 (0.0025) | 0.2195 (0.0025) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p200 | Default | - | 1 | 57.2 | 0.7 | 1.0 |
| fixed100_FriedmanSparseDir_p200 | MTMH | - | 1 | 254.4 | 1.3 | 4.45 |
| fixed100_FriedmanSparseDir_p200 | Default+PT | 47 | 47 | 196.4 | 15.8 | 3.43 |
| fixed100_FriedmanSparseDir_p200 | MTMH+PT | 47 | 47 | 421.9 | 5.2 | 7.38 |

## fixed100_Superconduct_subsample5000

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.4411 (0.0360) | 1.9784 (0.0556) | 1.2958 (0.0140) | 1.2734 (0.1273) | 0.3592 (0.0203) | 0.3536 (0.0178) |
| Default+PT | 1.1733 (0.0670) | 1.3497 (0.1132) | 1.1388 (0.0273) | 0.3818 (0.1677) | 0.3463 (0.0246) | 0.3352 (0.0206) |
| MTMH | 1.4028 (0.0268) | 1.8705 (0.0863) | 1.2717 (0.0144) | 1.1872 (0.2054) | 0.3752 (0.0147) | 0.3578 (0.0106) |
| MTMH+PT | 1.1420 (0.0423) | 1.3418 (0.0988) | 1.1302 (0.0267) | 0.3304 (0.1728) | 0.3411 (0.0258) | 0.3245 (0.0188) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Superconduct_subsample5000 | Default | - | 1 | 81.7 | 0.7 | 1.0 |
| fixed100_Superconduct_subsample5000 | MTMH | - | 1 | 426.4 | 4.9 | 5.22 |
| fixed100_Superconduct_subsample5000 | Default+PT | 46 | 46 | 255.0 | 2.3 | 3.12 |
| fixed100_Superconduct_subsample5000 | MTMH+PT | 46 | 46 | 638.9 | 4.9 | 7.82 |

## fixed100_SeoulBike

| method | cross R-hat median | cross R-hat q90 | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.4664 (0.0308) | 1.9129 (0.0889) | 1.3183 (0.0125) | 1.1017 (0.1835) | 0.3511 (0.0277) | 0.3189 (0.0135) |
| Default+PT | 1.2195 (0.0418) | 1.4624 (0.1138) | 1.1919 (0.0564) | 0.3659 (0.0786) | 0.3504 (0.0269) | 0.3136 (0.0137) |
| MTMH | 1.4147 (0.0388) | 1.8420 (0.0407) | 1.2599 (0.0168) | 1.2027 (0.1825) | 0.3685 (0.0235) | 0.3190 (0.0129) |
| MTMH+PT | 1.2782 (0.0560) | 1.5697 (0.1227) | 1.2142 (0.0146) | 0.6755 (0.3398) | 0.3728 (0.0111) | 0.3147 (0.0049) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_SeoulBike | Default | - | 1 | 88.3 | 0.6 | 1.0 |
| fixed100_SeoulBike | MTMH | - | 1 | 470.8 | 4.6 | 5.33 |
| fixed100_SeoulBike | Default+PT | 46 | 46 | 315.9 | 22.1 | 3.58 |
| fixed100_SeoulBike | MTMH+PT | 46 | 46 | 740.8 | 22.8 | 8.39 |

## Metric definitions

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Method RMSE is sqrt(mean chain MSE) within each run; setting summaries average run scores.
- Segment R-hat summarizes temporal stability across consecutive segments.
- Raw RMSE and CRPS remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
