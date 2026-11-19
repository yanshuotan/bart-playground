# Four-method comparison summary

All diagnostic and predictive entries summarize five paired runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `short_burn`: 3000
- `chain_geometry_draws_per_chain`: all post-burn draws
- `table_rhat_draws`: all post-burn draws

`short_burn` is also the start of the worst-direction calculation.
Table cross R-hat uses every retained draw, with the median and q90 taken across test inputs, and the rank-normalized split-and-folded estimator. Rolling-window R-hat is used only in the temporal figures, not in these table columns.

## fixed100_Abalone

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.0972 (0.0053) | 1.2478 (0.0232) | 0.1393 (0.0182) | 0.6592 (0.0204) | 0.6546 (0.0142) |
| Default+PT | 1.0287 (0.0072) | 1.0713 (0.0193) | 0.0464 (0.0199) | 0.6553 (0.0208) | 0.6506 (0.0151) |
| MTMH | 1.0312 (0.0053) | 1.0872 (0.0070) | 0.0476 (0.0155) | 0.6524 (0.0196) | 0.6482 (0.0128) |
| MTMH+PT | 1.0130 (0.0024) | 1.0349 (0.0081) | 0.0156 (0.0045) | 0.6524 (0.0196) | 0.6476 (0.0139) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 64.1 | 1.1 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 327.2 | 1.1 | 5.1 |
| fixed100_Abalone | Default+PT | 23 | 23 | 188.6 | 3.9 | 2.94 |
| fixed100_Abalone | MTMH+PT | 23 | 23 | 467.4 | 3.0 | 7.29 |

## fixed100_Airfoil

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.2975 (0.0637) | 1.6747 (0.0532) | 0.5817 (0.2363) | 0.4989 (0.0388) | 0.4957 (0.0438) |
| Default+PT | 1.0640 (0.0140) | 1.1299 (0.0183) | 0.0827 (0.0282) | 0.4808 (0.0475) | 0.4671 (0.0517) |
| MTMH | 1.2088 (0.0510) | 1.5567 (0.0578) | 0.4346 (0.1719) | 0.4908 (0.0400) | 0.4832 (0.0480) |
| MTMH+PT | 1.0557 (0.0178) | 1.1280 (0.0317) | 0.0828 (0.0291) | 0.4830 (0.0507) | 0.4698 (0.0563) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Airfoil | Default | - | 1 | 45.1 | 0.1 | 1.0 |
| fixed100_Airfoil | MTMH | - | 1 | 181.6 | 0.9 | 4.03 |
| fixed100_Airfoil | Default+PT | 55 | 55 | 163.7 | 13.5 | 3.63 |
| fixed100_Airfoil | MTMH+PT | 55 | 55 | 325.5 | 12.0 | 7.22 |

## fixed100_CCPP

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.4394 (0.0105) | 1.8121 (0.0549) | 0.8047 (0.0939) | 0.1926 (0.0044) | 0.1890 (0.0044) |
| Default+PT | 1.1972 (0.0407) | 1.3916 (0.0658) | 0.2935 (0.0266) | 0.1912 (0.0038) | 0.1876 (0.0038) |
| MTMH | 1.3134 (0.0290) | 1.6304 (0.0419) | 0.5629 (0.0699) | 0.1926 (0.0050) | 0.1876 (0.0057) |
| MTMH+PT | 1.2129 (0.0531) | 1.4296 (0.1064) | 0.3349 (0.0743) | 0.1900 (0.0039) | 0.1852 (0.0036) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CCPP | Default | - | 1 | 112.9 | 1.4 | 1.0 |
| fixed100_CCPP | MTMH | - | 1 | 712.2 | 3.1 | 6.31 |
| fixed100_CCPP | Default+PT | 26 | 26 | 290.5 | 7.4 | 2.57 |
| fixed100_CCPP | MTMH+PT | 26 | 26 | 891.5 | 4.1 | 7.9 |

## fixed100_CalHousing_subsample10000

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.3883 (0.0355) | 1.7343 (0.0708) | 0.7374 (0.0572) | 0.3474 (0.0130) | 0.3434 (0.0138) |
| Default+PT | 1.1865 (0.0555) | 1.3747 (0.0968) | 0.2813 (0.0958) | 0.3411 (0.0137) | 0.3367 (0.0131) |
| MTMH | 1.2879 (0.0112) | 1.6208 (0.0339) | 0.6153 (0.0811) | 0.3532 (0.0150) | 0.3460 (0.0157) |
| MTMH+PT | 1.1814 (0.0305) | 1.3741 (0.0822) | 0.3396 (0.1179) | 0.3478 (0.0086) | 0.3410 (0.0086) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CalHousing_subsample10000 | Default | - | 1 | 114.1 | 0.6 | 1.0 |
| fixed100_CalHousing_subsample10000 | MTMH | - | 1 | 620.6 | 7.7 | 5.44 |
| fixed100_CalHousing_subsample10000 | Default+PT | 36 | 36 | 335.3 | 5.9 | 2.94 |
| fixed100_CalHousing_subsample10000 | MTMH+PT | 36 | 36 | 895.8 | 1.8 | 7.85 |

## fixed100_Concrete

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.1733 (0.0268) | 1.3938 (0.0936) | 0.3066 (0.1116) | 0.2944 (0.0088) | 0.2786 (0.0081) |
| Default+PT | 1.0362 (0.0070) | 1.0819 (0.0125) | 0.0501 (0.0092) | 0.2851 (0.0161) | 0.2668 (0.0141) |
| MTMH | 1.1084 (0.0377) | 1.2654 (0.0721) | 0.1823 (0.0925) | 0.2914 (0.0201) | 0.2732 (0.0182) |
| MTMH+PT | 1.0317 (0.0104) | 1.0646 (0.0159) | 0.0405 (0.0149) | 0.2803 (0.0169) | 0.2609 (0.0148) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 45.6 | 0.9 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 174.3 | 0.1 | 3.83 |
| fixed100_Concrete | Default+PT | 37 | 37 | 143.0 | 9.6 | 3.14 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 298.7 | 4.6 | 6.56 |

## fixed100_CPUAct

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.5176 (0.0527) | 1.9266 (0.1162) | 1.0914 (0.2695) | 0.1371 (0.0074) | 0.1887 (0.0094) |
| Default+PT | 1.1717 (0.0558) | 1.3218 (0.1273) | 0.2660 (0.1242) | 0.1301 (0.0048) | 0.1800 (0.0064) |
| MTMH | 1.4279 (0.0240) | 1.8216 (0.0289) | 1.1184 (0.1884) | 0.1414 (0.0063) | 0.1912 (0.0071) |
| MTMH+PT | 1.1827 (0.0544) | 1.3897 (0.1591) | 0.3520 (0.1740) | 0.1343 (0.0097) | 0.1831 (0.0115) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CPUAct | Default | - | 1 | 104.6 | 0.4 | 1.0 |
| fixed100_CPUAct | MTMH | - | 1 | 612.6 | 12.4 | 5.86 |
| fixed100_CPUAct | Default+PT | 60 | 60 | 359.5 | 19.2 | 3.44 |
| fixed100_CPUAct | MTMH+PT | 60 | 60 | 981.7 | 10.1 | 9.38 |

## fixed100_Friedman

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.2330 (0.0181) | 1.4820 (0.0227) | 0.3445 (0.0399) | 0.2122 (0.0070) | 0.2101 (0.0064) |
| Default+PT | 1.0518 (0.0029) | 1.1001 (0.0062) | 0.0609 (0.0079) | 0.2081 (0.0073) | 0.2064 (0.0070) |
| MTMH | 1.1082 (0.0122) | 1.2538 (0.0270) | 0.1460 (0.0273) | 0.2120 (0.0056) | 0.2099 (0.0061) |
| MTMH+PT | 1.0275 (0.0049) | 1.0576 (0.0095) | 0.0315 (0.0045) | 0.2094 (0.0086) | 0.2078 (0.0087) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 53.9 | 0.6 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 244.5 | 1.1 | 4.54 |
| fixed100_Friedman | Default+PT | 44 | 44 | 174.6 | 8.2 | 3.24 |
| fixed100_Friedman | MTMH+PT | 44 | 44 | 389.2 | 0.9 | 7.22 |

## fixed100_FriedmanSparseDir_p20

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.1993 (0.0083) | 1.3988 (0.0220) | 0.2554 (0.0113) | 0.2260 (0.0023) | 0.2248 (0.0025) |
| Default+PT | 1.0388 (0.0077) | 1.0787 (0.0077) | 0.0381 (0.0050) | 0.2225 (0.0029) | 0.2214 (0.0028) |
| MTMH | 1.0655 (0.0074) | 1.1387 (0.0119) | 0.0700 (0.0092) | 0.2229 (0.0013) | 0.2218 (0.0012) |
| MTMH+PT | 1.0175 (0.0040) | 1.0347 (0.0069) | 0.0160 (0.0038) | 0.2210 (0.0026) | 0.2197 (0.0023) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p20 | Default | - | 1 | 57.1 | 1.1 | 1.0 |
| fixed100_FriedmanSparseDir_p20 | MTMH | - | 1 | 252.9 | 0.9 | 4.43 |
| fixed100_FriedmanSparseDir_p20 | Default+PT | 47 | 47 | 181.4 | 3.2 | 3.18 |
| fixed100_FriedmanSparseDir_p20 | MTMH+PT | 47 | 47 | 406.9 | 2.2 | 7.13 |

## fixed100_FriedmanSparseDir_p100

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.1734 (0.0107) | 1.3534 (0.0610) | 0.2123 (0.0282) | 0.2264 (0.0040) | 0.2251 (0.0039) |
| Default+PT | 1.0410 (0.0092) | 1.0857 (0.0222) | 0.0423 (0.0108) | 0.2215 (0.0027) | 0.2203 (0.0022) |
| MTMH | 1.0700 (0.0038) | 1.1392 (0.0140) | 0.0732 (0.0054) | 0.2212 (0.0030) | 0.2201 (0.0033) |
| MTMH+PT | 1.0206 (0.0039) | 1.0422 (0.0116) | 0.0205 (0.0077) | 0.2210 (0.0029) | 0.2197 (0.0030) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p100 | Default | - | 1 | 56.9 | 0.8 | 1.0 |
| fixed100_FriedmanSparseDir_p100 | MTMH | - | 1 | 255.7 | 1.0 | 4.49 |
| fixed100_FriedmanSparseDir_p100 | Default+PT | 41 | 41 | 181.1 | 10.1 | 3.18 |
| fixed100_FriedmanSparseDir_p100 | MTMH+PT | 41 | 41 | 404.9 | 5.3 | 7.12 |

## fixed100_FriedmanSparseDir_p200

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.1841 (0.0157) | 1.3586 (0.0169) | 0.2160 (0.0317) | 0.2250 (0.0014) | 0.2233 (0.0016) |
| Default+PT | 1.0331 (0.0070) | 1.0662 (0.0158) | 0.0349 (0.0069) | 0.2220 (0.0036) | 0.2209 (0.0037) |
| MTMH | 1.0676 (0.0066) | 1.1429 (0.0185) | 0.0738 (0.0154) | 0.2223 (0.0021) | 0.2208 (0.0026) |
| MTMH+PT | 1.0152 (0.0027) | 1.0292 (0.0055) | 0.0130 (0.0024) | 0.2207 (0.0025) | 0.2195 (0.0025) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_FriedmanSparseDir_p200 | Default | - | 1 | 57.2 | 0.7 | 1.0 |
| fixed100_FriedmanSparseDir_p200 | MTMH | - | 1 | 254.4 | 1.3 | 4.45 |
| fixed100_FriedmanSparseDir_p200 | Default+PT | 47 | 47 | 196.4 | 15.8 | 3.43 |
| fixed100_FriedmanSparseDir_p200 | MTMH+PT | 47 | 47 | 421.9 | 5.2 | 7.38 |

## fixed100_SeoulBike

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.4664 (0.0308) | 1.9129 (0.0889) | 1.1017 (0.1835) | 0.3511 (0.0277) | 0.3189 (0.0135) |
| Default+PT | 1.2195 (0.0418) | 1.4624 (0.1138) | 0.3659 (0.0786) | 0.3504 (0.0269) | 0.3136 (0.0137) |
| MTMH | 1.4147 (0.0388) | 1.8420 (0.0407) | 1.2027 (0.1825) | 0.3685 (0.0235) | 0.3190 (0.0129) |
| MTMH+PT | 1.2782 (0.0560) | 1.5697 (0.1227) | 0.6755 (0.3398) | 0.3728 (0.0111) | 0.3147 (0.0049) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_SeoulBike | Default | - | 1 | 88.3 | 0.6 | 1.0 |
| fixed100_SeoulBike | MTMH | - | 1 | 470.8 | 4.6 | 5.33 |
| fixed100_SeoulBike | Default+PT | 46 | 46 | 315.9 | 22.1 | 3.58 |
| fixed100_SeoulBike | MTMH+PT | 46 | 46 | 740.8 | 22.8 | 8.39 |

## fixed100_Superconduct_subsample10000

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.5617 (0.0490) | 2.0872 (0.0896) | 1.5651 (0.3243) | 0.3253 (0.0174) | 0.3253 (0.0134) |
| Default+PT | 1.1924 (0.0170) | 1.4069 (0.0598) | 0.4059 (0.0895) | 0.3112 (0.0237) | 0.3073 (0.0148) |
| MTMH | 1.5063 (0.0375) | 2.0387 (0.0969) | 1.7312 (0.3595) | 0.3325 (0.0298) | 0.3262 (0.0229) |
| MTMH+PT | 1.2173 (0.0789) | 1.4459 (0.1671) | 0.4440 (0.2156) | 0.2995 (0.0212) | 0.2937 (0.0162) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Superconduct_subsample10000 | Default | - | 1 | 125.4 | 2.1 | 1.0 |
| fixed100_Superconduct_subsample10000 | MTMH | - | 1 | 726.1 | 4.0 | 5.79 |
| fixed100_Superconduct_subsample10000 | Default+PT | 48 | 48 | 411.8 | 9.9 | 3.28 |
| fixed100_Superconduct_subsample10000 | MTMH+PT | 48 | 48 | 1110.5 | 2.8 | 8.86 |

## fixed100_Protein_subsample10000

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.2286 (0.0209) | 1.4688 (0.0201) | 0.3340 (0.0266) | 0.7214 (0.0179) | 0.7314 (0.0198) |
| Default+PT | 1.2144 (0.0105) | 1.4122 (0.0348) | 0.3074 (0.0278) | 0.7245 (0.0169) | 0.7331 (0.0179) |
| MTMH | 1.1219 (0.0101) | 1.2712 (0.0232) | 0.1701 (0.0141) | 0.7136 (0.0198) | 0.7220 (0.0228) |
| MTMH+PT | 1.1309 (0.0088) | 1.2543 (0.0353) | 0.1603 (0.0208) | 0.7136 (0.0147) | 0.7210 (0.0167) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Protein_subsample10000 | Default | - | 1 | 124.7 | 0.7 | 1.0 |
| fixed100_Protein_subsample10000 | MTMH | - | 1 | 785.7 | 4.7 | 6.3 |
| fixed100_Protein_subsample10000 | Default+PT | 17 | 17 | 264.8 | 5.9 | 2.12 |
| fixed100_Protein_subsample10000 | MTMH+PT | 17 | 17 | 940.3 | 2.0 | 7.54 |

## fixed100_SpaceGA

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.2451 (0.0187) | 1.5233 (0.0486) | 0.3796 (0.0613) | 0.5290 (0.0253) | 0.4841 (0.0179) |
| Default+PT | 1.0777 (0.0352) | 1.1728 (0.0851) | 0.1271 (0.0673) | 0.5201 (0.0193) | 0.4719 (0.0150) |
| MTMH | 1.1370 (0.0476) | 1.2914 (0.0976) | 0.2216 (0.0879) | 0.5291 (0.0233) | 0.4836 (0.0165) |
| MTMH+PT | 1.0529 (0.0205) | 1.1144 (0.0505) | 0.0644 (0.0249) | 0.5240 (0.0168) | 0.4736 (0.0098) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_SpaceGA | Default | - | 1 | 57.8 | 0.7 | 1.0 |
| fixed100_SpaceGA | MTMH | - | 1 | 280.8 | 1.3 | 4.86 |
| fixed100_SpaceGA | Default+PT | 32 | 32 | 177.3 | 6.8 | 3.07 |
| fixed100_SpaceGA | MTMH+PT | 32 | 32 | 426.2 | 4.2 | 7.37 |

## fixed100_QSARFishToxicity

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.0132 (0.0039) | 1.0319 (0.0095) | 0.0132 (0.0039) | 0.6284 (0.0095) | 0.6240 (0.0072) |
| Default+PT | 1.0054 (0.0008) | 1.0133 (0.0027) | 0.0055 (0.0015) | 0.6287 (0.0094) | 0.6246 (0.0071) |
| MTMH | 1.0048 (0.0009) | 1.0109 (0.0025) | 0.0044 (0.0011) | 0.6287 (0.0086) | 0.6244 (0.0067) |
| MTMH+PT | 1.0032 (0.0011) | 1.0074 (0.0044) | 0.0035 (0.0028) | 0.6296 (0.0086) | 0.6252 (0.0068) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_QSARFishToxicity | Default | - | 1 | 46.8 | 0.2 | 1.0 |
| fixed100_QSARFishToxicity | MTMH | - | 1 | 174.2 | 0.3 | 3.72 |
| fixed100_QSARFishToxicity | Default+PT | 21 | 21 | 136.1 | 6.0 | 2.91 |
| fixed100_QSARFishToxicity | MTMH+PT | 21 | 21 | 287.9 | 6.3 | 6.15 |

## fixed100_Cars

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.0292 (0.0107) | 1.0867 (0.0226) | 0.0563 (0.0224) | 0.2388 (0.0034) | 0.2421 (0.0043) |
| Default+PT | 1.0077 (0.0016) | 1.0181 (0.0055) | 0.0123 (0.0047) | 0.2386 (0.0033) | 0.2415 (0.0043) |
| MTMH | 1.0102 (0.0021) | 1.0312 (0.0135) | 0.0190 (0.0065) | 0.2388 (0.0025) | 0.2420 (0.0036) |
| MTMH+PT | 1.0040 (0.0010) | 1.0093 (0.0038) | 0.0052 (0.0011) | 0.2384 (0.0026) | 0.2413 (0.0037) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Cars | Default | - | 1 | 34.8 | 0.5 | 1.0 |
| fixed100_Cars | MTMH | - | 1 | 140.5 | 0.6 | 4.03 |
| fixed100_Cars | Default+PT | 28 | 28 | 117.1 | 8.8 | 3.36 |
| fixed100_Cars | MTMH+PT | 28 | 28 | 268.5 | 3.6 | 7.71 |

## fixed100_CPS88Wages_subsample10000

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.0927 (0.0187) | 1.2031 (0.0435) | 0.1077 (0.0292) | 0.7267 (0.0037) | 0.7581 (0.0039) |
| Default+PT | 1.0304 (0.0064) | 1.0633 (0.0121) | 0.0311 (0.0085) | 0.7268 (0.0031) | 0.7580 (0.0031) |
| MTMH | 1.0422 (0.0161) | 1.1011 (0.0349) | 0.0530 (0.0175) | 0.7264 (0.0035) | 0.7577 (0.0021) |
| MTMH+PT | 1.0133 (0.0021) | 1.0284 (0.0053) | 0.0121 (0.0033) | 0.7266 (0.0036) | 0.7579 (0.0021) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_CPS88Wages_subsample10000 | Default | - | 1 | 85.4 | 0.7 | 1.0 |
| fixed100_CPS88Wages_subsample10000 | MTMH | - | 1 | 512.3 | 1.3 | 6.0 |
| fixed100_CPS88Wages_subsample10000 | Default+PT | 12 | 12 | 247.0 | 5.8 | 2.89 |
| fixed100_CPS88Wages_subsample10000 | MTMH+PT | 12 | 12 | 657.5 | 4.4 | 7.7 |

## fixed100_BlackFriday_subsample10000

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.0418 (0.0057) | 1.2061 (0.0359) | 0.1252 (0.0349) | 0.7410 (0.0028) | 0.7418 (0.0025) |
| Default+PT | 1.0289 (0.0063) | 1.1220 (0.0199) | 0.0636 (0.0184) | 0.7424 (0.0030) | 0.7432 (0.0028) |
| MTMH | 1.0331 (0.0092) | 1.1608 (0.0883) | 0.0934 (0.0512) | 0.7420 (0.0036) | 0.7427 (0.0034) |
| MTMH+PT | 1.0176 (0.0063) | 1.0813 (0.0497) | 0.0465 (0.0346) | 0.7424 (0.0023) | 0.7430 (0.0019) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_BlackFriday_subsample10000 | Default | - | 1 | 97.5 | 1.5 | 1.0 |
| fixed100_BlackFriday_subsample10000 | MTMH | - | 1 | 625.2 | 1.5 | 6.41 |
| fixed100_BlackFriday_subsample10000 | Default+PT | 18 | 18 | 256.6 | 3.3 | 2.63 |
| fixed100_BlackFriday_subsample10000 | MTMH+PT | 18 | 18 | 787.2 | 6.6 | 8.08 |

## fixed100_Fiat500

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.0271 (0.0086) | 1.0540 (0.0155) | 0.0253 (0.0110) | 0.3796 (0.0091) | 0.3784 (0.0074) |
| Default+PT | 1.0086 (0.0017) | 1.0191 (0.0046) | 0.0093 (0.0035) | 0.3793 (0.0104) | 0.3782 (0.0087) |
| MTMH | 1.0107 (0.0038) | 1.0239 (0.0075) | 0.0094 (0.0043) | 0.3800 (0.0093) | 0.3786 (0.0074) |
| MTMH+PT | 1.0046 (0.0005) | 1.0093 (0.0006) | 0.0036 (0.0004) | 0.3796 (0.0100) | 0.3783 (0.0081) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Fiat500 | Default | - | 1 | 48.0 | 0.6 | 1.0 |
| fixed100_Fiat500 | MTMH | - | 1 | 201.9 | 0.2 | 4.2 |
| fixed100_Fiat500 | Default+PT | 20 | 20 | 144.4 | 2.9 | 3.0 |
| fixed100_Fiat500 | MTMH+PT | 20 | 20 | 316.5 | 6.2 | 6.59 |

## fixed100_Kin8nm

| method | cross R-hat median | cross R-hat q90 | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- |
| Default | 1.6039 (0.0270) | 1.9948 (0.0830) | 1.3155 (0.1606) | 0.4719 (0.0173) | 0.4662 (0.0176) |
| Default+PT | 1.2586 (0.0725) | 1.4920 (0.1199) | 0.4639 (0.1603) | 0.4410 (0.0097) | 0.4333 (0.0112) |
| MTMH | 1.5165 (0.0193) | 1.8680 (0.0447) | 1.0220 (0.0842) | 0.4670 (0.0152) | 0.4631 (0.0159) |
| MTMH+PT | 1.2313 (0.0777) | 1.4210 (0.1414) | 0.3576 (0.1331) | 0.4274 (0.0151) | 0.4230 (0.0157) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Kin8nm | Default | - | 1 | 108.7 | 1.0 | 1.0 |
| fixed100_Kin8nm | MTMH | - | 1 | 654.7 | 3.7 | 6.02 |
| fixed100_Kin8nm | Default+PT | 40 | 40 | 346.9 | 3.1 | 3.19 |
| fixed100_Kin8nm | MTMH+PT | 40 | 40 | 951.4 | 1.2 | 8.75 |

## Metric definitions

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Method RMSE is sqrt(mean chain MSE) within each run; setting summaries average run scores.
- Raw RMSE and CRPS remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
