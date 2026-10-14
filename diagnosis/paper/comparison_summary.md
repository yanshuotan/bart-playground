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

`short_burn` is also the start of the worst-direction calculation.

## fixed100_Abalone

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.3698 (0.1580) | 1.2894 (0.0103) | 1.2247 (0.0067) | 0.1393 (0.0182) | 0.6592 (0.0205) | 0.6546 (0.0142) |
| Default+PT | 1.2163 (0.1107) | 1.1230 (0.0371) | 1.1100 (0.0310) | 0.0473 (0.0125) | 0.6534 (0.0179) | 0.6491 (0.0128) |
| MTMH | 1.2057 (0.1497) | 1.1368 (0.0070) | 1.1197 (0.0043) | 0.0476 (0.0155) | 0.6524 (0.0196) | 0.6482 (0.0128) |
| MTMH+PT | 1.0815 (0.0833) | 1.0764 (0.0180) | 1.0715 (0.0160) | 0.0274 (0.0122) | 0.6515 (0.0194) | 0.6470 (0.0137) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 75.0 | 1.1 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 548.5 | 3.4 | 7.31 |
| fixed100_Abalone | Default+PT | 20 | 20 | 288.9 | 5.0 | 3.85 |
| fixed100_Abalone | MTMH+PT | 20 | 20 | 756.2 | 3.8 | 10.08 |

## fixed100_Airfoil

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.5552 (0.3238) | 1.5883 (0.1169) | 1.3367 (0.0629) | 0.5817 (0.2363) | 0.4981 (0.0385) | 0.4957 (0.0438) |
| Default+PT | 1.5769 (0.2575) | 1.1751 (0.0268) | 1.1259 (0.0139) | 0.1120 (0.0419) | 0.4822 (0.0432) | 0.4702 (0.0464) |
| MTMH | 2.6762 (0.2991) | 1.4191 (0.0950) | 1.2423 (0.0547) | 0.4346 (0.1719) | 0.4903 (0.0399) | 0.4832 (0.0480) |
| MTMH+PT | 1.4525 (0.0635) | 1.1727 (0.0416) | 1.1266 (0.0348) | 0.1271 (0.0476) | 0.4835 (0.0423) | 0.4691 (0.0457) |

## fixed100_CCPP

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.6297 (0.4899) | 1.7608 (0.0151) | 1.3839 (0.0168) | 0.8047 (0.0939) | 0.1925 (0.0044) | 0.1890 (0.0044) |
| Default+PT | 2.1337 (0.2865) | 1.4557 (0.0509) | 1.2738 (0.0304) | 0.3380 (0.0966) | 0.1925 (0.0037) | 0.1885 (0.0027) |
| MTMH | 2.6369 (0.1492) | 1.5481 (0.0395) | 1.2742 (0.0067) | 0.5629 (0.0699) | 0.1925 (0.0050) | 0.1876 (0.0057) |
| MTMH+PT | 2.2908 (0.3123) | 1.4375 (0.0585) | 1.2441 (0.0143) | 0.4503 (0.1447) | 0.1901 (0.0058) | 0.1852 (0.0056) |

## fixed100_CalHousing_subsample5000

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.3257 (0.2028) | 1.4956 (0.0308) | 1.2837 (0.0067) | 0.4720 (0.0811) | 0.3876 (0.0073) | 0.3814 (0.0050) |
| Default+PT | 1.8858 (0.2174) | 1.3025 (0.0600) | 1.1948 (0.0338) | 0.2515 (0.0654) | 0.3821 (0.0103) | 0.3750 (0.0079) |
| MTMH | 2.0520 (0.1633) | 1.3425 (0.0321) | 1.2136 (0.0041) | 0.2852 (0.0372) | 0.3757 (0.0077) | 0.3682 (0.0069) |
| MTMH+PT | 1.9168 (0.3068) | 1.2567 (0.0464) | 1.1834 (0.0279) | 0.1951 (0.0500) | 0.3700 (0.0108) | 0.3619 (0.0093) |

## fixed100_Concrete

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.1029 (0.5306) | 1.3887 (0.0442) | 1.2548 (0.0130) | 0.3066 (0.1116) | 0.2940 (0.0088) | 0.2786 (0.0081) |
| Default+PT | 1.2865 (0.1018) | 1.1242 (0.0268) | 1.1045 (0.0197) | 0.0653 (0.0291) | 0.2855 (0.0140) | 0.2669 (0.0128) |
| MTMH | 1.5933 (0.1683) | 1.2720 (0.0593) | 1.1889 (0.0169) | 0.1823 (0.0925) | 0.2911 (0.0202) | 0.2732 (0.0182) |
| MTMH+PT | 1.0940 (0.0617) | 1.1046 (0.0233) | 1.0906 (0.0181) | 0.0379 (0.0132) | 0.2804 (0.0188) | 0.2612 (0.0179) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 51.7 | 0.2 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 333.2 | 0.3 | 6.44 |
| fixed100_Concrete | Default+PT | 37 | 37 | 345.5 | 1.1 | 6.68 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 824.9 | 6.0 | 15.95 |

## fixed100_CPUAct

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.8752 (0.1264) | 1.8670 (0.0420) | 1.4266 (0.0103) | 1.0914 (0.2695) | 0.1370 (0.0073) | 0.1887 (0.0094) |
| Default+PT | 2.0019 (0.2749) | 1.3359 (0.0631) | 1.1990 (0.0279) | 0.3167 (0.0999) | 0.1321 (0.0045) | 0.1823 (0.0053) |
| MTMH | 2.6826 (0.3445) | 1.7072 (0.0284) | 1.3317 (0.0176) | 1.1184 (0.1884) | 0.1410 (0.0062) | 0.1912 (0.0071) |
| MTMH+PT | 2.2101 (0.1336) | 1.3758 (0.0549) | 1.2078 (0.0243) | 0.5445 (0.1866) | 0.1363 (0.0075) | 0.1848 (0.0081) |

## fixed100_Friedman

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.1742 (0.0502) | 1.4700 (0.0140) | 1.2712 (0.0063) | 0.3445 (0.0399) | 0.2122 (0.0070) | 0.2101 (0.0064) |
| Default+PT | 1.3282 (0.1437) | 1.1570 (0.0315) | 1.1242 (0.0193) | 0.0746 (0.0221) | 0.2097 (0.0073) | 0.2081 (0.0073) |
| MTMH | 1.6790 (0.1553) | 1.2997 (0.0181) | 1.2143 (0.0082) | 0.1460 (0.0273) | 0.2119 (0.0056) | 0.2099 (0.0061) |
| MTMH+PT | 1.1235 (0.0622) | 1.1204 (0.0203) | 1.1058 (0.0177) | 0.0416 (0.0136) | 0.2095 (0.0066) | 0.2078 (0.0069) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 52.6 | 0.6 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 375.6 | 1.9 | 7.14 |
| fixed100_Friedman | Default+PT | 36 | 36 | 348.0 | 18.1 | 6.62 |
| fixed100_Friedman | MTMH+PT | 36 | 36 | 798.1 | 3.7 | 15.18 |

## fixed100_FriedmanSparseDir_p20

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.9772 (0.2305) | 1.4763 (0.0210) | 1.3203 (0.0083) | 0.2554 (0.0113) | 0.2260 (0.0023) | 0.2248 (0.0025) |
| Default+PT | 1.1845 (0.0349) | 1.1501 (0.0039) | 1.1332 (0.0037) | 0.0463 (0.0126) | 0.2216 (0.0032) | 0.2205 (0.0029) |
| MTMH | 1.3363 (0.1201) | 1.2386 (0.0119) | 1.2045 (0.0044) | 0.0700 (0.0092) | 0.2229 (0.0014) | 0.2218 (0.0012) |
| MTMH+PT | 1.0417 (0.0219) | 1.0959 (0.0072) | 1.0902 (0.0041) | 0.0177 (0.0045) | 0.2207 (0.0019) | 0.2195 (0.0016) |

## fixed100_FriedmanSparseDir_p100

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.8905 (0.2146) | 1.4735 (0.0195) | 1.3269 (0.0080) | 0.2123 (0.0282) | 0.2264 (0.0040) | 0.2251 (0.0039) |
| Default+PT | 1.1321 (0.0642) | 1.1416 (0.0164) | 1.1291 (0.0135) | 0.0396 (0.0079) | 0.2214 (0.0021) | 0.2202 (0.0021) |
| MTMH | 1.3072 (0.1106) | 1.2448 (0.0070) | 1.2047 (0.0063) | 0.0732 (0.0054) | 0.2212 (0.0030) | 0.2201 (0.0033) |
| MTMH+PT | 1.0564 (0.0418) | 1.0946 (0.0105) | 1.0912 (0.0113) | 0.0188 (0.0055) | 0.2209 (0.0020) | 0.2197 (0.0021) |

## fixed100_FriedmanSparseDir_p200

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 1.7511 (0.2278) | 1.4909 (0.0223) | 1.3422 (0.0079) | 0.2160 (0.0317) | 0.2250 (0.0014) | 0.2233 (0.0016) |
| Default+PT | 1.0961 (0.0276) | 1.1180 (0.0194) | 1.1086 (0.0190) | 0.0323 (0.0120) | 0.2210 (0.0031) | 0.2198 (0.0030) |
| MTMH | 1.2461 (0.0988) | 1.2407 (0.0048) | 1.2040 (0.0048) | 0.0738 (0.0154) | 0.2223 (0.0021) | 0.2208 (0.0026) |
| MTMH+PT | 1.0474 (0.0597) | 1.0777 (0.0089) | 1.0758 (0.0080) | 0.0117 (0.0029) | 0.2209 (0.0021) | 0.2198 (0.0019) |

## fixed100_SeoulBike

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 2.6879 (0.2918) | 1.7573 (0.0253) | 1.3536 (0.0115) | 1.1017 (0.1835) | 0.3508 (0.0277) | 0.3189 (0.0135) |
| Default+PT | 2.1852 (0.4166) | 1.3458 (0.0516) | 1.1840 (0.0275) | 0.4374 (0.0908) | 0.3452 (0.0199) | 0.3068 (0.0077) |
| MTMH | 2.9862 (0.1697) | 1.6548 (0.0224) | 1.2757 (0.0097) | 1.2027 (0.1825) | 0.3678 (0.0235) | 0.3190 (0.0129) |
| MTMH+PT | 2.3057 (0.5161) | 1.3522 (0.0278) | 1.1883 (0.0147) | 0.4127 (0.0855) | 0.3707 (0.0209) | 0.3143 (0.0085) |

## Interpretation guardrails

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Segment R-hat uses temporally dependent pseudo-chains and is a stability diagnostic, not a formal convergence certificate.
- Raw RMSE and CRPS remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
