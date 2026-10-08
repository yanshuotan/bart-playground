# Four-method comparison summary

All diagnostic and predictive entries summarize five paired runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `segment_length`: 1000
- `segments_per_block`: 4
- `short_burn`: 3000
- `long_burn`: 3000
- `chain_separation_draws_per_chain`: 1500
- `energy_chain_pairs`: 4 x 4
- `energy_draws_per_chain`: 1000

`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.

## fixed100_Abalone

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 1.9488 (0.2048) | 1.2447 (0.0095) | 1.2006 (0.0096) | 0.1026 (0.0104) | 0.0193 (0.0030) | 304.6x | 0.6493 (0.0192) | 0.6858 (0.0254) |
| Default+PT | 1.6225 (0.1152) | 1.1090 (0.0283) | 1.1013 (0.0276) | 0.0360 (0.0079) | 0.0089 (0.0049) | 140.7x | 0.6416 (0.0139) | 0.6761 (0.0232) |
| MTMH | 1.6052 (0.1024) | 1.1161 (0.0067) | 1.1054 (0.0045) | 0.0401 (0.0117) | 0.0078 (0.0021) | 123.6x | 0.6455 (0.0166) | 0.6764 (0.0244) |
| MTMH+PT | 1.3112 (0.0744) | 1.0719 (0.0148) | 1.0678 (0.0122) | 0.0200 (0.0082) | 0.0039 (0.0012) | 61.4x | 0.6439 (0.0171) | 0.6742 (0.0240) |

Reference floor (scaled energy among the six pairs of long chains): 0.00006 (SD 0.00002 across runs); the smallest ratio above is 61.4, so the comparison is resolved.

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 75.0 | 1.1 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 548.5 | 3.4 | 7.31 |
| fixed100_Abalone | Default+PT | 20 | 20 | 288.9 | 5.0 | 3.85 |
| fixed100_Abalone | MTMH+PT | 20 | 20 | 756.2 | 3.8 | 10.08 |

## fixed100_Airfoil

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.4644 (0.3427) | 1.5883 (0.1169) | 1.3367 (0.0629) | 0.5821 (0.2382) | 0.1291 (0.0151) | 3.4x | 0.4981 (0.0385) | 0.5335 (0.0516) |
| Default+PT | 2.0193 (0.3105) | 1.1751 (0.0268) | 1.1259 (0.0139) | 0.1119 (0.0433) | 0.0612 (0.0130) | 1.6x | 0.4822 (0.0432) | 0.4926 (0.0511) |
| MTMH | 2.3899 (0.4911) | 1.4191 (0.0950) | 1.2423 (0.0547) | 0.4322 (0.1718) | 0.1112 (0.0329) | 2.9x | 0.4903 (0.0399) | 0.5121 (0.0551) |
| MTMH+PT | 1.9595 (0.1066) | 1.1727 (0.0416) | 1.1266 (0.0348) | 0.1280 (0.0479) | 0.0550 (0.0110) | 1.4x | 0.4835 (0.0423) | 0.4892 (0.0503) |

Reference floor (scaled energy among the six pairs of long chains): 0.03840 (SD 0.00645 across runs); the smallest ratio above is 1.4, so the reference disagrees with itself by as much as the closest method differs from it and no ordering should be read from this dataset.

## fixed100_CCPP

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.6997 (0.2120) | 1.7608 (0.0151) | 1.3839 (0.0168) | 0.8005 (0.0934) | 0.0803 (0.0066) | 1.9x | 0.1925 (0.0044) | 0.2223 (0.0062) |
| Default+PT | 2.6667 (0.1584) | 1.4557 (0.0509) | 1.2738 (0.0304) | 0.3371 (0.0974) | 0.0637 (0.0034) | 1.5x | 0.1925 (0.0037) | 0.2161 (0.0025) |
| MTMH | 2.4333 (0.1650) | 1.5481 (0.0395) | 1.2742 (0.0067) | 0.5622 (0.0692) | 0.0640 (0.0070) | 1.5x | 0.1925 (0.0050) | 0.2170 (0.0079) |
| MTMH+PT | 2.0730 (0.5429) | 1.4375 (0.0585) | 1.2441 (0.0143) | 0.4496 (0.1477) | 0.0580 (0.0094) | 1.4x | 0.1901 (0.0058) | 0.2113 (0.0088) |

Reference floor (scaled energy among the six pairs of long chains): 0.04141 (SD 0.00334 across runs); the smallest ratio above is 1.4, so the reference disagrees with itself by as much as the closest method differs from it and no ordering should be read from this dataset.

## fixed100_CPUAct

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.6702 (0.2507) | 1.8670 (0.0420) | 1.4266 (0.0103) | 1.0867 (0.2689) | 0.0734 (0.0103) | 1.2x | 0.1370 (0.0073) | 0.2147 (0.0126) |
| Default+PT | 2.4183 (0.3092) | 1.3359 (0.0631) | 1.1990 (0.0279) | 0.3185 (0.0984) | 0.0560 (0.0072) | 0.9x | 0.1321 (0.0045) | 0.1990 (0.0072) |
| MTMH | 2.3929 (0.3396) | 1.7072 (0.0284) | 1.3317 (0.0176) | 1.1193 (0.1936) | 0.0708 (0.0064) | 1.1x | 0.1410 (0.0062) | 0.2143 (0.0084) |
| MTMH+PT | 2.4723 (0.1831) | 1.3758 (0.0549) | 1.2078 (0.0243) | 0.5430 (0.1845) | 0.0576 (0.0058) | 0.9x | 0.1363 (0.0075) | 0.2022 (0.0091) |

Reference floor (scaled energy among the six pairs of long chains): 0.06327 (SD 0.01169 across runs); the smallest ratio above is 0.9, so the reference disagrees with itself by as much as the closest method differs from it and no ordering should be read from this dataset.

## fixed100_CalHousing_subsample5000

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.5100 (0.3485) | 1.4956 (0.0308) | 1.2837 (0.0067) | 0.4715 (0.0807) | 0.0855 (0.0085) | 6.8x | 0.3876 (0.0073) | 0.4114 (0.0040) |
| Default+PT | 2.0761 (0.3196) | 1.3025 (0.0600) | 1.1948 (0.0338) | 0.2523 (0.0662) | 0.0607 (0.0145) | 4.8x | 0.3821 (0.0103) | 0.3968 (0.0091) |
| MTMH | 2.2055 (0.2947) | 1.3425 (0.0321) | 1.2136 (0.0041) | 0.2840 (0.0372) | 0.0553 (0.0100) | 4.4x | 0.3757 (0.0077) | 0.3887 (0.0080) |
| MTMH+PT | 2.2360 (0.2345) | 1.2567 (0.0464) | 1.1834 (0.0279) | 0.1943 (0.0499) | 0.0404 (0.0095) | 3.2x | 0.3700 (0.0108) | 0.3790 (0.0110) |

Reference floor (scaled energy among the six pairs of long chains): 0.01252 (SD 0.00364 across runs); the smallest ratio above is 3.2, so the comparison is resolved.

## fixed100_Concrete

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.2827 (0.3728) | 1.3887 (0.0442) | 1.2548 (0.0130) | 0.3064 (0.1120) | 0.0448 (0.0164) | floor unresolved | 0.2940 (0.0088) | 0.2957 (0.0095) |
| Default+PT | 1.7459 (0.1347) | 1.1242 (0.0268) | 1.1045 (0.0197) | 0.0658 (0.0299) | 0.0140 (0.0049) | floor unresolved | 0.2855 (0.0140) | 0.2776 (0.0141) |
| MTMH | 1.8456 (0.3061) | 1.2720 (0.0593) | 1.1889 (0.0169) | 0.1814 (0.0907) | 0.0250 (0.0089) | floor unresolved | 0.2911 (0.0202) | 0.2866 (0.0203) |
| MTMH+PT | 1.5486 (0.1226) | 1.1046 (0.0233) | 1.0906 (0.0181) | 0.0384 (0.0133) | 0.0090 (0.0045) | floor unresolved | 0.2804 (0.0188) | 0.2705 (0.0203) |

Reference floor (scaled energy among the six pairs of long chains): 0.00155 (SD 0.00174 across runs); the reference chains are not distinguishable from each other at this sample size, so the floor is below detection and the ratios are omitted.

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 51.7 | 0.2 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 333.2 | 0.3 | 6.44 |
| fixed100_Concrete | Default+PT | 37 | 37 | 345.5 | 1.1 | 6.68 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 824.9 | 6.0 | 15.95 |

## fixed100_Friedman

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.3150 (0.4671) | 1.4700 (0.0140) | 1.2712 (0.0063) | 0.3426 (0.0396) | 0.0368 (0.0016) | 94.1x | 0.2122 (0.0070) | 0.2318 (0.0089) |
| Default+PT | 1.8286 (0.2414) | 1.1570 (0.0315) | 1.1242 (0.0193) | 0.0754 (0.0221) | 0.0106 (0.0021) | 27.1x | 0.2097 (0.0073) | 0.2247 (0.0101) |
| MTMH | 2.3504 (0.2138) | 1.2997 (0.0181) | 1.2143 (0.0082) | 0.1464 (0.0272) | 0.0157 (0.0025) | 40.0x | 0.2119 (0.0056) | 0.2290 (0.0089) |
| MTMH+PT | 1.5295 (0.0455) | 1.1204 (0.0203) | 1.1058 (0.0177) | 0.0426 (0.0137) | 0.0051 (0.0017) | 13.1x | 0.2095 (0.0066) | 0.2242 (0.0096) |

Reference floor (scaled energy among the six pairs of long chains): 0.00039 (SD 0.00010 across runs); the smallest ratio above is 13.1, so the comparison is resolved.

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 52.6 | 0.6 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 375.6 | 1.9 | 7.14 |
| fixed100_Friedman | Default+PT | 36 | 36 | 348.0 | 18.1 | 6.62 |
| fixed100_Friedman | MTMH+PT | 36 | 36 | 798.1 | 3.7 | 15.18 |

## fixed100_FriedmanSparseDir_p100

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.5177 (0.2212) | 1.4735 (0.0195) | 1.3269 (0.0080) | 0.2125 (0.0290) | 0.0214 (0.0019) | 179.4x | 0.2264 (0.0040) | 0.2483 (0.0060) |
| Default+PT | 1.6595 (0.0981) | 1.1416 (0.0164) | 1.1291 (0.0135) | 0.0399 (0.0077) | 0.0050 (0.0009) | 42.0x | 0.2214 (0.0021) | 0.2378 (0.0031) |
| MTMH | 1.8136 (0.1770) | 1.2448 (0.0070) | 1.2047 (0.0063) | 0.0730 (0.0050) | 0.0081 (0.0006) | 67.7x | 0.2212 (0.0030) | 0.2386 (0.0054) |
| MTMH+PT | 1.3748 (0.0770) | 1.0946 (0.0105) | 1.0912 (0.0113) | 0.0193 (0.0056) | 0.0021 (0.0005) | 18.0x | 0.2209 (0.0020) | 0.2367 (0.0037) |

Reference floor (scaled energy among the six pairs of long chains): 0.00012 (SD 0.00006 across runs); the smallest ratio above is 18.0, so the comparison is resolved.

## fixed100_FriedmanSparseDir_p20

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.5198 (0.2336) | 1.4763 (0.0210) | 1.3203 (0.0083) | 0.2549 (0.0129) | 0.0237 (0.0016) | floor unresolved | 0.2260 (0.0023) | 0.2487 (0.0042) |
| Default+PT | 1.7234 (0.1628) | 1.1501 (0.0039) | 1.1332 (0.0037) | 0.0470 (0.0130) | 0.0055 (0.0013) | floor unresolved | 0.2216 (0.0032) | 0.2384 (0.0046) |
| MTMH | 1.9080 (0.1687) | 1.2386 (0.0119) | 1.2045 (0.0044) | 0.0701 (0.0086) | 0.0076 (0.0005) | floor unresolved | 0.2229 (0.0014) | 0.2408 (0.0024) |
| MTMH+PT | 1.3699 (0.0996) | 1.0959 (0.0072) | 1.0902 (0.0041) | 0.0180 (0.0044) | 0.0021 (0.0005) | floor unresolved | 0.2207 (0.0019) | 0.2364 (0.0028) |

Reference floor (scaled energy among the six pairs of long chains): 0.00007 (SD 0.00004 across runs); the reference chains are not distinguishable from each other at this sample size, so the floor is below detection and the ratios are omitted.

## fixed100_FriedmanSparseDir_p200

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.4533 (0.2308) | 1.4909 (0.0223) | 1.3422 (0.0079) | 0.2160 (0.0316) | 0.0230 (0.0021) | 236.5x | 0.2250 (0.0014) | 0.2453 (0.0039) |
| Default+PT | 1.5052 (0.1121) | 1.1180 (0.0194) | 1.1086 (0.0190) | 0.0327 (0.0123) | 0.0039 (0.0010) | 40.0x | 0.2210 (0.0031) | 0.2372 (0.0045) |
| MTMH | 1.8073 (0.2442) | 1.2407 (0.0048) | 1.2040 (0.0048) | 0.0742 (0.0146) | 0.0081 (0.0015) | 82.7x | 0.2223 (0.0021) | 0.2392 (0.0044) |
| MTMH+PT | 1.2556 (0.0632) | 1.0777 (0.0089) | 1.0758 (0.0080) | 0.0123 (0.0027) | 0.0015 (0.0003) | 15.7x | 0.2209 (0.0021) | 0.2368 (0.0033) |

Reference floor (scaled energy among the six pairs of long chains): 0.00010 (SD 0.00003 across runs); the smallest ratio above is 15.7, so the comparison is resolved.

## fixed100_SeoulBike

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | energy / floor | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.5936 (0.2334) | 1.6351 (0.0485) | 1.3119 (0.0065) | 0.8689 (0.0751) | 0.2644 (0.0279) | 1.2x | 0.5054 (0.0130) | 0.5043 (0.0098) |
| Default+PT | 2.3310 (0.3535) | 1.4116 (0.0322) | 1.2322 (0.0227) | 0.5597 (0.0785) | 0.2398 (0.0079) | 1.1x | 0.4936 (0.0152) | 0.4901 (0.0176) |
| MTMH | 2.4216 (0.3465) | 1.5012 (0.0373) | 1.2371 (0.0104) | 0.9121 (0.1563) | 0.2357 (0.0334) | 1.1x | 0.4979 (0.0205) | 0.4920 (0.0201) |
| MTMH+PT | 2.4030 (0.1940) | 1.4092 (0.0468) | 1.2218 (0.0114) | 0.5414 (0.2130) | 0.2093 (0.0181) | 1.0x | 0.4987 (0.0196) | 0.4767 (0.0179) |

Reference floor (scaled energy among the six pairs of long chains): 0.21672 (SD 0.02189 across runs); the smallest ratio above is 1.0, so the reference disagrees with itself by as much as the closest method differs from it and no ordering should be read from this dataset.

## Interpretation guardrails

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Segment R-hat uses temporally dependent pseudo-chains and is a stability diagnostic, not a formal convergence certificate.
- Energy distance is averaged over all 4 x 4 short-chain/long-chain pairs, using 1,000 randomly sampled draws from each chain.
- The chain-pair mean is divided by training-target SD times sqrt(number of test points); the within-run SD across the 16 pairs is retained in the numerical tables.
- The reference floor is the mean energy distance over the six pairs of long reference chains, scored with the same draw count and scale. `energy / floor` below about two is not a resolvable difference.
- Raw RMSE, CRPS, and energy distance remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
