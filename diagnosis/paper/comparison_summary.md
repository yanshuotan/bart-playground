# Four-method comparison summary

All diagnostic and predictive entries summarize five paired runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `segment_length`: 1000
- `segments_per_block`: 4
- `short_burn`: 3000
- `long_burn`: 30
- `chain_separation_draws_per_chain`: 1500
- `energy_chain_pairs`: 4 x 4
- `energy_draws_per_chain`: 1000

`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.

## fixed100_Abalone

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 1.9488 (0.2048) | 1.2447 (0.0095) | 1.2006 (0.0096) | 0.1026 (0.0104) | 0.0199 (0.0031) | 0.6493 (0.0192) | 0.6858 (0.0254) |
| Default+PT | 1.6225 (0.1152) | 1.1090 (0.0283) | 1.1013 (0.0276) | 0.0360 (0.0079) | 0.0093 (0.0048) | 0.6416 (0.0139) | 0.6761 (0.0232) |
| MTMH | 1.6052 (0.1024) | 1.1161 (0.0067) | 1.1054 (0.0045) | 0.0401 (0.0117) | 0.0081 (0.0022) | 0.6455 (0.0166) | 0.6764 (0.0244) |
| MTMH+PT | 1.3112 (0.0744) | 1.0719 (0.0148) | 1.0678 (0.0122) | 0.0200 (0.0082) | 0.0042 (0.0013) | 0.6439 (0.0171) | 0.6742 (0.0240) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | Default | - | 1 | 75.0 | 1.1 | 1.0 |
| fixed100_Abalone | MTMH | - | 1 | 548.5 | 3.4 | 7.31 |
| fixed100_Abalone | Default+PT | 20 | 20 | 288.9 | 5.0 | 3.85 |
| fixed100_Abalone | MTMH+PT | 20 | 20 | 756.2 | 3.8 | 10.08 |

## fixed100_Concrete

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.2827 (0.3728) | 1.3887 (0.0442) | 1.2548 (0.0130) | 0.3079 (0.1106) | 0.0443 (0.0151) | 0.2940 (0.0088) | 0.2957 (0.0095) |
| Default+PT | 1.7459 (0.1347) | 1.1242 (0.0268) | 1.1045 (0.0197) | 0.0658 (0.0291) | 0.0132 (0.0046) | 0.2855 (0.0140) | 0.2776 (0.0141) |
| MTMH | 1.8456 (0.3061) | 1.2720 (0.0593) | 1.1889 (0.0169) | 0.1833 (0.0929) | 0.0245 (0.0084) | 0.2911 (0.0202) | 0.2866 (0.0203) |
| MTMH+PT | 1.5486 (0.1226) | 1.1046 (0.0233) | 1.0906 (0.0181) | 0.0383 (0.0131) | 0.0080 (0.0034) | 0.2804 (0.0188) | 0.2705 (0.0203) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Concrete | Default | - | 1 | 51.7 | 0.2 | 1.0 |
| fixed100_Concrete | MTMH | - | 1 | 333.2 | 0.3 | 6.44 |
| fixed100_Concrete | Default+PT | 37 | 37 | 345.5 | 1.1 | 6.68 |
| fixed100_Concrete | MTMH+PT | 37 | 37 | 824.9 | 6.0 | 15.95 |

## fixed100_Friedman

| method | worst projected R-hat | cross R-hat | within R-hat | B/W ratio | scaled energy | relative RMSE | relative CRPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 2.3150 (0.4671) | 1.4700 (0.0140) | 1.2712 (0.0063) | 0.3467 (0.0401) | 0.0368 (0.0018) | 0.2122 (0.0070) | 0.2318 (0.0089) |
| Default+PT | 1.8286 (0.2414) | 1.1570 (0.0315) | 1.1242 (0.0193) | 0.0745 (0.0218) | 0.0106 (0.0021) | 0.2097 (0.0073) | 0.2247 (0.0101) |
| MTMH | 2.3504 (0.2138) | 1.2997 (0.0181) | 1.2143 (0.0082) | 0.1465 (0.0277) | 0.0156 (0.0024) | 0.2119 (0.0056) | 0.2290 (0.0089) |
| MTMH+PT | 1.5295 (0.0455) | 1.1204 (0.0203) | 1.1058 (0.0177) | 0.0429 (0.0142) | 0.0051 (0.0014) | 0.2095 (0.0066) | 0.2242 (0.0096) |

### Computational cost

| dataset | method | temperatures | workers | mean_seconds_per_chain | sd_seconds_per_chain | relative_to_default |
| --- | --- | --- | --- | --- | --- | --- |
| fixed100_Friedman | Default | - | 1 | 52.6 | 0.6 | 1.0 |
| fixed100_Friedman | MTMH | - | 1 | 375.6 | 1.9 | 7.14 |
| fixed100_Friedman | Default+PT | 36 | 36 | 348.0 | 18.1 | 6.62 |
| fixed100_Friedman | MTMH+PT | 36 | 36 | 798.1 | 3.7 | 15.18 |

## Interpretation guardrails

- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.
- Segment R-hat uses temporally dependent pseudo-chains and is a stability diagnostic, not a formal convergence certificate.
- Energy distance is averaged over all 4 x 4 short-chain/long-chain pairs, using 1,000 randomly sampled draws from each chain.
- The chain-pair mean is divided by training-target SD times sqrt(number of test points); the within-run SD across the 16 pairs is retained in the numerical tables.
- Raw RMSE, CRPS, and energy distance remain available in the CSV tables.
- Timing reports the actual parallel PT implementation and omits serial PT speed-up.
