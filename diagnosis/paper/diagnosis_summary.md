# Default-sampler diagnosis summary

All values summarize five runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `segment_length`: 1000
- `segments_per_block`: 4
- `short_burn`: 3000
- `long_burn`: 3000

`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.


## Dataset-level results

| dataset | worst_projected_rhat_mean | cross_rhat_median_mean | within_rhat_median_mean | short_centroid_distance_mean | long_rhat_median_mean |
| --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 1.9488 | 1.2447 | 1.2006 | 3.0620 | 1.0001 |
| fixed100_Airfoil | 2.4644 | 1.5883 | 1.3367 | 15.0097 | 1.0415 |
| fixed100_CCPP | 2.6997 | 1.7608 | 1.3839 | 14.3999 | 1.1261 |
| fixed100_CPUAct | 2.6702 | 1.8670 | 1.4266 | 12.7371 | 1.2363 |
| fixed100_CalHousing_subsample5000 | 2.5100 | 1.4956 | 1.2837 | 1.8482 | 1.0177 |
| fixed100_Concrete | 2.2827 | 1.3887 | 1.2548 | 19.5801 | 1.0035 |
| fixed100_Friedman | 2.3150 | 1.4700 | 1.2712 | 5.0399 | 1.0016 |
| fixed100_FriedmanSparseDir_p100 | 2.5177 | 1.4735 | 1.3269 | 3.8009 | 1.0004 |
| fixed100_FriedmanSparseDir_p20 | 2.5198 | 1.4763 | 1.3203 | 4.1051 | 1.0004 |
| fixed100_FriedmanSparseDir_p200 | 2.4533 | 1.4909 | 1.3422 | 3.9360 | 1.0004 |
| fixed100_SeoulBike | 2.5936 | 1.6351 | 1.3119 | 1166.5787 | 1.2146 |

The long chains are empirical references; their own R-hat summaries qualify PCA-based interpretation.
