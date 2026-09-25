# Default-sampler diagnosis summary

All values summarize five runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `cross_chain_window`: 1000
- `rolling_step`: 100
- `segment_length`: 1000
- `segments_per_block`: 4
- `short_burn`: 3000
- `long_burn`: 30

`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.


## Dataset-level results

| dataset | worst_projected_rhat_mean | cross_rhat_median_mean | within_rhat_median_mean | short_centroid_distance_mean | long_rhat_median_mean |
| --- | --- | --- | --- | --- | --- |
| fixed100_Abalone | 1.9488 | 1.2447 | 1.2006 | 3.0620 | 1.0012 |
| fixed100_Concrete | 2.2827 | 1.3887 | 1.2548 | 19.5801 | 1.0026 |
| fixed100_Friedman | 2.3150 | 1.4700 | 1.2712 | 5.0399 | 1.0013 |

The long chains are empirical references; their own R-hat summaries qualify PCA-based interpretation.
