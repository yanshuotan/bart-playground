# Default-sampler diagnosis summary

All values summarize five runs per dataset.

## Settings

- `runs_per_dataset`: 5
- `short_burn`: 3000
- `long_burn`: 3000

`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.


## Dataset-level results

| dataset | worst_lambda_max_mean | cross_rhat_median_mean | long_rhat_median_mean |
| --- | --- | --- | --- |
| fixed100_Abalone | 25.4743 | 1.0972 | 1.0002 |
| fixed100_Concrete | 45.4533 | 1.1733 | 1.0035 |
| fixed100_Friedman | 53.7944 | 1.2330 | 1.0016 |

The long chains are empirical references; their own R-hat summaries qualify comparisons with the production-length chains.
