# Fixed100 rerun and TODO3 benchmarks

This folder contains scripts and analysis notebooks for the July 2026 fixed100 BART rerun and TODO3 external StochTree benchmarks.

## Fixed100 long/short rerun

- Long-chain reference: `default_long`.
- Short-chain methods: `default`, `default_pt`, `mtmh`, `mtmh_pt`.
- Main datasets:
  - `fixed100_Abalone`
  - `fixed100_Concrete`
  - `fixed100_Friedman`

Analysis notebooks:
- `real1_Abalone_fixed100_partial.ipynb`
- `real4_Concrete_fixed100_full_after_correct_long01.ipynb`
- `syn1_Friedman_fixed100_partial.ipynb`

## TODO3

TODO3a:
External StochTree BART reference with `num_gfr=0`, burn-in 1000, 4 chains × 2500 retained draws.

TODO3b:
100-chain multistart benchmark with burn-in 500, 100 chains × 100 retained draws.

TODO3c:
Preliminary external StochTree GFR/XBART warm-start BART benchmark with `num_gfr=40`, burn-in 0, 4 chains × 2500 retained draws. This is not yet integration into the custom `bart_playground` sampler.

Scripts:
- `todo3ab_stochtree_fixed100.py`
- `todo3c_gfr_fixed100.py`

Large result stores, logs, tarballs, and prediction arrays are intentionally not committed.
