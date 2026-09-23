# PT serial vs parallel timing

run_id=0, chains per dataset=4 (run one after another), ndpost=10000, nskip=0, n_trees=100, swap_interval=50, chain seeds=2024+c, updated 2026-09-23T18:00:00

`parallel CPUs` = peak number of live PT worker processes measured during each parallel fit (1 for serial / non-PT rows). Times are per chain; `x default` = mean per-chain time / default's mean per-chain time.

## fixed100_Abalone (run 000)

n=4177, train=3057, test=100, temperatures=20, allocated CPUs=-, affinity CPUs=40, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 75.0 | 1.1 | 1.00 |
| mtmh | - | 1 | 548.5 | 3.4 | 7.31 |
| default_pt (serial) | 20 | 1 | 1864.7 | 12.6 | 24.87 |
| default_pt (parallel) | 20 | 20 | 288.9 | 5.0 | 3.85 |
| mtmh_pt (serial) | 20 | 1 | 11376.3 | 68.6 | 151.70 |
| mtmh_pt (parallel) | 20 | 20 | 756.2 | 3.8 | 10.08 |
