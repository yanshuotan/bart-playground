# PT serial vs parallel timing

run_id=0, chains per dataset=4 (run one after another), ndpost=10000, nskip=0, n_trees=100, swap_interval=50, chain seeds=2024+c, updated 2026-09-24T13:15:55

`parallel CPUs` = peak number of live PT worker processes measured during each parallel fit (1 for serial / non-PT rows). Times are per chain; `x default` = mean per-chain time / default's mean per-chain time.

## fixed100_Abalone (run 000)

n=4177, train=3057, test=100, temperatures=20, allocated CPUs=-, affinity CPUs=40, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 75.0 | 1.1 | 1.00 |
| mtmh | - | 1 | 548.5 | 3.4 | 7.31 |
| default_pt (serial) | 20 | 1 | 1864.7 | 12.6 | 24.87 |
| default_pt (parallel) | 20 | 20 | 288.9 | 5.0 | **3.85** |
| mtmh_pt (serial) | 20 | 1 | 11376.3 | 68.6 | 151.70 |
| mtmh_pt (parallel) | 20 | 20 | 756.2 | 3.8 | **10.08** |

## fixed100_Concrete (run 000)

n=1030, train=697, test=100, temperatures=37, allocated CPUs=-, affinity CPUs=40, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 51.7 | 0.2 | 1.00 |
| mtmh | - | 1 | 333.2 | 0.3 | 6.44 |
| default_pt (serial) | 37 | 1 | 2567.9 | 7.2 | 49.64 |
| default_pt (parallel) | 37 | 37 | 345.5 | 1.1 | **6.68** |
| mtmh_pt (serial) | 37 | 1 | 12942.0 | 27.2 | 250.16 |
| mtmh_pt (parallel) | 37 | 37 | 824.9 | 6.0 | **15.95** |

## fixed100_Friedman (run 000)

n=2000, train=1425, test=100, temperatures=36, allocated CPUs=-, affinity CPUs=40, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 52.6 | 0.6 | 1.00 |
| mtmh | - | 1 | 375.6 | 1.9 | 7.14 |
| default_pt (serial) | 36 | 1 | 2499.0 | 2.7 | 47.53 |
| default_pt (parallel) | 36 | 36 | 348.0 | 18.1 | **6.62** |
| mtmh_pt (serial) | 36 | 1 | 14116.9 | 8.4 | 268.50 |
| mtmh_pt (parallel) | 36 | 36 | 798.1 | 3.7 | **15.18** |
