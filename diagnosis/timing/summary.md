# PT serial vs parallel timing

run_id=0, chains per dataset=4 (run one after another), ndpost=10000, nskip=0, n_trees=100, swap_interval=50, chain seeds=2024+c, updated 2026-10-12T17:27:28

`parallel CPUs` = peak number of live PT worker processes measured during each parallel fit (1 for serial / non-PT rows). Times are per chain; `x default` = mean per-chain time / default's mean per-chain time.

## fixed100_Abalone (run 000)

n=4177, train=3057, test=100, temperatures=21, allocated CPUs=-, affinity CPUs=21, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 62.9 | 0.8 | 1.00 |
| mtmh | - | 1 | 477.0 | 3.5 | 7.59 |
| default_pt (parallel) | 21 | 21 | 229.0 | 6.8 | 3.64 |
| mtmh_pt (parallel) | 21 | 21 | 677.7 | 22.3 | 10.78 |

## fixed100_Airfoil (run 000)

n=1503, train=1052, test=100, temperatures=45, allocated CPUs=-, affinity CPUs=45, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 45.3 | 0.1 | 1.00 |
| mtmh | - | 1 | 301.8 | 0.4 | 6.66 |
| default_pt (parallel) | 45 | 45 | 262.7 | 5.8 | 5.80 |
| mtmh_pt (parallel) | 45 | 45 | 599.9 | 3.4 | 13.24 |

## fixed100_CalHousing_subsample5000 (run 000)

n=5000, train=3675, test=100, temperatures=28, allocated CPUs=-, affinity CPUs=28, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 77.7 | 0.3 | 1.00 |
| mtmh | - | 1 | 590.6 | 3.2 | 7.60 |
| default_pt (parallel) | 28 | 28 | 310.6 | 1.2 | 4.00 |
| mtmh_pt (parallel) | 28 | 28 | 908.8 | 0.7 | 11.69 |

## fixed100_CCPP (run 000)

n=9568, train=7101, test=100, temperatures=25, allocated CPUs=-, affinity CPUs=25, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 112.5 | 1.2 | 1.00 |
| mtmh | - | 1 | 998.5 | 5.7 | 8.88 |
| default_pt (parallel) | 25 | 25 | 414.5 | 6.5 | 3.68 |
| mtmh_pt (parallel) | 25 | 25 | 1408.0 | 3.5 | 12.51 |

## fixed100_Concrete (run 000)

n=1030, train=697, test=100, temperatures=37, allocated CPUs=-, affinity CPUs=37, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 45.7 | 0.3 | 1.00 |
| mtmh | - | 1 | 295.5 | 0.9 | 6.47 |
| default_pt (parallel) | 37 | 37 | 230.9 | 10.6 | 5.06 |
| mtmh_pt (parallel) | 37 | 37 | 574.7 | 2.6 | 12.59 |

## fixed100_CPUAct (run 000)

n=8192, train=6069, test=100, temperatures=52, allocated CPUs=-, affinity CPUs=52, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 104.4 | 0.5 | 1.00 |
| mtmh | - | 1 | 876.6 | 12.9 | 8.40 |
| default_pt (parallel) | 52 | 52 | 726.4 | 40.1 | 6.96 |
| mtmh_pt (parallel) | 52 | 52 | 1536.7 | 1.8 | 14.72 |

## fixed100_Friedman (run 000)

n=2000, train=1425, test=100, temperatures=36, allocated CPUs=-, affinity CPUs=36, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 54.0 | 0.5 | 1.00 |
| mtmh | - | 1 | 387.0 | 3.5 | 7.16 |
| default_pt (parallel) | 36 | 36 | 285.3 | 4.9 | 5.28 |
| mtmh_pt (parallel) | 36 | 36 | 703.0 | 14.5 | 13.01 |

## fixed100_FriedmanSparseDirP20K5 (run 000)

n=2000, train=1425, test=100, temperatures=36, allocated CPUs=-, affinity CPUs=36, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 56.2 | 0.4 | 1.00 |
| mtmh | - | 1 | 422.9 | 1.3 | 7.53 |
| default_pt (parallel) | 36 | 36 | 295.4 | 9.4 | 5.26 |
| mtmh_pt (parallel) | 36 | 36 | 741.8 | 4.6 | 13.21 |

## fixed100_FriedmanSparseDir (run 000)

n=2000, train=1425, test=100, temperatures=38, allocated CPUs=-, affinity CPUs=38, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 59.6 | 2.3 | 1.00 |
| mtmh | - | 1 | 439.9 | 9.3 | 7.38 |
| default_pt (parallel) | 38 | 38 | 369.2 | 94.0 | 6.19 |
| mtmh_pt (parallel) | 38 | 38 | 799.7 | 49.2 | 13.41 |

## fixed100_FriedmanSparseDirP200K5 (run 000)

n=2000, train=1425, test=100, temperatures=43, allocated CPUs=-, affinity CPUs=43, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 57.3 | 0.5 | 1.00 |
| mtmh | - | 1 | 436.4 | 2.1 | 7.61 |
| default_pt (parallel) | 43 | 43 | 373.9 | 37.5 | 6.52 |
| mtmh_pt (parallel) | 43 | 43 | 787.3 | 0.9 | 13.73 |

## fixed100_SeoulBike (run 000)

n=8760, train=6495, test=100, temperatures=36, allocated CPUs=-, affinity CPUs=40, chains timed so far=4/4

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 87.7 | 0.8 | 1.00 |
| mtmh | - | 1 | 688.7 | 19.0 | 7.85 |
| default_pt (parallel) | 36 | 36 | 523.6 | 12.7 | 5.97 |
| mtmh_pt (parallel) | 36 | 36 | 1192.8 | 21.5 | 13.60 |
