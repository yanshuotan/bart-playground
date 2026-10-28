# PT serial vs parallel timing

run_id=0, chains per dataset=4 (run one after another), ndpost=10000, nskip=0, n_trees=100, swap_interval=50, chain seeds=2024+c, updated 2026-10-28T20:04:48

`parallel CPUs` = peak number of live PT worker processes measured during each parallel fit (1 for serial / non-PT rows). Times are per chain; `x default` = mean per-chain time / default's mean per-chain time. When the ladder was searched in the job, the `ladder search (parallel)` row is that one search, also divided by default's mean per-chain time.

## fixed100_Abalone (run 000)

n=4177, train=3057, test=100, temperatures=23, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 113.3s, 6 rounds, 23 temperatures, peak workers=23, 1000 search points; stored ladder has 23 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 64.1 | 1.1 | 1.00 |
| mtmh | - | 1 | 327.2 | 1.1 | 5.10 |
| default_pt (parallel) | 23 | 23 | 188.6 | 3.9 | 2.94 |
| mtmh_pt (parallel) | 23 | 23 | 467.4 | 3.0 | 7.29 |
| ladder search (parallel) | 23 | 23 | 113.3 | - | 1.77 |

## fixed100_Airfoil (run 000)

n=1503, train=1052, test=100, temperatures=55, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 192.7s, 8 rounds, 55 temperatures, peak workers=55, 1000 search points; stored ladder has 55 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 45.1 | 0.1 | 1.00 |
| mtmh | - | 1 | 181.6 | 0.9 | 4.03 |
| default_pt (parallel) | 55 | 55 | 163.7 | 13.5 | 3.63 |
| mtmh_pt (parallel) | 55 | 55 | 325.5 | 12.0 | 7.22 |
| ladder search (parallel) | 55 | 55 | 192.7 | - | 4.28 |

## fixed100_CalHousing_subsample5000 (run 000)

n=5000, train=3675, test=100, temperatures=35, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 178.1s, 8 rounds, 35 temperatures, peak workers=35, 1000 search points; stored ladder has 35 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 76.9 | 0.5 | 1.00 |
| mtmh | - | 1 | 389.4 | 2.1 | 5.07 |
| default_pt (parallel) | 35 | 35 | 235.2 | 2.6 | 3.06 |
| mtmh_pt (parallel) | 35 | 35 | 570.4 | 5.2 | 7.42 |
| ladder search (parallel) | 35 | 35 | 178.1 | - | 2.32 |

## fixed100_CCPP (run 000)

n=9568, train=7101, test=100, temperatures=26, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 91.8s, 5 rounds, 26 temperatures, peak workers=26, 1000 search points; stored ladder has 26 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 112.9 | 1.4 | 1.00 |
| mtmh | - | 1 | 712.2 | 3.1 | 6.31 |
| default_pt (parallel) | 26 | 26 | 290.5 | 7.4 | 2.57 |
| mtmh_pt (parallel) | 26 | 26 | 891.5 | 4.1 | 7.90 |
| ladder search (parallel) | 26 | 26 | 91.8 | - | 0.81 |

## fixed100_Concrete (run 000)

n=1030, train=697, test=100, temperatures=37, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 107.7s, 6 rounds, 37 temperatures, peak workers=37, 697 search points; stored ladder has 37 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 45.6 | 0.9 | 1.00 |
| mtmh | - | 1 | 174.3 | 0.1 | 3.83 |
| default_pt (parallel) | 37 | 37 | 143.0 | 9.6 | 3.14 |
| mtmh_pt (parallel) | 37 | 37 | 298.7 | 4.6 | 6.56 |
| ladder search (parallel) | 37 | 37 | 107.7 | - | 2.36 |

## fixed100_CPUAct (run 000)

n=8192, train=6069, test=100, temperatures=60, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 229.8s, 10 rounds, 60 temperatures, peak workers=59, 1000 search points; stored ladder has 60 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 104.6 | 0.4 | 1.00 |
| mtmh | - | 1 | 612.6 | 12.4 | 5.86 |
| default_pt (parallel) | 60 | 60 | 359.5 | 19.2 | 3.44 |
| mtmh_pt (parallel) | 60 | 60 | 981.7 | 10.1 | 9.38 |
| ladder search (parallel) | 60 | 59 | 229.8 | - | 2.20 |

## fixed100_Friedman (run 000)

n=2000, train=1425, test=100, temperatures=44, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 143.3s, 7 rounds, 44 temperatures, peak workers=44, 1000 search points; stored ladder has 44 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 53.9 | 0.6 | 1.00 |
| mtmh | - | 1 | 244.5 | 1.1 | 4.54 |
| default_pt (parallel) | 44 | 44 | 174.6 | 8.2 | 3.24 |
| mtmh_pt (parallel) | 44 | 44 | 389.2 | 0.9 | 7.22 |
| ladder search (parallel) | 44 | 44 | 143.3 | - | 2.66 |

## fixed100_FriedmanSparseDirP20K5 (run 000)

n=2000, train=1425, test=100, temperatures=47, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 207.5s, 9 rounds, 47 temperatures, peak workers=47, 1000 search points; stored ladder has 47 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 57.1 | 1.1 | 1.00 |
| mtmh | - | 1 | 252.9 | 0.9 | 4.43 |
| default_pt (parallel) | 47 | 47 | 181.4 | 3.2 | 3.18 |
| mtmh_pt (parallel) | 47 | 47 | 406.9 | 2.2 | 7.13 |
| ladder search (parallel) | 47 | 47 | 207.5 | - | 3.64 |

## fixed100_FriedmanSparseDir (run 000)

n=2000, train=1425, test=100, temperatures=41, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 128.7s, 6 rounds, 41 temperatures, peak workers=41, 1000 search points; stored ladder has 41 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 56.9 | 0.8 | 1.00 |
| mtmh | - | 1 | 255.7 | 1.0 | 4.49 |
| default_pt (parallel) | 41 | 41 | 181.1 | 10.1 | 3.18 |
| mtmh_pt (parallel) | 41 | 41 | 404.9 | 5.3 | 7.12 |
| ladder search (parallel) | 41 | 41 | 128.7 | - | 2.26 |

## fixed100_FriedmanSparseDirP200K5 (run 000)

n=2000, train=1425, test=100, temperatures=47, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 158.7s, 7 rounds, 47 temperatures, peak workers=47, 1000 search points; stored ladder has 47 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 57.2 | 0.7 | 1.00 |
| mtmh | - | 1 | 254.4 | 1.3 | 4.45 |
| default_pt (parallel) | 47 | 47 | 196.4 | 15.8 | 3.43 |
| mtmh_pt (parallel) | 47 | 47 | 421.9 | 5.2 | 7.38 |
| ladder search (parallel) | 47 | 47 | 158.7 | - | 2.78 |

## fixed100_SeoulBike (run 000)

n=8760, train=6495, test=100, temperatures=46, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 202.7s, 9 rounds, 46 temperatures, peak workers=46, 1000 search points; stored ladder has 46 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 88.3 | 0.6 | 1.00 |
| mtmh | - | 1 | 470.8 | 4.6 | 5.33 |
| default_pt (parallel) | 46 | 46 | 315.9 | 22.1 | 3.58 |
| mtmh_pt (parallel) | 46 | 46 | 740.8 | 22.8 | 8.39 |
| ladder search (parallel) | 46 | 46 | 202.7 | - | 2.30 |

## fixed100_Superconduct_subsample5000 (run 000)

n=5000, train=3675, test=100, temperatures=46, allocated CPUs=-, affinity CPUs=60, chains timed so far=4/4

Ladder search: 215.9s, 10 rounds, 46 temperatures, peak workers=44, 1000 search points; stored ladder has 46 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 81.7 | 0.7 | 1.00 |
| mtmh | - | 1 | 426.4 | 4.9 | 5.22 |
| default_pt (parallel) | 46 | 46 | 255.0 | 2.3 | 3.12 |
| mtmh_pt (parallel) | 46 | 46 | 638.9 | 4.9 | 7.82 |
| ladder search (parallel) | 46 | 44 | 215.9 | - | 2.64 |
