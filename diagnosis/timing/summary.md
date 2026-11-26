# PT serial vs parallel timing

run_id=0, chains per dataset=4 (run one after another), ndpost=10000, nskip=0, n_trees=100, swap_interval=50, chain seeds=2024+c, updated 2026-11-19T10:09:07

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

## fixed100_CalHousing_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=36, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 163.6s, 8 rounds, 36 temperatures, peak workers=36, 1000 search points; stored ladder has 36 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 114.1 | 0.6 | 1.00 |
| mtmh | - | 1 | 620.6 | 7.7 | 5.44 |
| default_pt (parallel) | 36 | 36 | 335.3 | 5.9 | 2.94 |
| mtmh_pt (parallel) | 36 | 36 | 895.8 | 1.8 | 7.85 |
| ladder search (parallel) | 36 | 36 | 163.6 | - | 1.43 |

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

## fixed100_Superconduct_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=48, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 196.1s, 9 rounds, 48 temperatures, peak workers=48, 1000 search points; stored ladder has 48 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 125.4 | 2.1 | 1.00 |
| mtmh | - | 1 | 726.1 | 4.0 | 5.79 |
| default_pt (parallel) | 48 | 48 | 411.8 | 9.9 | 3.28 |
| mtmh_pt (parallel) | 48 | 48 | 1110.5 | 2.8 | 8.86 |
| ladder search (parallel) | 48 | 48 | 196.1 | - | 1.56 |

## fixed100_Protein_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=17, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 69.7s, 4 rounds, 17 temperatures, peak workers=17, 1000 search points; stored ladder has 17 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 124.7 | 0.7 | 1.00 |
| mtmh | - | 1 | 785.7 | 4.7 | 6.30 |
| default_pt (parallel) | 17 | 17 | 264.8 | 5.9 | 2.12 |
| mtmh_pt (parallel) | 17 | 17 | 940.3 | 2.0 | 7.54 |
| ladder search (parallel) | 17 | 17 | 69.7 | - | 0.56 |

## fixed100_Diamonds_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=60, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 163.8s, 7 rounds, 60 temperatures, peak workers=60, 1000 search points; stored ladder has 60 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 95.8 | 1.6 | 1.00 |
| mtmh | - | 1 | 538.2 | 3.8 | 5.62 |
| default_pt (parallel) | 60 | 60 | 385.6 | 1.5 | 4.02 |
| mtmh_pt (parallel) | 60 | 60 | 950.1 | 2.4 | 9.91 |
| ladder search (parallel) | 60 | 60 | 163.8 | - | 1.71 |

## fixed100_Sarcos_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=60, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 197.9s, 9 rounds, 60 temperatures, peak workers=60, 1000 search points; stored ladder has 60 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 112.0 | 1.3 | 1.00 |
| mtmh | - | 1 | 613.1 | 6.8 | 5.48 |
| default_pt (parallel) | 60 | 60 | 412.5 | 7.1 | 3.68 |
| mtmh_pt (parallel) | 60 | 60 | 1016.7 | 1.2 | 9.08 |
| ladder search (parallel) | 60 | 60 | 197.9 | - | 1.77 |

## fixed100_MiamiHousing_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=60, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 155.3s, 7 rounds, 60 temperatures, peak workers=60, 1000 search points; stored ladder has 60 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 106.8 | 1.8 | 1.00 |
| mtmh | - | 1 | 519.8 | 6.2 | 4.87 |
| default_pt (parallel) | 60 | 60 | 447.8 | 11.5 | 4.19 |
| mtmh_pt (parallel) | 60 | 60 | 984.6 | 1.2 | 9.22 |
| ladder search (parallel) | 60 | 60 | 155.3 | - | 1.45 |

## fixed100_SpaceGA (run 000)

n=3107, train=2255, test=100, temperatures=32, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 198.3s, 10 rounds, 32 temperatures, peak workers=31, 1000 search points; stored ladder has 32 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 57.8 | 0.7 | 1.00 |
| mtmh | - | 1 | 280.8 | 1.3 | 4.86 |
| default_pt (parallel) | 32 | 32 | 177.3 | 6.8 | 3.07 |
| mtmh_pt (parallel) | 32 | 32 | 426.2 | 4.2 | 7.37 |
| ladder search (parallel) | 32 | 31 | 198.3 | - | 3.43 |

## fixed100_VideoTranscoding_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=60, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 131.6s, 6 rounds, 60 temperatures, peak workers=60, 1000 search points; stored ladder has - temperatures, identical=.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 131.3 | 10.3 | 1.00 |
| mtmh | - | 1 | 627.1 | 21.4 | 4.78 |
| default_pt (parallel) | 60 | 60 | 465.5 | 16.5 | 3.55 |
| mtmh_pt (parallel) | 60 | 60 | 1135.6 | 54.4 | 8.65 |
| ladder search (parallel) | 60 | 60 | 131.6 | - | 1.00 |

## fixed100_CPS88Wages_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=12, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 37.2s, 2 rounds, 12 temperatures, peak workers=12, 1000 search points; stored ladder has 12 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 85.4 | 0.7 | 1.00 |
| mtmh | - | 1 | 512.3 | 1.3 | 6.00 |
| default_pt (parallel) | 12 | 12 | 247.0 | 5.8 | 2.89 |
| mtmh_pt (parallel) | 12 | 12 | 657.5 | 4.4 | 7.70 |
| ladder search (parallel) | 12 | 12 | 37.2 | - | 0.44 |

## fixed100_BlackFriday_subsample10000 (run 000)

n=10000, train=7425, test=100, temperatures=18, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 74.0s, 4 rounds, 18 temperatures, peak workers=18, 1000 search points; stored ladder has 18 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 97.5 | 1.5 | 1.00 |
| mtmh | - | 1 | 625.2 | 1.5 | 6.41 |
| default_pt (parallel) | 18 | 18 | 256.6 | 3.3 | 2.63 |
| mtmh_pt (parallel) | 18 | 18 | 787.2 | 6.6 | 8.08 |
| ladder search (parallel) | 18 | 18 | 74.0 | - | 0.76 |

## fixed100_QSARFishToxicity (run 000)

n=908, train=606, test=100, temperatures=21, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 101.4s, 6 rounds, 21 temperatures, peak workers=21, 606 search points; stored ladder has 21 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 46.8 | 0.2 | 1.00 |
| mtmh | - | 1 | 174.2 | 0.3 | 3.72 |
| default_pt (parallel) | 21 | 21 | 136.1 | 6.0 | 2.91 |
| mtmh_pt (parallel) | 21 | 21 | 287.9 | 6.3 | 6.15 |
| ladder search (parallel) | 21 | 21 | 101.4 | - | 2.16 |

## fixed100_Cars (run 000)

n=804, train=528, test=100, temperatures=28, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 147.1s, 8 rounds, 28 temperatures, peak workers=28, 528 search points; stored ladder has 28 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 34.8 | 0.5 | 1.00 |
| mtmh | - | 1 | 140.5 | 0.6 | 4.03 |
| default_pt (parallel) | 28 | 28 | 117.1 | 8.8 | 3.36 |
| mtmh_pt (parallel) | 28 | 28 | 268.5 | 3.6 | 7.71 |
| ladder search (parallel) | 28 | 28 | 147.1 | - | 4.23 |

## fixed100_Fiat500 (run 000)

n=1538, train=1078, test=100, temperatures=20, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 89.6s, 5 rounds, 20 temperatures, peak workers=20, 1000 search points; stored ladder has 20 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 48.0 | 0.6 | 1.00 |
| mtmh | - | 1 | 201.9 | 0.2 | 4.20 |
| default_pt (parallel) | 20 | 20 | 144.4 | 2.9 | 3.00 |
| mtmh_pt (parallel) | 20 | 20 | 316.5 | 6.2 | 6.59 |
| ladder search (parallel) | 20 | 20 | 89.6 | - | 1.86 |

## fixed100_Kin8nm (run 000)

n=8192, train=6069, test=100, temperatures=40, allocated CPUs=-, affinity CPUs=64, chains timed so far=4/4

Ladder search: 184.7s, 9 rounds, 40 temperatures, peak workers=40, 1000 search points; stored ladder has 40 temperatures, identical=True.

| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |
|---|---:|---:|---:|---:|---:|
| default | - | 1 | 108.7 | 1.0 | 1.00 |
| mtmh | - | 1 | 654.7 | 3.7 | 6.02 |
| default_pt (parallel) | 40 | 40 | 346.9 | 3.1 | 3.19 |
| mtmh_pt (parallel) | 40 | 40 | 951.4 | 1.2 | 8.75 |
| ladder search (parallel) | 40 | 40 | 184.7 | - | 1.70 |
