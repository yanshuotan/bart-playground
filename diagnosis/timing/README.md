Run-0 timing used by the paper (the default --timing-dir of the section scripts),
next to the benchmark script benchmark_pt_parallelization.py.

All twelve settings are timed with the current code (fused MTMH kernels,
batched PT swap refresh). Each setting ran on an exclusive node with its four
chains timed one after another; PT uses one worker per temperature. Every
searched ladder matches the stored run-0 ladder.

Per setting: `*_per_chain.csv` (one row per chain and method),
`*_summary.csv` (means and ratios to Default) and `*_ladder_search.csv`;
`summary.md` collects all twelve.
