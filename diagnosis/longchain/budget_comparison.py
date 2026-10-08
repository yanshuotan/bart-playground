#!/usr/bin/env python
"""Chain separation for the Default sampler at two computational budgets.

The short chains run 10^4 iterations and the long chains 10^7, and both are
stored as roughly 10^4 draws. For each the separation index of
`long_chain_mixing.separation_index` is computed on the same number of draws per
chain, so the two columns differ only in the span of iterations behind them.

    .venv/Scripts/python diagnosis/longchain/budget_comparison.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chains import dataset_tag, load_with_shape, long_predictions, separation_index

DISPLAY = {
    "fixed100_Abalone": "Abalone",
    "fixed100_Airfoil": "Airfoil",
    "fixed100_CPUAct": "CPUAct",
    "fixed100_CCPP": "CCPP",
    "fixed100_CalHousing_subsample5000": "CalHousing",
    "fixed100_Concrete": "Concrete",
    "fixed100_Friedman": "Friedman",
    "fixed100_FriedmanSparseDir_p20": "Friedman-S p20",
    "fixed100_FriedmanSparseDir_p100": "Friedman-S p100",
    "fixed100_FriedmanSparseDir_p200": "Friedman-S p200",
    "fixed100_SeoulBike": "SeoulBike",
}


def short_default(store: Path, dataset: str, run: int, burn: int) -> np.ndarray:
    tag = dataset_tag(store, dataset)
    path = store / dataset / "preds" / f"{tag}__run{run:03d}__default__preds.csv"
    return load_with_shape(path).transpose(0, 2, 1)[:, burn:, :]


def thin(draws: np.ndarray, count: int) -> np.ndarray:
    """Evenly spaced draws spanning the whole chain, so the span is preserved."""
    if draws.shape[1] <= count:
        return draws
    return draws[:, np.linspace(0, draws.shape[1] - 1, count, dtype=int), :]


def verify_null(blocks: int, n_draws: int = 7000, n_points: int = 100) -> int:
    """Four independent AR(1) chains from one distribution: the index should give 1."""
    rng = np.random.default_rng(0)
    for rho in (0.0, 0.9, 0.99):
        values = []
        for _ in range(20):
            noise = rng.normal(scale=np.sqrt(1 - rho**2), size=(4, n_draws, n_points))
            chains = np.empty_like(noise)
            chains[:, 0] = rng.normal(size=(4, n_points))
            for t in range(1, n_draws):
                chains[:, t] = rho * chains[:, t - 1] + noise[:, t]
            values.append(separation_index(chains, blocks)[0])
        print(f"rho={rho:4.2f}  index {np.mean(values):.3f} (sd {np.std(values):.3f})")
    return 0


def main() -> int:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=script.parent.parent / "store")
    parser.add_argument("--out-dir", type=Path, default=script.parent)
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=3000)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Recompute every run instead of reusing the cache.")
    parser.add_argument("--verify-null", action="store_true",
                        help="Simulate exchangeable AR(1) chains and print the index's null.")
    args = parser.parse_args()

    if args.verify_null:
        return verify_null(args.blocks)

    store = args.store_root.resolve()
    cache_path = args.out_dir / "tables" / "budget_comparison_cache.csv"
    fingerprint = f"short_burn={args.short_burn};long_burn={args.long_burn};blocks={args.blocks}"
    cached = {}
    if cache_path.is_file() and not args.force:
        frame = pd.read_csv(cache_path)
        frame = frame[frame["settings"] == fingerprint]  # stale settings drop out
        cached = {(r.dataset, int(r.run)): r._asdict() for r in frame.itertuples(index=False)}
    rows = []
    for directory in sorted(store.glob("fixed100_*")):
        dataset = directory.name
        if not sorted((directory / "preds").glob("*__default_long__preds.csv")):
            continue
        tag = dataset_tag(store, dataset)
        for path in sorted((directory / "preds").glob(f"{tag}__run*__default__preds.csv")):
            run = int(re.search(r"__run(\d+)__", path.name).group(1))
            if (dataset, run) in cached:
                hit = cached[(dataset, run)]
                rows.append({k: hit[k] for k in
                             ("dataset", "run", "draws_per_chain", "short_index", "long_index")})
                print(f"[budget] {dataset} run {run:03d}  cached", flush=True)
                continue
            short = short_default(store, dataset, run, args.short_burn)
            long = long_predictions(store, dataset, run, args.long_burn)
            count = min(short.shape[1], long.shape[1])
            short_index, _ = separation_index(thin(short, count), args.blocks)
            long_index, _ = separation_index(thin(long, count), args.blocks)
            rows.append({
                "dataset": dataset, "run": run, "draws_per_chain": count,
                "short_index": short_index, "long_index": long_index,
            })
            print(f"[budget] {dataset} run {run:03d}  short {short_index:6.2f}  "
                  f"long {long_index:6.2f}", flush=True)

    runs = pd.DataFrame(rows)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    runs.assign(settings=fingerprint).to_csv(cache_path, index=False)
    runs.to_csv(args.out_dir / "tables" / "budget_comparison_runs.csv", index=False)

    grouped = runs.groupby("dataset").agg(
        runs=("run", "count"), draws=("draws_per_chain", "min"),
        short_lo=("short_index", "min"), short_hi=("short_index", "max"),
        long_lo=("long_index", "min"), long_hi=("long_index", "max"),
    ).reset_index().sort_values("long_lo")
    grouped["dataset"] = grouped["dataset"].map(DISPLAY)
    grouped.to_csv(args.out_dir / "tables" / "budget_comparison_datasets.csv", index=False)

    lines = [
        "# Chain separation at two computational budgets",
        "",
        "The separation index divides the across-chain between/within ratio by the same",
        f"statistic on {args.blocks} consecutive blocks of one chain and multiplies by",
        f"{args.blocks}. Perfect mixing gives 1 at any autocorrelation. Both columns use the",
        "same number of draws per chain, so they differ only in the iterations those draws span:",
        "10^4 for the short chains and 10^7 for the long ones.",
        "",
        "| dataset | runs | draws/chain | short index | long index |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in grouped.iterrows():
        lines.append(f"| {row.dataset} | {row.runs} | {row.draws} | "
                     f"{row.short_lo:.2f}-{row.short_hi:.2f} | {row.long_lo:.2f}-{row.long_hi:.2f} |")
    lines.append("")
    (args.out_dir / "budget_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
