#!/usr/bin/env python
"""Pointwise R-hat for the Default sampler at two computational budgets.

The short chains run 10^4 iterations and the long chains 10^7, and both are
stored as roughly 10^4 draws. Modern rank-normalized split R-hat is computed at
each fixed test point using the same number of post-burn draws per chain, so the
two budgets differ only in the span of iterations behind the stored draws.

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
from chains import dataset_tag, load_with_shape, long_predictions, pointwise_rhat_summary

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


def main() -> int:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=script.parent.parent / "store")
    parser.add_argument("--out-dir", type=Path, default=script.parent)
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=3000)
    parser.add_argument("--force", action="store_true",
                        help="Recompute every run instead of reusing the cache.")
    args = parser.parse_args()

    store = args.store_root.resolve()
    cache_path = args.out_dir / "tables" / "budget_comparison_cache.csv"
    fingerprint = f"metric=rank_rhat_v1;short_burn={args.short_burn};long_burn={args.long_burn}"
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
                rows.append({k: hit[k] for k in (
                    "dataset", "run", "draws_per_chain",
                    "short_rhat_median", "short_rhat_q90", "short_rhat_max",
                    "long_rhat_median", "long_rhat_q90", "long_rhat_max",
                )})
                print(f"[budget] {dataset} run {run:03d}  cached", flush=True)
                continue
            short = short_default(store, dataset, run, args.short_burn)
            long = long_predictions(store, dataset, run, args.long_burn)
            count = min(short.shape[1], long.shape[1])
            short_rhat = pointwise_rhat_summary(thin(short, count))
            long_rhat = pointwise_rhat_summary(thin(long, count))
            rows.append({
                "dataset": dataset, "run": run, "draws_per_chain": count,
                "short_rhat_median": short_rhat["median"],
                "short_rhat_q90": short_rhat["q90"],
                "short_rhat_max": short_rhat["maximum"],
                "long_rhat_median": long_rhat["median"],
                "long_rhat_q90": long_rhat["q90"],
                "long_rhat_max": long_rhat["maximum"],
            })
            print(f"[budget] {dataset} run {run:03d}  "
                  f"short median/q90 {short_rhat['median']:.3f}/{short_rhat['q90']:.3f}  "
                  f"long {long_rhat['median']:.3f}/{long_rhat['q90']:.3f}", flush=True)

    runs = pd.DataFrame(rows)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    runs.assign(settings=fingerprint).to_csv(cache_path, index=False)
    runs.to_csv(args.out_dir / "tables" / "budget_comparison_runs.csv", index=False)

    grouped = runs.groupby("dataset").agg(
        runs=("run", "count"), draws=("draws_per_chain", "min"),
        short_median_lo=("short_rhat_median", "min"), short_median_hi=("short_rhat_median", "max"),
        long_median_lo=("long_rhat_median", "min"), long_median_hi=("long_rhat_median", "max"),
        short_q90_lo=("short_rhat_q90", "min"), short_q90_hi=("short_rhat_q90", "max"),
        long_q90_lo=("long_rhat_q90", "min"), long_q90_hi=("long_rhat_q90", "max"),
    ).reset_index().sort_values("long_q90_hi")
    grouped["dataset"] = grouped["dataset"].map(DISPLAY)
    grouped.to_csv(args.out_dir / "tables" / "budget_comparison_datasets.csv", index=False)

    lines = [
        "# Pointwise R-hat at two computational budgets",
        "",
        "Modern rank-normalized split R-hat is calculated at each fixed test point and",
        "summarized by its median and 90th percentile. Both budgets use the same number",
        "of post-burn draws per chain, so they differ only in the iterations those draws",
        "span: 10^4 for the short chains and 10^7 for the long ones.",
        "",
        "| dataset | runs | draws/chain | short median | long median | short q90 | long q90 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in grouped.iterrows():
        lines.append(f"| {row.dataset} | {row.runs} | {row.draws} | "
                     f"{row.short_median_lo:.3f}-{row.short_median_hi:.3f} | "
                     f"{row.long_median_lo:.3f}-{row.long_median_hi:.3f} | "
                     f"{row.short_q90_lo:.3f}-{row.short_q90_hi:.3f} | "
                     f"{row.long_q90_lo:.3f}-{row.long_q90_hi:.3f} |")
    lines.append("")
    (args.out_dir / "budget_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
