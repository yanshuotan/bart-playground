#!/usr/bin/env python3
"""Production swap acceptance of the adjacent temperature pairs of both PT arms.

A pair rate is the proportion of accepted exchanges among the post-warm-up
attempts for one adjacent pair in one chain, as stored in ``swap_accept_rates``
(shape: chain x pair). Per setting and PT arm the summary reports the range of
ladder sizes J over runs, the median of all pair rates pooled over runs and
chains, the share of pair rates below 0.4, and the lowest chain-averaged pair
rate over pairs and runs.

Writes ``swap_rates_pairs.csv`` (one row per run, pair and chain, with the
pair's two temperatures) and ``swap_rates_summary.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path[:0] = [str(HERE), str(DIAG)]
from core import dataset_tag, load_with_shape  # noqa: E402
from section7_costs import DISPLAY  # noqa: E402
from section7_methods import DATASETS  # noqa: E402

ARMS = {"default_pt": "Default+PT", "mtmh_pt": "MTMH+PT"}
THRESHOLD = 0.4


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    return parser.parse_args(supplied)


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for dataset in args.datasets:
        tag = dataset_tag(args.store_root, dataset)
        for run in args.runs:
            for arm, label in ARMS.items():
                base = args.store_root / dataset
                rates = np.atleast_2d(load_with_shape(
                    base / "swap_accept_rates" / f"{tag}__run{run:03d}__{arm}__swap_accept_rates.csv"))
                temps = np.atleast_2d(load_with_shape(
                    base / "swap_temperatures" / f"{tag}__run{run:03d}__{arm}__swap_temperatures.csv"))
                if temps.shape[1] != rates.shape[1] + 1:
                    raise ValueError(f"{dataset} run {run} {arm}: {temps.shape[1]} temperatures "
                                     f"for {rates.shape[1]} pairs")
                for chain in range(rates.shape[0]):
                    for pair in range(rates.shape[1]):
                        pairs.append(dict(dataset=dataset, setting=DISPLAY.get(dataset, dataset), method=label,
                                          run=run, chain=chain, pair=pair + 1, n_temperatures=temps.shape[1],
                                          t_low=temps[chain, pair], t_high=temps[chain, pair + 1],
                                          rate=rates[chain, pair]))
        print(f"[swap] {dataset}", flush=True)
    frame = pd.DataFrame(pairs)
    frame.to_csv(args.table_dir / "swap_rates_pairs.csv", index=False)

    rows = []
    for (dataset, method), part in frame.groupby(["dataset", "method"], sort=False):
        averaged = part.groupby(["run", "pair"]).rate.mean()
        sizes = part.groupby("run").n_temperatures.first()
        rows.append(dict(dataset=dataset, setting=DISPLAY.get(dataset, dataset), method=method,
                         temperatures_min=int(sizes.min()), temperatures_max=int(sizes.max()),
                         median=float(part.rate.median()),
                         share_below=float((part.rate < THRESHOLD).mean()),
                         lowest_chain_averaged=float(averaged.min()),
                         lowest_run=int(averaged.idxmin()[0]), lowest_pair=int(averaged.idxmin()[1])))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.table_dir / "swap_rates_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
