#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from todo3ab_stochtree_fixed100 import (
    DATASET_CANON,
    load_dataset,
    validate_split_and_data,
    fit_stochtree_bart,
    normalize_prediction_array,
    compute_metrics,
    log,
)


def run_one(args, dataset: str, run_id: int):
    index_store = Path(args.index_store).resolve()
    out_root = Path(args.out_dir).resolve()
    canon = DATASET_CANON[dataset]

    X, y = load_dataset(dataset, index_store, run_id)
    train_idx, test_idx = validate_split_and_data(dataset, run_id, index_store, X, y)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    method_name = f"todo3c_gfr{args.num_gfr}_warmstart_bart"
    out_dir = out_root / canon / f"run{run_id:03d}" / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    params = dict(
        num_gfr=args.num_gfr,
        num_burnin=args.num_burnin,
        num_mcmc=args.num_mcmc,
        num_chains=args.num_chains,
    )

    log(f"[todo3c] {canon} run{run_id:03d}")
    log(f"X_train={X_train.shape}, X_test={X_test.shape}, params={params}")

    t0 = time.time()
    pred = fit_stochtree_bart(
        X_train,
        y_train,
        X_test,
        num_gfr=args.num_gfr,
        num_burnin=args.num_burnin,
        num_mcmc=args.num_mcmc,
        num_chains=args.num_chains,
        num_threads=args.num_threads,
        seed=args.seed + run_id,
    )
    elapsed = time.time() - t0

    draws = normalize_prediction_array(pred, n_test=len(y_test))
    metrics = compute_metrics(draws, y_test)

    metrics.update(params)
    metrics.update({
        "task": "todo3c",
        "dataset": canon,
        "run_id": run_id,
        "method": method_name,
        "elapsed_sec": float(elapsed),
        "split_source": str(index_store),
        "hard_split_match": True,
        "note": "Preliminary external StochTree GFR/XBART warm-start BART, not yet custom bart_playground initialization.",
    })

    np.save(out_dir / "pred_draws.npy", draws)
    np.savetxt(out_dir / "pred_mean.csv", draws.mean(axis=0), delimiter=",")
    np.savetxt(out_dir / "y_test.csv", y_test, delimiter=",")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"done: rmse={metrics['rmse']:.4f}, coverage={metrics['coverage_95']:.3f}, elapsed={elapsed/60:.1f} min")
    return metrics


def rebuild_summary(out_root: Path):
    records = []
    for mp in sorted(out_root.rglob("metrics.json")):
        with open(mp, "r") as f:
            rec = json.load(f)
        rec["_metrics_path"] = str(mp)
        records.append(rec)

    df = pd.DataFrame(records)
    sort_cols = [c for c in ["task", "dataset", "run_id", "method"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    summary_path = out_root / "todo3c_summary.csv"
    df.to_csv(summary_path, index=False)
    log(f"Wrote TODO3c summary: {summary_path} with n_rows={len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["abalone", "concrete", "friedman"], required=True)
    parser.add_argument("--run-ids", nargs="+", type=int, required=True)
    parser.add_argument("--index-store", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9000)

    parser.add_argument("--num-gfr", type=int, default=40)
    parser.add_argument("--num-burnin", type=int, default=0)
    parser.add_argument("--num-mcmc", type=int, default=2500)
    parser.add_argument("--num-chains", type=int, default=4)

    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for run_id in args.run_ids:
            run_one(args, dataset, run_id)

    rebuild_summary(out_root)


if __name__ == "__main__":
    main()
