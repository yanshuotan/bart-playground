#!/usr/bin/env python3
"""Longer Default chains for the equal-wall-time budget axis.

The equal-time comparison is denominated in Default wall time, so Default's own
curve is the one that runs out first: its stored 10,000-iteration run only
reaches a budget of one. This script reruns Default alone at a larger iteration
count for a single run id, so prefixes of the result extend that curve.

Two details make the result a true extension of the stored short run rather
than a different experiment:

* the chain seeds are the short-chain seeds, ``base + 1000*run + chain``, not
  the offset seeds the 10^7 long chains use; and
* every draw is kept, so a prefix of this run is treated exactly like the
  stored short run.

Memory is the reason this does not go through the short-chain path: that path
holds every draw of the fit at once, which is about 0.33 MB per state, so
100,000 draws on four chains would need of the order of 100 GB. The streaming
long-chain runner keeps one chunk in memory, writes its predictions and frees
it, so the footprint is set by ``--chunk-size`` instead of by the run length.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment_fixed100 as exp  # noqa: E402
from experiment_fixed100 import (  # noqa: E402
    _load_2d_or_empty,
    _save_numeric_csv,
    _write_key_value_csv,
    run_long_default_chain_streaming,
)
from fixed100_support import (  # noqa: E402
    DATASET_CONFIGS,
    SPECS,
    generate_sparse_variant,
    load_dataset,
)
from run_fixed100 import repo_nested_friedman1_generator  # noqa: E402

# The DART variants are generated rather than loaded, and always run with the
# Dirichlet prior on splitting proportions.
SPARSE = {name: spec.dataset_tag for name, spec in SPECS.items()}


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", required=True,
                        choices=sorted(set(DATASET_CONFIGS) | set(SPARSE)))
    parser.add_argument("--store-dir", type=Path, required=True,
                        help="Written separately so the 10^7 long chains are untouched.")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--ndpost", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--method-name", default="default_ext")
    parser.add_argument("--base-chain-seed", type=int, default=exp.GLOBAL_BASE_CHAIN_SEED)
    return parser.parse_args(supplied)


def inputs_for(dataset: str) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Inputs, store tag and prior settings for a named or generated dataset."""
    if dataset in SPARSE:
        X, y, _ = generate_sparse_variant(
            dataset, n_samples=2000, seed=42, noise_sd=1.0,
            friedman_generator=repo_nested_friedman1_generator,
        )
        return X, y, SPARSE[dataset], {"dirichlet_prior": True, "s_alpha": 1.0}
    config = DATASET_CONFIGS[dataset]
    X, y = load_dataset(dataset)
    return X, y, config["dataset_tag"], config


def extend_one(args: argparse.Namespace, dataset: str) -> None:
    X, y, tag, config = inputs_for(dataset)
    splits = exp.make_fixed100_splits(X, y, n_runs=args.run_id + 1)
    split = splits[args.run_id]

    root = args.store_dir / tag
    tmp_root = root / "_stream_tmp" / f"run{args.run_id:03d}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[extend] {tag} run {args.run_id:03d}: {args.ndpost:,} iterations, "
          f"{args.n_chains} chains, chunk {args.chunk_size:,}", flush=True)
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_long_default_chain_streaming)(
            chain_id=chain_id,
            # the short-chain seeds, so this run extends the stored short run
            chain_seed=args.base_chain_seed + args.run_id * 1000 + chain_id,
            run_id=args.run_id,
            X_train=split["X_train"],
            y_train=split["y_train"],
            X_test_fixed=split["X_test_fixed"],
            y_test_fixed=split["y_test_fixed"],
            ndpost=args.ndpost,
            nskip=0,
            n_trees=args.n_trees,
            proposal_probs_default=exp.default_proposal_probs,
            store_every=1,
            chunk_size=args.chunk_size,
            tmp_dir=tmp_root,
            store_preds=True,
            dirichlet_prior=bool(config.get("dirichlet_prior", False)),
            s_alpha=float(config.get("s_alpha", 1.0)),
        )
        for chain_id in range(args.n_chains)
    )

    sigmas, rmses, preds = [], [], []
    n_test = split["X_test_fixed"].shape[0]
    for result in results:
        chain_dir = Path(result["tmp_dir"])
        sigmas.append(_load_2d_or_empty(chain_dir / "sigmas_rows.csv", 1).reshape(-1, 1))
        rmses.append(_load_2d_or_empty(chain_dir / "rmses_rows.csv", 1).reshape(-1))
        preds.append(_load_2d_or_empty(chain_dir / "preds_rows.csv", n_test).T)

    name = args.method_name
    stacked = {"sigmas": np.asarray(sigmas), "rmses": np.asarray(rmses),
               "preds": np.asarray(preds)}
    for kind, values in stacked.items():
        _save_numeric_csv(root / kind / f"{tag}__run{args.run_id:03d}__{name}__{kind}.csv",
                          values)
    _save_numeric_csv(root / "indices" / f"{tag}__run{args.run_id:03d}__train_idx.csv",
                      split["train_idx"])
    _save_numeric_csv(root / "indices" / f"{tag}__run{args.run_id:03d}__fixed_test_idx.csv",
                      split["test_idx"])
    _write_key_value_csv(
        root / "metadata" / f"{tag}__run{args.run_id:03d}__{name}_metadata.csv",
        {"phase": name, "run_id": args.run_id, "ndpost": args.ndpost,
         "store_every": 1, "chunk_size": args.chunk_size,
         "n_chains": args.n_chains, "n_trees": args.n_trees,
         "base_chain_seed": args.base_chain_seed,
         "seed_convention": "short-chain seeds, base + 1000*run + chain",
         "train_seed": split["train_seed"],
         "preds_shape": str(stacked["preds"].shape)},
    )
    for result in results:
        shutil.rmtree(result["tmp_dir"], ignore_errors=True)
    print(f"[extend] {tag} run {args.run_id:03d} wrote {stacked['preds'].shape}", flush=True)


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    if args.ndpost % args.chunk_size != 0:
        raise SystemExit("--ndpost must be a multiple of --chunk-size")
    for dataset in args.datasets:
        extend_one(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
