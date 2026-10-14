#!/usr/bin/env python3
"""Default-sampler agreement against training-set size and signal strength.

The two-budget table sorts settings into a group whose disagreement resolves at
a thousandfold budget and a group whose does not, but it does not say what
separates them. The two persistent real datasets are also the two largest,
which is what hitting-time analyses of BART predict, while Abalone settles at
$n=4177$ with a weak signal. This sweep tests that reading directly by holding
everything else fixed and moving the training-set size, and separately the
noise level on a generated design.

The mechanism under test is posterior concentration: as $n$ grows the posterior
narrows, so the within-chain spread falls, and if the chain centroids do not
approach one another at the same rate the agreement diagnostics deteriorate.
The within- and between-chain terms are therefore recorded separately by the
analysis; a ratio alone cannot distinguish a wider disagreement from a narrower
posterior.

Two conventions make the points along a sweep comparable:

* the 100 fixed test inputs are chosen once per source and reused at every
  training-set size, so one panel of inputs is scored throughout; and
* the training subsets are nested, each size being a prefix of one permutation
  of the pool, so increasing $n$ adds rows rather than exchanging them.

Only the Default sampler is run. This is a question about the baseline's
failure mode, not a comparison of samplers.
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
from fixed100_support import DATASET_CONFIGS, _load_generator_dataset, load_dataset  # noqa: E402

SOURCE_TAG = {"ccpp": "CCPP", "calhousing": "CalHousing", "friedman": "Friedman"}
GENERATED = {"friedman"}


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_TAG))
    parser.add_argument("--train-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[1.0],
                        help="Generated sources only; every level runs at every size.")
    parser.add_argument("--n-generated", type=int, default=20_100,
                        help="Rows generated once for a generated source.")
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--ndpost", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--fixed-test-seed", type=int, default=exp.GLOBAL_FIXED_TEST_SEED)
    parser.add_argument("--base-train-seed", type=int, default=exp.GLOBAL_BASE_TRAIN_SEED)
    parser.add_argument("--base-chain-seed", type=int, default=exp.GLOBAL_BASE_CHAIN_SEED)
    return parser.parse_args(supplied)


def load_source(args: argparse.Namespace, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows for one source, at one noise level when the source is generated."""
    if args.source == "friedman":
        config = {**DATASET_CONFIGS["friedman"],
                  "n_samples": args.n_generated, "noise": noise}
        return _load_generator_dataset(config, "friedman1")
    if args.source == "calhousing":
        # the full design rather than the 5,000-row subsample the main
        # experiment uses, so the sweep has room above that size
        from sklearn.datasets import fetch_california_housing

        X, y = fetch_california_housing(return_X_y=True)
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    return load_dataset(args.source)


def nested_splits(X: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    """The fixed test panel, and nested training subsets for every run and size."""
    rows = X.shape[0]
    largest = max(args.train_sizes)
    if largest + args.n_fixed_test_points > rows:
        raise SystemExit(
            f"{args.source} has {rows} rows, too few for {largest} training rows "
            f"plus {args.n_fixed_test_points} test inputs")
    rng = np.random.default_rng(args.fixed_test_seed)
    test_idx = np.sort(rng.choice(rows, size=args.n_fixed_test_points, replace=False))
    mask = np.ones(rows, dtype=bool)
    mask[test_idx] = False
    pool = np.flatnonzero(mask)

    subsets = {}
    for run in args.runs:
        order = np.random.default_rng(args.base_train_seed + run).permutation(pool)
        for size in args.train_sizes:
            subsets[(run, size)] = np.sort(order[:size])
    return test_idx, subsets


def tag_for(args: argparse.Namespace, size: int, noise: float) -> str:
    tag = f"scal_{SOURCE_TAG[args.source]}_n{size:05d}"
    if args.source in GENERATED:
        tag += f"_sd{int(round(100 * noise)):03d}"
    return tag


def fit_one(args: argparse.Namespace, tag: str, run: int, X: np.ndarray, y: np.ndarray,
            train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    root = args.store_dir / tag
    tmp_root = root / "_stream_tmp" / f"run{run:03d}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"[scaling] {tag} run {run:03d}: {len(train_idx)} training rows", flush=True)
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_long_default_chain_streaming)(
            chain_id=chain_id,
            chain_seed=args.base_chain_seed + run * 1000 + chain_id,
            run_id=run,
            X_train=X_train,
            y_train=y_train,
            X_test_fixed=X_test,
            y_test_fixed=y_test,
            ndpost=args.ndpost,
            nskip=0,
            n_trees=args.n_trees,
            proposal_probs_default=exp.default_proposal_probs,
            store_every=1,
            chunk_size=args.chunk_size,
            tmp_dir=tmp_root,
            store_preds=True,
            dirichlet_prior=False,
            s_alpha=1.0,
        )
        for chain_id in range(args.n_chains)
    )

    sigmas, rmses, preds = [], [], []
    for result in results:
        chain_dir = Path(result["tmp_dir"])
        sigmas.append(_load_2d_or_empty(chain_dir / "sigmas_rows.csv", 1).reshape(-1, 1))
        rmses.append(_load_2d_or_empty(chain_dir / "rmses_rows.csv", 1).reshape(-1))
        preds.append(_load_2d_or_empty(chain_dir / "preds_rows.csv", len(test_idx)).T)

    stacked = {"sigmas": np.asarray(sigmas), "rmses": np.asarray(rmses),
               "preds": np.asarray(preds)}
    for kind, values in stacked.items():
        _save_numeric_csv(root / kind / f"{tag}__run{run:03d}__default__{kind}.csv", values)
    _save_numeric_csv(root / "indices" / f"{tag}__run{run:03d}__train_idx.csv", train_idx)
    _save_numeric_csv(root / "indices" / f"{tag}__run{run:03d}__fixed_test_idx.csv", test_idx)
    _write_key_value_csv(
        root / "metadata" / f"{tag}__run{run:03d}__default_metadata.csv",
        {"phase": "scaling", "source": args.source, "run_id": run,
         "n_train": len(train_idx), "n_test": len(test_idx),
         "ndpost": args.ndpost, "chunk_size": args.chunk_size,
         "n_chains": args.n_chains, "n_trees": args.n_trees,
         "base_chain_seed": args.base_chain_seed,
         "subsets": "nested prefixes of one permutation per run",
         "preds_shape": str(stacked["preds"].shape)},
    )
    for result in results:
        shutil.rmtree(result["tmp_dir"], ignore_errors=True)
    print(f"[scaling] {tag} run {run:03d} wrote {stacked['preds'].shape}", flush=True)


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    if args.ndpost % args.chunk_size != 0:
        raise SystemExit("--ndpost must be a multiple of --chunk-size")
    if args.source not in GENERATED and args.noise_levels != [1.0]:
        raise SystemExit("--noise-levels applies to generated sources only")

    for noise in args.noise_levels:
        X, y = load_source(args, noise)
        test_idx, subsets = nested_splits(X, args)
        for size in args.train_sizes:
            tag = tag_for(args, size, noise)
            for run in args.runs:
                fit_one(args, tag, run, X, y, subsets[(run, size)], test_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
