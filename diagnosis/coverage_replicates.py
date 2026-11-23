#!/usr/bin/env python3
"""Frequentist interval coverage of the latent function on Friedman #1.

Each replicate draws a fresh training set of 1,425 rows from the Friedman #1
design of the main experiment (ten U(0,1) inputs, five active, Gaussian noise of
standard deviation 1), with its own data seed. The 100 prediction inputs are the
fixed panel of the main Friedman setting in every replicate, so the true f at
each panel input is the same throughout and pointwise coverage can be averaged
over replicates.

Within a replicate everything follows the production short runs: a ladder search
on 1,000 training rows (ten temperatures from 1 to 100, target swap rate 0.4, at
most ten rounds, at most 60 temperatures, swap interval 50), then four chains,
each running Default, MTMH, Default+PT and MTMH+PT one after another for 10,000
iterations, with the PT replicas updated serially inside the chain.

Results are stored like the main store under ``<store-dir>/cov_Friedman`` with
the replicate number in the run field (``cov_Friedman__run007__...``). The
stored test response is the true f at the panel, so the stored RMSE is the error
against f. The training data of each replicate are stored as well.

    python diagnosis/coverage_replicates.py --replicates 0-9 --n-jobs 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment_fixed100 as exp  # noqa: E402
from experiment_fixed100 import (  # noqa: E402
    METHODS_SHORT,
    _save_numeric_csv,
    _write_key_value_csv,
    quick_ladder_search,
    run_short_chain,
    save_short_run,
)
from fixed100_support import DATASET_CONFIGS, _load_generator_dataset  # noqa: E402
from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs  # noqa: E402

TAG = "cov_Friedman"
DATA_SEED_BASE = 300_000
CHAIN_SEED_BASE = 700_000
LADDER_SEED_BASE = 123


def parse_replicates(text: list[str]) -> list[int]:
    out = []
    for item in text:
        if "-" in item:
            lo, hi = (int(v) for v in item.split("-"))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", nargs="+", required=True, help="e.g. 0-9 or 3 7")
    parser.add_argument("--store-dir", type=Path, default=HERE / "store_coverage")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=1425)
    parser.add_argument("--ndpost", type=int, default=10_000)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--ladder-max-temperatures", type=int, default=60)
    parser.add_argument("--ladder-search-points", type=int, default=1000)
    parser.add_argument("--methods", nargs="+", default=list(METHODS_SHORT), choices=list(METHODS_SHORT))
    parser.add_argument("--dry-run", action="store_true", help="Build the data and print the plan only.")
    return parser.parse_args(supplied)


def fixed_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The main Friedman panel: its indices, inputs and true f."""
    cfg = DATASET_CONFIGS["friedman"]
    X, _ = _load_generator_dataset(cfg, "friedman1")
    _, signal = _load_generator_dataset({**cfg, "noise": 0.0}, "friedman1")
    test_idx = np.sort(np.random.default_rng(exp.GLOBAL_FIXED_TEST_SEED).choice(len(X), 100, replace=False))
    return test_idx, X[test_idx], signal[test_idx]


def replicate_data(rep: int, n_train: int) -> tuple[np.ndarray, np.ndarray, int]:
    seed = DATA_SEED_BASE + rep
    cfg = {**DATASET_CONFIGS["friedman"], "n_samples": n_train, "seed": seed}
    X, y = _load_generator_dataset(cfg, "friedman1")
    return X, y, seed


def finished(args, rep: int) -> bool:
    return all((args.store_dir / TAG / "preds" / f"{TAG}__run{rep:03d}__{m}__preds.csv").is_file()
               for m in args.methods)


def ladder_task(args, rep, X, y, data_seed):
    rng = np.random.default_rng(data_seed)
    pick = rng.choice(len(X), args.ladder_search_points, replace=False)
    temps, rates, history = quick_ladder_search(
        X[pick], y[pick], n_trees=args.n_trees, tree_alpha=0.95, tree_beta=2.0,
        proposal_probs=default_proposal_probs, target_rate=0.4, max_rounds=10,
        max_temperatures=args.ladder_max_temperatures, ndpost=500, nskip=500,
        random_state=LADDER_SEED_BASE + 10000 * rep, swap_interval=args.swap_interval,
        post_swap_repair_steps=0, initial_temperatures=tuple(np.geomspace(1.0, 100.0, 10).tolist()),
        dirichlet_prior=False, s_alpha=1.0, progress_print=False)
    return rep, list(temps)


def chain_task(args, rep, chain_id, X, y, X_test, f_test, temps):
    t0 = time.time()
    out = run_short_chain(
        chain_id=chain_id, chain_seed=CHAIN_SEED_BASE + 1000 * rep + chain_id, methods=args.methods,
        X_train=X, y_train=y, X_test_fixed=X_test, y_test_fixed=f_test,
        ndpost=args.ndpost, nskip=0, n_trees=args.n_trees, tree_alpha=0.95, tree_beta=2.0,
        proposal_probs_default=default_proposal_probs, proposal_probs_mtmh=mtmh_proposal_probs,
        temperatures=temps, swap_interval=args.swap_interval, post_swap_repair_steps=0,
        multi_tries=args.multi_tries, pt_n_jobs=1, dirichlet_prior=False, s_alpha=1.0, store_preds=True)
    print(f"[coverage] replicate {rep:03d} chain {chain_id} done in {time.time() - t0:.0f}s", flush=True)
    return rep, out


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    reps = parse_replicates(args.replicates)
    test_idx, X_test, f_test = fixed_panel()
    todo = [rep for rep in reps if not finished(args, rep)]
    data = {rep: replicate_data(rep, args.n_train) for rep in todo}
    print(f"[coverage] replicates {reps}: {len(todo)} to run, n_train={args.n_train}, "
          f"{args.n_chains} chains, methods={args.methods}, {args.n_jobs} workers", flush=True)
    if args.dry_run:
        for rep in todo:
            X, y, seed = data[rep]
            print(f"  replicate {rep:03d}: data seed {seed}, y mean {y.mean():.3f}, sd {y.std():.3f}", flush=True)
        return 0

    wants_ladder = any(m in ("default_pt", "mtmh_pt") for m in args.methods)
    temps = {rep: [1.0] for rep in todo}
    if wants_ladder:
        # a ladder searched by an earlier, interrupted job is reused as it is
        ladder_path = lambda rep: args.store_dir / TAG / "ladder" / f"{TAG}__run{rep:03d}__ladder_temperatures.csv"
        searched = [rep for rep in todo if not ladder_path(rep).is_file()]
        for rep in todo:
            if rep not in searched:
                temps[rep] = np.loadtxt(ladder_path(rep), delimiter=",", comments="#").reshape(-1).tolist()
                print(f"[coverage] replicate {rep:03d} ladder reused: {len(temps[rep])} temperatures", flush=True)
        if searched:
            for rep, ladder in Parallel(n_jobs=min(args.n_jobs, len(searched)))(
                    delayed(ladder_task)(args, rep, *data[rep]) for rep in searched):
                temps[rep] = ladder
                _save_numeric_csv(ladder_path(rep), np.asarray(ladder))
                print(f"[coverage] replicate {rep:03d} ladder: {len(ladder)} temperatures", flush=True)

    results: dict[int, list] = {rep: [] for rep in todo}
    jobs = [delayed(chain_task)(args, rep, chain, data[rep][0], data[rep][1], X_test, f_test, temps[rep])
            for rep in todo for chain in range(args.n_chains)]
    for rep, out in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
        results[rep].append(out)
        if len(results[rep]) < args.n_chains:
            continue
        X, y, seed = data[rep]
        chains = sorted(results.pop(rep), key=lambda r: r["chain_id"])
        save_short_run(
            store_root=args.store_dir, dataset_tag=TAG, run_id=rep,
            split_info={"X_test_fixed": X_test, "y_test_fixed": f_test,
                        "train_idx": np.arange(len(y)), "test_idx": test_idx},
            chain_results=chains, methods=args.methods,
            metadata={"phase": "coverage", "design": "friedman1", "replicate": rep,
                      "data_seed": seed, "n_train": len(y), "n_test": len(test_idx),
                      "test_response": "true f at the main Friedman panel",
                      "chain_seed": f"{CHAIN_SEED_BASE} + 1000*replicate + chain",
                      "ladder_random_state": LADDER_SEED_BASE + 10000 * rep,
                      "temperatures": [round(t, 6) for t in temps[rep]],
                      "ndpost": args.ndpost, "swap_interval": args.swap_interval,
                      "multi_tries": args.multi_tries, "pt_n_jobs": 1})
        root = args.store_dir / TAG
        _save_numeric_csv(root / "train_X" / f"{TAG}__run{rep:03d}__train_X.csv", X)
        _save_numeric_csv(root / "train_y" / f"{TAG}__run{rep:03d}__train_y.csv", y)
        print(f"[coverage] replicate {rep:03d} saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
