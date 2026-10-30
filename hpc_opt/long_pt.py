"""One long PT chain, run in chunks that continue from each other, with checkpoints.

Usage: python hpc_opt/long_pt.py --dataset seoul_bike --run 0 --chain 0 \
           --method mtmh_pt --iterations 1000000 --out /scratch/e1377027/long_pt

Each chunk is one ParallelTemperingBART fit started from the previous chunk's
final states (as the ladder search continues its rounds); chunk k is seeded with
SeedSequence([chain_seed, k]). After every chunk the final states and then the
stored draws are written, so a rerun resumes after the last finished chunk.

Stored per chunk, for every `thin`-th cold-chain draw: predictions at the fixed
test panel (response scale), eps_sigma2, the per-feature split counts, and the
collapsed log posterior of the draw split into its three terms (leaf-integrated
log likelihood at temperature 1 including its normalizing constant, log tree prior, log prior of eps_sigma2). The
last chunk also stores every draw's predictions, sigma and split counts (the
contiguous tail). Swap attempt/accept counts are accumulated per pair.

The temperature ladder is the run's ladder from the short-chain rerun store.
"""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnosis"))

TAGS = {"concrete": "fixed100_Concrete", "seoul_bike": "fixed100_SeoulBike", "abalone": "fixed100_Abalone",
        "airfoil": "fixed100_Airfoil", "calhousing": "fixed100_CalHousing_subsample5000", "ccpp": "fixed100_CCPP"}
SHORT_STORE = Path("/scratch/e1377027/rerun_ae")
LONG_SEED_OFFSET = 100_000  # run_fixed100.py: long chains add 100000 to the chain seed


def run_ladder(dataset, run_id):
    tag = TAGS[dataset]
    meta = dict(csv.reader(open(SHORT_STORE / tag / "metadata" / f"{tag}__run{run_id:03d}__short_metadata.csv")))
    return [float(t) for t in json.loads(meta["temperatures"])]


def _with_cache(tree, X):
    """Rebuild leaf ids, node counts and fitted values of a cache-free tree.

    (Tree.update_data starts from a fixed-size count array, too short for deep trees.)
    """
    tree.dataX = X
    tree.leaf_ids = tree.traverse_tree(X).astype(np.int32)
    leaf_counts = np.bincount(tree.leaf_ids, minlength=len(tree.vars))
    counts = np.zeros(len(tree.vars), dtype=np.int32)
    for leaf in np.flatnonzero(leaf_counts):
        node = int(leaf)
        while True:
            counts[node] += leaf_counts[leaf]
            if node == 0:
                break
            node = (node - 1) // 2
    tree.n = counts
    tree.update_outputs()
    return tree


def log_posterior_terms(model, state):
    """Collapsed log posterior of a stored (cache-free) state, split into its terms."""
    from scipy.stats import invgamma
    from bart_playground.params import Parameters
    sampler, data = model.sampler, model.data
    trees = [_with_cache(tree.copy(copy_cache=False), data.X) for tree in state.trees]
    params = Parameters(trees, state.global_params.copy())
    loglik = sampler.likelihood.trees_log_marginal_lkhd(params, data.y, np.arange(params.n_trees), temp=1.0)
    sigma2 = float(state.global_params["eps_sigma2"][0])
    # trees_log_marginal_lkhd leaves out the Gaussian normalizing term; PT adds it the same way.
    loglik = float(np.asarray(loglik).reshape(-1)[0]) - 0.5 * len(data.y) * np.log(2.0 * np.pi * sigma2)
    tree_prior = float(sampler.tree_prior.trees_log_prior(params, range(params.n_trees)))
    prior = sampler.global_prior
    sigma_prior = float(invgamma.logpdf(sigma2, a=prior.eps_nu / 2, scale=prior.eps_nu * prior.eps_lambda / 2))
    return loglik, tree_prior, sigma_prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TAGS))
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--chain", type=int, required=True)
    ap.add_argument("--method", choices=("mtmh_pt", "default_pt"), default="mtmh_pt")
    ap.add_argument("--iterations", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--thin", type=int, default=100)
    ap.add_argument("--pt-n-jobs", type=int, default=0, help="0 gives every temperature its own worker")
    ap.add_argument("--temperatures", type=float, nargs="+", help="override the stored ladder (smoke tests)")
    ap.add_argument("--out", default="/scratch/e1377027/long_pt")
    args = ap.parse_args()
    if args.iterations % args.chunk or args.chunk % args.thin or args.chunk % 50:
        raise ValueError("iterations must be a multiple of chunk, and chunk of thin and of the swap interval")

    import experiment_fixed100 as exp
    from fixed100_support import load_dataset
    from bart_playground import ParallelTemperingBART
    from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs

    tag = TAGS[args.dataset]
    out = Path(args.out) / tag / f"run{args.run:03d}" / args.method / f"chain{args.chain}"
    out.mkdir(parents=True, exist_ok=True)
    temps = args.temperatures or run_ladder(args.dataset, args.run)
    seed = exp.GLOBAL_BASE_CHAIN_SEED + LONG_SEED_OFFSET + args.run * 1000 + args.chain
    X, y = load_dataset(args.dataset)
    split = exp.make_fixed100_splits(X, y, n_runs=args.run + 1)[args.run]
    n_chunks = args.iterations // args.chunk

    config = {k: v for k, v in vars(args).items()}
    config.update(temperatures=temps, chain_seed=seed, n_chunks=n_chunks,
                  code_commit=os.environ.get("CODE_COMMIT", ""))
    config_path = out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ("dataset", "run", "chain", "method", "iterations", "chunk", "thin", "temperatures", "chain_seed"):
            if old[key] != config[key]:
                raise ValueError(f"{out} was started with {key}={old[key]}, not {config[key]}")
    else:
        config_path.write_text(json.dumps(config, indent=2))

    done = sorted(int(p.stem[5:]) for p in out.glob("chunk[0-9][0-9][0-9][0-9].npz"))
    start = done[-1] + 1 if done else 0
    if done and done != list(range(start)):
        raise ValueError(f"missing chunks in {out}: {done}")
    states = None
    if start:
        # Written before its chunk file, so it exists for every finished chunk.
        with open(out / f"states{start - 1:04d}.pkl", "rb") as fh:
            states = pickle.load(fh)
    print(f"[LONG] {tag} run {args.run} chain {args.chain} {args.method}: J={len(temps)}, "
          f"chunks {start}..{n_chunks - 1} of {n_chunks}, seed {seed}", flush=True)

    method_kwargs = (dict(proposal_probs=mtmh_proposal_probs, sampler_kind="multi", multi_tries=10)
                     if args.method == "mtmh_pt" else dict(proposal_probs=default_proposal_probs))
    for k in range(start, n_chunks):
        t0 = time.perf_counter()
        model = ParallelTemperingBART(
            ndpost=args.chunk, nskip=0, n_trees=100, tol=1, tree_alpha=0.95, tree_beta=2.0,
            random_state=np.random.SeedSequence([seed, k]), temperatures=temps, swap_interval=50,
            post_swap_repair_steps=0, n_jobs=args.pt_n_jobs or None, store_chain_traces=False,
            store_swap_diagnostics=False, print_swap_diagnostics=False, dirichlet_prior=False,
            **method_kwargs)
        model.fit(split["X_train"], split["y_train"], quietly=True, init_states=states, keep_final_states=True)
        fit_s = time.perf_counter() - t0

        kept = list(range(args.thin - 1, args.chunk, args.thin))
        trace = [model.trace[i] for i in model.range_post]
        X_test = split["X_test_fixed"]
        record = {
            "iteration": np.array([k * args.chunk + i + 1 for i in kept]),
            "preds": np.array([model.predict_trace(model.range_post[i], X_test) for i in kept]),
            "sigmas": np.array([float(trace[i].global_params["eps_sigma2"][0]) for i in kept]),
            "vars_histogram": np.array([trace[i].vars_histogram for i in kept]),
            "log_post_terms": np.array([log_posterior_terms(model, trace[i]) for i in kept]),
            "swap_attempts": model.swap_attempt_counts.copy(),
            "swap_accepts": model.swap_accept_counts.copy(),
        }
        if k == n_chunks - 1:
            record["tail_preds"] = np.asarray(model.posterior_f(X_test, backtransform=True)).T
            record["tail_sigmas"] = np.array([float(s.global_params["eps_sigma2"][0]) for s in trace])
            record["tail_vars_histogram"] = np.array([s.vars_histogram for s in trace])
        states = model.final_states
        # States first, then the chunk file that marks the chunk as finished, so a
        # job killed between the two writes resumes from a consistent point.
        with open(out / "states.tmp", "wb") as fh:
            pickle.dump(states, fh)
        os.replace(out / "states.tmp", out / f"states{k:04d}.pkl")
        np.savez_compressed(out / f"chunk{k:04d}.tmp.npz", **record)
        os.replace(out / f"chunk{k:04d}.tmp.npz", out / f"chunk{k:04d}.npz")
        old_states = out / f"states{k - 2:04d}.pkl"
        if old_states.exists():
            old_states.unlink()
        rates = record["swap_accepts"] / np.maximum(1, record["swap_attempts"])
        print(f"[LONG] chunk {k + 1}/{n_chunks}: fit {fit_s:.0f}s, total {time.perf_counter() - t0:.0f}s, "
              f"mean swap rate {rates.mean():.3f}, log post {record['log_post_terms'].sum(axis=1).mean():.1f}",
              flush=True)
        del model, trace, record
    print("[LONG] done", flush=True)


if __name__ == "__main__":
    main()
