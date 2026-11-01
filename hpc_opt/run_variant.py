"""Short PT/MTMH chains for one speed-up variant, in the production store layout.

Mirrors diagnosis/experiment_fixed100.py (same splits, chain seeds, ladder
search and summaries) and adds three switches plus wall times:
  --fast / --no-fast        fused MTMH scoring (A; bit-identical draws)
  --batched / --no-batched  batched PT swap refresh (E'; same in distribution)
  --ladder baseline         reuse each run's ladder from the baseline store
  --ladder search --target-rate R   rerun the production ladder search with target R

Usage example:
  python hpc_opt/run_variant.py --dataset concrete --variant ae --ladder baseline \
      --batched --fast --runs 0 1 2 3 4 --out /scratch/e1377027/opt/variants
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
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnosis"))

TAGS = {"concrete": "fixed100_Concrete", "seoul_bike": "fixed100_SeoulBike", "abalone": "fixed100_Abalone",
        "airfoil": "fixed100_Airfoil", "calhousing": "fixed100_CalHousing_subsample5000", "ccpp": "fixed100_CCPP"}
BASELINE = Path("/scratch/e1377027/share_pt")
METHOD_ORDER = ("default", "default_pt", "mtmh", "mtmh_pt")


def _read_kv(path):
    return dict(csv.reader(open(path)))


def run_ladder(dataset, run_id, target_rate):
    """The production ladder search for this run, with `target_rate` swapped in."""
    import experiment_fixed100 as exp
    from fixed100_support import load_dataset
    from bart_playground.samplers import default_proposal_probs
    tag = TAGS[dataset]
    ds_meta = _read_kv(BASELINE / tag / "metadata" / f"{tag}__dataset_metadata.csv")
    X, y = load_dataset(dataset)
    split = exp.make_fixed100_splits(X, y, n_runs=run_id + 1)[run_id]
    X_train, y_train = split["X_train"], split["y_train"]
    points = int(ds_meta["ladder_search_points"])
    if points < X_train.shape[0]:
        idx = np.random.default_rng(split["train_seed"]).choice(X_train.shape[0], points, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    t = time.perf_counter()
    temps, rates, history = exp.quick_ladder_search(
        X_train, y_train, n_trees=100, tree_alpha=0.95, tree_beta=2.0,
        proposal_probs=default_proposal_probs, target_rate=target_rate,
        max_rounds=int(ds_meta["ladder_max_rounds"]), max_temperatures=None,
        ndpost=int(ds_meta["ladder_ndpost"]), nskip=int(ds_meta["ladder_nskip"]),
        random_state=123 + 10000 * run_id, swap_interval=int(ds_meta["swap_interval"]),
        post_swap_repair_steps=0, initial_temperatures=json.loads(ds_meta["temperatures"]),
    )
    return {"temperatures": temps, "ladder_swap_rates": rates, "ladder_history": history,
            "ladder_wall_s": time.perf_counter() - t}


def baseline_ladder(dataset, run_id):
    tag = TAGS[dataset]
    meta = _read_kv(BASELINE / tag / "metadata" / f"{tag}__run{run_id:03d}__short_metadata.csv")
    return {"temperatures": json.loads(meta["temperatures"]),
            "ladder_swap_rates": json.loads(meta["ladder_swap_rates"]),
            "ladder_history": json.loads(meta["ladder_history"]), "ladder_wall_s": None}


def run_task(dataset, run_id, chain_id, method, temps, args):
    import experiment_fixed100 as exp
    from fixed100_support import load_dataset
    from bart_playground import DefaultBART, MultiBART, ParallelTemperingBART, moves
    from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs
    moves.FAST_MTM_SCORING = args.fast
    X, y = load_dataset(dataset)
    split = exp.make_fixed100_splits(X, y, n_runs=run_id + 1)[run_id]
    seed = exp.GLOBAL_BASE_CHAIN_SEED + run_id * 1000 + chain_id
    common = dict(ndpost=args.ndpost, nskip=0, n_trees=100, tol=1, random_state=seed,
                  dirichlet_prior=False, s_alpha=1.0)
    pt = dict(tree_alpha=0.95, tree_beta=2.0, n_jobs=args.pt_n_jobs or None, temperatures=temps,
              swap_interval=50, post_swap_repair_steps=0, store_chain_traces=False,
              store_swap_diagnostics=False, print_swap_diagnostics=False,
              batched_swap_refresh=args.batched)
    if method == "default":
        model = DefaultBART(proposal_probs=default_proposal_probs, **common)
    elif method == "default_pt":
        model = ParallelTemperingBART(proposal_probs=default_proposal_probs, **pt, **common)
    elif method == "mtmh_pt":
        model = ParallelTemperingBART(proposal_probs=mtmh_proposal_probs, sampler_kind="multi",
                                      multi_tries=10, **pt, **common)
    elif method == "mtmh":
        model = MultiBART(tree_alpha=0.95, tree_beta=2.0, proposal_probs=mtmh_proposal_probs,
                          multi_tries=10, **common)
    else:
        raise ValueError(method)
    t = time.perf_counter()
    model.fit(split["X_train"], split["y_train"], quietly=True)
    wall = time.perf_counter() - t
    result = exp.summarize_model_outputs(model, split["X_test_fixed"], split["y_test_fixed"], store_preds=True)
    result["wall_s"] = wall
    return {"run_id": run_id, "chain_id": chain_id, "method": method, "result": result,
            "host": os.uname().nodename, "n_temperatures": len(temps) if method.endswith("_pt") else "",
            "flags": {"fast_mtm_scoring": args.fast, "batched_swap_refresh": args.batched,
                      "code_commit": os.environ.get("CODE_COMMIT", "")}}


def _task_path(task_dir, run_id, chain_id, method):
    return task_dir / f"run{run_id:03d}__chain{chain_id}__{method}.pkl"


def run_and_keep(task_dir, dataset, run_id, chain_id, method, temps, args):
    result = run_task(dataset, run_id, chain_id, method, temps, args)
    path = _task_path(task_dir, run_id, chain_id, method)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(result, fh)
    path.with_suffix(".tmp").rename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TAGS))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ladder", choices=("baseline", "search"), default="baseline")
    ap.add_argument("--target-rate", type=float, default=0.4)
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batched", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--methods", nargs="+", default=["default_pt", "mtmh_pt"])
    ap.add_argument("--runs", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--ndpost", type=int, default=10_000)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--pt-n-jobs", type=int, default=1, help="0 gives every temperature its own worker")
    ap.add_argument("--out", default="/scratch/e1377027/opt/variants")
    ap.add_argument("--save-only", action="store_true",
                    help="Skip fitting; rebuild the store from every finished task on disk.")
    args = ap.parse_args()

    import experiment_fixed100 as exp
    from fixed100_support import load_dataset
    tag = TAGS[args.dataset]
    store_root = Path(args.out) / args.variant
    print(f"[VARIANT] {vars(args)} host={os.uname().nodename}", flush=True)

    if args.ladder == "baseline":
        ladders = {r: baseline_ladder(args.dataset, r) for r in args.runs}
    else:
        found = Parallel(n_jobs=min(args.n_jobs, len(args.runs)))(
            delayed(run_ladder)(args.dataset, r, args.target_rate) for r in args.runs)
        ladders = dict(zip(args.runs, found))
    for r in args.runs:
        print(f"[LADDER] run {r}: J={len(ladders[r]['temperatures'])} "
              f"wall={ladders[r]['ladder_wall_s']}", flush=True)

    tasks = [(r, c, m) for r in args.runs for m in args.methods for c in range(args.chains)]
    # Longest tasks first so the pool drains evenly.
    tasks.sort(key=lambda t: (t[2] != "mtmh_pt", -len(ladders[t[0]]["temperatures"])))
    # Each finished task is kept on disk, so a rerun after a crash only redoes the rest.
    task_dir = store_root / tag / "_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    todo = [] if args.save_only else [t for t in tasks if not _task_path(task_dir, *t).exists()]
    print(f"[VARIANT] {len(tasks) - len(todo)} of {len(tasks)} tasks already done", flush=True)
    t0 = time.perf_counter()
    Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_and_keep)(task_dir, args.dataset, r, c, m, ladders[r]["temperatures"], args)
        for r, c, m in todo)
    print(f"[VARIANT] all tasks done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Save every method finished on disk, not only this invocation's: runs of
    # different methods share the per-run metadata and wall-time files.
    results = [pickle.load(open(path, "rb")) for path in sorted(task_dir.glob("run*__chain*__*.pkl"))]
    results = [res for res in results if res["run_id"] in args.runs]
    complete = [m for m in METHOD_ORDER
                if sum(res["method"] == m for res in results) == len(args.runs) * args.chains]
    results = [res for res in results if res["method"] in complete]
    print(f"[VARIANT] saving methods {complete}", flush=True)

    X, y = load_dataset(args.dataset)
    splits = exp.make_fixed100_splits(X, y, n_runs=max(args.runs) + 1)
    for r in args.runs:
        chain_results = []
        for c in range(args.chains):
            out = {"chain_id": c, "chain_seed": exp.GLOBAL_BASE_CHAIN_SEED + r * 1000 + c}
            for res in results:
                if res["run_id"] == r and res["chain_id"] == c:
                    out[res["method"]] = res["result"]
            chain_results.append(out)
        exp.save_short_run(
            store_root=store_root, dataset_tag=tag, run_id=r, split_info=splits[r],
            chain_results=chain_results, methods=complete,
            metadata={"phase": "short_methods", "run_id": r, "variant": args.variant,
                      "ladder": args.ladder, "ladder_target_rate": args.target_rate,
                      "short_ndpost": args.ndpost, "n_chains": args.chains, "methods": complete,
                      "method_flags": {res["method"]: res.get("flags") for res in results},
                      "pt_n_jobs": args.pt_n_jobs, "n_jobs": args.n_jobs,
                      "code_commit": os.environ.get("CODE_COMMIT", ""),
                      **{k: v for k, v in ladders[r].items()}},
        )
        wall_dir = store_root / tag / "wall_times"
        wall_dir.mkdir(parents=True, exist_ok=True)
        with open(wall_dir / f"{tag}__run{r:03d}__wall_times.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "chain_id", "wall_s", "n_temperatures", "host"])
            for res in results:
                if res["run_id"] == r:
                    w.writerow([res["method"], res["chain_id"], res["result"]["wall_s"],
                                res.get("n_temperatures", len(ladders[r]["temperatures"])), res["host"]])
    print("[VARIANT] saved", store_root, flush=True)


if __name__ == "__main__":
    main()
