"""Decompose PT / MTMH wall time on one node, without modifying the library.

Usage: python hpc_opt/prof_pt.py DATASET N_ITER OUT_DIR
  DATASET in {concrete, seoul_bike}; run-0 split, the store's run-0 PT ladder.

Measures, in this order:
  1. contention: one Default chain alone, then J Default chains concurrently;
  2. parallel PT (one worker per temperature), timing block advances vs swap steps
     by wrapping the two driver methods from the outside;
  3. cProfile of serial PT and of MTMH (shorter runs), top functions by cumtime.
"""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import cProfile
import csv
import io
import json
import pstats
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnosis"))

TAGS = {"concrete": "fixed100_Concrete", "seoul_bike": "fixed100_SeoulBike"}
TEAM = Path("/scratch/e0950116/bart_store")


def split_and_ladder(dataset):
    from fixed100_support import load_dataset
    from experiment_fixed100 import make_fixed100_splits
    X, y = load_dataset(dataset)
    split = make_fixed100_splits(X, y, n_runs=1)[0]
    tag = TAGS[dataset]
    meta = dict(csv.reader(open(TEAM / tag / "metadata" / f"{tag}__run000__short_metadata.csv")))
    return split, [float(t) for t in json.loads(meta["temperatures"])]


def default_model(n_iter, seed):
    from bart_playground import DefaultBART
    from bart_playground.samplers import default_proposal_probs
    return DefaultBART(ndpost=n_iter, nskip=0, n_trees=100, tol=1, random_state=seed,
                       proposal_probs=default_proposal_probs)


def time_default(args):
    dataset, n_iter, seed = args
    split, _ = split_and_ladder(dataset)
    model = default_model(n_iter, seed)
    t = time.perf_counter()
    model.fit(split["X_train"], split["y_train"], quietly=True)
    return time.perf_counter() - t


def pt_model(temps, n_iter, seed, n_jobs, sampler_kind="default"):
    from bart_playground import ParallelTemperingBART
    from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs
    extra = {}
    if sampler_kind == "multi":
        extra = dict(sampler_kind="multi", multi_tries=10, proposal_probs=mtmh_proposal_probs)
    else:
        extra = dict(proposal_probs=default_proposal_probs)
    return ParallelTemperingBART(
        ndpost=n_iter, nskip=0, n_trees=100, tol=1, random_state=seed, tree_alpha=0.95,
        tree_beta=2.0, n_jobs=n_jobs, temperatures=temps, swap_interval=50,
        post_swap_repair_steps=0, store_chain_traces=False, store_swap_diagnostics=False,
        print_swap_diagnostics=False, **extra)


def profile_call(fn, top=25):
    prof = cProfile.Profile()
    t = time.perf_counter()
    prof.enable()
    fn()
    prof.disable()
    wall = time.perf_counter() - t
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(top)
    return wall, buf.getvalue()


def main():
    dataset, n_iter, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    split, temps = split_and_ladder(dataset)
    J = len(temps)
    report = {"dataset": dataset, "n_train": int(len(split["y_train"])), "J": J, "n_iter": n_iter}

    # 1. contention
    report["default_alone_s"] = time_default((dataset, n_iter, 2024))
    with ProcessPoolExecutor(max_workers=J) as pool:
        times = list(pool.map(time_default, [(dataset, n_iter, 2024 + c) for c in range(J)]))
    report["default_concurrent_J_s"] = {"mean": float(np.mean(times)), "max": float(np.max(times))}
    print(json.dumps(report), flush=True)

    # 2. parallel PT, driver-side decomposition
    from bart_playground import bart as bart_module
    cls = bart_module.ParallelTemperingBART
    acc = {"advance": 0.0, "swap": 0.0, "n_swap": 0}
    orig_adv, orig_swap = cls._advance_all_chains_workers, cls._swap_step_with_workers

    def adv(self, *a, **k):
        t = time.perf_counter(); r = orig_adv(self, *a, **k); acc["advance"] += time.perf_counter() - t; return r

    def swp(self, *a, **k):
        t = time.perf_counter(); r = orig_swap(self, *a, **k)
        acc["swap"] += time.perf_counter() - t; acc["n_swap"] += 1; return r

    cls._advance_all_chains_workers, cls._swap_step_with_workers = adv, swp
    for kind in ("default", "multi"):
        acc.update(advance=0.0, swap=0.0, n_swap=0)
        model = pt_model(temps, n_iter, 2024, n_jobs=J, sampler_kind=kind)
        t = time.perf_counter()
        model.fit(split["X_train"], split["y_train"], quietly=True)
        total = time.perf_counter() - t
        report[f"pt_parallel_{kind}"] = {"total_s": total, "advance_s": acc["advance"],
                                         "swap_s": acc["swap"], "swap_steps": acc["n_swap"],
                                         "other_s": total - acc["advance"] - acc["swap"]}
        print(json.dumps(report), flush=True)
    cls._advance_all_chains_workers, cls._swap_step_with_workers = orig_adv, orig_swap

    # 3. cProfile: serial PT (short) and MTMH single chain (short)
    short = max(100, n_iter // 10)
    serial = pt_model(temps, short, 2024, n_jobs=1)
    wall, text = profile_call(lambda: serial.fit(split["X_train"], split["y_train"], quietly=True))
    report["pt_serial_profile_wall_s"] = wall
    (out / f"{dataset}_pt_serial_profile.txt").write_text(text)

    from bart_playground import MultiBART
    from bart_playground.samplers import mtmh_proposal_probs
    mtmh = MultiBART(ndpost=short, nskip=0, n_trees=100, tol=1, random_state=2024, tree_alpha=0.95,
                     tree_beta=2.0, proposal_probs=mtmh_proposal_probs, multi_tries=10)
    wall, text = profile_call(lambda: mtmh.fit(split["X_train"], split["y_train"], quietly=True))
    report["mtmh_profile_wall_s"] = wall
    (out / f"{dataset}_mtmh_profile.txt").write_text(text)
    dflt = default_model(short, 2024)
    wall, text = profile_call(lambda: dflt.fit(split["X_train"], split["y_train"], quietly=True))
    report["default_profile_wall_s"] = wall
    (out / f"{dataset}_default_profile.txt").write_text(text)

    (out / f"{dataset}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
