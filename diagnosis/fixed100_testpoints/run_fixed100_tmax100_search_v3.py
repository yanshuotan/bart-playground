from __future__ import annotations

"""
Run fixed100 short-chain experiments with a safer T_max=100 ladder search.

This script DOES NOT modify experiment_fixed100.py or run_fixed100.py.
It imports the original fixed100 pipeline, monkey-patches only quick_ladder_search
at runtime, and then calls run_fixed100_dataset exactly as the original runner does.

Changes relative to the public fixed100 runner:
  - initial ladder defaults to np.geomspace(1, 100, 10)
  - adaptive insertion still uses the harmonic mean / beta-space midpoint
  - optional max ladder size cap defaults to 32; pass 0 to disable
  - memory logger is disabled by default on macOS to avoid BSD ps warnings
  - pilot ladder-search defaults are lighter; override them from CLI
  - all short-chain model fitting/output logic stays in experiment_fixed100.py
"""

import argparse
import gc
import threading
import time
from pathlib import Path

import numpy as np

import experiment_fixed100 as exp
from run_fixed100 import (
    DATASET_CONFIGS,
    GLOBAL_BASE_CHAIN_SEED,
    GLOBAL_BASE_TRAIN_SEED,
    GLOBAL_FIXED_TEST_SEED,
    load_dataset,
)


def make_capped_harmonic_ladder_search(max_temperatures: int | None):
    """Return a quick_ladder_search replacement with harmonic insertion + cap."""

    def quick_ladder_search(
        X,
        y,
        *,
        n_trees,
        tree_alpha,
        tree_beta,
        proposal_probs,
        target_rate=0.4,
        max_rounds=10,
        ndpost=500,
        nskip=500,
        n_repeats=3,
        random_state=123,
        swap_interval=5,
        post_swap_repair_steps=0,
        initial_temperatures=(1.0, 3.0),
        progress_print: bool = False,
        progress_prefix: str = "",
    ):
        temps = sorted({float(t) for t in initial_temperatures})
        if not temps:
            raise ValueError("initial_temperatures cannot be empty")
        if temps[0] != 1.0:
            temps = [1.0] + [t for t in temps if t != 1.0]

        cap_enabled = max_temperatures is not None and int(max_temperatures) > 0
        cap_value = int(max_temperatures) if cap_enabled else None
        if cap_enabled and len(temps) > cap_value:
            raise ValueError(
                f"Initial ladder already has {len(temps)} temperatures, "
                f"which exceeds max_temperatures={cap_value}."
            )

        history = []
        final_mean_rates = np.array([], dtype=float)

        if progress_print:
            print(
                f"{progress_prefix}[LADDER-TMAX100] start: n_points={X.shape[0]}, "
                f"rounds<={max_rounds}, repeats={n_repeats}, target_rate={target_rate}, "
                f"max_temperatures={cap_value if cap_enabled else 'disabled'}, "
                f"init_temps={np.round(temps, 6).tolist()}",
                flush=True,
            )

        for round_id in range(max_rounds):
            round_rates = []
            for rep in range(n_repeats):
                model = exp.ParallelTemperingBART(
                    ndpost=ndpost,
                    nskip=nskip,
                    n_trees=n_trees,
                    tree_alpha=tree_alpha,
                    tree_beta=tree_beta,
                    proposal_probs=proposal_probs,
                    random_state=random_state + 1000 * round_id + rep,
                    temperatures=temps,
                    swap_interval=swap_interval,
                    post_swap_repair_steps=post_swap_repair_steps,
                    store_chain_traces=False,
                    store_swap_diagnostics=False,
                    print_swap_diagnostics=False,
                )
                model.fit(X, y, quietly=True)
                rates = np.asarray(model.get_params().get("swap_accept_rates", []), dtype=float)
                if rates.size == len(temps) - 1:
                    round_rates.append(rates)
                del model
                gc.collect()

            if round_rates:
                mean_rates = np.mean(np.vstack(round_rates), axis=0)
            else:
                mean_rates = np.array([], dtype=float)

            if progress_print:
                rates_preview = np.round(mean_rates, 4).tolist() if mean_rates.size > 0 else []
                print(
                    f"{progress_prefix}[LADDER-TMAX100] round {round_id + 1}/{max_rounds}: "
                    f"n_temps={len(temps)}, mean_rates={rates_preview}",
                    flush=True,
                )

            # Candidate insertions are ordered by bottleneck severity: lowest swap rate first.
            candidate_insertions = []
            if mean_rates.size > 0:
                for i, rate in enumerate(mean_rates):
                    if rate < target_rate:
                        t_low = float(temps[i])
                        t_high = float(temps[i + 1])
                        # Harmonic mean = midpoint in inverse temperature beta = 1 / T.
                        t_new = 2.0 * t_low * t_high / (t_low + t_high)
                        candidate_insertions.append(
                            {
                                "interval_index": int(i),
                                "rate": float(rate),
                                "t_low": t_low,
                                "t_high": t_high,
                                "t_new": float(t_new),
                            }
                        )

            history.append(
                {
                    "round": int(round_id),
                    "temperatures": [float(t) for t in temps],
                    "mean_swap_rates": mean_rates.tolist(),
                    "all_swap_rates": [r.tolist() for r in round_rates],
                    "candidate_insertions": candidate_insertions,
                    "max_temperatures": cap_value if cap_enabled else None,
                }
            )
            final_mean_rates = mean_rates

            if mean_rates.size == 0 or np.all(mean_rates >= target_rate):
                if progress_print:
                    print(
                        f"{progress_prefix}[LADDER-TMAX100] stop: target reached or no valid rates. "
                        f"final_temps={np.round(temps, 6).tolist()}",
                        flush=True,
                    )
                break

            if not candidate_insertions:
                if progress_print:
                    print(f"{progress_prefix}[LADDER-TMAX100] stop: no low-rate intervals.", flush=True)
                break

            candidate_insertions = sorted(candidate_insertions, key=lambda d: d["rate"])
            if cap_enabled:
                remaining_slots = cap_value - len(temps)
                if remaining_slots <= 0:
                    if progress_print:
                        print(
                            f"{progress_prefix}[LADDER-TMAX100] stop: max_temperatures={cap_value} reached "
                            f"before all rates met target.",
                            flush=True,
                        )
                    break
                candidate_insertions = candidate_insertions[:remaining_slots]

            new_temps = set(temps)
            for item in candidate_insertions:
                new_temps.add(float(item["t_new"]))

            updated_temps = sorted(new_temps)
            if len(updated_temps) == len(temps):
                if progress_print:
                    print(f"{progress_prefix}[LADDER-TMAX100] stop: no new temperature inserted.", flush=True)
                break
            temps = updated_temps

        if progress_print:
            print(f"{progress_prefix}[LADDER-TMAX100] done: final n_temps={len(temps)}", flush=True)
        return [float(t) for t in temps], final_mean_rates.tolist(), history

    return quick_ladder_search


def main():
    parser = argparse.ArgumentParser(
        description="Fixed100 benchmark using T_max=100 adaptive harmonic ladder search."
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=["abalone", "concrete", "friedman"])
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--store-dir", type=str, default="store_ladder_tmax100_pilot")
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fixed-test-seed", type=int, default=GLOBAL_FIXED_TEST_SEED)
    parser.add_argument("--base-train-seed", type=int, default=GLOBAL_BASE_TRAIN_SEED)
    parser.add_argument("--base-chain-seed", type=int, default=GLOBAL_BASE_CHAIN_SEED)
    parser.add_argument("--short-ndpost", type=int, default=2000)
    parser.add_argument("--short-nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--long-chunk-size", type=int, default=10000)
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--memory-log-interval", type=int, default=60)
    parser.add_argument("--enable-memory-log", action="store_true", help="Enable original memory logger. Disabled by default to avoid macOS ps warnings.")
    parser.add_argument("--skip-long", action="store_true", help="Run only the four short-chain methods.")
    parser.add_argument("--skip-short", action="store_true", help="Run only default_long.")

    # Ladder-search controls.
    parser.add_argument("--ladder-tmax", type=float, default=100.0)
    parser.add_argument("--ladder-init-size", type=int, default=10)
    parser.add_argument("--ladder-max-temperatures", type=int, default=32, help="Use 0 to disable the cap.")
    parser.add_argument("--ladder-target-rate", type=float, default=0.4)
    parser.add_argument("--ladder-max-rounds", type=int, default=6)
    parser.add_argument("--ladder-ndpost", type=int, default=300)
    parser.add_argument("--ladder-nskip", type=int, default=200)
    parser.add_argument("--ladder-repeats", type=int, default=1)
    parser.add_argument("--ladder-search-points", type=int, default=500)
    args = parser.parse_args()

    if args.skip_long and args.skip_short:
        parser.error("--skip-long and --skip-short cannot be used together.")
    if args.ladder_init_size < 2:
        parser.error("--ladder-init-size must be at least 2.")
    if args.ladder_tmax <= 1.0:
        parser.error("--ladder-tmax must be greater than 1.0.")

    max_temps = None if args.ladder_max_temperatures <= 0 else int(args.ladder_max_temperatures)
    exp.quick_ladder_search = make_capped_harmonic_ladder_search(max_temps)

    initial_temperatures = tuple(np.geomspace(1.0, args.ladder_tmax, args.ladder_init_size).tolist())

    here = Path(__file__).resolve().parent
    store_root = here / args.store_dir
    print(f"Writing outputs to: {store_root}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"n_runs={args.n_runs}, n_chains={args.n_chains}, n_jobs={args.n_jobs}", flush=True)
    print(f"skip_short={args.skip_short}, skip_long={args.skip_long}", flush=True)
    print(
        "Tmax100 ladder controls: "
        f"initial_temperatures={np.round(initial_temperatures, 6).tolist()}, "
        f"target_rate={args.ladder_target_rate}, "
        f"max_temperatures={max_temps if max_temps is not None else 'disabled'}, "
        f"insertion=harmonic_mean",
        flush=True,
    )
    print(
        "Ladder pilot controls: "
        f"search_points={args.ladder_search_points}, "
        f"max_rounds={args.ladder_max_rounds}, "
        f"repeats={args.ladder_repeats}, "
        f"ndpost={args.ladder_ndpost}, "
        f"nskip={args.ladder_nskip}",
        flush=True,
    )

    stop_event = None
    mem_thread = None
    if args.enable_memory_log:
        try:
            from run_fixed100 import memory_logger
            stop_event = threading.Event()
            mem_thread = threading.Thread(
                target=memory_logger,
                args=(store_root / "memory_log.csv", stop_event, args.memory_log_interval),
                daemon=True,
            )
            mem_thread.start()
        except Exception as e:
            print(f"WARNING: could not start memory logger: {e}", flush=True)
            stop_event = None
            mem_thread = None
    else:
        print("Memory logger disabled. This avoids macOS 'ps: illegal argument: python' warnings.", flush=True)

    start = time.time()
    try:
        for name in args.datasets:
            cfg = DATASET_CONFIGS[name]
            X, y = load_dataset(name)
            print(f"\n=== DATASET {name} ===", flush=True)
            print(f"X={X.shape}, y={y.shape}", flush=True)
            print(
                f"long_ndpost={cfg['long_ndpost']}, long_store_every={cfg['long_store_every']}",
                flush=True,
            )
            exp.run_fixed100_dataset(
                X=X,
                y=y,
                dataset_tag=cfg["dataset_tag"],
                store_dir=store_root,
                n_runs=args.n_runs,
                n_chains=args.n_chains,
                n_jobs=args.n_jobs,
                n_trees=args.n_trees,
                short_ndpost=args.short_ndpost,
                short_nskip=args.short_nskip,
                long_ndpost=cfg["long_ndpost"],
                long_store_every=cfg["long_store_every"],
                long_chunk_size=args.long_chunk_size,
                n_fixed_test_points=args.n_fixed_test_points,
                train_fraction=args.train_fraction,
                fixed_test_seed=args.fixed_test_seed,
                base_train_seed=args.base_train_seed,
                base_chain_seed=args.base_chain_seed,
                temperatures=initial_temperatures,
                ladder_target_rate=args.ladder_target_rate,
                ladder_max_rounds=args.ladder_max_rounds,
                ladder_ndpost=args.ladder_ndpost,
                ladder_nskip=args.ladder_nskip,
                ladder_repeats=args.ladder_repeats,
                ladder_search_points=args.ladder_search_points,
                swap_interval=args.swap_interval,
                multi_tries=args.multi_tries,
                store_preds=True,
                progress_print=True,
                run_short=not args.skip_short,
                run_long=not args.skip_long,
                dirichlet_prior=cfg.get("dirichlet_prior", False),
                s_alpha=float(cfg.get("s_alpha", 1.0)),
            )
    finally:
        if stop_event is not None:
            stop_event.set()
        if mem_thread is not None:
            mem_thread.join(timeout=5)

    runtime_min = (time.time() - start) / 60
    print(f"DONE fixed100 Tmax100 ladder run in {runtime_min:.2f} min", flush=True)
    if args.enable_memory_log:
        print(f"Memory log: {store_root / 'memory_log.csv'}", flush=True)


if __name__ == "__main__":
    main()
