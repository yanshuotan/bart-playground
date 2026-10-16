"""Abalone run-0: independent SMC starts followed by chronological BART chains.

Run from the repository root with .venv/Scripts/python.exe diagnosis/run_smc.py.
Four processes each use one computational thread. Results use the fixed-100
CSV convention, with raw streaming arrays and separate initialization timing.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAG = "fixed100_Abalone"
METHOD = "smc_init"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers may briefly hold the destination without delete sharing.
    # Progress observation must not abort a statistical computation.
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)


def save_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    np.savetxt(path, values.reshape(values.shape[0], -1), delimiter=",", fmt="%.10g",
               header=f"original_shape={values.shape}")


def run_chain(chain_id, X, y, X_test, y_test, output, settings):
    from bart_playground import SMCBART

    folder = Path(output) / "chains" / f"chain{chain_id:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    seed = 2024 + chain_id
    production_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    started, cpu_started = time.perf_counter(), time.process_time()
    progress_file = folder / "progress.json"
    write_json(progress_file, {"phase": "initializing", "chain": chain_id, "seed": seed})
    with threadpool_limits(limits=1):
        smc = SMCBART(
            n_particles=settings["n_particles"], ndpost=settings["n_particles"],
            n_trees=100, n_mcmc_steps=settings["n_mcmc_steps"],
            ess_target=0.8, resample_threshold=0.5, random_state=seed,
        )
        choose_beta = smc._next_beta

        def reporting_beta(beta, log_weights, log_likelihoods):
            next_beta = choose_beta(beta, log_weights, log_likelihoods)
            stage = len(smc.stage_diagnostics_) + 1
            if stage == 1 or stage % 5 == 0 or next_beta == 1:
                write_json(progress_file, {"phase": "annealing", "chain": chain_id,
                    "stage": stage, "next_beta": next_beta,
                    "elapsed_seconds": time.perf_counter() - started})
            return next_beta

        smc._next_beta = reporting_beta
        smc.fit(X, y, quietly=True)
        smc_seconds = time.perf_counter() - started
        lineages = np.arange(settings["n_particles"])
        diagnostics = []
        for stage in smc.stage_diagnostics_:
            lineages = lineages[stage["ancestors"]]
            diagnostics.append({**{k: v for k, v in stage.items() if k != "ancestors"},
                "ancestors": stage["ancestors"].tolist(),
                "unique_initial_ancestors": int(len(np.unique(lineages)))})
        write_json(folder / "annealing.json", diagnostics)
        np.savez(folder / "population.npz", weights=smc.weights_,
                 preds=np.array([smc.preprocessor.backtransform_y(p.evaluate(X_test))
                                 for p in smc.particles_]),
                 sigmas=np.array([p.global_params["eps_sigma2"][0] for p in smc.particles_]))
        model = smc.initialize_posterior_chain(production_seed)
        init_info = {
            "chain": chain_id, "smc_seed": seed, "production_seed": production_seed,
            "smc_seconds": smc_seconds, "smc_stages": len(diagnostics),
            "smc_sweeps": len(diagnostics) * settings["n_particles"] * settings["n_mcmc_steps"],
            "smc_resamples": sum(int(d["resampled"]) for d in diagnostics),
            "smc_weight_ess": float(1 / np.sum(smc.weights_**2)),
            "smc_unique_initial_ancestors": int(len(np.unique(lineages))),
            "selected_particle": model.smc_particle_index_,
            "log_evidence_transformed_y": smc.log_evidence_,
        }
        # Break the reporting closure cycle and release the complete population.
        smc._next_beta = choose_beta
        del reporting_beta, choose_beta, smc
        import gc
        gc.collect()
        count = settings["ndpost"]
        arrays = {
            "preds": np.lib.format.open_memmap(folder / "preds.npy", mode="w+", dtype=float,
                                              shape=(count, len(X_test))),
            "sigmas": np.lib.format.open_memmap(folder / "sigmas.npy", mode="w+", dtype=float, shape=(count,)),
            "rmses": np.lib.format.open_memmap(folder / "rmses.npy", mode="w+", dtype=float, shape=(count,)),
            "vars_histogram": np.lib.format.open_memmap(folder / "vars_histogram.npy", mode="w+", dtype=np.int32,
                                                       shape=(count, X.shape[1])),
        }
        state = model.trace[0]
        production_started = time.perf_counter()
        sampling_seconds = 0.0
        for draw in range(count):
            tick = time.perf_counter()
            state = model.sampler.one_iter(state, temp=1.0)
            sampling_seconds += time.perf_counter() - tick
            preds = model.preprocessor.backtransform_y(state.evaluate(X_test))
            arrays["preds"][draw] = preds
            arrays["sigmas"][draw] = state.global_params["eps_sigma2"][0]
            arrays["rmses"][draw] = np.sqrt(np.mean((preds - y_test)**2))
            arrays["vars_histogram"][draw] = state.vars_histogram
            if (draw + 1) % 500 == 0 or draw + 1 == count:
                for values in arrays.values():
                    values.flush()
                model.sampler.accepted_moves_logmh.clear()
                write_json(progress_file, {"phase": "production", "chain": chain_id,
                    "draws_completed": draw + 1, "draws_requested": count,
                    "elapsed_seconds": time.perf_counter() - started})
        production_seconds = time.perf_counter() - production_started
        with (folder / "final_state.pkl").open("wb") as stream:
            pickle.dump({"state": state, "rng_state": model.sampler.generator.bit_generator.state}, stream)
        result = {**init_info, "production_seconds": production_seconds,
                  "production_sampling_seconds": sampling_seconds,
                  "total_seconds": time.perf_counter() - started,
                  "cpu_seconds": time.process_time() - cpu_started,
                  "draws": count, "production_kernel": "DefaultSampler, tol=1, temperature=1",
                  "move_selected": model.sampler.move_selected_counts,
                  "move_accepted": model.sampler.move_accepted_counts}
        write_json(folder / "timing.json", result)
        write_json(progress_file, {"phase": "complete", **result})
        return result


def aggregate(output, split, settings, timings, elapsed):
    base = Path(output)
    for name in ("preds", "sigmas", "rmses", "vars_histogram"):
        arrays = [np.load(base / "chains" / f"chain{chain:03d}" / f"{name}.npy") for chain in range(4)]
        values = np.stack(arrays)
        if name == "preds":
            values = values.transpose(0, 2, 1)
        elif name == "sigmas":
            values = values[:, :, None]
        save_csv(base / name / f"{TAG}__run000__{METHOD}__{name}.csv", values)
    for name in ("train_idx", "fixed_test_idx"):
        save_csv(base / "indices" / f"{TAG}__run000__{name}.csv",
                 split["train_idx" if name == "train_idx" else "test_idx"])
    for name, key in (("subsample_X_test", "X_test_fixed"), ("subsample_y_test", "y_test_fixed")):
        save_csv(base / name / f"{TAG}__run000__{name}.csv", split[key])
    write_json(base / "metadata.json", {"method": "SMC-initialized Default MCMC",
        "settings": settings, "chains": sorted(timings, key=lambda x: x["chain"]),
        "four_process_wall_seconds": elapsed, "n_train": len(split["y_train"]),
        "n_features": split["X_train"].shape[1], "test_points": len(split["y_test_fixed"]),
        "seeds": [2024, 2025, 2026, 2027], "production_draws_are_chronological": True,
        "initial_selected_state_is_not_counted": True,
        "recovery": (json.loads((base / "recovery.json").read_text())
                     if (base / "recovery.json").exists() else []),
        "batch_wall_includes_recovery": (base / "recovery.json").exists(),
        "cost_note": "SMC initialization is charged to total cost, not hidden as free warm-up."})


def analyze(output, source, burn):
    sys.path.insert(0, str(ROOT / "diagnosis" / "paper"))
    from core import load_with_shape, pointwise_rhat, summarize_rhat, worst_direction_summary
    from section3_default import plot_diagnosis, rolling_cross_rhat, rolling_within_rhat
    from section6_methods import (full_draw_rhat, chain_geometry, predictive_metrics,
                                  predictive_baselines)
    from section7_costs import scalar_summary
    import arviz as az
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr

    output, source = Path(output), Path(source)
    figures = output / "diagnostics"
    figures.mkdir(exist_ok=True)
    timings = [json.loads((output / "chains" / f"chain{c:03d}" / "timing.json").read_text()) for c in range(4)]
    pd.DataFrame([{key: value for key, value in row.items()
                   if not isinstance(value, dict)} for row in timings]).to_csv(figures / "timing.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout="constrained")
    for chain in range(4):
        stages = json.loads((output / "chains" / f"chain{chain:03d}" / "annealing.json").read_text())
        x = np.arange(1, len(stages)+1)
        for ax, key in zip(axes, ("beta", "ess", "unique_initial_ancestors")):
            ax.plot(x, [stage[key] for stage in stages], label=f"chain {chain+1}")
            ax.set_xlabel("annealing stage")
    axes[0].set_ylabel("beta"); axes[0].legend(ncol=4)
    axes[1].set_ylabel("weight ESS before resampling")
    axes[2].set_ylabel("distinct initial ancestors"); axes[2].set_yscale("log")
    fig.suptitle("SMC initialization | weight ESS and ancestry are not MCMC mixing diagnostics")
    fig.savefig(figures / "annealing.png", dpi=160)
    plt.close(fig)
    all_smc = None
    truth = load_with_shape(source / "subsample_y_test" / f"{TAG}__run000__subsample_y_test.csv").reshape(-1)
    # Reuse exactly the training response scale used by existing analyses.
    sys.path.insert(0, str(ROOT / "diagnosis"))
    from fixed100_support import load_dataset
    _, full_y = load_dataset("abalone")
    train_idx = load_with_shape(source / "indices" / f"{TAG}__run000__train_idx.csv").reshape(-1).astype(int)
    train_y = full_y[train_idx]
    y_scale = float(np.ptp(train_y)) or 1.0
    baseline = predictive_baselines(train_y, truth)
    method_roots = {METHOD: output, "default": source, "default_pt": source,
                    "mtmh": source, "mtmh_pt": source, "default_long": source}
    rows, scalars, scalar_chains, intervals, worsts = [], [], [], [], {}
    for method, location in method_roots.items():
        preds = load_with_shape(location / "preds" / f"{TAG}__run000__{method}__preds.csv").transpose(0, 2, 1)
        sigmas = load_with_shape(location / "sigmas" / f"{TAG}__run000__{method}__sigmas.csv").reshape(4, -1)
        if preds.shape != (4, 10000, 100) or sigmas.shape != (4, 10000):
            raise ValueError(f"Unexpected saved shape for {method}: {preds.shape}, {sigmas.shape}")
        if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError(f"Invalid posterior values for {method}")
        post, variances = preds[:, burn:], sigmas[:, burn:]
        worst = worst_direction_summary(post, 1e-8)
        worsts[method] = worst
        rmse, crps = predictive_metrics(post, variances, truth, y_scale, seed=20261012)
        data = xr.DataArray(post, dims=("chain", "draw", "test_point"), name="prediction")
        ess = np.asarray(az.ess(data, method="bulk")["prediction"])
        chain_means = post.mean(axis=1)
        pooled_mean = chain_means.mean(axis=0)
        within_variance = post.var(axis=1).mean(axis=0)
        between_variance = chain_means.var(axis=0)
        single_mse = float(np.mean((chain_means - truth)**2))
        pooled_mse = float(np.mean((pooled_mean - truth)**2))
        chain_specific_mse = float(np.mean((chain_means - pooled_mean)**2))
        np.testing.assert_allclose(single_mse, pooled_mse + chain_specific_mse, rtol=1e-10)
        rows.append({"method": method, **full_draw_rhat(post), **chain_geometry(post),
                     "worst_projected_rhat": worst["projected_rhat"],
                     "worst_lambda_max": worst["lambda_max"], "rmse": rmse, "crps": crps,
                     "relative_rmse": rmse / baseline["naive_rmse"],
                     "relative_crps": crps / baseline["climatology_crps"],
                     "pooled_rmse": float(np.sqrt(pooled_mse)),
                     "average_single_chain_mse": single_mse,
                     "pooled_mse": pooled_mse, "chain_specific_mse": chain_specific_mse,
                     "between_variance_fraction_median": float(np.median(
                         between_variance / (between_variance + within_variance))),
                     "pointwise_bulk_ess_median": float(np.median(ess)),
                     "pointwise_bulk_ess_minimum": float(np.min(ess))})
        for quantity, values in (("panel_mean_f", post.mean(axis=2)),
                                 ("noise_variance", variances * y_scale**2)):
            scalars.append({"method": method, "quantity": quantity, **scalar_summary(values)})
            for chain in range(4):
                scalar_chains.append({"method": method, "quantity": quantity, "chain": chain,
                                      **scalar_summary(values[chain:chain+1])})
        for chain in range(5):
            sample = post[chain] if chain < 4 else post.reshape(-1, post.shape[-1])
            low, high = np.quantile(sample, [0.05, 0.95], axis=0)
            intervals.append({"method": method, "scope": f"chain{chain}" if chain < 4 else "pooled",
                              "latent_90pct_width_median": float(np.median(high-low)),
                              "latent_90pct_width_mean": float(np.mean(high-low))})
        if method == METHOD:
            all_smc = preds
            pointwise = pointwise_rhat(post)
            pd.DataFrame({"test_point": np.arange(100), "rhat": pointwise,
                          "bulk_ess": ess}).to_csv(figures / "pointwise.csv", index=False)
            cross = rolling_cross_rhat(preds, 1000, 100)
            within = rolling_within_rhat(preds, 1000, 4, 100)
            cross.to_csv(figures / "rolling_cross_rhat.csv", index=False)
            within.to_csv(figures / "rolling_within_rhat.csv", index=False)
            fig = plot_diagnosis(preds, post, worst, cross, within, burn, 2000,
                                "Abalone run 0 | SMC-initialized Default MCMC | 100 particles")
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_text(text.get_text().replace("default chain", "chain"))
            fig.savefig(figures / "smc_diagnosis.png", dpi=160)
            plt.close(fig)
    comparison = pd.DataFrame(rows)
    scalar_frame = pd.DataFrame(scalars)
    comparison.to_csv(figures / "method_comparison.csv", index=False)
    scalar_frame.to_csv(figures / "scalar_diagnostics.csv", index=False)
    pd.DataFrame(scalar_chains).to_csv(figures / "scalar_chain_diagnostics.csv", index=False)
    pd.DataFrame(intervals).to_csv(figures / "latent_interval_widths.csv", index=False)
    method_labels = {METHOD: "SMC-init + Default", "default": "Default",
                     "default_pt": "Default+PT", "mtmh": "MTMH", "mtmh_pt": "MTMH+PT"}
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), layout="constrained")
    colours = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for row, (method, label) in enumerate(method_labels.items()):
        found = worsts[method]
        for chain in range(4):
            trace = found["projected"][chain]
            axes[row, 0].plot(np.arange(burn, burn+len(trace)), trace,
                              color=colours[chain], alpha=0.7, linewidth=0.5)
            coords = found["projected_2d"][chain]
            idx = np.linspace(0, len(coords)-1, min(750, len(coords)), dtype=int)
            axes[row, 1].scatter(coords[idx, 0], coords[idx, 1], s=6,
                                 color=colours[chain], alpha=0.15)
            axes[row, 1].scatter(*coords.mean(axis=0), s=70, marker="X", color=colours[chain])
        axes[row, 0].axvline(burn+found["fit_draws"], color="0.4", linestyle="--")
        axes[row, 0].set_title(f"{label}: held-half projected R-hat={found['projected_rhat']:.3f}")
        axes[row, 1].set_title(f"{label}: fitted-half leading eigenvalue={found['lambda_max']:.3g}")
        axes[row, 0].set_ylabel("projected prediction")
        axes[row, 1].set(xlabel="direction 1", ylabel="direction 2")
    axes[-1, 0].set_xlabel("production iteration")
    fig.suptitle("Abalone run 0 | method-specific exploratory directions")
    fig.savefig(figures / "worst_direction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    labels = ["SMC-init", "Default", "Default+PT", "MTMH", "MTMH+PT", "Default long"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    metrics = [("cross_rhat_median", "Pointwise R-hat: median"),
               ("cross_rhat_q90", "Pointwise R-hat: 90th percentile"),
               ("between_within_ratio", "Prediction-space between/within ratio"),
               ("rmse", "Mean single-chain posterior-mean RMSE"),
               ("pointwise_bulk_ess_median", "Pointwise bulk ESS: median"),
               (None, "Panel mean and noise variance R-hat")]
    positions = np.arange(len(labels))
    for ax, (metric, title) in zip(axes.flat, metrics):
        if metric:
            ax.scatter(comparison[metric], positions, color="#26764a")
            if "rhat" in metric:
                ax.axvline(1.01, color="0.4", linestyle="--", linewidth=0.8)
        else:
            for quantity, marker in (("panel_mean_f", "o"), ("noise_variance", "x")):
                values = scalar_frame.loc[scalar_frame.quantity == quantity, "rhat"]
                ax.scatter(values, positions, marker=marker, label=quantity)
            ax.axvline(1.01, color="0.4", linestyle="--", linewidth=0.8)
            ax.legend(fontsize=8)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Abalone run 0 | 4 chains, 7,000 retained draws each\n"
                 "Equal saved counts; SMC initialization and long-chain budgets differ")
    fig.savefig(figures / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    panel_mean = all_smc.mean(axis=2)
    smc_sigmas = load_with_shape(output / "sigmas" / f"{TAG}__run000__{METHOD}__sigmas.csv").reshape(4, -1) * y_scale**2
    for chain in range(4):
        axes[0].plot(panel_mean[chain], alpha=0.65, linewidth=0.5, label=f"chain {chain+1}")
        axes[1].plot(smc_sigmas[chain], alpha=0.65, linewidth=0.5)
    for ax in axes:
        ax.axvline(burn, color="0.3", linestyle="--")
        ax.set_xlabel("production iteration (SMC initialization precedes zero)")
    axes[0].set_ylabel("fixed-panel mean f"); axes[0].legend(ncol=4)
    axes[1].set_ylabel("noise variance (response units squared)")
    fig.savefig(figures / "scalar_traces.png", dpi=160)
    plt.close(fig)
    write_json(figures / "analysis_settings.json", {"burn": burn, "retained_per_chain": 10000-burn,
        "cross_window": 1000, "step": 100, "within_segments": 4,
        "within_rolling_segment_length": 1000, "full_post_burn_segment_length": (10000-burn)//4,
        "notes": ["SMC initialization followed by a single continuous Default MCMC trajectory per run.",
                  "Four independent populations; final-particle copies are not counted as draws.",
                  "Default long spans 10^7 iterations; other stored methods span 10^4.",
                  "Stored baseline timings came from another machine; do not divide local time by those timings."]})
    print(comparison.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "diagnosis/store_smc" / TAG / "run000")
    parser.add_argument("--source", type=Path, default=ROOT / "diagnosis/store" / TAG)
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--n-mcmc-steps", type=int, default=2)
    parser.add_argument("--ndpost", type=int, default=10000)
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Recover unfinished chains without replacing completed chains.")
    parser.add_argument("--chain-ids", nargs="+", type=int, choices=range(4), default=list(range(4)))
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.output, args.source, args.burn)
        return
    from fixed100_support import load_dataset
    from experiment_fixed100 import make_fixed100_splits
    sys.path.insert(0, str(ROOT / "diagnosis/paper"))
    from core import load_with_shape
    X, y = load_dataset("abalone")
    split = make_fixed100_splits(X, y, n_runs=1)[0]
    for name, key in (("train_idx", "train_idx"), ("fixed_test_idx", "test_idx")):
        saved = load_with_shape(args.source / "indices" / f"{TAG}__run000__{name}.csv").reshape(-1)
        np.testing.assert_array_equal(split[key], saved)
    saved_X = load_with_shape(args.source / "subsample_X_test" / f"{TAG}__run000__subsample_X_test.csv")
    np.testing.assert_allclose(split["X_test_fixed"], saved_X)
    settings = {"n_particles": args.n_particles, "n_mcmc_steps": args.n_mcmc_steps,
                "ndpost": args.ndpost, "jobs": args.jobs, "n_trees": 100,
                "ess_target": 0.8, "resample_threshold": 0.5, "burn_for_analysis": args.burn,
                "tree_alpha": 0.95, "tree_beta": 2.0, "proposal_tol": 1}
    if args.resume:
        saved_settings = json.loads((args.output / "settings.json").read_text())
        if any(settings[key] != value for key, value in saved_settings.items() if key != "jobs"):
            raise ValueError("Resume settings must match the original statistical configuration.")
        settings = saved_settings
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "settings.json", settings)
    requested = []
    recovery = []
    for chain in args.chain_ids:
        folder = args.output / "chains" / f"chain{chain:03d}"
        if (folder / "timing.json").exists():
            if not args.resume:
                raise FileExistsError(folder)
            continue
        if folder.exists():
            if not args.resume:
                raise FileExistsError(folder)
            destination = folder.with_name(folder.name + f"_failed_{int(time.time())}")
            # Both directory targets must remain under this exact run's chains.
            parent = (args.output / "chains").resolve()
            if folder.resolve().parent != parent or destination.resolve().parent != parent:
                raise ValueError("Recovery paths escaped the run directory.")
            folder.rename(destination)
            recovery.append({"chain": chain, "preserved_failed_attempt": destination.name})
        requested.append(chain)
    if recovery:
        write_json(args.output / "recovery.json", recovery)
    started = time.perf_counter()
    timings = []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(run_chain, c, split["X_train"], split["y_train"],
                   split["X_test_fixed"], split["y_test_fixed"], args.output, settings) for c in requested]
        for future in as_completed(futures):
            timing = future.result()
            timings.append(timing)
            print(json.dumps(timing), flush=True)
    timings = [json.loads((args.output / "chains" / f"chain{c:03d}" / "timing.json").read_text()) for c in range(4)]
    elapsed = (time.time() - (args.output / "settings.json").stat().st_mtime
               if args.resume else time.perf_counter()-started)
    aggregate(args.output, split, settings, timings, elapsed)
    if args.ndpost == 10000:
        analyze(args.output, args.source, args.burn)


if __name__ == "__main__":
    main()
