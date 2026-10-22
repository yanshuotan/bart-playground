"""Retain all SMC particles; compare unordered posterior ensembles on fixed-100 data.

Four independent SMC populations, no selected-particle initialization. At beta
one, terminal resampling is followed by configurable population warm-up and
collection sweeps. No flattened-chain R-hat or MCMC ESS.
"""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnosis"))
from run_smc import write_json, save_csv

TAG = "fixed100_Abalone"
METHOD = "smc_population"


def unordered_scores(draws, variances, truth, y_scale, seed=20261012):
    """RMSE and predictive MC-CRPS, invariant to joint state permutations.

    Canonical ordering is only used to assign reproducible Gaussian noise to
    each latent state/variance pair; it is never interpreted as a time series.
    """
    draws = np.asarray(draws)
    variances = np.asarray(variances)
    if draws.ndim != 2 or variances.shape != (len(draws),):
        raise ValueError("Expected draws (sample, point) and variances (sample,).")
    rng = np.random.default_rng(seed)
    crps = []
    n = len(draws)
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    for point, target in enumerate(truth):
        order = np.lexsort((variances, draws[:, point]))
        predictive = (draws[order, point] + y_scale * np.sqrt(variances[order])
                      * rng.standard_normal(n))
        crps.append(np.mean(np.abs(predictive-target))
                    - np.sum(coefficients * np.sort(predictive)) / n**2)
    return float(np.sqrt(np.mean((draws.mean(axis=0)-truth)**2))), float(np.mean(crps))


def unordered_geometry(draws):
    """The paper's exact finite-mixture R_B/W, with absolute components."""
    centers = np.mean(draws, axis=1)
    grand = centers.mean(axis=0)
    between = float(np.mean(np.sum((centers-grand)**2, axis=1)))
    within = float(np.mean(np.sum((draws-centers[:, None, :])**2, axis=2)))
    return {"R_BW": between / within, "between_centroid_variance": between,
            "within_population_variance": within}


def balanced_population_indices(n_particles, n_sweeps, target=7000):
    """Match state counts using entire, evenly spaced population sweeps."""
    if target % n_particles or target > n_particles * n_sweeps:
        raise ValueError("Target must fit the population and contain complete sweeps.")
    rounds = np.linspace(0, n_sweeps - 1, target // n_particles, dtype=int)
    return (rounds[:, None] * n_particles + np.arange(n_particles)).ravel()


def retained_block_diagnostics(draws, variances, truth, y_scale, block_size=1000):
    """Describe changes along each method's own retained record, not stationarity."""
    draws, variances = np.asarray(draws), np.asarray(variances)
    if draws.ndim != 3 or variances.shape != draws.shape[:2]:
        raise ValueError("Expected replicate/sample/point predictions and matching variances.")
    records = []
    for replicate in range(len(draws)):
        for start in range(0,draws.shape[1],block_size):
            end = min(start+block_size,draws.shape[1])
            block = draws[replicate,start:end]
            records.append({"replicate":replicate,"first_sample":start,
                "last_sample_exclusive":end,"block":start//block_size,
                "noise_variance":float(variances[replicate,start:end].mean()*y_scale**2),
                "panel_mean_f":float(block.mean()),
                "rmse":float(np.sqrt(np.mean((block.mean(axis=0)-truth)**2)))})
    return records


def validate_stored_split(split, source, tag):
    """Require identical rows, encoded predictors and targets to the baselines."""
    sys.path.insert(0,str(ROOT/"diagnosis/paper"))
    from core import load_with_shape
    source = Path(source)
    for name,key in (("train_idx","train_idx"),("fixed_test_idx","test_idx")):
        saved=load_with_shape(source/"indices"/f"{tag}__run000__{name}.csv").ravel()
        np.testing.assert_array_equal(saved,split[key])
    for name,key in (("subsample_X_test","X_test_fixed"),("subsample_y_test","y_test_fixed")):
        saved=load_with_shape(source/name/f"{tag}__run000__{name}.csv")
        np.testing.assert_allclose(saved.reshape(split[key].shape),split[key])


def worker(replicate, split, output, settings):
    from bart_playground import SMCBART

    folder = Path(output) / "replicates" / f"replicate{replicate:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    seed = 2024 + replicate
    posterior_seed = int(np.random.SeedSequence([seed, 2]).generate_state(1)[0])
    with threadpool_limits(limits=1):
        model = SMCBART(n_particles=settings["n_particles"], n_trees=100,
                        n_mcmc_steps=2, ess_target=0.8, resample_threshold=0.5,
                        random_state=seed, n_jobs=settings.get("posterior_n_jobs", 1))
        choose_beta = model._next_beta

        def reporting_beta(beta, weights, likelihoods):
            value = choose_beta(beta, weights, likelihoods)
            stage = len(model.stage_diagnostics_) + 1
            if stage == 1 or stage % 5 == 0 or value == 1:
                write_json(folder / "progress.json", {"replicate": replicate,
                    "phase": "annealing", "stage": stage, "next_beta": value,
                    "elapsed_seconds": time.perf_counter()-started})
            return value

        model._next_beta = reporting_beta
        model.fit(split["X_train"], split["y_train"], quietly=True)
        annealing_seconds = time.perf_counter()-started
        model._next_beta = choose_beta
        del reporting_beta, choose_beta
        ancestry = np.arange(settings["n_particles"])
        stages = []
        for stage in model.stage_diagnostics_:
            ancestry = ancestry[stage["ancestors"]]
            stages.append({**{k:v for k,v in stage.items() if k != "ancestors"},
                           "ancestors": stage["ancestors"].tolist(),
                           "initial_ancestors": int(len(np.unique(ancestry)))})
        write_json(folder / "annealing.json", stages)
        np.savez(folder / "annealed_population.npz", weights=model.weights_,
                 preds=np.array([model.preprocessor.backtransform_y(p.evaluate(split["X_test_fixed"]))
                                 for p in model.particles_]))
        # Annealing acceptance logs are not required to propagate the population.
        model.sampler.accepted_moves_logmh.clear()
        n, sweeps, warm = settings["n_particles"], settings["retained_sweeps"], settings["posterior_warmup_sweeps"]
        count = n * sweeps
        shapes = {"preds": (count, len(split["y_test_fixed"])), "sigmas": (count,),
                  "rmses": (count,), "vars_histogram": (count, split["X_train"].shape[1])}
        arrays = {name: np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+",
                   dtype=np.int32 if name == "vars_histogram" else float, shape=shape)
                  for name, shape in shapes.items()}
        population_started = time.perf_counter()

        def progress(completed, total):
            if completed == 1 or completed % 5 == 0 or completed == total:
                for a in arrays.values():
                    a.flush()
                write_json(folder / "progress.json", {"replicate": replicate,
                    "phase": "posterior_particle_paths",
                    "particle_paths_completed": completed, "particle_paths_total": total,
                    "posterior_workers": model.posterior_workers_,
                    "mutation_sweeps_in_completed_paths": completed*(warm+sweeps),
                    "elapsed_seconds": time.perf_counter()-started})

        model.sample_posterior(split["X_test_fixed"], sweeps, warm,
            random_state=posterior_seed, progress_callback=progress, outputs=arrays,
            temp_dir=settings.get("posterior_temp_dir"))
        # Bound temporary prediction storage when collecting long server runs.
        for start in range(0, count, 10000):
            end = min(start+10000, count)
            arrays["rmses"][start:end] = np.sqrt(np.mean(
                (arrays["preds"][start:end]-split["y_test_fixed"])**2, axis=1))
        for a in arrays.values():
            a.flush()
            if not np.all(np.isfinite(a)):
                raise ValueError("Non-finite posterior output.")
        if not np.all(arrays["sigmas"] > 0):
            raise ValueError("Nonpositive posterior variance.")
        np.save(folder / "terminal_resampling_indices.npy", model.posterior_resampling_indices_)
        # Preserve particle/sweep identity even though the comparison ignores order.
        np.save(folder / "particle_id.npy", np.tile(np.arange(n), sweeps))
        np.save(folder / "population_sweep.npy", np.repeat(np.arange(sweeps), n))
        result = {"replicate": replicate, "smc_seed": seed, "posterior_seed": posterior_seed,
                  "annealing_seconds": annealing_seconds, "annealing_stages": len(stages),
                  "annealing_sweeps": len(stages)*n*2,
                  "posterior_warmup_and_first_sweep_seconds": None,
                  "posterior_population_seconds": time.perf_counter()-population_started,
                  "posterior_workers": model.posterior_workers_,
                  "posterior_rng_scheme": "seedsequence_per_offspring_v1",
                  "posterior_population_sweeps": (warm+sweeps)*n,
                  "retained_samples": count, "total_seconds": time.perf_counter()-started,
                  "parent_cpu_seconds": time.process_time()-cpu_started,
                  "posterior_path_cpu_seconds": model.posterior_path_cpu_seconds_,
                  "cpu_seconds": time.process_time()-cpu_started + (
                      model.posterior_path_cpu_seconds_ if model.posterior_workers_ > 1 else 0),
                  "cpu_seconds_scope": "parent plus parallel posterior tasks; excludes child startup",
                  "parameters": model.get_params(),
                  "mutation_selected": model.posterior_move_counts_["move_selected_counts"]}
        write_json(folder / "timing.json", result)
        write_json(folder / "progress.json", {"phase": "complete", **result})
        return result


def aggregate(output, split, settings, timings, elapsed):
    output = Path(output)
    tag = settings.get("dataset_tag", TAG)
    for name in ("preds", "sigmas", "rmses", "vars_histogram", "particle_id", "population_sweep"):
        values = np.stack([np.load(output / "replicates" / f"replicate{r:03d}" / f"{name}.npy") for r in range(4)])
        if name == "preds":
            values = values.transpose(0, 2, 1)
        elif name == "sigmas":
            values = values[:, :, None]
        save_csv(output / name / f"{tag}__run000__{METHOD}__{name}.csv", values)
    for name, key in (("train_idx", "train_idx"), ("fixed_test_idx", "test_idx")):
        save_csv(output / "indices" / f"{tag}__run000__{name}.csv", split[key])
    for name, key in (("subsample_X_test", "X_test_fixed"), ("subsample_y_test", "y_test_fixed")):
        save_csv(output / name / f"{tag}__run000__{name}.csv", split[key])
    write_json(output / "metadata.json", {"method": "Annealed SMC with full-population posterior rejuvenation",
        "settings": settings, "replicates": sorted(timings, key=lambda t:t["replicate"]),
        "sampling_wall_seconds": elapsed,
        "four_process_sampling_wall_seconds": elapsed,
        "sample_layout": "replicate, point, sample; sample = sweep*n_particles + particle",
        "samples_are_iid": False, "samples_are_one_chronological_chain": False,
        "all_retained_samples_beta": 1.0, "retained_sample_weights": "equal",
        "one_particle_long_chain_used": False,
        "rhat_and_mcmc_ess_reported": False})


def analyze(output, source, reference_population=None):
    sys.path.insert(0, str(ROOT / "diagnosis/paper"))
    from core import load_with_shape
    from scipy.stats import wasserstein_distance
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fixed100_support import DATASET_CONFIGS, load_dataset

    output, source = Path(output), Path(source)
    settings = json.loads((output / "settings.json").read_text())
    dataset = settings.get("dataset", "abalone")
    tag = DATASET_CONFIGS[dataset]["dataset_tag"]
    if settings.get("dataset_tag", tag) != tag:
        raise ValueError("Dataset name and dataset tag disagree.")
    n_particles, n_sweeps = settings["n_particles"], settings["retained_sweeps"]
    dest = output / "diagnostics"
    dest.mkdir(exist_ok=True)
    _, y = load_dataset(dataset)
    idx = load_with_shape(source / "indices" / f"{tag}__run000__train_idx.csv").astype(int).ravel()
    train_y = y[idx]
    truth = load_with_shape(source / "subsample_y_test" / f"{tag}__run000__subsample_y_test.csv").ravel()
    y_scale = float(np.ptp(train_y)) or 1.0
    methods = {METHOD: output, "default": source, "default_pt": source,
               "mtmh": source, "mtmh_pt": source, "default_long": source}
    population_settings = {METHOD: settings}
    tail_available = (source / "preds" / f"{tag}__run000__default_long_tail__preds.csv").is_file()
    if tail_available:
        methods["default_long_tail"] = source
    if reference_population is not None:
        reference_population = Path(reference_population)
        methods["smc_population_reference"] = reference_population
        population_settings["smc_population_reference"] = json.loads(
            (reference_population / "settings.json").read_text())
        if population_settings["smc_population_reference"].get("dataset", "abalone") != dataset:
            raise ValueError("Reference population must use the same dataset.")
    rows, equal_rows, scalar_rows, interval_rows, collection = [], [], [], [], {}
    retained_blocks = []
    predictive_rows = []
    for method, location in methods.items():
        file_method = METHOD if method in population_settings else method
        preds = load_with_shape(location / "preds" / f"{tag}__run000__{file_method}__preds.csv").transpose(0, 2, 1)
        sigmas = load_with_shape(location / "sigmas" / f"{tag}__run000__{file_method}__sigmas.csv").reshape(4, -1)
        if method not in population_settings:
            preds, sigmas = preds[:, 3000:], sigmas[:, 3000:]
        if not np.all(np.isfinite(preds)) or not np.all(sigmas > 0):
            raise ValueError(f"Invalid outputs for {method}")
        collection[method] = (preds, sigmas)

        def score(p, v, selection="full"):
            single = [unordered_scores(a,b,truth,y_scale) for a,b in zip(p,v)]
            for replicate, (rmse, crps) in enumerate(single):
                predictive_rows.append({"method":method,"selection":selection,
                    "replicate":replicate,"samples":p.shape[1],"rmse":rmse,"crps":crps})
            pooled = unordered_scores(p.reshape(-1,100), v.reshape(-1), truth, y_scale)
            return {"method": method, "samples_per_replicate": p.shape[1],
                **unordered_geometry(p), "rmse_single_mean": float(np.mean([a for a,b in single])),
                "rmse_root_mean_mse": float(np.sqrt(np.mean([a*a for a,b in single]))),
                "crps_single_mean": float(np.mean([b for a,b in single])),
                "rmse_pooled": pooled[0], "crps_pooled": pooled[1]}

        rows.append(score(preds, sigmas))
        if method in population_settings:
            config = population_settings[method]
            choose = balanced_population_indices(config["n_particles"], config["retained_sweeps"])
            equal_rows.append(score(preds[:, choose], sigmas[:, choose], "matched7000"))
            block_preds, block_sigmas = preds[:,choose], sigmas[:,choose]
        else:
            equal_rows.append(rows[-1].copy())
            block_preds, block_sigmas = preds, sigmas
        retained_blocks.extend({"method":method,**row} for row in
            retained_block_diagnostics(block_preds,block_sigmas,truth,y_scale))
        for quantity, values in (("panel_mean_f", preds.mean(axis=2)),
                                 ("noise_variance", sigmas*y_scale**2)):
            centers = values.mean(axis=1)
            distances = [wasserstein_distance(values[i], values[j])
                         for i in range(4) for j in range(i+1,4)]
            scalar_rows.append({"method": method, "quantity": quantity,
                "replicate_mean_sd": float(np.std(centers, ddof=1)),
                "mean_pairwise_wasserstein": float(np.mean(distances)),
                "pooled_mean": float(values.mean()), "pooled_q05": float(np.quantile(values,.05)),
                "pooled_q95": float(np.quantile(values,.95))})
            for replicate in range(4):
                scalar_rows.append({"method":method,"quantity":quantity,"replicate":replicate,
                    "pooled_mean":float(values[replicate].mean()),
                    "pooled_q05":float(np.quantile(values[replicate],.05)),
                    "pooled_q95":float(np.quantile(values[replicate],.95))})
        for replicate in range(5):
            values = preds[replicate] if replicate < 4 else preds.reshape(-1,100)
            low, high = np.quantile(values,[.05,.95],axis=0)
            interval_rows.append({"method":method,"scope":f"replicate{replicate}" if replicate<4 else "pooled",
                "latent_90pct_width_median":float(np.median(high-low)),
                "latent_90pct_width_mean":float(np.mean(high-low))})
    full = pd.DataFrame(rows); matched = pd.DataFrame(equal_rows)
    full.to_csv(dest / "comparison_full_population.csv", index=False)
    matched.to_csv(dest / "comparison_matched7000.csv", index=False)
    pd.DataFrame(predictive_rows).to_csv(dest / "replicate_predictive_scores.csv", index=False)
    pd.DataFrame(scalar_rows).to_csv(dest / "scalar_distribution_agreement.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(dest / "latent_interval_widths.csv", index=False)
    labels = [f"SMC ({n_particles} particles)", "Default", "Default+PT", "MTMH", "MTMH+PT", "Default long"]
    if tail_available:
        labels.append("Default long tail")
    if reference_population is not None:
        labels.append(f"SMC reference ({population_settings['smc_population_reference']['n_particles']} particles)")
    retained_frame = pd.DataFrame(retained_blocks)
    retained_frame.to_csv(dest / "collection_block_comparison.csv",index=False)
    drift = []
    fig, axes = plt.subplots(2,4,figsize=(14,7),layout="constrained",sharex=True,sharey=True)
    for ax, (method,label) in zip(axes.flat,zip(methods,labels)):
        frame = retained_frame.loc[retained_frame.method==method]
        for replicate, values in frame.groupby("replicate"):
            ax.plot(values.block+1,values.noise_variance,marker="o",label=f"repeat {replicate+1}")
            first, last = values.iloc[0], values.iloc[-1]
            drift.append({"method":method,"replicate":int(replicate),
                "first_block_noise_variance":first.noise_variance,
                "last_block_noise_variance":last.noise_variance,
                "relative_noise_change":last.noise_variance/first.noise_variance-1,
                "first_block_rmse":first.rmse,"last_block_rmse":last.rmse})
        ax.set(title=label,xlabel="block of 1,000 retained states",ylabel="noise variance")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Changes within each method's retained record; block positions are not equal computation time")
    fig.savefig(dest / "collection_block_comparison.png",dpi=160); plt.close(fig)
    pd.DataFrame(drift).to_csv(dest / "collection_drift_summary.csv",index=False)
    fig, axes = plt.subplots(1,3,figsize=(13,5.2),layout="constrained")
    for ax, key, title in zip(axes, ("R_BW","rmse_root_mean_mse","crps_single_mean"),
                             ("Prediction-space R_B/W", "Posterior-mean RMSE", "Predictive MC-CRPS")):
        ax.scatter(matched[key],np.arange(len(labels)),color="#26764a")
        ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis()
        ax.set_title(title); ax.grid(axis="x",alpha=.2)
    fig.suptitle(f"{tag.removeprefix('fixed100_')} run 0 | 4 independent replicates, 7,000 samples each\n"
                 "SMC: balanced sweep/particle selection; others: paper post-burn records")
    fig.savefig(dest / "comparison.png",dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for row, method in enumerate((METHOD,"default")):
        p,v = collection[method]
        for r in range(4):
            axes[row,0].hist(p[r].mean(axis=1),bins=45,density=True,histtype="step",label=f"replicate {r+1}")
            axes[row,1].hist(v[r]*y_scale**2,bins=45,density=True,histtype="step")
        axes[row,0].set(xlabel="fixed-panel mean f",ylabel=f"{method}: density")
        axes[row,1].set(xlabel="noise variance (response units squared)",ylabel="density")
    axes[0,0].legend(fontsize=8)
    fig.savefig(dest / "scalar_distributions.png",dpi=160); plt.close(fig)
    p = collection[METHOD][0]
    sweep_rows = []
    for retained in sorted({r for r in (10,20,40,60,80,100,n_sweeps) if r <= n_sweeps}):
        sweep_rows.append({"retained_population_sweeps":retained,
                           **unordered_geometry(p[:,:retained*n_particles])})
    pd.DataFrame(sweep_rows).to_csv(dest / "population_prefix_geometry.csv",index=False)
    # Known population-sweep labels permit a sensitivity check without inventing
    # a chronological chain from the flattened sample array.
    blocks = []
    p,v = collection[METHOD]
    for start in range(0,n_sweeps,20):
        end = min(start+20,n_sweeps)
        for replicate in range(4):
            choose = slice(start*n_particles,end*n_particles)
            blocks.append({"replicate":replicate,"first_retained_sweep":start,
                           "last_retained_sweep_exclusive":end,
                           "panel_mean_f":float(p[replicate,choose].mean()),
                           "noise_variance":float(v[replicate,choose].mean()*y_scale**2)})
    block_frame = pd.DataFrame(blocks)
    block_frame.to_csv(dest / "population_round_blocks.csv",index=False)
    scalar_frame = pd.DataFrame(scalar_rows)
    pooled_scalars = scalar_frame.loc[scalar_frame.replicate.isna()]
    smc_noise = float(pooled_scalars.loc[(pooled_scalars.method==METHOD)
                                       & (pooled_scalars.quantity=="noise_variance"),"pooled_mean"].iloc[0])
    long_noise = float(pooled_scalars.loc[(pooled_scalars.method=="default_long")
                                        & (pooled_scalars.quantity=="noise_variance"),"pooled_mean"].iloc[0])
    smc_row = next(row for row in rows if row["method"] == METHOD)
    default_row = next(row for row in rows if row["method"] == "default")
    write_json(dest / "summary.json", {
        "R_BW_relative_reduction_vs_default":1-smc_row["R_BW"]/default_row["R_BW"],
        "between_centroid_variance_relative_reduction":1-smc_row["between_centroid_variance"]/default_row["between_centroid_variance"],
        "within_population_variance_relative_increase":smc_row["within_population_variance"]/default_row["within_population_variance"]-1,
        "smc_noise_variance_mean":smc_noise,"long_reference_noise_variance_mean":long_noise,
        "noise_variance_relative_difference_from_long_reference":smc_noise/long_noise-1,
        "interpretation":"Replicate R_B/W alone does not establish posterior accuracy. "
            "Inspect noise-variance displacement and population-round stability for residual finite-budget effects."})
    fig, ax = plt.subplots(figsize=(8,4.5),layout="constrained")
    for replicate, frame in block_frame.groupby("replicate"):
        ax.plot((frame.first_retained_sweep+frame.last_retained_sweep_exclusive)/2,
                frame.noise_variance,marker="o",label=f"replicate {replicate+1}")
    ax.axhline(long_noise,color="0.3",linestyle="--",label="long-Default pooled mean")
    ax.set(xlabel="retained population sweep (20-sweep block center)",
           ylabel="population mean noise variance (response units squared)",
           title="Population noise variance across retained sweeps")
    ax.legend(fontsize=8,ncol=2)
    fig.savefig(dest / "population_noise_sensitivity.png",dpi=160); plt.close(fig)
    if reference_population is not None:
        comparison_blocks = []
        fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained",sharex=True,sharey=True)
        for ax, (method, config) in zip(axes, population_settings.items()):
            particles, sweeps = config["n_particles"], config["retained_sweeps"]
            _, variances = collection[method]
            for replicate in range(4):
                centers, means = [], []
                for start in range(0,sweeps,20):
                    end = min(start+20,sweeps)
                    center = config["posterior_warmup_sweeps"] + (start+end)/2
                    value = float(variances[replicate,start*particles:end*particles].mean()*y_scale**2)
                    centers.append(center); means.append(value)
                    comparison_blocks.append({"method":method,"replicate":replicate,
                        "particles":particles,"posterior_updates_per_particle":center,
                        "noise_variance":value})
                ax.plot(centers,means,marker="o",label=f"replicate {replicate+1}")
            ax.axhline(long_noise,color="0.3",linestyle="--",label="long-Default mean")
            ax.set(title=f"{particles} particles; warm-up {config['posterior_warmup_sweeps']}",
                   xlabel="posterior updates per particle (block center)")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("noise variance (response units squared)")
        fig.savefig(dest / "population_count_sensitivity.png",dpi=160); plt.close(fig)
        pd.DataFrame(comparison_blocks).to_csv(dest / "population_comparison_blocks.csv",index=False)
    write_json(dest / "analysis_settings.json", {"reported_metrics_require_draw_order":False,
        "dataset":dataset,"dataset_tag":tag,
        "ordinary_rhat_reported":False,"ordinary_mcmc_ess_reported":False,
        "rank_normalized_split_folded_rhat_reported":False,
        "smc_prefix_burn_after_collection":0,"baseline_burn":3000,
        "matched_smc_selection":f"{7000//n_particles} evenly spaced sweeps, all {n_particles} particles per sweep",
        "reference_population":str(reference_population) if reference_population is not None else None,
        "crps":"Posterior predictive; canonical state ordering then fixed-seed Gaussian noise, seed 20261012",
        "comparison_is_equal_wall_time":False,
        "notes":["Finite-population dependence and shared ancestry remain.",
                 "Retained-block diagnostics describe changes within each method's record; they are not a stationarity acceptance rule or an equal-time comparison.",
                 "R_B/W is replicate agreement, not a proof that all modes were explored.",
                 "Default long spans 10^7 iterations; short baselines span 10^4.",
                 "Default long tail, when available, is a contiguous 10^4-state endpoint record; its first 3,000 records are excluded for count matching.",
                 "Long runs are comparison references, not assumed stationary ground truth."]})
    print("FULL POPULATION:",flush=True); print(full.to_string(index=False),flush=True)
    print("MATCHED COUNTS:",flush=True); print(matched.to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("abalone","seoul_bike"),default="abalone")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--jobs",type=int,choices=range(1,5),default=4)
    parser.add_argument("--posterior-n-jobs",type=int,default=1,
                        help="Processes per population after annealing (default: 1, serial). "
                             "-1 uses available physical CPUs PER chain; positive values cap at N particles.")
    parser.add_argument("--posterior-temp-dir",type=Path,
                        help="Existing local scratch directory for read-only shared array mappings.")
    parser.add_argument("--n-particles",type=int,default=100)
    parser.add_argument("--retained-sweeps",type=int,default=100)
    parser.add_argument("--warmup-sweeps",type=int,default=50)
    parser.add_argument("--analyze-only",action="store_true")
    parser.add_argument("--reference-population",type=Path)
    parser.add_argument("--skip-analysis",action="store_true",
                        help="Save the complete run for a separate window analysis.")
    args=parser.parse_args()
    from fixed100_support import DATASET_CONFIGS, load_dataset
    if args.analyze_only and args.output is not None:
        args.dataset = json.loads((args.output/"settings.json").read_text()).get("dataset","abalone")
    tag = DATASET_CONFIGS[args.dataset]["dataset_tag"]
    args.output = args.output or ROOT/"diagnosis/store_smc"/tag/"run000"
    args.source = args.source or ROOT/"diagnosis/store"/tag
    if args.analyze_only:
        analyze(args.output,args.source,args.reference_population); return
    from bart_playground.smc import resolve_smc_workers, _positive_int
    posterior_workers = resolve_smc_workers(args.posterior_n_jobs, args.n_particles)
    _positive_int("retained_sweeps", args.retained_sweeps)
    if args.warmup_sweeps < 0:
        parser.error("--warmup-sweeps must be non-negative")
    if args.posterior_temp_dir is not None and not args.posterior_temp_dir.is_dir():
        parser.error("--posterior-temp-dir must be an existing directory")
    from experiment_fixed100 import make_fixed100_splits
    X,y=load_dataset(args.dataset); split=make_fixed100_splits(X,y,n_runs=1)[0]
    validate_stored_split(split,args.source,tag)
    settings={"dataset":args.dataset,"dataset_tag":tag,
              "training_rows":len(split["y_train"]),"n_features":X.shape[1],
              "fixed_test_rows":len(split["y_test_fixed"]),
              "n_particles":args.n_particles,"n_trees":100,"annealing_mcmc_steps":2,
              "conditional_ess_target":.8,"resample_threshold":.5,
              "posterior_warmup_sweeps":args.warmup_sweeps,"retained_sweeps":args.retained_sweeps,
              "samples_per_replicate":args.n_particles*args.retained_sweeps,"jobs":args.jobs,
              "posterior_n_jobs":args.posterior_n_jobs,"posterior_workers_per_chain":posterior_workers,
              "max_active_posterior_workers":args.jobs*posterior_workers,
              "posterior_temp_dir":str(args.posterior_temp_dir.resolve()) if args.posterior_temp_dir else None,
              "posterior_rng_scheme":"seedsequence_per_offspring_v1"}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"settings.json",settings)
    print(f"{args.jobs} concurrent populations; {posterior_workers} posterior workers each "
          f"({args.jobs*posterior_workers} active posterior workers maximum). "
          "Annealing is serial within each population.", flush=True)
    started=time.perf_counter(); timings=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context("spawn")) as executor:
        futures=[executor.submit(worker,r,split,args.output,settings) for r in range(4)]
        for future in as_completed(futures):
            result=future.result(); timings.append(result); print(json.dumps(result),flush=True)
    aggregate(args.output,split,settings,timings,time.perf_counter()-started)
    if not args.skip_analysis and settings["samples_per_replicate"] >= 7000 and 7000 % args.n_particles == 0:
        analyze(args.output,args.source,args.reference_population)


if __name__=="__main__":
    main()
