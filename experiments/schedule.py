import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import hashlib
import pandas as pd
import wandb
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator
from bart_playground.samplers import TemperatureSchedule, default_proposal_probs

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- Mixing Diagnostic Functions (copied from initialization_experiment.py) ---
def gelman_rubin(chains):
    # chains shape: (m_chains, n_samples_per_chain)
    if chains.ndim == 1:  # Single chain case
        LOGGER.warning("Gelman-Rubin diagnostic requires multiple chains. Returning NaN.")
        return np.nan
    if chains.shape[0] < 2:
        LOGGER.warning("Gelman-Rubin diagnostic requires at least 2 chains. Returning NaN.")
        return np.nan

    m, n = chains.shape
    if n == 0:  # No samples in chain
        LOGGER.warning("Gelman-Rubin diagnostic requires samples in chains. Returning NaN.")
        return np.nan

    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W == 0:  # Avoid division by zero if within-chain variance is zero
        LOGGER.warning("Within-chain variance is zero, Gelman-Rubin may be undefined. Returning NaN.")
        return np.nan
    V_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(V_hat / W)


def get_gelman_rubin_statistic(preds_chains):
    n_samples, n_chains, n_post = preds_chains.shape
    mu_chains = np.mean(preds_chains, axis=2)
    mu_all = np.mean(mu_chains, axis=1)
    B = n_post * np.sum((mu_chains - mu_all[:, None]) ** 2, axis=1) / (n_chains - 1)
    W = np.mean(np.var(preds_chains, axis=2), axis=1)
    R = (B / n_post + W * (n_post / (n_post - 1))) / W
    mean_R = np.mean(R)
    return float(mean_R)


def autocorrelation(chain, lag):
    n = len(chain)
    if lag == 0:
        return 1.0
    if lag >= n or n < 2:  # Added n < 2 check
        return np.nan
    # Ensure chains are not constant
    if np.all(chain == chain[0]):
        return np.nan
    return np.corrcoef(chain[:-lag], chain[lag:])[0, 1]


def effective_sample_size(chains, step=1):
    # chains shape: (m_chains, n_samples_per_chain)
    if chains.ndim == 1:  # Handle single chain by reshaping
        chains = chains.reshape(1, -1)

    m, n = chains.shape
    if n == 0:
        LOGGER.warning("ESS calculation requires samples in chains. Returning NaN.")
        return np.nan

    total_ess = 0.0
    for i in range(m):
        chain = chains[i, :]
        ac_sum = 0.0
        # Check if chain is constant
        if np.all(chain == chain[0]):
            # For a constant chain, ESS is problematic. Could return 1 or NaN.
            # Let's return NaN as it indicates an issue with the chain.
            LOGGER.warning(f"Chain {i} is constant. ESS is undefined. Returning NaN for this chain.")
            total_ess += np.nan  # Or simply skip adding to total_ess if one chain is bad
            continue

        for lag in range(1, n, step):
            ac = autocorrelation(chain, lag)
            if np.isnan(ac) or ac < 0:  # Stop if ACF is negative or NaN
                break
            ac_sum += step * ac  # Corrected: was ac_sum += ac , now step * ac

        # Denominator for ESS calculation
        ess_denominator = 1 + 2 * ac_sum
        if ess_denominator <= 0:  # Avoid division by zero or negative
            total_ess += np.nan  # Or some other indicator of problematic ESS
        else:
            total_ess += n / ess_denominator

    # If all chains were constant or had issues, total_ess might be NaN
    if np.isnan(total_ess) and m > 0:
        return np.nan
        # If averaging ESS, divide by m; if summing, this is total.
    # The original script implies sum, but per-chain ESS is often averaged.
    # Let's return total ESS as per original script's logic.
    return total_ess


# --- Run One Chain ---
def run_chain_bart(X_train_raw, X_test_raw, y_train_raw, y_test_raw,
                   cfg: DictConfig, seed: int, temp_schedule: TemperatureSchedule):
    bart_cfg = cfg.bart_params
    preprocessor = DefaultPreprocessor(max_bins=X_train_raw.shape[0])
    train_data = preprocessor.fit_transform(X_train_raw, y_train_raw)
    model = DefaultBART(
        ndpost=bart_cfg.ndpost, nskip=bart_cfg.nskip, n_trees=bart_cfg.n_trees,
        random_state=seed, temperature=temp_schedule, proposal_probs=default_proposal_probs
    )
    model.preprocessor = preprocessor
    model.data = train_data
    model.sampler.add_data(train_data)
    model.sampler.add_thresholds(preprocessor.thresholds)
    model.is_fitted = True
    start_time = time.perf_counter()
    trace = model.sampler.run(n_iter=bart_cfg.ndpost + bart_cfg.nskip, quietly=True, n_skip=bart_cfg.nskip)
    runtime = time.perf_counter() - start_time
    model.trace = trace
    post_raw = model.posterior_f(X_test_raw)
    if post_raw.shape[0] != X_test_raw.shape[0] and post_raw.shape[1] == X_test_raw.shape[0]:
        post_raw = post_raw.T
    preds_mean_raw = np.mean(post_raw, axis=1)
    final_mse = mean_squared_error(y_test_raw, preds_mean_raw)
    lower, upper = np.percentile(post_raw, [2.5, 97.5], axis=1)
    coverage = np.mean((y_test_raw >= lower) & (y_test_raw <= upper))
    chain_sample_for_diag = post_raw[0, :] if post_raw.shape[0] > 0 else np.array([])
    return {'final_mse': final_mse, 'coverage': coverage, 'chain_sample_for_diag': chain_sample_for_diag, 'runtime': runtime,
            'sigma2_samples': np.array([p.global_params['eps_sigma2'] for p in trace]),
            'post_raw': post_raw}


# --- Run Multi-chain Experiment ---
def run_full_experiment(X_train_specific, y_train_specific, X_test_specific, y_test_specific,
                        cfg: DictConfig, temp_schedule: TemperatureSchedule):
    exp_cfg = cfg.experiment_params
    all_results = []
    for i in range(exp_cfg.n_chains):
        seed_string = f"{exp_cfg.main_seed}_chain_{i}"
        seed_hash = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
        chain_specific_seed = int(seed_hash, 16) % (2 ** 32)
        chain_res = run_chain_bart(X_train_specific, X_test_specific, y_train_specific, y_test_specific,
                                   cfg, chain_specific_seed, temp_schedule)
        all_results.append(chain_res)
    aggregated_chain_samples = np.array([r['chain_sample_for_diag'] for r in all_results if r['chain_sample_for_diag'].size > 0])
    aggregated_sigma2_samples = np.array([r['sigma2_samples'] for r in all_results if r['sigma2_samples'].size > 0])
    
    # Each chain's posterior predictions for all test points
    preds_chains = np.stack([r['post_raw'] for r in all_results], axis=1)
    
    gr_fx, ess_fx = np.nan, np.nan
    if aggregated_chain_samples.size > 0:
        gr_fx = gelman_rubin(aggregated_chain_samples)
        ess_fx = effective_sample_size(aggregated_chain_samples)
    results_dict = {
        'final_mse': np.mean([r['final_mse'] for r in all_results]),
        'coverage': np.mean([r['coverage'] for r in all_results]),
        'runtime': np.mean([r['runtime'] for r in all_results]),
        'chain_samples_for_diag': aggregated_chain_samples,
        'sigma2_samples': aggregated_sigma2_samples,
        'gr_fx': gr_fx, 'ess_fx': ess_fx
    }
    return results_dict, preds_chains


# --- Plotting Functions ---
def plot_results(all_results, cfg: DictConfig, plot_dir_for_run: str, n_train_samples: int):
    os.makedirs(plot_dir_for_run, exist_ok=True)
    metrics_to_plot = ['final_mse', 'coverage', 'runtime', 'ess_fx']
    labels = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for metric in metrics_to_plot:
        values = [res.get(metric, np.nan) for res in all_results.values()]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color=colors)
        plt.title(f'{metric.replace("_", " ").title()} Comparison for DGP: {cfg.dgp} (N_train={n_train_samples})')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir_for_run, f"{cfg.dgp}_ntrain{n_train_samples}_{metric}_comparison.png"))
        plt.close()
    
    for config_name, res_data in all_results.items():
        gr_fx = res_data.get('gr_fx')
        ess_fx = res_data.get('ess_fx')

        if ess_fx is not None and not np.isnan(ess_fx):
            log_msg = f"DGP {cfg.dgp} (N_train={n_train_samples}) | {config_name} | f(x_test[0]): ESS = {ess_fx:.2f}"
            if gr_fx is not None and not np.isnan(gr_fx):
                log_msg += f", Gelman-Rubin = {gr_fx:.4f}"
            LOGGER.info(log_msg)
        else:
            LOGGER.warning(
                f"Could not compute diagnostics for f(x_test[0]) for {config_name} of DGP {cfg.dgp} (N_train={n_train_samples}).")


def log_wandb_artifacts(cfg: DictConfig, run_type: str, experiment_name: str, run_results: dict,
                        preds_chains: np.ndarray, y_true: np.ndarray,
                        run_specific_artifact_dir: str, current_n_train: int):
    """Logs experiment artifacts to Weights & Biases."""

    dgp_name_safe = "".join(c if c.isalnum() else "_" for c in cfg.dgp)

    full_exp_name = (f"{run_type}_{dgp_name_safe}_{experiment_name}_ntrain_{current_n_train}_"
                     f"seed_{cfg.experiment_params.main_seed}")

    wandb.init(
        project="bart-playground",
        entity="bart_playground",
        name=full_exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[run_type, cfg.dgp, experiment_name, f"seed_{cfg.experiment_params.main_seed}"],
        reinit=True
    )

    wandb_dir = os.path.join(run_specific_artifact_dir, f"wandb_artifacts_{experiment_name}")
    os.makedirs(wandb_dir, exist_ok=True)

    y_true_df = pd.DataFrame({"y_true": y_true})
    y_true_file = os.path.join(wandb_dir, "y_true.csv")
    y_true_df.to_csv(y_true_file, index=False)

    n_samples, n_chains, n_post = preds_chains.shape
    chain_files = []
    for chain_idx in range(n_chains):
        chain_preds = preds_chains[:, chain_idx, :]
        chain_df = pd.DataFrame(chain_preds, columns=[f"posterior_sample_{i}" for i in range(n_post)])
        chain_df["sample_index"] = np.arange(n_samples)
        chain_file = os.path.join(wandb_dir, f"predictions_chain_{chain_idx}.csv")
        chain_df.to_csv(chain_file, index=False)
        chain_files.append(chain_file)

    wandb.log({k: v for k, v in run_results.items() if 'samples' not in k})

    artifact = wandb.Artifact(
        name=f"{run_type}_results_{dgp_name_safe}_{experiment_name}_seed_{cfg.experiment_params.main_seed}_ntrain_{current_n_train}",
        type="experiment_results",
        description=f"BART {run_type} experiment for {cfg.dgp} with setting {experiment_name}"
    )

    config_file = os.path.join(wandb_dir, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False)
    artifact.add_file(config_file, name="config.yaml")
    artifact.add_file(y_true_file, name="y_true.csv")

    for chain_idx, chain_file in enumerate(chain_files):
        artifact.add_file(chain_file, name=f"predictions_chain_{chain_idx}.csv")

    wandb.log_artifact(artifact)
    wandb.finish()
    LOGGER.info(f"Finished logging to W&B for {experiment_name}.")


def create_schedule(schedule_cfg, total_iters):
    schedule_type = schedule_cfg.type
    params = OmegaConf.to_container(schedule_cfg.params, resolve=True)
    if schedule_type == "constant":
        return TemperatureSchedule(lambda t: params['temp'])
    elif schedule_type == "cosine":
        t_max, t_min = params['t_max'], params['t_min']
        return TemperatureSchedule(lambda t: t_min + 0.5 * (t_max - t_min) * (1 + np.cos(np.pi * t / total_iters)))
    elif schedule_type == "linear":
        t_max, t_min = params['t_max'], params['t_min']
        return TemperatureSchedule(lambda t: t_max - (t_max - t_min) * (t / total_iters))
    elif schedule_type == "exponential":
        t_max, gamma = params['t_max'], params['gamma']
        return TemperatureSchedule(lambda t: t_max * (gamma ** t))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def run_and_analyze(cfg: DictConfig):
    exp_params = cfg.experiment_params
    main_seed = exp_params.main_seed
    dgp_name_safe = "".join(c if c.isalnum() else "_" for c in cfg.dgp)
    dgp_and_seed_base_artifact_dir = os.path.join(cfg.artifacts_dir, dgp_name_safe, f"seed_{main_seed}")
    os.makedirs(dgp_and_seed_base_artifact_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(dgp_and_seed_base_artifact_dir, "config_main_seed.yaml"))

    dgp_params_test_dict = OmegaConf.to_container(cfg.dgp_params, resolve=True)
    dgp_params_test_dict['n_samples'] = exp_params.n_test_samples
    dgp_params_test_dict['noise'] = 0.0
    seed_string_test = f"{main_seed}_dgp_test_n{exp_params.n_test_samples}"
    seed_hash_test = hashlib.sha256(seed_string_test.encode('utf-8')).hexdigest()
    dgp_seed_test = int(seed_hash_test, 16) % (2 ** 32)
    dgp_params_test_dict['random_seed'] = dgp_seed_test
    generator_test = DataGenerator(**dgp_params_test_dict)
    X_test_run, y_test_run_true = generator_test.generate(scenario=cfg.dgp)

    for idx, current_n_train in enumerate(exp_params.n_train_samples_list):
        LOGGER.info(f"--- Starting run for DGP: {cfg.dgp} with N_train = {current_n_train} ---")
        dgp_params_train_dict = OmegaConf.to_container(cfg.dgp_params, resolve=True)
        dgp_params_train_dict['n_samples'] = current_n_train
        seed_string_train = f"{main_seed}_dgp_train_n{current_n_train}"
        seed_hash_train = hashlib.sha256(seed_string_train.encode('utf-8')).hexdigest()
        dgp_seed_train = int(seed_hash_train, 16) % (2 ** 32)
        dgp_params_train_dict['random_seed'] = dgp_seed_train
        generator_train = DataGenerator(**dgp_params_train_dict)
        X_train_run, y_train_run = generator_train.generate(scenario=cfg.dgp)

        run_specific_artifact_dir = os.path.join(dgp_and_seed_base_artifact_dir, f"ntrain_{current_n_train}")
        os.makedirs(run_specific_artifact_dir, exist_ok=True)

        all_results = {}
        schedules_to_test = exp_params.get("schedules", [])
        print(f"Schedules to test: {schedules_to_test}")
        total_mcmc_iters = cfg.bart_params.ndpost + cfg.bart_params.nskip

        for schedule_cfg in schedules_to_test:
            schedule_name = schedule_cfg.name
            LOGGER.info(f"--- Running Experiment: Schedule={schedule_name} ---")
            temp_schedule = create_schedule(schedule_cfg, total_mcmc_iters)
            results, preds_chains = run_full_experiment(X_train_run, y_train_run, X_test_run, y_test_run_true, cfg, temp_schedule)
            all_results[schedule_name] = results
            with open(os.path.join(run_specific_artifact_dir, f"results_{schedule_name}.pkl"), "wb") as f:
                pickle.dump(results, f)
            
            if cfg.experiment_params.get("log_to_wandb", False):
                log_wandb_artifacts(cfg, "schedule", schedule_name, results, preds_chains,
                                    y_test_run_true, run_specific_artifact_dir, current_n_train)

            LOGGER.info(f"Results ({schedule_name}) for DGP {cfg.dgp}, N_train={current_n_train}:")
            for k, v in results.items():
                if k not in ['chain_samples_for_diag', 'sigma2_samples']:
                    LOGGER.info(f"  {k}: {v}")

        if exp_params.get("plot_results", True):
            LOGGER.info(f"--- Generating Plots for N_train={current_n_train} ---")
            plot_results(all_results, cfg, run_specific_artifact_dir, current_n_train)
        LOGGER.info(f"Experiment for N_train={current_n_train} complete.")


@hydra.main(config_path="configs", config_name="schedule", version_base=None)
def main(cfg: DictConfig):
    run_and_analyze(cfg)


if __name__ == "__main__":
    main()