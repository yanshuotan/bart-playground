import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
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
from bart_playground.samplers import default_proposal_probs

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
    R = (B/n_post + W * (n_post / (n_post - 1))) / W
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
                   cfg: DictConfig, seed: int, init_from_xgb: bool):
    """
    Runs one chain of BART.
    """
    bart_cfg = cfg.bart_params
    xgb_cfg = cfg.get('xgb_params', OmegaConf.create({'max_depth': 4, 'learning_rate': 0.2}))

    preprocessor = DefaultPreprocessor(max_bins=X_train_raw.shape[0])
    train_data = preprocessor.fit_transform(X_train_raw, y_train_raw)

    # X_test_raw is passed directly to evaluate and posterior_f, as per initialization_experiment.py
    # No explicit X_test_scaled = preprocessor.transform(X_test_raw) call needed here.

    model = DefaultBART(
        ndpost=bart_cfg.ndpost,
        nskip=bart_cfg.nskip,
        n_trees=bart_cfg.n_trees,
        random_state=seed,
        proposal_probs=bart_cfg.get('proposal_probs', default_proposal_probs),
        temperature=bart_cfg.get('temperature', 1.0),
        dirichlet_prior=False
    )
    # Manually set data and preprocessor as in original script for init_from_xgboost
    model.preprocessor = preprocessor
    model.data = train_data  # BART model expects preprocessed data
    model.sampler.add_data(train_data)
    model.sampler.add_thresholds(preprocessor.thresholds)
    model.is_fitted = True  # To allow init_from_xgboost

    if init_from_xgb:
        xgb_model_instance = xgb.XGBRegressor(
            n_estimators=bart_cfg.n_trees,  # Match BART trees
            max_depth=xgb_cfg.max_depth,
            learning_rate=xgb_cfg.learning_rate,
            random_state=seed,
            tree_method="exact",  # As in original
            grow_policy="depthwise",  # As in original
            base_score=0.0  # As in original
        )
        # XGBoost should be trained on the same preprocessed X as BART will use internally for init
        xgb_model_instance.fit(train_data.X, train_data.y)  # Use preprocessed X,y
        model.init_from_xgboost(xgb_model_instance, train_data.X, train_data.y, debug=cfg.get('debug_xgb_init', False))

    # Capture initial BART prediction (scaled)
    # get_init_state() gets the state *before* any MCMC steps by the .run() method
    # If init_from_xgb=True, this state is the one derived from XGBoost.
    # If init_from_xgb=False, this is a random initial state (e.g. stumps).
    init_params = model.sampler.get_init_state()
    # Evaluate expects raw X, output is on potentially scaled y-axis
    init_pred_values_on_model_scale = init_params.evaluate(X_test_raw)
    init_pred = preprocessor.backtransform_y(init_pred_values_on_model_scale)  # Backtransform y
    # init_mse = mean_squared_error(y_test_raw, init_pred) # Already removed

    # Run sampler
    total_iters_for_sampler = bart_cfg.ndpost + bart_cfg.nskip
    start_time = time.perf_counter()
    # The sampler's run method handles nskip internally.
    # We pass ndpost to DefaultBART, which then passes total_iters and nskip to sampler.run
    # However, here we are calling sampler.run directly after manual setup.
    # model.sampler.run takes total iterations and then n_skip *within those iterations*
    # DefaultBART's constructor receives ndpost and nskip.
    # model.fit would call model.sampler.run(self.ndpost + self.nskip, n_skip=self.nskip)

    # For direct sampler call after manual setup and potential XGB init:
    # The trace from init_from_xgboost is a single state.
    # We want to run 'ndpost' more iterations, treating previous as burn-in effectively.
    # The original script's DefaultBART used nskip=0, so all iterations were kept.
    # Let's match DefaultBART's nskip behavior for clarity.

    trace = model.sampler.run(
        n_iter=bart_cfg.ndpost + bart_cfg.nskip,  # Total MCMC iterations
        quietly=True,
        n_skip=bart_cfg.nskip  # Sampler will discard these initial samples
    )
    runtime = time.perf_counter() - start_time
    model.trace = trace  # Store the trace in the model

    # Posterior predictions on RAW test X.
    # model.posterior_f internally calls preprocessor.backtransform_y.
    post_raw = model.posterior_f(X_test_raw)

    # Ensure post_raw is (n_test_samples, n_draws)
    if post_raw.shape[0] != X_test_raw.shape[0] and post_raw.shape[1] == X_test_raw.shape[0]:
        post_raw = post_raw.T

    # Manual back-transformation loop is NOT needed here as posterior_f handles it.

    preds_mean_raw = np.mean(post_raw, axis=1)
    final_mse = mean_squared_error(y_test_raw, preds_mean_raw)

    lower_percentile = np.percentile(post_raw, 2.5, axis=1)
    upper_percentile = np.percentile(post_raw, 97.5, axis=1)
    coverage = np.mean((y_test_raw >= lower_percentile) & (y_test_raw <= upper_percentile))

    # For diagnostics, use one of the posterior samples (e.g., for one test point)
    # The original script used post[0, :], which is the posterior draws for the first test point.
    # This should be from *scaled* predictions if diagnostics are on scaled values, or raw if on raw.
    # For ACF of f(x), raw predictions are fine.
    # Using raw predictions for the first test point for ACF.
    chain_sample_for_diag = post_raw[0, :] if post_raw.shape[0] > 0 else np.array([])

    return {
        'final_mse': final_mse,
        'coverage': coverage,
        'chain_sample_for_diag': chain_sample_for_diag,  # Samples for one observation point
        'runtime': runtime,
        'sigma2_samples': np.array([p.global_params['eps_sigma2'] for p in trace]),  # Store sigma^2 samples
        'post_raw': post_raw
    }


# --- Run Multi-chain Experiment ---
def run_full_experiment(X_train_specific, y_train_specific, X_test_specific, y_test_specific,
                        cfg: DictConfig, init_from_xgb: bool, current_n_train: int):
    exp_cfg = cfg.experiment_params
    all_results = []

    # This function now receives already split data for a specific n_train run
    # The train_test_split is now done in run_and_analyze before calling this.

    for i in range(exp_cfg.n_chains):
        # Seed for each BART chain: derived from main_seed via hashing + a larger offset + chain index
        # Use a deterministic hashing function (hashlib) instead of Python's built-in hash()
        seed_string = f"{exp_cfg.main_seed}_chain_{i}_ntrain_{current_n_train}"
        seed_hash = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
        chain_specific_seed = int(seed_hash, 16) % (2 ** 32)
        LOGGER.info(
            f"Running chain {i + 1}/{exp_cfg.n_chains} with chain_specific_seed {chain_specific_seed}, init_from_xgb={init_from_xgb}")
        chain_res = run_chain_bart(X_train_specific, X_test_specific, y_train_specific, y_test_specific,
                                   cfg, chain_specific_seed, init_from_xgb)
        all_results.append(chain_res)

    # Aggregate results
    # chain_samples_for_diag will be a list of arrays, stack them for multi-chain diagnostics
    # Each element of all_results['chain_sample_for_diag'] is for one specific chain (ndpost samples for 1st test point)
    # So, aggregated_chain_samples should be (n_chains, ndpost)
    aggregated_chain_samples = np.array(
        [r['chain_sample_for_diag'] for r in all_results if r['chain_sample_for_diag'].size > 0])

    # Each chain's posterior predictions for all test points
    # Stacking to get shape: (n_test_samples, n_chains, n_posterior_draws)
    preds_chains = np.stack([r['post_raw'] for r in all_results], axis=1)

    # Sigma^2 samples from each chain, shape (n_chains, ndpost)
    aggregated_sigma2_samples = np.array([r['sigma2_samples'] for r in all_results if r['sigma2_samples'].size > 0])

    gr_fx, ess_fx = np.nan, np.nan
    
    # Calculate Gelman-Rubin and ESS for each test point's predictions and average
    if preds_chains.ndim == 3 and preds_chains.shape[0] > 0:
        gr_values = [gelman_rubin(preds_chains[i, :, :]) for i in range(preds_chains.shape[0])]
        ess_values = [effective_sample_size(preds_chains[i, :, :]) for i in range(preds_chains.shape[0])]
        
        gr_fx = np.nanmean(gr_values)
        ess_fx = np.nanmean(ess_values)

    results_dict = {
        'final_mse': np.mean([r['final_mse'] for r in all_results]),
        'coverage': np.mean([r['coverage'] for r in all_results]),
        'runtime': np.mean([r['runtime'] for r in all_results]),
        'chain_samples_for_diag': aggregated_chain_samples,  # For f(x_test[0])
        'sigma2_samples': aggregated_sigma2_samples,  # For sigma^2
        'gr_fx': gr_fx,
        'ess_fx': ess_fx
    }
    return results_dict, preds_chains


# --- Plotting Functions ---
def plot_results(all_results, cfg: DictConfig, plot_dir_for_run: str, n_train_samples: int):
    os.makedirs(plot_dir_for_run, exist_ok=True)

    # Bar plots for overall metrics
    metrics_to_plot = ['final_mse', 'coverage', 'runtime']
    labels = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    for metric in metrics_to_plot:
        values = [res[metric] for res in all_results.values()]
        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, color=colors)
        plt.title(f'{metric.replace("_", " ").title()} Comparison for DGP: {cfg.dgp} (N_train={n_train_samples})')
        plt.ylabel(metric.replace("_", " ").title())
        plt.savefig(os.path.join(plot_dir_for_run, f"{cfg.dgp}_ntrain{n_train_samples}_{metric}_comparison.png"))
        plt.close()

    # Diagnostic plots
    diag_cfg_map = all_results

    for config_name, res_data in diag_cfg_map.items():
        # Retrieve pre-calculated diagnostics
        gr_fx = res_data.get('gr_fx')
        ess_fx = res_data.get('ess_fx')

        # Log diagnostics for f(x_test[0])
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

    chain_files = []
    if cfg.experiment_params.get("save_full_chains", False):
        n_samples, n_chains, n_post = preds_chains.shape
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


def run_and_analyze(cfg: DictConfig):
    """
    Main function to run the experiment and analysis.
    Iterates over different training sample sizes, regenerating data for each.
    """
    exp_params = cfg.experiment_params
    main_seed = exp_params.main_seed  # Get the main seed

    # Load the pre-calculated variances
    with open("experiments/data_variances.yaml", 'r') as f:
        data_variances = yaml.safe_load(f)

    dgp_name_safe = "".join(c if c.isalnum() else "_" for c in cfg.dgp)

    # Base directory for the DGP, now includes the main_seed
    dgp_and_seed_base_artifact_dir = os.path.join(cfg.artifacts_dir, dgp_name_safe, f"seed_{main_seed}")
    os.makedirs(dgp_and_seed_base_artifact_dir, exist_ok=True)

    # Save the main config for this seed run at the seed level directory
    OmegaConf.save(cfg, os.path.join(dgp_and_seed_base_artifact_dir, "config_main_seed.yaml"))

    # Generate Test Data (X_test_run, y_test_run_true) - ONCE per main_seed
    dgp_params_test_dict = OmegaConf.to_container(cfg.dgp_params, resolve=True)
    dgp_params_test_dict['n_samples'] = exp_params.n_test_samples
    dgp_params_test_dict['snr'] = np.inf  # Critical: use snr=inf for true y_test
    seed_string_test = f"{main_seed}_dgp_test_n{exp_params.n_test_samples}"
    seed_hash_test = hashlib.sha256(seed_string_test.encode('utf-8')).hexdigest()
    dgp_seed_test = int(seed_hash_test, 16) % (2 ** 32)
    dgp_params_test_dict['random_seed'] = dgp_seed_test
    generator_test = DataGenerator(**dgp_params_test_dict)
    LOGGER.info(
        f"Generating test data (snr=inf): N_test={exp_params.n_test_samples}, DGP params: {dgp_params_test_dict}")
    X_test_run, y_test_run_true = generator_test.generate(scenario=cfg.dgp)
    LOGGER.info(f"Test data generated: X_test shape {X_test_run.shape}, y_test_true shape {y_test_run_true.shape}")

    for idx, current_n_train in enumerate(exp_params.n_train_samples_list):
        LOGGER.info(
            f"--- Starting run for DGP: {cfg.dgp} with N_train = {current_n_train} (run {idx + 1}/{len(exp_params.n_train_samples_list)} for seed {main_seed}) ---")

        # Generate Training Data (X_train_run, y_train_run)
        dgp_params_train_dict = OmegaConf.to_container(cfg.dgp_params, resolve=True)
        dgp_params_train_dict['n_samples'] = current_n_train
        
        # Set noise to the pre-calculated variance for an SNR of 1:3
        dgp_params_train_dict['noise'] = data_variances[cfg.dgp]['required_noise_for_snr_1'] * 3
        
        seed_string_train = f"{main_seed}_dgp_train_n{current_n_train}"
        seed_hash_train = hashlib.sha256(seed_string_train.encode('utf-8')).hexdigest()
        dgp_seed_train = int(seed_hash_train, 16) % (2 ** 32)
        dgp_params_train_dict['random_seed'] = dgp_seed_train

        generator_train = DataGenerator(**dgp_params_train_dict)
        LOGGER.info(f"Generating training data: N_train={current_n_train}, DGP params: {dgp_params_train_dict}")
        X_train_run, y_train_run = generator_train.generate(scenario=cfg.dgp)
        LOGGER.info(f"Training data generated: X_train shape {X_train_run.shape}, y_train shape {y_train_run.shape}")

        # The train_test_split call is no longer needed as data is generated separately.
        LOGGER.info(
            f"Data for N_train={current_n_train} prepared: X_train shape {X_train_run.shape}, y_train shape {y_train_run.shape}, X_test shape {X_test_run.shape}, y_test_true shape {y_test_run_true.shape}")

        # Define artifact directory for this specific run (e.g., dgp/seed_XXX/ntrain_YYY)
        run_specific_artifact_dir = os.path.join(dgp_and_seed_base_artifact_dir, f"ntrain_{current_n_train}")
        os.makedirs(run_specific_artifact_dir, exist_ok=True)

        all_results = {}
        temperatures_to_test = exp_params.get("temperatures", [1.0, 2.0, 5.0])

        for temp in temperatures_to_test:
            result_key = f"Temp_{temp}"
            results_path = os.path.join(run_specific_artifact_dir, f"results_{result_key}.pkl")
            if os.path.exists(results_path):
                LOGGER.info(f"Results file already exists, skipping: {results_path}")
                with open(results_path, "rb") as f:
                    all_results[result_key] = pickle.load(f)
                continue

            LOGGER.info(f"--- Running Experiment: Temperature={temp} for DGP {cfg.dgp}, N_train={current_n_train} ---")
            
            run_cfg = cfg.copy()
            OmegaConf.set_struct(run_cfg, False)
            run_cfg.bart_params.temperature = temp
            OmegaConf.set_struct(run_cfg, True)

            results, preds_chains = run_full_experiment(X_train_run, y_train_run, X_test_run, y_test_run_true, run_cfg,
                                                  init_from_xgb=False, current_n_train=current_n_train)
            
            if not exp_params.get("save_full_chains", False):
                if 'chain_samples_for_diag' in results:
                    del results['chain_samples_for_diag']
                if 'sigma2_samples' in results:
                    del results['sigma2_samples']
            
            all_results[result_key] = results

            with open(results_path, "wb") as f:
                pickle.dump(results, f)

            if cfg.experiment_params.get("log_to_wandb", False):
                log_wandb_artifacts(cfg, "temperature", result_key, results, preds_chains,
                                    y_test_run_true, run_specific_artifact_dir, current_n_train)

            LOGGER.info(f"Results ({result_key}) for DGP {cfg.dgp}, N_train={current_n_train}:")
            for k, v in results.items():
                if k not in ['chain_samples_for_diag', 'sigma2_samples']:
                    LOGGER.info(f"  {k}: {v}")

        # Plotting
        if exp_params.get("plot_results", True) and all_results:
            LOGGER.info(f"--- Generating Plots for DGP {cfg.dgp} with N_train={current_n_train} ---")
            plot_results(all_results, cfg, run_specific_artifact_dir, current_n_train)

        LOGGER.info(
            f"Experiment for DGP {cfg.dgp} with N_train={current_n_train} complete. Artifacts saved to {run_specific_artifact_dir}")


@hydra.main(config_path="configs", config_name="temperature", version_base=None)
def main(cfg: DictConfig):
    run_and_analyze(cfg)


if __name__ == "__main__":
    main()