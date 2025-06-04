import os
import yaml
import hydra
import logging
import numpy as np
import pandas as pd
import wandb

import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import train_test_split


from experiments import LOGGER

from bart_playground import DataGenerator
from bart_playground.bart import DefaultBART


# set seaborn stype for an academic paper
sns.set_context("paper")
logging.basicConfig(level=logging.INFO)
FS = 16

plt.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "figure.dpi": 300,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "legend.title_fontsize": FS,
        "font.size": FS,
        # 'lines.linewidth': 2,s
    }
)


def get_gelman_rubin_statistic(preds_chains):
    _, n_chains, n_post = preds_chains.shape
    if n_chains == 1:
        return -1.0
    mu_chains = np.mean(preds_chains, axis=2)
    mu_all = np.mean(mu_chains, axis=1)
    B = n_post * np.sum((mu_chains - mu_all[:, None]) ** 2, axis=1) / (n_chains - 1)
    W = np.mean(np.var(preds_chains, axis=2, ddof=1), axis=1)
    R = (B / n_post + W * (n_post / (n_post - 1))) / W
    mean_R = np.mean(R)
    return float(mean_R)


def get_bart_rmse_and_coverage_gr(X, y, params, random_state=42):
    proposal_probs = params["proposal_probs"]
    n_trees = params["n_trees"]
    ndpost = params["ndpost"]
    nskip = params["nskip"]
    alpha = params["alpha"]
    n_chains = params["n_chains"]
    temperature = params["temperature"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    preds_chains = np.zeros((y_test.shape[0], n_chains, ndpost))

    for chain in range(n_chains):
        bart = DefaultBART(
            ndpost=ndpost,
            nskip=nskip,
            n_trees=n_trees,
            proposal_probs=proposal_probs,
            random_state=random_state,
            temperature=temperature,
        )
        bart.fit(X_train, y_train)
        preds = bart.posterior_f(X_test)
        preds_chains[:, chain, :] = preds

    return y_test, preds_chains


def run_main_experiment(cfg: DictConfig):
    dgp = cfg.dgp
    dgp_params = cfg.dgp_params
    bart_params = cfg.bart_params
    generator = DataGenerator(**dgp_params)
    X, y = generator.generate(scenario=dgp)
    y_test, preds_chains = get_bart_rmse_and_coverage_gr(X, y, bart_params)
    return y_test, preds_chains
    # results = {sample_size: {} for sample_size in sample_sizes}
    # for sample_size in sample_sizes:
    #     # chains_results = {
    #     #     "rmse": {c: [] for c in n_chains_range},
    #     #     "coverage": {c: [] for c in n_chains_range},
    #     #     "gelman_rubin": {c: [] for c in n_chains_range},
    #     # }
    #     for n_chains in n_chains_range:
    #         coverage_results = []
    #         rmse_results = []
    #         gelman_rubin_results = []
    #         for _ in tqdm(range(cfg.n_reps), desc=f"Running reps for {n_chains} chains"):
    #             generator = DataGenerator(**dgp_params)
    #             X, y = generator.generate(scenario=dgp)
    #             bart_params["n_chains"] = n_chains
    #             rmse, coverage, gelman_rubin = get_bart_rmse_and_coverage_gr(X, y, bart_params)
    #             rmse_results.append(float(rmse))
    #             coverage_results.append(float(coverage))
    #             gelman_rubin_results.append(float(gelman_rubin))
    #         chains_results["rmse"][n_chains] = rmse_results
    #         chains_results["coverage"][n_chains] = coverage_results
    #         chains_results["gelman_rubin"][n_chains] = gelman_rubin_results
    #     results[sample_size] = chains_results
    # return results


def run_and_analyze(cfg: DictConfig):
    # Initialize wandb
    # set the experiment name to have the dgp, temperature, nskip alpha and n_trees
    experiment_name = f"mixing_paper_{cfg.dgp}_temp_{cfg.bart_params.temperature}_nskip_{cfg.bart_params.nskip}_alpha_{cfg.bart_params.alpha}_n_trees_{cfg.bart_params.n_trees}"
    wandb.init(
        project="bart-playground",
        entity="bart_playground",
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["mixing_paper", cfg.dgp, f"temp_{cfg.bart_params.temperature}"],
    )

    artifacts_dir = cfg.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    experiment_dir = os.path.join(artifacts_dir, f"temperature_{cfg.bart_params.temperature}", cfg.dgp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Run single experiment
    y_test, preds_chains = run_main_experiment(cfg)

    # Save configuration as YAML
    config_file = os.path.join(experiment_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False)

    # Get dimensions
    n_samples, n_chains, n_post = preds_chains.shape

    # Save y_true as CSV for analysis
    y_true_df = pd.DataFrame({"y_true": y_test})
    y_true_file = os.path.join(experiment_dir, "y_true.csv")
    y_true_df.to_csv(y_true_file, index=False)

    # Save each chain's predictions as CSV for analysis
    chain_files = []
    for chain_idx in range(n_chains):
        chain_preds = preds_chains[:, chain_idx, :]
        chain_df = pd.DataFrame(chain_preds, columns=[f"posterior_sample_{i}" for i in range(n_post)])
        chain_df["sample_index"] = np.arange(n_samples)
        chain_file = os.path.join(experiment_dir, f"predictions_chain_{chain_idx}.csv")
        chain_df.to_csv(chain_file, index=False)
        chain_files.append(chain_file)

    # Log basic statistics to wandb
    mean_preds = np.mean(preds_chains, axis=(1, 2))
    rmse = np.sqrt(np.mean((mean_preds - y_test) ** 2))
    wandb.log({"rmse": rmse, "n_samples": n_samples, "n_chains": n_chains, "n_posterior_samples": n_post})

    # Create wandb artifact with both CSV files and config
    artifact = wandb.Artifact(
        name=f"bart_results_{cfg.dgp}_temp_{cfg.bart_params.temperature}",
        type="experiment_data",
        description=f"BART experiment data for {cfg.dgp} with temperature {cfg.bart_params.temperature}",
    )
    artifact.add_file(config_file, name="config.yaml")
    artifact.add_file(y_true_file, name="y_true.csv")

    # Add each chain CSV file to the artifact
    for chain_idx, chain_file in enumerate(chain_files):
        artifact.add_file(chain_file, name=f"predictions_chain_{chain_idx}.csv")

    # Log the artifact
    wandb.log_artifact(artifact)

    # Finish wandb run
    wandb.finish()


@hydra.main(config_path="configs", config_name="bart_run", version_base=None)
def main(cfg: DictConfig):
    run_and_analyze(cfg)


if __name__ == "__main__":
    main()
