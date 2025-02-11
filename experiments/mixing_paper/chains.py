import os
import yaml
import hydra

import numpy as np
import pandas as pd

import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import train_test_split

from bart_playground import DataGenerator
from bart_playground.bart import DefaultBART


# set seaborn stype for an academic paper
sns.set_context("paper")

FS = 16

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'axes.labelsize': FS,
    'axes.titlesize': FS,
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'legend.fontsize': FS,
    'legend.title_fontsize': FS,
    'font.size': FS,
    # 'lines.linewidth': 2,s

})


def get_bart_rmse_and_coverage(X, y, params, random_state=42):
    proposal_probs = params["proposal_probs"]
    n_trees = params["n_trees"]
    ndpost = params["ndpost"]
    nskip = params["nskip"]
    alpha = params["alpha"]
    n_chains = params["n_chains"]
    temperature = params["temperature"]
    
    bart = DefaultBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        proposal_probs=proposal_probs,
        random_state=random_state,
        temperature=temperature
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    preds_chains = np.zeros((y_test.shape[0], n_chains, ndpost))
    
    for chain in range(n_chains):
        bart.fit(X_train, y_train, quietly=True)
        preds = bart.posterior_f(X_test)
        preds_chains[:, chain, :] = preds

    # Average predictions over chains and posterior samples.
    average_preds = np.mean(preds_chains, axis=(1, 2))
    rmse = np.sqrt(np.mean((average_preds - y_test) ** 2))
    
    # Compute the symmetric prediction interval.
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile
    lower_bound = np.quantile(preds_chains, lower_quantile, axis=(1, 2))
    upper_bound = np.quantile(preds_chains, upper_quantile, axis=(1, 2))
    
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    return float(rmse), float(coverage)

def run_main_experiment(cfg: DictConfig):
    sample_sizes = cfg.sample_sizes
    dgp = cfg.dgp
    dgp_params = cfg.dgp_params
    bart_params = cfg.bart_params
    n_chains_range = cfg.n_chains_range
    results = {sample_size: {} for sample_size in sample_sizes}
    for sample_size in sample_sizes:
        chains_results = {"rmse": [], "coverage": []}
        for n_chains in n_chains_range:
            coverage_results = []
            rmse_results = []
            for _ in range(cfg.n_reps):
            
                generator = DataGenerator(**dgp_params)
                X, y = generator.generate(scenario=dgp)
                bart_params["n_chains"] = n_chains
                rmse, coverage = get_bart_rmse_and_coverage(X, y, bart_params)
                rmse_results.append(rmse)
                coverage_results.append(coverage)
            chains_results["rmse"] = rmse_results
            chains_results["coverage"] = coverage_results
        results[sample_size] = chains_results
    return results

def run_and_analyze(cfg: DictConfig):
    artifacts_dir = cfg.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    experiment_dir = os.path.join(artifacts_dir, cfg.dgp)   
    os.makedirs(experiment_dir, exist_ok=True)
    results_file = os.path.join(experiment_dir, "results.yaml")

    # Load existing results if available.
    if os.path.exists(results_file):
        results = yaml.load(results_file)
    else:
        results = run_main_experiment(cfg)
        with open(results_file, "w") as f:
            yaml.dump(results, f)

    # make for every number of chains plot of rmse and coverage as a function of sample size
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(cfg.n_chains_range)))
    rmse_vec_one_chain = [results[sample_size][1]["rmse"] for sample_size in cfg.sample_sizes]
    fig, ax = plt.subplots()
    for n_chains, color in zip(cfg.n_chains_range, colors):
        rmse_vec = [results[sample_size][n_chains]["rmse"] for sample_size in cfg.sample_sizes]
        relative_rmse_vec = [rmse_vec[i] / rmse_vec_one_chain[i] for i in range(len(rmse_vec))]
        # plot using seaborn
        sns.lineplot(x=cfg.sample_sizes, y=relative_rmse_vec, label=f"{n_chains}", ax=ax, color=color)
    ax.legend(title="Chains")
    plt.savefig(os.path.join(experiment_dir, f"n_chains_{n_chains}.png"))
    plt.close()

    # make plot of coverage as a function of sample size for each number of chains
    for n_chains in cfg.n_chains_range:
        fig, ax = plt.subplots()
        coverage_vec = [results[sample_size][n_chains]["coverage"] for sample_size in cfg.sample_sizes]
        sns.lineplot(x=cfg.sample_sizes, y=coverage_vec, label=f"{n_chains}", ax=ax, color=color)
    # add horizontal line at y=0.95
    ax.axhline(y=0.95, color="red", linestyle="--")
    ax.legend(title="Chains")
    plt.savefig(os.path.join(experiment_dir, f"n_chains_{n_chains}.png"))
    plt.close()

@hydra.main(config_path="configs", config_name="chains")
def main(cfg: DictConfig):
    run_and_analyze(cfg)

if __name__ == "__main__":
    main()
