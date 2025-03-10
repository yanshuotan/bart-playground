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


def get_bart_rmse_and_coverage(X, y, params):
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
        random_state=42,
        temperature=temperature
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
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
    return rmse, coverage

def _convert_results_to_df(results):
    data = {}
    for sample_size, _results in results.items():
        for setup_str, (rmse, coverage) in _results.items():
            n_trees = int(setup_str.split("_")[1])
            dgp = setup_str.split("_")[-1]
            new_str = f"s_{sample_size}_t_{n_trees}_dgp_{dgp}"
            data[new_str] = (rmse, coverage)
    return pd.DataFrame.from_dict(data, orient="index", columns=["rmse", "coverage"])

def run_main_experiment(cfg: DictConfig):
    sample_sizes = cfg.sample_sizes
    n_trees_range = cfg.n_trees_range
    dgp = cfg.dgp
    dgp_params = cfg.dgp_params
    bart_params = cfg.bart_params
    temperature = cfg.bart_params.temperature
    artifacts_dir = os.path.join(cfg.artifacts_dir, f"temperature_{temperature}")
    os.makedirs(artifacts_dir, exist_ok=True)
    results = {s: {} for s in sample_sizes}
    for sample_size in sample_sizes:
        sample_size_dir = os.path.join(artifacts_dir, f"s_{sample_size}")
        os.makedirs(sample_size_dir, exist_ok=True)
        results_yaml_file = os.path.join(sample_size_dir, "results.yaml")
        if os.path.exists(results_yaml_file):
            with open(results_yaml_file, "r") as f:
                results = yaml.load(f)
                results[sample_size] = results
                continue
        _results = {}
        for n_trees in tqdm(n_trees_range, desc="Running Number of Trees"):
            # Update the number of trees for this experiment run.
            bart_params["n_trees"] = n_trees
            # print(f"Running experiment with {sample_size} samples and {n_trees} trees.")
            
            # Generate data and run BART.
            generator = DataGenerator(**dgp_params)
            X, y = generator.generate(scenario=dgp)
            rmse, coverage = get_bart_rmse_and_coverage(X, y, bart_params)
            
            setup_str = f"t_{n_trees}_dgp_{dgp}"
            _results[setup_str] = (rmse, coverage)
        results[sample_size] = _results
        # Save the results to a YAML file.
        with open(results_yaml_file, "w") as f:
            yaml.dump(_results, f)
    data_df = _convert_results_to_df(results)
    return data_df

def run_and_analyze(cfg: DictConfig):
    artifacts_dir = cfg.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    results_file = os.path.join(artifacts_dir, "results.csv")

    # Load existing results if available.
    if os.path.exists(results_file):
        data = pd.read_csv(results_file, index_col=0)
        data = data.to_dict(orient="index")
    else:
        data = run_main_experiment(cfg)
        # Convert dictionary to DataFrame for saving.
        data_df = pd.DataFrame.from_dict(data, orient="index", columns=["rmse", "coverage"])
        data_df.to_csv(results_file)

    # Create plots for RMSE and Coverage.
    def _make_plot(quantity, filename):
        fig, ax = plt.subplots()
        idx = 0 if quantity == "RMSE" else 1
        # I want a range of blues for the different sample sizes.
        colors = plt.cm.Blues(np.linspace(0.3, 1, len(cfg.sample_sizes)))
        for i, sample_size in enumerate(cfg.sample_sizes):
            values = [
                data[f"s_{sample_size}_t_{n_trees}_dgp_{cfg.dgp}"][idx]
                for n_trees in cfg.n_trees_range
            ]
            n_ks = int(sample_size / 1000)
            lbl = f"{n_ks}k"
            sns.lineplot(x=cfg.n_trees_range, y=values, label=lbl, color=colors[i])
        ax.set_xlabel("Number of trees")
        ax.set_ylabel(quantity)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, filename))
        plt.close()

    _make_plot("RMSE", "rmse_plot.png")
    _make_plot("Coverage", "coverage_plot.png")

@hydra.main(config_path="configs", config_name="additive")
def main(cfg: DictConfig):
    run_and_analyze(cfg)

if __name__ == "__main__":
    main()
