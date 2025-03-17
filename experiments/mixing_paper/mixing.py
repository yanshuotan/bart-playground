import os

import yaml
from experiments import LOGGER
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

FS = 16

# set seaborn stype for an academic paper
sns.set_context("paper")

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


def get_gelman_rubin(X, y, params):
    proposal_probs = params["proposal_probs"]
    n_trees = params["n_trees"]
    ndpost = params["ndpost"]
    nskip = params["nskip"]
    n_chains = params["n_chains"]
    random_state = np.random.randint(0, 1000000)
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
    return get_gelman_rubin_statistic(preds_chains)


def get_gelman_rubin_statistic(preds_chains):
    n_samples, n_chains, n_post = preds_chains.shape
    mu_chains = np.mean(preds_chains, axis=2)
    mu_all = np.mean(mu_chains, axis=1)
    B = n_post * np.sum((mu_chains - mu_all[:, None]) ** 2, axis=1) / (n_chains - 1)
    W = np.mean(np.var(preds_chains, axis=2), axis=1)
    R = (B/n_post + W * (n_post / (n_post - 1))) / W
    mean_R = np.mean(R)
    return float(mean_R)


def run_main_experiment(cfg: DictConfig):
    sample_sizes = cfg.sample_sizes
    dgps = cfg.dgps
    
    dgp_params = cfg.dgp_params
    bart_params = cfg.bart_params
    n_reps = cfg.n_reps
    temperature = cfg.bart_params.temperature

    artifacts_dir = os.path.join(cfg.artifacts_dir, f"temperature_{temperature}")
    os.makedirs(artifacts_dir, exist_ok=True)
    results = {}    
    for dgp in dgps:
        results[str(dgp)] = {}
        for sample_size in sample_sizes:
            sample_size_dir = os.path.join(artifacts_dir, dgp, f"s_{sample_size}")
            os.makedirs(sample_size_dir, exist_ok=True)
            results_file = os.path.join(sample_size_dir, "results.yaml")
            if os.path.exists(results_file):
                # LOGGER.info(f"Loading results from {results_file}")
                with open(results_file, "r") as f:
                    data = yaml.load(f, Loader=yaml.SafeLoader)
                    results[str(dgp)][str(sample_size)] = data
                    continue
            gr_vec = []
            dgp_params["n_samples"] = sample_size
            for rep in tqdm(range(n_reps), desc="Running Repetitions"):
                generator = DataGenerator(**dgp_params)
                X, y = generator.generate(scenario=dgp)
                gr_vec.append(float(get_gelman_rubin(X, y, bart_params)))
            results[str(dgp)][str(sample_size)] = gr_vec
            with open(results_file, "w") as f:
                yaml.dump(gr_vec, f)
    results_out = {}
    for dgp in dgps:
        results_out[str(dgp)] = {s: np.mean(results[str(dgp)][str(s)]) for s in sample_sizes}
    return results_out

def run_and_analyze(cfg: DictConfig):
    artifacts_dir = cfg.artifacts_dir

    results = run_main_experiment(cfg)
    LOGGER.info(results)
    fig, ax = plt.subplots()
    for dgp in cfg.dgps:
        r_dgp = results[str(dgp)]
        samples = list(r_dgp.keys())
        gr_values = list(r_dgp.values())
        sns.lineplot(x=samples, y=gr_values, label=dgp)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Gelman-Rubin Statistic")
    ax.legend()
    plt.savefig(os.path.join(artifacts_dir, f"temperature_{cfg.bart_params.temperature}", "gelman_rubin_density.png"))
    plt.close()

@hydra.main(config_path="configs", config_name="mixing")
def main(cfg: DictConfig):
    run_and_analyze(cfg)

if __name__ == "__main__":
    main()
