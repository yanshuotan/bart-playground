import os
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

def _get_correct_first_split(dgp):
    if dgp == "dgp_1":
        return 0
    elif dgp == "dgp_2":
        return 0
    else:
        raise ValueError(f"Unknown DGP: {dgp}")


def get_bcart_first_split(X, y, params):
    proposal_probs = params["proposal_probs"]
    n_trees = params["n_trees"]
    ndpost = params["ndpost"]
    nskip = params["nskip"]
    n_chains = params["n_chains"]
    temperature = params["temperature"]
    random_state = np.random.randint(0, 1000000)
    bart = DefaultBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        proposal_probs=proposal_probs,
        random_state=random_state,
        temperature=temperature,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    for chain in range(n_chains):
        bart.fit(X_train, y_train, quietly=True)
        first_splits = [t.trees[0].vars[0] for t in bart.trace]
    return np.array(first_splits)



def run_main_experiment(cfg: DictConfig):
    sample_sizes = cfg.sample_sizes
    dgp = cfg.dgp
    
    dgp_params = cfg.dgp_params
    bart_params = cfg.bart_params
    n_reps = cfg.n_reps

    results = {"pct_correct": {}, "changes_in_first_split": {}}
    correct_first_split = _get_correct_first_split(dgp)
    for sample_size in tqdm(sample_sizes, desc="Running Sample Sizes"):
        pct_correct = []
        changes_in_first_split = []
        dgp_params["n_samples"] = sample_size
        for _ in tqdm(range(n_reps), desc="Fitting BART Models"):
            
            # Generate data and run BART.
            generator = DataGenerator(**dgp_params)
            X, y = generator.generate(scenario=dgp)
            first_splits = get_bcart_first_split(X, y, bart_params)
            is_correct_vector = first_splits == correct_first_split
            pct_correct.append(float(np.mean(is_correct_vector)))
            n_changes = int(np.sum(first_splits[1:] != first_splits[:-1]))
            changes_in_first_split.append(n_changes)
        results["pct_correct"][str(sample_size)] = pct_correct
        results["changes_in_first_split"][str(sample_size)] = changes_in_first_split
    return results

def run_and_analyze(cfg: DictConfig):
    artifacts_dir = cfg.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    results_file = os.path.join(artifacts_dir, "results.yaml")

    # Load existing results if available.
    if os.path.exists(results_file):
        data = OmegaConf.load(results_file)
    else:
        data = run_main_experiment(cfg)
        OmegaConf.save(data, results_file)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(cfg.sample_sizes)))

    fig, ax = plt.subplots()
    for i, sample_size in enumerate(cfg.sample_sizes):
        # do a histogram of the number of changes in the first split
        # change to integer
        changes_in_first_split = [int(x) for x in data["changes_in_first_split"][str(sample_size)]]
        sns.histplot(changes_in_first_split, label=f"{sample_size} samples", color=colors[i])
    ax.legend()
    plt.savefig(os.path.join(artifacts_dir, "changes_in_first_split_histogram.png"))
    plt.close()

    fig, ax = plt.subplots()
    for i, sample_size in enumerate(cfg.sample_sizes):
        sns.kdeplot(data["pct_correct"][str(sample_size)], label=f"{sample_size} samples", color=colors[i])
    ax.legend()
    plt.savefig(os.path.join(artifacts_dir, "pct_correct_density.png"))
    plt.close()

@hydra.main(config_path="configs", config_name="bcart_root")
def main(cfg: DictConfig):
    run_and_analyze(cfg)

if __name__ == "__main__":
    main()
