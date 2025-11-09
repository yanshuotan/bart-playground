

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from bart_playground import DataGenerator
from bart_playground.bart import DefaultBART


def main():
    proposal_probs = {"grow" : 0.5,
                  "prune" : 0.5}
    n_chains = 2
    generator = DataGenerator(n_samples=160, n_features=2, noise=0.1, random_seed=42)
    X, y = generator.generate(scenario="piecewise_flat")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    y_test = np.expand_dims(y_test, axis=1)
    plot_data = {}

    for chain in range(n_chains):
        bart = DefaultBART(ndpost=12, nskip=20, n_trees=100, proposal_probs=proposal_probs, random_state=chain)
        bart.fit(X_train, y_train)
        preds = bart.posterior_f(X_test)
        mean_preds = np.mean(preds, axis=1)
        mse_squence = np.mean((mean_preds - y_test)**2, axis=0)
        rmse_sequence = np.sqrt(mse_squence)
        plot_data[chain] = np.squeeze(rmse_sequence)
        print(f"Chain {chain} RMSE: {rmse_sequence[-1]}")

    # do a plot for the RMSE sequence, each chain is a different line in a different color
    fig, ax = plt.subplots()
    for chain in range(n_chains):
        ax.plot(plot_data[chain], label=f"Chain {chain}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig("rmse_sequence.png")

if __name__ == "__main__":
    main()