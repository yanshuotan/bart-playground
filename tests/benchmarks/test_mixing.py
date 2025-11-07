import numpy as np
import pytest

from pmlb import fetch_data
from sklearn.datasets import fetch_california_housing

from bart_playground import DefaultBART  # our bart implementation
import bartz  # bartz package


pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.slow,
    pytest.mark.requires_pmlb,
    pytest.mark.requires_bartz,
]

# --- Helper: Gelman-Rubin (PSRF) statistic ---
def gelman_rubin(chains):
    """
    Compute the Gelman–Rubin statistic (PSRF) for a set of MCMC chains.

    Args:
        chains (np.ndarray): Array of shape (m, n) with m chains and n iterations.

    Returns:
        float: The PSRF.
    """
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    V = (n - 1) / n * W + B / n
    return np.sqrt(V / W)


def autocorrelation(chain, lag):
    """Compute the sample autocorrelation at the given lag."""
    n = len(chain)
    if lag >= n:
        return np.nan
    return np.corrcoef(chain[:-lag], chain[lag:])[0, 1]


def effective_sample_size(chains, step=1):
    """
    Compute the total effective sample size (ESS) across all chains,
    using a down-sampling parameter 'step' for the lag calculation.

    For each chain, ESS is computed as:
       ESS = n / (1 + 2 * sum_{lag=1, lag+=step}^{L} (step * ρ(lag)))
    where the summation stops at the first negative autocorrelation.

    Args:
        chains (np.ndarray): Array of shape (m, n) with m chains and n iterations.
        step (int): Down-sampling step for computing autocorrelations.

    Returns:
        float: Total effective sample size summed over all chains.
    """
    m, n = chains.shape
    total_ess = 0.0
    for chain in chains:
        ac_sum = 0.0
        for lag in range(1, n, step):
            ac = autocorrelation(chain, lag)
            if ac < 0:
                break
            ac_sum += step * ac
        total_ess += n / (1 + 2 * ac_sum)
    return total_ess


def geweke(chain, first=0.1, last=0.5):
    """
    Compute a simple Geweke z-score comparing the first 10% and last 50% of the chain.

    Args:
        chain (np.ndarray): 1D array of MCMC samples.
        first (float): Fraction of early iterations.
        last (float): Fraction of later iterations.

    Returns:
        float: Geweke z-score.
    """
    n = len(chain)
    n_first = int(first * n)
    n_last = int(last * n)
    mean_first = np.mean(chain[:n_first])
    mean_last = np.mean(chain[-n_last:])
    var_first = np.var(chain[:n_first], ddof=1)
    var_last = np.var(chain[-n_last:], ddof=1)
    z = (mean_first - mean_last) / np.sqrt(var_first / n_first + var_last / n_last)
    return z


def average_geweke(chains):
    """Compute the average absolute Geweke z-score over all chains."""
    return np.mean([abs(geweke(chain)) for chain in chains])


# --- Helper functions for running a single chain ---

def run_chain_bart(X, y, ndpost=100, n_trees=20, seed=0):
    """
    Run one MCMC chain using our bart implementation.
    Returns posterior samples (after 200 burn-in, 100 posterior samples)
    for the first observation.

    The proposal probabilities are set to 50% grow and 50% prune.
    """
    model = DefaultBART(
        ndpost=ndpost,
        nskip=200,
        n_trees=n_trees,
        random_state=seed,
        proposal_probs={'grow': 0.5, 'prune': 0.5}
    )
    model.fit(X, y)
    post = model.posterior_f(X)
    if post.shape[0] == X.shape[0]:
        samples = post[0, :]
    else:
        samples = post[:, 0]
    return samples


class BartzWrapper:
    """
    Wraps the bartz gbart output to provide a uniform interface.
    """

    def __init__(self, model):
        self.model = model

    def posterior_predictive(self, X):
        # bartz expects predictors as (p, n)
        return self.model.predict(X.T)


def run_chain_bartz(X, y, ndpost=100, n_trees=20, seed=0):
    """
    Run one MCMC chain using the bartz algorithm.
    Returns posterior samples (after 200 burn-in, 100 posterior samples)
    for the first observation.
    """
    X_t = X.T
    model = bartz.BART.gbart(
        X_t, y,
        x_test=X_t,
        ndpost=ndpost,
        nskip=200,
        ntree=n_trees,
        seed=seed
    )
    wrapper = BartzWrapper(model)
    post = wrapper.posterior_predictive(X)
    samples = post[:, 0]
    return samples


# --- Dataset metadata ---
DATASETS = [
    {"name": "1201_BNG_breastTumor", "n_features": 9, "source": "pmlb"},
    {"name": "california_housing", "n_features": 8, "source": "sklearn"},
    {"name": "1199_BNG_echoMonths", "n_features": 9, "source": "pmlb"},
    {"name": "294_satellite_image", "n_features": 36, "source": "pmlb"},
]


@pytest.mark.parametrize("dataset_info", DATASETS)
def test_mixing_diagnostics(dataset_info):
    # Load dataset based on the source.
    if dataset_info["source"] == "sklearn" and dataset_info["name"] == "california_housing":
        data_bunch = fetch_california_housing(return_X_y=False, as_frame=True)
        df = data_bunch.frame
    else:
        df = fetch_data(dataset_info["name"], return_X_y=False)

    # Subsample to at most 100 samples.
    if df.shape[0] > 100:
        df = df.sample(n=100, random_state=42)

    # Assume the target is the last column.
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    assert X.shape[1] == dataset_info["n_features"], (
        f"Dataset {dataset_info['name']} expected {dataset_info['n_features']} features but got {X.shape[1]}"
    )

    n_chains = 4
    ndpost = 100  # 100 posterior samples after burn-in
    n_trees = 10

    chains_bart = []
    chains_bartz = []
    for i in range(n_chains):
        chains_bart.append(run_chain_bart(X, y, ndpost=ndpost, n_trees=n_trees, seed=42 + i))
        chains_bartz.append(run_chain_bartz(X, y, ndpost=ndpost, n_trees=n_trees, seed=42 + i))

    chains_bart = np.array(chains_bart)  # shape: (n_chains, 100)
    chains_bartz = np.array(chains_bartz)
    total_samples = chains_bart.shape[0] * chains_bart.shape[1]

    Rhat_bart = gelman_rubin(chains_bart)
    Rhat_bartz = gelman_rubin(chains_bartz)
    ess_bart = effective_sample_size(chains_bart, step=2)
    ess_bartz = effective_sample_size(chains_bartz, step=2)
    geweke_bart = average_geweke(chains_bart)
    geweke_bartz = average_geweke(chains_bartz)

    print(f"Dataset {dataset_info['name']}:")
    print(f"  BART   -> PSRF: {Rhat_bart:.3f}, ESS: {ess_bart:.1f}/{total_samples}, Geweke: {geweke_bart:.3f}")
    print(f"  Bartz  -> PSRF: {Rhat_bartz:.3f}, ESS: {ess_bartz:.1f}/{total_samples}, Geweke: {geweke_bartz:.3f}")

    assert Rhat_bart < 1.1, f"BART mixing poor for {dataset_info['name']}: PSRF={Rhat_bart:.3f}"
    assert Rhat_bartz < 1.1, f"Bartz mixing poor for {dataset_info['name']}: PSRF={Rhat_bartz:.3f}"
    assert ess_bart > 0.5 * total_samples, f"BART ESS too low for {dataset_info['name']}: ESS={ess_bart:.1f}"
    assert ess_bartz > 0.5 * total_samples, f"Bartz ESS too low for {dataset_info['name']}: ESS={ess_bartz:.1f}"
    assert geweke_bart < 2, f"BART Geweke diagnostic indicates non-convergence for {dataset_info['name']}: {geweke_bart:.3f}"
    assert geweke_bartz < 2, f"Bartz Geweke diagnostic indicates non-convergence for {dataset_info['name']}: {geweke_bartz:.3f}"
