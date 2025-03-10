import time
import numpy as np
import pytest

from pmlb import fetch_data
from sklearn.datasets import fetch_california_housing

from bart_playground import DefaultBART

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
    Compute the total effective sample size (ESS) across all chains.
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

# --- Helper: Run one chain of DefaultBART ---
def run_chain_bart(X, y, ndpost=100, n_trees=20, seed=0, marginalize=False):
    """
    Run one MCMC chain using the DefaultBART implementation.
    Returns:
      - posterior samples (for the first observation)
      - runtime (in seconds) for fitting the model.
    """
    model = DefaultBART(
        ndpost=ndpost,
        nskip=200,
        n_trees=n_trees,
        random_state=seed,
        proposal_probs={'grow': 0.5, 'prune': 0.5},
        marginalize=marginalize
    )
    start_time = time.perf_counter()
    model.fit(X, y)
    runtime = time.perf_counter() - start_time
    post = model.posterior_f(X)
    # Assume the first row corresponds to the first observation
    if post.shape[0] == X.shape[0]:
        samples = post[0, :]
    else:
        samples = post[:, 0]
    return samples, runtime

# --- Dataset metadata ---
DATASETS = [
    {"name": "1201_BNG_breastTumor", "n_features": 9, "source": "pmlb"},
    {"name": "california_housing", "n_features": 8, "source": "sklearn"},
    {"name": "1199_BNG_echoMonths", "n_features": 9, "source": "pmlb"},
    {"name": "294_satellite_image", "n_features": 36, "source": "pmlb"},
]

@pytest.mark.parametrize("dataset_info", DATASETS)
def test_runtime_and_mixing(dataset_info):
    # --- Load dataset ---
    if dataset_info["source"] == "sklearn" and dataset_info["name"] == "california_housing":
        data_bunch = fetch_california_housing(return_X_y=False, as_frame=True)
        df = data_bunch.frame
    else:
        df = fetch_data(dataset_info["name"], return_X_y=False)

    # Subsample to at most 100 samples.
    if df.shape[0] > 100:
        df = df.sample(n=100, random_state=42)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    assert X.shape[1] == dataset_info["n_features"], (
        f"Dataset {dataset_info['name']} expected {dataset_info['n_features']} features but got {X.shape[1]}"
    )

    n_chains = 4
    ndpost = 100  # Number of posterior samples (after burn-in)
    n_trees = 10

    # --- Run chains for non-marginalized and marginalized configurations ---
    chains_nomarg = []
    chains_marg = []
    runtimes_nomarg = []
    runtimes_marg = []
    for i in range(n_chains):
        samples, runtime = run_chain_bart(X, y, ndpost=ndpost, n_trees=n_trees, seed=42 + i, marginalize=False)
        chains_nomarg.append(samples)
        runtimes_nomarg.append(runtime)

        samples, runtime = run_chain_bart(X, y, ndpost=ndpost, n_trees=n_trees, seed=42 + i, marginalize=True)
        chains_marg.append(samples)
        runtimes_marg.append(runtime)

    chains_nomarg = np.array(chains_nomarg)  # shape: (n_chains, ndpost)
    chains_marg = np.array(chains_marg)
    total_samples = chains_nomarg.shape[0] * chains_nomarg.shape[1]

    # --- Compute mixing diagnostics ---
    Rhat_nomarg = gelman_rubin(chains_nomarg)
    Rhat_marg = gelman_rubin(chains_marg)
    ess_nomarg = effective_sample_size(chains_nomarg, step=2)
    ess_marg = effective_sample_size(chains_marg, step=2)
    geweke_nomarg = average_geweke(chains_nomarg)
    geweke_marg = average_geweke(chains_marg)

    avg_runtime_nomarg = np.mean(runtimes_nomarg)
    avg_runtime_marg = np.mean(runtimes_marg)

    print(f"Dataset {dataset_info['name']}:")
    print("Non-Marginalized Configuration:")
    print(f"  PSRF: {Rhat_nomarg:.3f}, ESS: {ess_nomarg:.1f}/{total_samples}, Geweke: {geweke_nomarg:.3f}, Runtime: {avg_runtime_nomarg:.2f}s")
    print("Marginalized Configuration:")
    print(f"  PSRF: {Rhat_marg:.3f}, ESS: {ess_marg:.1f}/{total_samples}, Geweke: {geweke_marg:.3f}, Runtime: {avg_runtime_marg:.2f}s")

    # --- Assert diagnostics ---
    assert Rhat_nomarg < 1.1, f"Non-marginalized BART mixing poor for {dataset_info['name']}: PSRF={Rhat_nomarg:.3f}"
    assert Rhat_marg < 1.1, f"Marginalized BART mixing poor for {dataset_info['name']}: PSRF={Rhat_marg:.3f}"
    assert ess_nomarg > 0.5 * total_samples, f"Non-marginalized BART ESS too low for {dataset_info['name']}: ESS={ess_nomarg:.1f}"
    assert ess_marg > 0.5 * total_samples, f"Marginalized BART ESS too low for {dataset_info['name']}: ESS={ess_marg:.1f}"
    assert geweke_nomarg < 2, f"Non-marginalized BART Geweke diagnostic indicates non-convergence for {dataset_info['name']}: {geweke_nomarg:.3f}"
    assert geweke_marg < 2, f"Marginalized BART Geweke diagnostic indicates non-convergence for {dataset_info['name']}: {geweke_marg:.3f}"
