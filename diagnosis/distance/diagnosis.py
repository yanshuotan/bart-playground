import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss, acf
import warnings

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss, acf
import warnings


def sample_random_pairs(n, n_samples, random_state=None):
    rng = np.random.default_rng(random_state)
    pairs = set()
    while len(pairs) < n_samples:
        i, j = rng.integers(0, n, size=2)
        if i < j:
            pairs.add((i, j))
    return list(pairs)

def sample_between_chain_pairs(all_labels, n_samples, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(all_labels)
    pairs = set()
    while len(pairs) < n_samples:
        i, j = rng.integers(0, n, size=2)
        if i < j and all_labels[i] != all_labels[j]:
            pairs.add((i, j))
    return list(pairs)

def pairwise_l2_distances(arr, pairs):
    # arr: [n_samples, n_features], pairs: [(i, j), ...]
    return np.array([np.linalg.norm(arr[i] - arr[j]) for i, j in pairs])

def compute_within_and_between_chain_distances(preds, max_pairs=1000, random_state=None):
    n_chains, n_test_points, n_iter = preds.shape

    # within-chain pairwise
    within_chain_dists = []
    for chain in range(n_chains):
        chain_vectors = preds[chain].T  # [n_iter, n_test_points]
        pairs = sample_random_pairs(n_iter, max_pairs, random_state)
        dists = pairwise_l2_distances(chain_vectors, pairs)
        within_chain_dists.append(dists)
    mean_within_chain = np.mean(np.concatenate(within_chain_dists))

    # between-chain pairwise
    all_vectors = []
    all_labels = []
    for chain in range(n_chains):
        chain_vectors = preds[chain].T  # [n_iter, n_test_points]
        all_vectors.append(chain_vectors)
        all_labels.extend([chain] * n_iter)
    all_vectors = np.concatenate(all_vectors, axis=0)  # [n_chains * n_iter, n_test_points]
    all_labels = np.array(all_labels)

    pairs = sample_between_chain_pairs(all_labels, 4*max_pairs, random_state)
    between_chain_dists = pairwise_l2_distances(all_vectors, pairs)
    mean_between_chain = np.mean(between_chain_dists)

    return mean_within_chain, mean_between_chain

def segmented_kpss_test(chain, window_size=100, step=20, regression='c', alpha=0.05):
    """
    Perform segmented KPSS test with overlapping windows to detect MCMC convergence

    Parameters:
    -----------
    chain : array-like, shape (n_samples,) or (n_samples, n_dim)
        MCMC chain samples
    window_size : int
        Size of each window
    step : int
        Step size for moving window
    regression : str
        Type of regression ('c' for constant, 'ct' for constant and trend)
    alpha : float
        Significance level

    Returns:
    --------
    dict : convergence results
    """
    chain = np.array(chain)
    if chain.ndim == 1:
        chain = chain[:, None]  # shape (n_samples, 1)
    n_samples, n_dim = chain.shape

    n_windows = (n_samples - window_size) // step + 1
    results = [[] for _ in range(n_dim)]

    for d in range(n_dim):
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            segment = chain[start:end, d]

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    kpss_stat, pvalue, _, _ = kpss(segment, regression=regression)
                if pvalue >= 0.1:
                    pvalue = 0.1
            except Exception:
                kpss_stat = np.nan
                pvalue = 0.01

            converged = pvalue > alpha
            results[d].append({
                'window': i,
                'end_iter': end,
                'kpss_stat': kpss_stat,
                'pvalue': pvalue,
                'converged': converged
            })

    # Find convergence point: each dim has its own, take the max
    convergence_iters = []
    for d in range(n_dim):
        dim_converged = None
        for i in range(n_windows):
            if results[d][i]['converged']:
                dim_converged = results[d][i]['end_iter']
                break
        convergence_iters.append(dim_converged)
    convergence_iter = max([c for c in convergence_iters if c is not None]) if all(c is not None for c in convergence_iters) else None

    # Plot diagnostics for the first dimension
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(chain[:, 0])
    if convergence_iter:
        ax1.axvline(convergence_iter, color='red', linestyle='--',
                    label=f'Convergence at {convergence_iter}')
    ax1.set_title('MCMC Trace (dim 0)')
    ax1.set_xlabel('Iteration')
    ax1.legend()

    # P-values by window for first dimension
    end_iters = [r['end_iter'] for r in results[0]]
    pvalues = [r['pvalue'] for r in results[0]]
    converged_status = [r['converged'] for r in results[0]]
    colors = ['green' if c else 'red' for c in converged_status]
    ax2.scatter(end_iters, pvalues, c=colors)
    ax2.axhline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
    ax2.set_title('KPSS p-values by Window (dim 0)')
    ax2.set_xlabel('End Iteration')
    ax2.set_ylabel('p-value')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return {
        'converged': convergence_iter is not None,
        'convergence_iteration': convergence_iter,
        'convergence_rate': np.mean([[r['converged'] for r in dim_res] for dim_res in results]),
        'results': results
    }


def plot_autocorrelation(chain, nlags=50):
    chain = np.array(chain)
    
    # Calculate autocorrelation with confidence intervals
    autocorr_values = acf(chain, nlags=nlags, fft=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    lags = np.arange(len(autocorr_values))
    
    # Plot autocorrelation
    plt.plot(lags, autocorr_values, 'o-', markersize=3, linewidth=1)
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='±0.1')
    plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Function (up to {nlags} lags)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return autocorr_values