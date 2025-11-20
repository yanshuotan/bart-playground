import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss, acf
import warnings

def segmented_kpss_test(chain, window_length=100, step=5, regression='c', alpha=0.05):
    """
    Perform sliding window KPSS test to detect MCMC convergence.
    Window always ends at chain tail, start moves forward by step.
    """
    chain = np.array(chain)
    n_samples = len(chain)
    results = []

    # Sliding window: window always ends at chain tail, start moves from 0 to n_samples - window_length
    for start in range(0, n_samples - window_length + 1, step):
        end = n_samples
        window = chain[end-1:start:-1] # Window from end to start, reversed
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                kpss_stat, pvalue, lags, critical_values = kpss(window, regression=regression)
            
            # If p-value is at boundary (0.1), assume it's actually higher (more stationary)
            if pvalue >= 0.1:
                pvalue = 0.1  # Cap at 0.1 for display purposes
                
        except Exception as e:
            # If KPSS fails, assume non-stationary
            kpss_stat = np.nan
            pvalue = 0.01
        rejected = pvalue < alpha
        results.append({
            'start': start,
            'end': end,
            'kpss_stat': kpss_stat,
            'pvalue': pvalue,
            'rejected': rejected
        })

    # Find convergence point: first occurrence of 3 consecutive rejected windows from the tail
    convergence_iter = None
    for i in range(len(results) - 1, 1, -1):
        if all(results[j]['rejected'] for j in range(i, i-3, -1)):
            convergence_iter = results[i-2]['start']
            break

    # Plot diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Chain trace
    ax1.plot(chain)
    if convergence_iter is not None:
        ax1.axvline(convergence_iter, color='red', linestyle='--',
                    label=f'Convergence at {convergence_iter}')
    ax1.set_title('MCMC Trace')
    ax1.set_xlabel('Iteration')
    ax1.legend()

    # P-values by window start
    starts = [r['start'] for r in results]
    pvalues = [r['pvalue'] for r in results]
    colors = ['red' if r['rejected'] else 'green' for r in results]
    ax2.scatter(starts, pvalues, c=colors)
    ax2.axhline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
    ax2.set_title('KPSS p-values by Window Start')
    ax2.set_xlabel('Window Start')
    ax2.set_ylabel('p-value')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return {
        'converged': convergence_iter is not None,
        'convergence_iteration': convergence_iter,
        'rejection_rate': np.mean([r['rejected'] for r in results]),
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