import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss, acf
import warnings

def segmented_kpss_test(chain, segment_length=100, regression='c', alpha=0.05):
    """
    Perform segmented KPSS test to detect MCMC convergence
    
    Parameters:
    -----------
    chain : array-like
        MCMC chain samples
    segment_length : int
        Length of each segment
    regression : str
        Type of regression ('c' for constant, 'ct' for constant and trend)
    alpha : float
        Significance level
        
    Returns:
    --------
    dict : convergence results
    """
    chain = np.array(chain)
    n_samples = len(chain)
    n_segments = n_samples // segment_length
    
    results = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = chain[start:end]
        
        # KPSS test - tests null hypothesis of stationarity
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                kpss_stat, pvalue, lags, critical_values = kpss(segment, regression=regression)
            
            # If p-value is at boundary (0.1), assume it's actually higher (more stationary)
            if pvalue >= 0.1:
                pvalue = 0.1  # Cap at 0.1 for display purposes
                
        except Exception as e:
            # If KPSS fails, assume non-stationary
            kpss_stat = np.nan
            pvalue = 0.01
        
        # For KPSS, we want to NOT reject null (p > alpha means stationary/converged)
        converged = pvalue > alpha
        
        results.append({
            'segment': i,
            'end_iter': end,
            'kpss_stat': kpss_stat,
            'pvalue': pvalue,
            'converged': converged
        })
    
    # Find convergence point (3 consecutive converged segments)
    convergence_iter = None
    for i in range(len(results) - 2):
        if all(results[j]['converged'] for j in range(i, i+3)):
            convergence_iter = results[i+2]['end_iter']
            break
    
    # Plot diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chain trace
    ax1.plot(chain)
    if convergence_iter:
        ax1.axvline(convergence_iter, color='red', linestyle='--', 
                   label=f'Convergence at {convergence_iter}')
    ax1.set_title('MCMC Trace')
    ax1.set_xlabel('Iteration')
    ax1.legend()
    
    # P-values by segment
    end_iters = [r['end_iter'] for r in results]
    pvalues = [r['pvalue'] for r in results]
    converged_status = [r['converged'] for r in results]
    
    colors = ['green' if c else 'red' for c in converged_status]
    ax2.scatter(end_iters, pvalues, c=colors)
    ax2.axhline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
    ax2.set_title('KPSS p-values by Segment')
    ax2.set_xlabel('End Iteration')
    ax2.set_ylabel('p-value')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'converged': convergence_iter is not None,
        'convergence_iteration': convergence_iter,
        'convergence_rate': np.mean([r['converged'] for r in results]),
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