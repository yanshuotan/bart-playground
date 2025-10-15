import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
# Add logging configuration before importing arviz
import logging
logging.getLogger('arviz.preview').setLevel(logging.WARNING)
import arviz as az

def segmented_ljung_box_test(chain, segment_length=100, lags=10, alpha=0.05):
    """
    Perform segmented Ljung-Box test to detect MCMC convergence
    
    Parameters:
    -----------
    chain : array-like
        MCMC chain samples
    segment_length : int
        Length of each segment
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
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(segment, lags=lags, return_df=True)
        min_pvalue = lb_result['lb_pvalue'].min()
        converged = np.all(lb_result['lb_pvalue'] > alpha)
        
        results.append({
            'segment': i,
            'end_iter': end,
            'min_pvalue': min_pvalue,
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
    pvalues = [r['min_pvalue'] for r in results]
    converged_status = [r['converged'] for r in results]
    
    colors = ['green' if c else 'red' for c in converged_status]
    ax2.scatter(end_iters, pvalues, c=colors)
    ax2.axhline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
    ax2.set_title('Ljung-Box p-values by Segment')
    ax2.set_xlabel('End Iteration')
    ax2.set_ylabel('Min p-value')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'converged': convergence_iter is not None,
        'convergence_iteration': convergence_iter,
        'convergence_rate': np.mean([r['converged'] for r in results]),
        'results': results
    }

def segment_ess_efficiency(chain, segment_length=100):
    """
    Calculate ESS efficiency for non-overlapping segments.
    
    Parameters:
    -----------
    chain : array-like
        MCMC chain samples
    segment_length : int
        Length of each segment
        
    Returns:
    --------
    np.array
        Array of ESS/N efficiencies for each segment
    """
    chain = np.array(chain)
    n_segments = len(chain) // segment_length
    
    efficiencies = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = chain[start:end]
        ess = az.ess(segment.reshape(1, -1))
        efficiency = ess / segment_length
        efficiencies.append(efficiency)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(efficiencies)), efficiencies, 'o', markersize=4, alpha=0.7)
    plt.xlabel('Segment Index')
    plt.ylabel('ESS Efficiency')
    plt.title('ESS Efficiency by Segment')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=np.mean(efficiencies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(efficiencies):.3f}')
    plt.legend()
    plt.show()
    
    return np.array(efficiencies)


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