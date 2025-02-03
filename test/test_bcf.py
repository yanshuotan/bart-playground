
# Draft file
# Not a unit test

import numpy as np
import sys
from os.path import abspath, dirname
# Add the parent directory (module) to the search path
sys.path.append(abspath(dirname(dirname(__file__))))

#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split

from bart_playground.bcf.bcf import BCF

def generate_bcf_data(n=1000, p=10, noise_level=0.5, random_state=42):
    rng = np.random.default_rng(random_state)
    
    X = rng.normal(0, 1, (n, p))
    mu = 2 * np.sin(np.pi * X[:,0]) + X[:,1]**2
    tau = 0.5 * X[:,2] + 0.5 * (X[:,3] > 0)
    z = rng.binomial(1, 0.5, n)
    y = mu + z * tau + rng.normal(0, noise_level, n)
    
    return X, y, z, mu, tau

# Generate data
X, y, z, true_mu, true_tau = generate_bcf_data(n=2000)
X_train, X_test, y_train, y_test, z_train, z_test, mu_train, mu_test, tau_train, tau_test = \
    train_test_split(X, y, z, true_mu, true_tau, test_size=0.2, random_state=42)

bcf = BCF(
    n_mu_trees=200,       # Number of prognostic effect trees
    n_tau_trees=50,       # Number of treatment effect trees
    mu_alpha=0.95,        # Tree depth prior for mu
    mu_beta=2.0,          # Tree depth prior for mu
    tau_alpha=0.5,        # Simpler trees for treatment effects
    tau_beta=3.0,         # Penalize complex tau trees
    tau_k=0.5,            # Regularization for treatment effects
    ndpost=100,          # Posterior samples
    nskip=100,            # Burn-in iterations
    random_state=42
)

bcf.fit(X_train, y_train, z_train)
