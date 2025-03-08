import unittest
import numpy as np
from sklearn.model_selection import train_test_split

from bart_playground.bcf import *
from bart_playground.bcf.bcf import BCF

def generate_bcf_data(n=1000, p=10, noise_level=0.5, random_state=42):
    rng = np.random.default_rng(random_state)
    
    X = rng.normal(0, 1, (n, p))
    mu = 2 * np.sin(np.pi * X[:,0]) + X[:,1]**2
    tau = 0.5 * X[:,2] + 0.5 * (X[:,3] > 0)
    z = rng.binomial(1, 0.5, n)
    y = mu + z * tau + rng.normal(0, noise_level, n)
    
    return X, y, z, mu, tau

class TestBCF(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        X, y, z, true_mu, true_tau = generate_bcf_data(n=500)
        (self.X_train, self.X_test, 
         self.y_train, self.y_test, 
         self.z_train, self.z_test, 
         self.mu_train, self.mu_test, 
         self.tau_train, self.tau_test) = train_test_split(X, y, z, true_mu, true_tau,
                                                   test_size=0.2, random_state=42)
        
        # Initialize the BCF model
        self.bcf = BCF(
            n_mu_trees=100,       # Number of prognostic effect trees
            n_tau_trees=50,       # Number of treatment effect trees
            mu_alpha=0.95,        # Tree depth prior for mu
            mu_beta=2.0,          # Tree depth prior for mu
            tau_alpha=0.5,        # Simpler trees for treatment effects
            tau_beta=3.0,         # Penalize complex tau trees
            tau_k=0.5,            # Regularization for treatment effects
            ndpost=100,           # Posterior samples
            nskip=100,            # Burn-in iterations
            random_state=42
        )

    def test_fit_predict(self):
        # Fit the model on the training data
        self.bcf.fit(self.X_train, self.y_train, self.z_train)
        predictions = self.bcf.predict_components(self.X_test, self.z_test)
        
        for i in range(1,3):
            prediction = predictions[i]
            # Assert that predictions have the correct shape
            self.assertEqual(prediction.shape[0], self.X_test.shape[0])
            # Assert that predictions are a NumPy array
            self.assertIsInstance(prediction, np.ndarray)
            # Assert that all predicted values are finite
            self.assertTrue(np.all(np.isfinite(prediction)))

if __name__ == '__main__':
    unittest.main()
