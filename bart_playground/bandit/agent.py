
import numpy as np
from ..bart import DefaultBART

class BARTTSAgent:
    """
    A bandit agent that uses a BART model to choose an arm.
    """
    def __init__(self, n_arms, n_features, nskip=50):
        """
        Initialize the agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            nskip (int): Number of burn-in iterations for the BART sampler.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        # Initialize feature and outcome lists for each arm as empty arrays.
        self.features = [np.empty((0, n_features)) for _ in range(n_arms)]
        self.outcomes = [np.empty((0, 1)) for _ in range(n_arms)]
        self.nskip_default = nskip

    def choose_arm(self, x, nskip=None, binary=False):
        """
        Choose an arm based on input features x using a BART model for prediction.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm.
            nskip (int): Number of burn-in iterations (if None, use default).
            binary (bool): Unused in this implementation.
            
        Returns:
            int: The index of the selected arm.
        """
        if nskip is None:
            nskip = self.nskip_default
        
        # Ensure x is a 2D array with one row.
        x = np.array(x).reshape(1, -1)
        
        # If any arm has very little data, return that arm for exploration.
        for arm in range(self.n_arms):
            fts = self.features[arm]
            if fts.shape[0] <= 3:
                return arm
        
        # Compute a predicted value for each arm.
        u = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            fts = self.features[arm]
            outcomes = self.outcomes[arm].flatten()
            # Instantiate a DefaultBART model.
            # We set ndpost=1 to mimic a single MCMC draw.
            model = DefaultBART(ndpost=1, nskip=nskip)
            model.fit(fts, outcomes, quietly=True)
            # Obtain the posterior prediction; posterior_f returns an array of shape (n_test, ndpost)
            preds = model.posterior_f(x)
            # Check if any predictions are NaN.
            if np.any(np.isnan(preds)):
                raise ValueError("NaN values found in predictions.")
            # Sample one prediction from the posterior.
            sampled_pred = preds[0, np.random.randint(preds.shape[1])]
            u[arm] = sampled_pred
        
        # Choose the arm with the highest predicted value.
        return int(np.argmax(u))

    def update_state(self, arm, x, y):
        """
        Update the state for a given arm with new data.
        
        Parameters:
            arm (int): The index of the arm to update.
            x (array-like): Feature vector to append.
            y (float): Outcome to append.
            
        Returns:
            self: Updated instance.
        """
        # Convert x and y to proper array shapes.
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, 1)
        self.features[arm] = np.vstack([self.features[arm], x])
        self.outcomes[arm] = np.vstack([self.outcomes[arm], y])
        return self
