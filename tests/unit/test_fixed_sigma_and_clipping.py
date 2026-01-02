import unittest
import numpy as np
from unittest.mock import MagicMock

from bart_playground import DefaultBART
from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent

class TestFixedSigmaAndClipping(unittest.TestCase):

    def setUp(self):
        # Setup for simple data
        self.rng = np.random.default_rng(42)
        self.X = self.rng.random((20, 2))
        self.y = self.rng.random(20)
        self.n_arms = 3
        self.n_features = 2

    def test_fixed_sigma_prior_and_bart(self):
        """Test that passing fixed_eps_sigma2 to DefaultBART fixes the sigma."""
        fixed_val = 0.12345
        # Initialize model with fixed sigma
        bart = DefaultBART(fixed_eps_sigma2=fixed_val, n_trees=10, ndpost=10, nskip=5, random_state=42)
        
        # 1. Check if parameter is stored correctly
        params = bart.get_params()
        self.assertIn("fixed_eps_sigma2", params)
        self.assertEqual(params["fixed_eps_sigma2"], fixed_val)
        
        # 2. Check if prior is set correctly
        self.assertEqual(bart.sampler.prior.global_prior.fixed_eps_sigma2, fixed_val)
        
        # 3. Fit and check trace values
        bart.fit(self.X, self.y)
        
        # Verify that every sample in the trace has the fixed sigma
        for param in bart.trace:
            sigma2 = param.global_params["eps_sigma2"][0]
            self.assertEqual(sigma2, fixed_val)

    def test_agent_feel_good_mu_sets_fixed_sigma(self):
        """Test that DefaultBARTTSAgent correctly translates feel_good_mu to fixed_eps_sigma2."""
        mu = 0.5
        # Formula: sigma^2 = 1 / (2 * mu) = 1 / 1 = 1.0
        expected_sigma2 = 1.0
        
        agent = DefaultBARTTSAgent(
            n_arms=self.n_arms, 
            n_features=self.n_features,
            feel_good_mu=mu,
            random_state=42
        )
        
        # Check internal model prior
        # Note: For default n_chains=1, agent.model is a DefaultBART instance
        self.assertIsInstance(agent.model, DefaultBART)
        self.assertEqual(agent.model.sampler.prior.global_prior.fixed_eps_sigma2, expected_sigma2)
        
        # Check get_params return value
        params = agent.model.get_params()
        self.assertEqual(params.get("fixed_eps_sigma2"), expected_sigma2)

    def test_clipping_logic_in_agent(self):
        """Test that _feel_good_full_recompute correctly clips values at 0.5."""
        agent = DefaultBARTTSAgent(
            n_arms=self.n_arms,
            n_features=self.n_features,
            feel_good_lambda=1.0, # Enable feel good to ensure methods run
            random_state=42
        )
        
        # Mock dependencies
        agent.all_features = [np.zeros(2)] * 5 # Simulate 5 data points
        
        # Mock n_post via model behavior
        mock_model = MagicMock()
        mock_model.range_post = range(3) # len(range(3)) == 3
        agent.model = mock_model
        agent.is_model_fitted = True
        
        agent._fg_S = None # Force recompute
        
        # Construct mock return data (n_probes=5, n_arms=3, n_post=3)
        # We only care about max over axis 1 (arms)
        # Initialize with -10.0 
        f_by_arm = np.full((5, 3, 3), -10.0)
        
        # Probe 0: max 0.6 (> 0.5, should clip to 0.5)
        f_by_arm[0, 0, :] = 0.6 
        # Probe 1: max 0.4 (< 0.5, stay 0.4)
        f_by_arm[1, 0, :] = 0.4
        # Probe 2: max 0.5 (== 0.5, stay 0.5)
        f_by_arm[2, 0, :] = 0.5
        # Probe 3: max -0.1 (stay -0.1)
        f_by_arm[3, 0, :] = -0.1
        # Probe 4: max 1.0 (> 0.5, should clip to 0.5)
        f_by_arm[4, 0, :] = 1.0
        
        # Mock the method
        agent._posterior_f_by_arm = MagicMock(return_value=(f_by_arm, 5, 3, 3))
        
        # Execute
        agent._feel_good_full_recompute()
        
        # Expected calculation:
        # Sum = 0.5 + 0.4 + 0.5 - 0.1 + 0.5 = 1.8
        expected_sum = 1.8
        
        # Verify result
        np.testing.assert_allclose(agent._fg_S, expected_sum, rtol=1e-5)

    def test_clipping_logic_incremental(self):
        """Test that _feel_good_incremental_recompute also performs clipping."""
        agent = DefaultBARTTSAgent(
            n_arms=self.n_arms,
            n_features=self.n_features,
            feel_good_lambda=1.0,
            random_state=42
        )
        
        # Initialize _fg_S
        agent._fg_S = np.zeros(3)
        
        # Mock data: new point max value 0.8
        f_by_arm = np.zeros((1, 3, 3))
        f_by_arm[0, 0, :] = 0.8 # Should clip to 0.5
        
        agent._posterior_f_by_arm = MagicMock(return_value=(f_by_arm, 1, 3, 3))
        
        # Execute incremental update
        agent._feel_good_incremental_recompute(np.zeros(2))
        
        # Expected: 0 + 0.5 = 0.5
        np.testing.assert_allclose(agent._fg_S, 0.5, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
