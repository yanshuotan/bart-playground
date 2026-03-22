
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from bart_playground.bandit.agents.external_agents import BartzTSAgent, StochTreeTSAgent

class TestExternalAgents:
    
    def test_bartz_agent_init(self):
        # Mock the wrapper so we don't need the library installed for init test
        with patch('bart_playground.bandit.agents.external_agents.BartzWrapper') as mock_wrapper:
            agent = BartzTSAgent(n_arms=2, n_features=3)
            assert agent.n_arms == 2
            assert agent.n_features == 3
            
            # Check model factory
            model = agent.model_factory(new_ndpost=100)
            mock_wrapper.assert_called()
            # Verify ndpost was passed correctly
            args, kwargs = mock_wrapper.call_args
            assert kwargs['ndpost'] == 100

    def test_stochtree_agent_init(self):
        with patch('bart_playground.bandit.agents.external_agents.StochTreeWrapper') as mock_wrapper:
            agent = StochTreeTSAgent(n_arms=2, n_features=3, use_gfr=True)
            assert agent.n_arms == 2
            
            model = agent.model_factory(new_ndpost=100)
            mock_wrapper.assert_called()
            args, kwargs = mock_wrapper.call_args
            assert kwargs['ndpost'] == 100
            assert kwargs['use_gfr'] is True

    def test_external_agent_integration_mock(self):
        """Test that external agents can run through basic loops using mocks."""
        with patch('bart_playground.bandit.agents.external_agents.BartzWrapper') as mock_wrapper_cls:
            mock_model = MagicMock()
            mock_model.range_post = range(10)
            # When choose_arm is called, it calls _get_action_estimates -> predict_trace
            # In multi-encoding, predict_trace should return (n_arms,) predictions
            mock_model.predict_trace.return_value = np.zeros(2) 
            mock_wrapper_cls.return_value = mock_model
            
            agent = BartzTSAgent(n_arms=2, n_features=2, initial_random_selections=0)
            
            # Simulate one step
            x = np.array([0.5, 0.5])
            
            # Force fitted state
            agent.is_model_fitted = True
            # In multi-encoding (default), agent.models is a list containing the model
            agent.models = [mock_model] 
            
            arm = agent.choose_arm(x)
            assert 0 <= arm < 2
            
            # Update state
            agent.update_state(arm, x, 1.0)
            assert len(agent.all_rewards) == 1
