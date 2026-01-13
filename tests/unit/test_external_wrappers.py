
import numpy as np
import pytest
import sys
import os
from unittest.mock import MagicMock, call

from bart_playground.bandit.agents.bart_ts_agents import BARTTSAgent
from bart_playground.bandit.agents.external_wrappers import BartzWrapper, StochTreeWrapper

# Mock available flags
BARTZ_AVAILABLE = True
STOCHTREE_AVAILABLE = True

class ConcreteAgent(BARTTSAgent):
    """Concrete subclass of BARTTSAgent for testing."""
    def posterior_draws_on_probes(self, X_probes: np.ndarray):
        return np.zeros((10, 5, 3)), 10
    
    def _feel_good_full_recompute(self):
        pass
        
    def _feel_good_incremental_recompute(self, x_new):
        pass

class TestExternalWrappers:
    
    def test_bartz_wrapper_methods(self):
        """Test BartzWrapper initialization and method signatures."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("bart_playground.bandit.agents.external_wrappers.BARTZ_AVAILABLE", True)
            
            mock_bartz_model = MagicMock()
            mock_bartz_model.predict.return_value = np.zeros((10, 1)) 
            
            wrapper = BartzWrapper(ndpost=10, nskip=5)
            assert wrapper.ndpost == 10
            assert wrapper.nskip == 5
            
            wrapper.model = mock_bartz_model
            
            X = np.random.rand(1, 5)
            pred = wrapper.predict_trace(0, X)
            
            # Manually check call args to avoid numpy ambiguity error with MagicMock
            assert mock_bartz_model.predict.called
            args, _ = mock_bartz_model.predict.call_args
            np.testing.assert_array_equal(args[0], X.T)
            
            assert pred.shape == (1,)
            
            post_f = wrapper.posterior_f(X)
            assert post_f.shape == (1, 10)
            
            assert len(wrapper.range_post) == 10

    def test_stochtree_wrapper_methods(self):
        """Test StochTreeWrapper initialization and method signatures."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("bart_playground.bandit.agents.external_wrappers.STOCHTREE_AVAILABLE", True)
            
            mock_stoch_model = MagicMock()
            mock_stoch_model.predict.return_value = np.zeros((1, 10))
            
            wrapper = StochTreeWrapper(ndpost=10, nskip=5, use_gfr=True)
            wrapper.model = mock_stoch_model
            
            assert wrapper.use_gfr is True
            
            X = np.random.rand(1, 5)
            
            pred = wrapper.predict_trace(0, X)
            assert pred.shape == (1,)
            
            post_f = wrapper.posterior_f(X)
            assert post_f.shape == (1, 10)

    def test_agent_integration_mock(self):
        """Test that BARTTSAgent accepts these wrappers."""
        
        mock_wrapper = MagicMock()
        mock_wrapper.fit.return_value = mock_wrapper
        mock_wrapper.range_post = range(10)
        
        n_arms = 3
        n_features = 2
        
        def mock_predict_trace(k, X, backtransform=True):
            return np.zeros(X.shape[0]) 
            
        mock_wrapper.predict_trace.side_effect = mock_predict_trace
        
        # Use ConcreteAgent instead of abstract base class
        agent = ConcreteAgent(
            n_arms=n_arms,
            n_features=n_features,
            model_factory=lambda **kwargs: mock_wrapper,
            initial_random_selections=0,
            encoding='multi' 
        )
        
        agent.update_state(0, np.random.rand(n_features), 1.0)
        
        agent.is_model_fitted = True
        
        arm = agent.choose_arm(np.random.rand(n_features))
        assert 0 <= arm < n_arms
        
        assert mock_wrapper.predict_trace.called
