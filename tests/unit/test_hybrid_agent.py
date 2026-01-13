
import numpy as np
import pytest
from unittest.mock import patch
from bart_playground.bandit.agents.external_agents import HybridBARTTSAgent
from bart_playground.mcbart import MultiChainBART

class TestHybridAgent:
    
    def test_hybrid_agent_initialization(self):
        """Verify HybridBARTTSAgent correctly sets up its factory."""
        n_arms = 3
        n_features = 4
        switch_t = 50
        
        agent = HybridBARTTSAgent(
            n_arms=n_arms,
            n_features=n_features,
            switch_t=switch_t,
            initial_random_selections=0
        )
        
        assert agent.n_arms == n_arms
        assert agent.encoding == 'separate'
        assert agent.switch_t == switch_t
        assert callable(agent.model_factory)

    def test_hybrid_agent_transition_mock(self):
        """Mocked integration test for the transition boundary."""
        n_arms = 2
        n_features = 2
        switch_t = 5

        class FakeStochTree:
            def __init__(self, ndpost: int = 500, **kwargs):
                self.ndpost = ndpost
                self.range_post = range(10)

            def fit(self, X, y, quietly: bool = True):
                return self

            def predict_trace(self, k: int, X, backtransform: bool = True):
                return np.array([0.5])

            def get_params(self):
                return {"ndpost": self.ndpost}

        class FakeMultiChain(MultiChainBART):
            def __init__(self, n_models: int, ndpost: int = 500, max_bins: int = 100, **kwargs):
                self.n_models = n_models
                self._ndpost = ndpost
                self._max_bins = max_bins

            @property
            def range_post(self):
                return range(10)

            def set_ndpost(self, ndpost: int):
                self._ndpost = ndpost
                return self

            def set_max_bins(self, max_bins: int):
                self._max_bins = max_bins
                return self

            def __getitem__(self, arm: int):
                return self

            def fit(self, X, y, quietly: bool = True):
                return self

            def get_params(self):
                return {"ndpost": self._ndpost, "max_bins": self._max_bins, "n_models": self.n_models}

            def predict_trace_batch(self, k: int, X, backtransform: bool = True):
                return np.array([[0.1], [0.2]])

        stoch_calls = []
        mc_calls = []

        def stoch_ctor(*args, **kwargs):
            stoch_calls.append((args, kwargs))
            return FakeStochTree(*args, **kwargs)

        def mc_ctor(*args, **kwargs):
            mc_calls.append((args, kwargs))
            # external_agents passes n_models as kwarg; keep signature flexible
            return FakeMultiChain(**kwargs)

        with patch("bart_playground.bandit.agents.external_agents.StochTreeWrapper", new=stoch_ctor), \
             patch("bart_playground.bandit.agents.external_agents.MultiChainBART", new=mc_ctor):

            agent = HybridBARTTSAgent(
                n_arms=n_arms,
                n_features=n_features,
                switch_t=switch_t,
                initial_random_selections=0,
            )

            # Step 1: Early phase refresh uses StochTree backend
            agent.t = 1
            agent._refresh_model()
            assert len(stoch_calls) > 0

            # Step 2: Boundary phase initializes MultiChain exactly once
            agent.t = switch_t
            agent._refresh_model()
            assert len(mc_calls) == 1
            assert mc_calls[0][1]["n_models"] == n_arms

            # Step 3: Subsequent refresh reuses cached MultiChain (no new init)
            agent.t = switch_t + 1
            agent._refresh_model()
            assert len(mc_calls) == 1

            # Verify choose_arm works on multichain path
            agent.is_model_fitted = True
            agent.choose_arm(np.array([0.1, 0.1]))
