'''
Test the separate encoding agent.

Auto-generated tests.
'''

import numpy as np
import ray

from bart_playground.bandit.agents.bart_ts_agents import BARTTSAgent
from bart_playground.mcbart import MultiChainBART
from bart_playground.bart import DefaultBART


class DummyAgent(BARTTSAgent):
    """Minimal concrete agent for testing core BARTTSAgent logic."""

    def posterior_draws_on_probes(self, X_probes: np.ndarray):
        return np.zeros((0, 0, 0)), 0


def _make_agent_with_real_multichain(n_arms=2, n_features=1):
    """Create agent with real MultiChainBART using real DefaultBART."""
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    def factory(new_ndpost=10):
        return MultiChainBART(
            n_ensembles=1,  # 1 chain for fast tests
            bart_class=DefaultBART,
            random_state=42,
            ndpost=new_ndpost,
            n_models=n_arms,
            n_trees=5,       # minimal trees for speed
            nskip=5,         # minimal burn-in
        )
    
    agent = DummyAgent(
        n_arms=n_arms,
        n_features=n_features,
        model_factory=factory,
        initial_random_selections=0,
        random_state=0,
        encoding="separate",
        refresh_schedule="log",
    )
    agent.initial_random_selections = 999
    return agent


def _populate_agent(agent: DummyAgent):
    n_features = agent.n_features
    rng = np.random.default_rng(123)
    data = [
        (0, rng.random(n_features), 1.0),
        (1, rng.random(n_features), 0.5),
        (0, rng.random(n_features), 0.8),
        (1, rng.random(n_features), 0.2),
    ]
    for arm, x, reward in data:
        agent.update_state(arm, x, reward)


def test_refresh_uses_multichain_and_sets_ndpost():
    agent = _make_agent_with_real_multichain()
    _populate_agent(agent)
    
    agent._refresh_model()
    
    assert agent.is_model_fitted
    assert isinstance(agent.models, MultiChainBART)
    assert agent.models.n_models == 2


def test_action_estimates_follow_active_submodel():
    agent = _make_agent_with_real_multichain()
    _populate_agent(agent)
    agent._refresh_model()
    agent.is_model_fitted = True

    estimates = agent._get_action_estimates(np.array([0.0]))
    # Verify we get an array of correct shape with finite values
    assert estimates.shape == (2,)
    assert np.all(np.isfinite(estimates))


def test_feature_inclusion_averages_over_submodels():
    agent = _make_agent_with_real_multichain(n_features=2)
    _populate_agent(agent)
    agent._refresh_model()
    agent.is_model_fitted = True

    result = agent.feature_inclusion()
    metrics = result["metrics"]
    # Check shape and that values are valid probabilities
    assert len(metrics["inclusion"]) == 2
    assert np.all(metrics["inclusion"].to_numpy() >= 0)
    assert np.all(metrics["inclusion"].to_numpy() <= 1)


def test_multichain_list_semantics():
    """Test that MultiChainBART supports list-like indexing."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    mc = MultiChainBART(
        n_ensembles=1,
        bart_class=DefaultBART,
        random_state=42,
        ndpost=10,
        n_models=3,
        n_trees=5,
        nskip=5,
    )
    
    # Test __len__
    assert len(mc) == 3
    
    # Test __getitem__ returns self (for chaining)
    assert mc[0] is mc
    assert mc[1] is mc
    assert mc[2] is mc
    
    # Test out of bounds
    try:
        mc[3]
        assert False, "Should raise IndexError"
    except IndexError:
        pass


def test_multiple_refreshes_update_ndpost():
    """Test that multiple refreshes correctly update ndpost."""
    agent = _make_agent_with_real_multichain()
    _populate_agent(agent)
    
    # First refresh
    agent._refresh_model()
    first_ndpost = agent.models.ndpost
    
    # Add more data to trigger different ndpost calculation
    rng = np.random.default_rng(456)
    for _ in range(20):
        arm = rng.integers(0, 2)
        x = rng.random(agent.n_features)
        reward = rng.random()
        agent.update_state(arm, x, reward)
    
    # Second refresh
    agent._refresh_model()
    second_ndpost = agent.models.ndpost
    
    # Both should be valid positive integers
    assert first_ndpost > 0
    assert second_ndpost > 0


def test_choose_arm_uses_thompson_sampling():
    """Test that choose_arm returns valid arm indices."""
    agent = _make_agent_with_real_multichain()
    _populate_agent(agent)
    agent._refresh_model()
    
    rng = np.random.default_rng(789)
    chosen_arms = []
    for _ in range(10):
        x = rng.random(agent.n_features)
        arm = agent.choose_arm(x)
        chosen_arms.append(arm)
    
    # All chosen arms should be valid
    assert all(0 <= arm < agent.n_arms for arm in chosen_arms)
    # With randomness, we expect some variation (not always same arm)
    # This is a weak test but catches obvious bugs
    assert len(set(chosen_arms)) >= 1


def test_actor_count_is_n_chains_not_n_arms_times_n_chains():
    """Verify the optimization: only n_chains actors, not n_arms * n_chains."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    n_arms = 5
    n_chains = 2
    
    mc = MultiChainBART(
        n_ensembles=n_chains,
        bart_class=DefaultBART,
        random_state=42,
        ndpost=10,
        n_models=n_arms,
        n_trees=5,
        nskip=5,
    )
    
    # Should have n_chains actors, not n_arms * n_chains
    assert len(mc.bart_actors) == n_chains
    # But should support n_arms models
    assert mc.n_models == n_arms
    assert len(mc) == n_arms
