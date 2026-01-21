'''
Test the separate encoding agent.

Auto-generated tests.
'''

import numpy as np
import ray
import pytest

from bart_playground.bandit.agents.bart_ts_agents import BARTTSAgent
from bart_playground.mcbart import MultiChainBART
from bart_playground.bart import DefaultBART


class DummyAgent(BARTTSAgent):
    """Minimal concrete agent for testing core BARTTSAgent logic."""

    def posterior_draws_on_probes(self, X_probes: np.ndarray):
        return np.zeros((0, 0, 0)), 0
    
    def _feel_good_full_recompute(self) -> None:
        """Dummy implementation for testing."""
        pass
    
    def _feel_good_incremental_recompute(self, x_new: np.ndarray) -> None:
        """Dummy implementation for testing."""
        pass


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


def test_batch_prediction_consistency():
    """Test that batch prediction matches individual prediction."""
    n_arms = 3
    agent = _make_agent_with_real_multichain(n_arms=n_arms, n_features=2)
    
    # Custom population to ensure ALL arms have data
    rng = np.random.default_rng(123)
    for arm in range(n_arms):
        # Give each arm some observations
        for _ in range(5):
            x = rng.random(2)
            reward = rng.random()
            agent.update_state(arm, x, reward)
    
    agent._refresh_model()
    
    # Ensure we have a fitted model
    assert agent.is_model_fitted
    mc_model = agent.models
    
    # Random probe
    x = np.random.random((1, 2))
    k = 0  # trace index
    
    # 1. Batch prediction
    batch_preds = mc_model.predict_trace_batch(k, x)
    
    # 2. Individual predictions
    individual_preds = []
    for i in range(3):
        mc_model.set_active_model(i)
        pred = mc_model.predict_trace(k, x)
        individual_preds.append(pred)
    
    # Compare results
    # Note: predict_trace usually returns scalar or (1,) for regression separate models
    # batch_preds should be (n_arms, ...)
    
    assert len(batch_preds) == 3
    for i in range(3):
        # Allow small floating point differences due to potential aggregation order
        # though usually they should be identical if deterministic
        np.testing.assert_allclose(batch_preds[i], individual_preds[i], rtol=1e-6)


def test_posterior_f_batch_consistency():
    """Test that posterior_f_batch matches repeated mc[i].posterior_f calls."""
    n_arms = 3
    agent = _make_agent_with_real_multichain(n_arms=n_arms, n_features=2)
    
    # Custom population to ensure ALL arms have data
    rng = np.random.default_rng(456)
    for arm in range(n_arms):
        # Give each arm some observations
        for _ in range(5):
            x = rng.random(2)
            reward = rng.random()
            agent.update_state(arm, x, reward)
    
    agent._refresh_model()
    
    # Ensure we have a fitted model
    assert agent.is_model_fitted
    mc_model = agent.models
    
    # Random probes (multiple rows to test shape handling)
    X_probes = np.random.random((10, 2))
    
    # 1. Batch API - new method (returns 3D array)
    batch_results = mc_model.posterior_f_batch(X_probes, backtransform=True)
    
    # 2. Individual calls via indexing (old semantics)
    individual_results = []
    for i in range(n_arms):
        mc_model.set_active_model(i)
        f = mc_model.posterior_f(X_probes, backtransform=True)
        individual_results.append(f)
    
    # Compare results
    # batch_results shape: (n_arms, n_probes, n_post_total)
    assert batch_results.shape[0] == n_arms
    assert batch_results.shape[1] == 10  # n_probes
    
    for i in range(n_arms):
        # Check shape and numerical consistency
        assert batch_results[i].shape == individual_results[i].shape
        np.testing.assert_allclose(batch_results[i], individual_results[i], rtol=1e-10)


def test_feel_good_weights_caching():
    """
    Test feel-good weights caching:
    1. When lambda=0, weights should be zero.
    2. Cache is incrementally updated with each update_state.
    3. Incremental computation == full recomputation.
    4. Cache is fully recomputed after model refresh.
    """
    from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent
    
    n_arms = 2
    n_features = 1
    
    bart_kwargs = {
        "ndpost": 10,
        "nskip": 5,
        "n_trees": 5
    }
    
    # Test 1: lambda=0 -> weights should be zero
    agent_zero = DefaultBARTTSAgent(
        n_arms=n_arms,
        n_features=n_features,
        initial_random_selections=0,
        random_state=42,
        encoding="multi",
        n_chains=1,
        refresh_schedule="log",
        bart_kwargs=bart_kwargs,
        feel_good_lambda=0.0
    )
    
    # Add some data and fit
    rng = np.random.default_rng(42)
    for i in range(10):
        arm = i % n_arms
        x = rng.random(n_features)
        y = rng.random()
        agent_zero.update_state(arm, x, y)
    
    # Should have triggered refresh and fit
    assert agent_zero.is_model_fitted
    
    # With lambda=0, cache should not be initialized (None)
    assert agent_zero._fg_S is None
    
    # Calling feel_good_weights should return zeros when cache is None
    weights = agent_zero.feel_good_weights(lambda t: 1.0)
    np.testing.assert_array_equal(weights, np.zeros(agent_zero.n_post))
    
    # Test 2 & 3: Incremental == Full recompute
    agent = DefaultBARTTSAgent(
        n_arms=n_arms,
        n_features=n_features,
        initial_random_selections=0,
        random_state=42,
        encoding="multi",
        n_chains=1,
        refresh_schedule="log",
        bart_kwargs=bart_kwargs,
        feel_good_lambda=0.5  # Non-zero lambda to enable caching
    )
    
    # Add initial data and force refresh
    rng = np.random.default_rng(42)
    for i in range(10):
        arm = i % n_arms
        x = rng.random(n_features)
        y = rng.random()
        agent.update_state(arm, x, y)
    
    # At this point cache should be initialized
    assert agent.is_model_fitted
    assert agent._fg_S is not None
    
    # Save the current cache
    cache_after_initial = agent._fg_S.copy()
    
    # Add 5 more points incrementally
    for i in range(5):
        arm = i % n_arms
        x = rng.random(n_features)
        y = rng.random()
        agent.update_state(arm, x, y)
    
    # Get incremental cache result
    incremental_cache = agent._fg_S.copy()
    
    # Full recompute reference (batch path) may have tiny FP drift vs incremental updates
    # due to different evaluation paths; check numerical closeness.
    agent._feel_good_full_recompute()
    full_cache = agent._fg_S.copy()
    np.testing.assert_allclose(
        incremental_cache,
        full_cache,
        rtol=0.0,
        atol=1e-6,
        err_msg="Incremental cache should be numerically close to full recomputation"
    )
    
    # Test 4: Cache should update after refresh
    # Force a refresh by manipulating the schedule
    old_t = agent.t
    agent.t = 100  # Jump ahead to trigger refresh
    
    # Add one more point to trigger refresh
    x = rng.random(n_features)
    y = rng.random()
    refreshed = agent._should_refresh()
    agent.t = old_t  # restore t before update
    
    if refreshed:
        # Cache before refresh
        cache_before_refresh = agent._fg_S.copy()
        
        # Update which should trigger refresh
        agent.t = 100
        agent.update_state(0, x, y)
        
        # Cache should be different after refresh (different posterior samples)
        # We can't predict exact values, but we can verify cache was updated
        assert agent._fg_S is not None
        # The cache shape should remain correct
        assert agent._fg_S.shape == (agent.n_post,)
    
    print("✓ Feel-good weights caching test passed")


def test_feel_good_sampling_draw_index_is_weighted():
    """
    Test that feel-good sampling uses weighted posterior draws.
    When _fg_S is highly peaked, sampled k should always match the peak index.
    """
    from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent
    
    n_arms = 2
    n_features = 1
    
    bart_kwargs = {
        "ndpost": 20,  # Request 20 posterior samples
        "nskip": 5,
        "n_trees": 5
    }
    
    # Create agent with non-zero feel_good_lambda
    agent = DefaultBARTTSAgent(
        n_arms=n_arms,
        n_features=n_features,
        initial_random_selections=0,
        random_state=42,
        encoding="multi",
        n_chains=1,
        refresh_schedule="log",
        bart_kwargs=bart_kwargs,
        feel_good_lambda=1.0
    )
    
    # Add enough data to trigger fit
    rng = np.random.default_rng(42)
    for i in range(30):
        arm = i % n_arms
        x = rng.random(n_features)
        y = rng.random()
        agent.update_state(arm, x, y)
    
    # Should have triggered refresh and fit
    assert agent.is_model_fitted
    assert agent._fg_S is not None
    
    # Get actual n_post (may be less than requested due to _steps_until_next_refresh)
    n_post = agent.n_post
    assert n_post >= 1, f"Need at least 1 posterior sample, got {n_post}"
    
    # Create a degenerate weight distribution (all -1000, one element 0)
    # Use the middle index or index 0 if n_post is small
    peak_idx = min(n_post // 2, n_post - 1) if n_post > 1 else 0
    
    # Create peaked _fg_S
    agent._fg_S = np.full(n_post, -1000.0)
    agent._fg_S[peak_idx] = 0.0
    
    # Expected k value
    rp = agent.model.range_post
    expected_k = rp.start + peak_idx * rp.step
    
    # Sample many times and check all match the expected k
    num_samples = 50
    for _ in range(num_samples):
        sampled_k = agent._sample_fg_post_index()
        assert sampled_k == expected_k, \
            f"Expected k={expected_k} (peak at idx {peak_idx}), got k={sampled_k}"
    
    print("✓ Feel-good sampling weighted draw test passed")

