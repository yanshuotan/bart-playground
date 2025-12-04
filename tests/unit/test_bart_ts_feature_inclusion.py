import numpy as np
import pytest

from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent


def _fit_separate_agent(
    n_arms: int = 2,
    n_features: int = 3,
    rng_seed: int = 0,
) -> DefaultBARTTSAgent:
    """
    Helper: build a small separate-encoding agent and force a refresh so that
    feature_inclusion can be queried deterministically.
    """
    agent = DefaultBARTTSAgent(
        n_arms=n_arms,
        n_features=n_features,
        encoding="separate",
        initial_random_selections=0,
        random_state=rng_seed,
        bart_kwargs={"ndpost": 10, "nskip": 10, "n_trees": 20},
    )

    rng = np.random.default_rng(rng_seed)
    # Provide enough observations per arm so that _refresh_model can fit.
    for arm in range(n_arms):
        for _ in range(8):
            x = rng.normal(size=n_features)
            reward = x[0] * (arm + 1) + rng.normal(scale=0.1)
            agent.update_state(arm, x, reward)

    agent._refresh_model()
    assert agent.is_model_fitted
    return agent


def test_feature_inclusion_returns_normalized_vector():
    agent = _fit_separate_agent()
    result = agent.feature_inclusion()

    assert set(result.keys()) == {"meta", "metrics"}
    df = result["metrics"]
    assert len(df) == agent.n_features
    assert np.all(df["inclusion"].to_numpy() >= 0.0)
    total = float(df["inclusion"].sum())
    assert total > 0.0
    np.testing.assert_allclose(total, 1.0, atol=1e-6)

    # Calling again without modifying the model should be deterministic.
    result2 = agent.feature_inclusion()
    np.testing.assert_allclose(
        df["inclusion"].to_numpy(),
        result2["metrics"]["inclusion"].to_numpy(),
        atol=0,
        rtol=0,
    )

    for model in agent.models:
        if hasattr(model, "clean_up"):
            model.clean_up()


def test_feature_inclusion_requires_separate_encoding():
    agent = DefaultBARTTSAgent(
        n_arms=2,
        n_features=2,
        encoding="multi",
        initial_random_selections=0,
        bart_kwargs={"ndpost": 10, "nskip": 10, "n_trees": 10},
    )

    with pytest.raises(NotImplementedError):
        agent.feature_inclusion()

