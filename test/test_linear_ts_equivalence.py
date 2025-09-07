import numpy as np
import pytest

from bart_playground.bandit.agents.basic_agents import LinearAgentStable, AgentType
from bart_playground.bandit.agents._deprecated.deprecated_agents import LinearTSAgent

def _round_robin_arms(n_arms, t):
    arms = np.arange(n_arms)
    return int(arms[t % n_arms])


@pytest.mark.parametrize("n_arms,n_features,steps,v,seed", [
    (2, 3, 50, 1.7, 123),
    (3, 5, 100, 0.9, 0),
])
def test_linear_ts_posterior_equivalence(n_arms, n_features, steps, v, seed):
    rng = np.random.default_rng(seed)

    # Initialize agents with matching parameters
    aggr = LinearTSAgent(n_arms=n_arms, n_features=n_features, v=v, random_state=seed)
    stab = LinearAgentStable(agent_type=AgentType.TS, n_arms=n_arms, n_features=n_features, v=v, random_state=seed)

    # Drive identical deterministic updates (no reliance on choose_arm randomness)
    for t in range(steps):
        arm = _round_robin_arms(n_arms, t)
        x = rng.standard_normal(n_features)
        y = float(rng.standard_normal())
        aggr.update_state(arm, x, y)
        stab.update_state(arm, x, y)

    d = aggr.n_features  # equals n_features + 1 due to intercept

    for a in range(n_arms):
        # Posterior mean: B^{-1} m
        theta_aggr = aggr.mean[a].reshape(-1)
        theta_stab = np.linalg.solve(stab.B[a], stab.m2_r[a]).reshape(-1)

        assert np.allclose(theta_aggr, theta_stab, rtol=1e-6, atol=1e-8), \
            f"Posterior means differ for arm {a}"

        # Posterior covariance equality: B_inv from both must match
        B_inv_aggr = aggr.B_inv[a]
        B_inv_stab = np.linalg.inv(stab.B[a])
        assert np.allclose(B_inv_aggr, B_inv_stab, rtol=1e-6, atol=1e-8), \
            f"Posterior covariances differ for arm {a}"

        # Validate that stored sqrt indeed squares back to B_inv
        C1 = aggr.B_inv_sqrt[a]
        assert np.allclose(C1 @ C1.T, B_inv_aggr, rtol=1e-6, atol=1e-8), \
            f"Stored sqrt does not reconstruct B_inv for arm {a}"


@pytest.mark.parametrize("n_arms,n_features,steps,v,seed", [
    (2, 4, 60, 1.3, 999),
])
def test_linear_ts_score_distribution_stats(n_arms, n_features, steps, v, seed):
    rng = np.random.default_rng(seed)

    aggr = LinearTSAgent(n_arms=n_arms, n_features=n_features, v=v, random_state=seed)
    stab = LinearAgentStable(agent_type=AgentType.TS, n_arms=n_arms, n_features=n_features, v=v, random_state=seed)

    # Deterministic updates
    for t in range(steps):
        arm = _round_robin_arms(n_arms, t)
        x = rng.standard_normal(n_features)
        y = float(rng.standard_normal())
        aggr.update_state(arm, x, y)
        stab.update_state(arm, x, y)

    d = aggr.n_features
    num_samples = 20000
    rng_z = np.random.default_rng(42)
    Z = rng_z.standard_normal((d, num_samples))

    x0 = rng.standard_normal(n_features)
    x_aug = np.append(x0, 1.0).reshape(-1, 1)
    for a in range(n_arms):
        theta_aggr = aggr.mean[a].reshape(-1, 1)
        theta_stab = np.linalg.solve(stab.B[a], stab.m2_r[a]).reshape(-1, 1)
        B_inv = aggr.B_inv[a]

        # Two valid covariance square-roots leading to same distribution
        C1 = aggr.B_inv_sqrt[a]
        L = np.linalg.cholesky(stab.B[a])
        C2 = np.linalg.solve(L.T, np.eye(d))

        # Sample score distributions using same Z for variance estimation
        S1 = (theta_aggr.T @ x_aug) + v * (x_aug.T @ C1 @ Z)
        S2 = (theta_stab.T @ x_aug) + v * (x_aug.T @ C2 @ Z)

        # Theoretical stats
        mean_theory = (theta_aggr.T @ x_aug).item()
        var_theory = (v * v * (x_aug.T @ B_inv @ x_aug)).item()

        # Empirical stats should be close and consistent across both agents
        m1, m2 = float(np.mean(S1)), float(np.mean(S2))
        v1, v2 = float(np.var(S1)), float(np.var(S2))

        assert np.isclose(m1, mean_theory, rtol=5e-3, atol=5e-3)
        assert np.isclose(m2, mean_theory, rtol=5e-3, atol=5e-3)
        assert np.isclose(v1, var_theory, rtol=5e-2, atol=1e-2)
        assert np.isclose(v2, var_theory, rtol=5e-2, atol=1e-2)


