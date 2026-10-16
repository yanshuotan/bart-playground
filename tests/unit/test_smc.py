"""Numerical and posterior checks for ensemble-level annealed SMC."""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from bart_playground import SMCBART
from bart_playground.smc import (
    _copy_particle, _effective_sample_size, _normalize_log_weights,
    _systematic_resample,
)


def toy_data():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(30, 2))
    return X, X[:, 0] + rng.normal(scale=0.25, size=len(X))


def prepare_prior(model, X, y):
    model.data = model.preprocessor.fit_transform(X, y)
    model.sampler.add_data(model.data)
    model.sampler.add_thresholds(model.preprocessor.thresholds)
    model.sampler.global_prior.fit_hyperparameters(model.data)


def test_log_weights_and_resampling_stay_stable():
    weights, normalizer = _normalize_log_weights(np.array([-10001.0, -10000.0]))
    np.testing.assert_allclose(np.exp(weights), [1 / (1 + np.e), np.e / (1 + np.e)])
    assert np.isfinite(normalizer)
    assert _effective_sample_size(np.full(4, -np.log(4))) == pytest.approx(4)
    indices = _systematic_resample(np.array([-np.inf, 0.0, -np.inf]), np.random.default_rng(1), 50)
    np.testing.assert_array_equal(indices, np.ones(50, dtype=int))
    with pytest.raises(FloatingPointError):
        _normalize_log_weights([-np.inf, -np.inf])


def test_adaptive_beta_uses_conditional_ess_with_unequal_weights():
    model = SMCBART(n_particles=3, n_trees=1, ess_target=0.8)
    log_weights = np.log([0.8, 0.15, 0.05])
    likelihoods = np.array([-1000.0, -10.0, -1.0])
    beta = model._next_beta(0, log_weights, likelihoods)
    assert 0 < beta < 1
    log_fraction = (2 * logsumexp(log_weights + beta * likelihoods)
                    - logsumexp(log_weights + 2 * beta * likelihoods))
    assert np.exp(log_fraction) == pytest.approx(0.8, abs=1e-10)
    assert model._next_beta(0.2, log_weights, np.zeros(3)) == 1


def test_fixed_schedule_weights_and_evidence_use_full_likelihood(monkeypatch):
    X, y = toy_data()
    model = SMCBART(n_particles=4, n_trees=2, betas=[0, 0.2, 0.6, 1],
                    fixed_eps_sigma2=0.2, resample_threshold=0.01)
    prepare_prior(model, X, y)
    particles = [model._sample_prior_state() for _ in range(4)]
    log_likelihoods = model._log_likelihoods(particles)
    # With identity mutation and no resampling, all increments telescope.
    iterator = iter(particles)
    monkeypatch.setattr(model, "_sample_prior_state", lambda: next(iterator))
    monkeypatch.setattr(model.sampler, "one_iter", lambda state, temp: state)
    model.fit_with_data(model.data, quietly=True)
    np.testing.assert_allclose(model.weights_, np.exp(log_likelihoods - logsumexp(log_likelihoods)))
    assert model.log_evidence_ == pytest.approx(logsumexp(log_likelihoods) - np.log(4))
    assert model.beta_history_ == [0, 0.2, 0.6, 1]
    assert not any(stage["resampled"] for stage in model.stage_diagnostics_)


@pytest.mark.parametrize("fixed_sigma", [None, 0.08])
def test_fit_reaches_posterior_and_preserves_prediction_api(fixed_sigma):
    X, y = toy_data()
    model = SMCBART(n_particles=12, ndpost=17, n_trees=4, random_state=12,
                    fixed_eps_sigma2=fixed_sigma, resample_threshold=0.95)
    assert model.fit(X, y, quietly=True) is model
    assert model.is_fitted
    assert model.beta_history_[0] == 0 and model.beta_history_[-1] == 1
    assert np.all(np.diff(model.beta_history_) > 0)
    assert any(stage["resampled"] for stage in model.stage_diagnostics_)
    assert np.isfinite(model.log_evidence_)
    assert model.weights_.sum() == pytest.approx(1)
    assert len(model.particles_) == 12
    assert len(model.trace) == 17
    assert model.posterior_f(X).shape == (30, 17)
    assert model.posterior_predict(X).shape == (30, 17)
    assert np.all(np.isfinite(model.predict(X)))
    weighted = sum(w * state.evaluate(X) for w, state in zip(model.weights_, model.particles_))
    np.testing.assert_allclose(model.predict(X), model.preprocessor.backtransform_y(weighted))
    for state in model.trace + model.particles_:
        assert state.cache is None
        if fixed_sigma is not None:
            assert state.global_params["eps_sigma2"][0] == fixed_sigma


def test_reproducible_population_and_independent_copies():
    X, y = toy_data()
    kwargs = dict(n_particles=8, n_trees=3, random_state=142, betas=[0, 0.1, 0.5, 1])
    first = SMCBART(**kwargs).fit(X, y, quietly=True)
    second = SMCBART(**kwargs).fit(X, y, quietly=True)
    np.testing.assert_array_equal(first.weights_, second.weights_)
    np.testing.assert_array_equal(first.posterior_f(X), second.posterior_f(X))
    assert first.log_evidence_ == second.log_evidence_
    prepare_prior(first, X, y)
    state = first._sample_prior_state()
    a, b = _copy_particle(state), _copy_particle(state)
    a.global_params["eps_sigma2"][0] = 999
    a.trees[0].leaf_vals[a.trees[0].leaves] = 999
    assert b.global_params["eps_sigma2"][0] != 999
    assert np.all(b.trees[0].leaf_vals[b.trees[0].leaves] != 999)
    assert not np.shares_memory(first.trace[0].global_params["eps_sigma2"],
                                first.trace[1].global_params["eps_sigma2"])


def test_prior_tree_frequencies_match_enumerated_admissible_prior():
    # Both thresholds (0, 0.5) split the two observations under the tree's <=
    # convention. Children must stop. The admissible prior is enumerable.
    alpha, beta = 0.6, 2.0
    model = SMCBART(n_trees=1, tree_alpha=alpha, tree_beta=beta, max_bins=2)
    prepare_prior(model, np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    split_mass = alpha * (1 - alpha / 2**beta)**2
    expected = split_mass / (1 - alpha + split_mass)
    trees = [model._sample_prior_tree() for _ in range(2000)]
    observed = np.mean([tree.n_leaves == 2 for tree in trees])
    assert observed == pytest.approx(expected, abs=0.035)
    assert all(np.all(tree.n[tree.leaves] > 0) for tree in trees)


def test_constant_features_match_gaussian_analytic_posterior_and_evidence():
    # Admissible trees are stumps: f is exactly a scalar Gaussian prior.
    X = np.zeros((10, 1))
    y = np.linspace(-0.2, 0.5, len(X))
    sigma2 = 0.1
    model = SMCBART(n_particles=1024, n_trees=1, tree_alpha=0.2,
                    fixed_eps_sigma2=sigma2, random_state=831,
                    resample_threshold=1).fit(X, y, quietly=True)
    scaled_y = model.data.y.astype(float)
    prior_var = model.sampler.tree_prior.f_sigma2
    post_var = 1 / (1 / prior_var + len(y) / sigma2)
    post_mean = post_var * scaled_y.sum() / sigma2
    draws = np.array([state.evaluate(X[:1])[0] for state in model.particles_])
    assert np.mean(draws) == pytest.approx(post_mean, abs=5 * np.sqrt(post_var / len(draws)))
    assert np.var(draws) == pytest.approx(post_var, rel=0.15)
    exact_log_evidence = multivariate_normal.logpdf(
        scaled_y, mean=np.zeros(len(y)), cov=sigma2 * np.eye(len(y)) + prior_var,
    )
    assert model.log_evidence_ == pytest.approx(exact_log_evidence, abs=0.2)


def test_mutation_refreshes_leaf_values_when_no_structure_move_is_possible():
    model = SMCBART(n_particles=2, n_trees=1, proposal_probs={"change": 1.0},
                    fixed_eps_sigma2=0.1)
    prepare_prior(model, np.zeros((10, 1)), np.linspace(0, 1, 10))
    state = model._sample_prior_state()
    before = state.trees[0].leaf_vals[0]
    updated = model.sampler.one_iter(state, temp=2)
    assert updated.trees[0].leaf_vals[0] != before
    assert updated.trees[0].n_leaves == 1


def test_noise_prior_draw_and_mutation_are_not_clipped():
    model = SMCBART(n_trees=1, tree_alpha=0.2)
    prepare_prior(model, np.zeros((10, 1)), np.zeros(10))
    prior_sigma2 = model._sample_prior_state().global_params["eps_sigma2"][0]
    mutation_sigma2 = model.sampler.global_prior._sample_eps_sigma2(np.zeros(10))[0]
    assert 0 < prior_sigma2 < 1e-8
    assert 0 < mutation_sigma2 < 1e-8


def test_two_observations_match_enumerated_structure_posterior():
    X, y = np.array([[0.0], [1.0]]), np.array([-1.0, 1.0])
    alpha, beta, sigma2 = 0.6, 2.0, 0.1
    model = SMCBART(n_particles=1600, n_trees=1, tree_alpha=alpha,
                    tree_beta=beta, max_bins=2, fixed_eps_sigma2=sigma2,
                    resample_threshold=1, random_state=938).fit(X, y, quietly=True)
    v = model.sampler.tree_prior.f_sigma2
    scaled_y = model.data.y.astype(float)
    stump_log_mass = np.log(1 - alpha) + multivariate_normal.logpdf(
        scaled_y, cov=sigma2 * np.eye(2) + v,
    )
    split_log_mass = np.log(alpha * (1 - alpha / 2**beta)**2) + multivariate_normal.logpdf(
        scaled_y, cov=(sigma2 + v) * np.eye(2),
    )
    expected = np.exp(split_log_mass - np.logaddexp(stump_log_mass, split_log_mass))
    actual = sum(w * (state.trees[0].n_leaves == 2)
                 for w, state in zip(model.weights_, model.particles_))
    assert actual == pytest.approx(expected, abs=0.055)


def test_stage_limit_never_claims_a_posterior_fit():
    X, y = toy_data()
    model = SMCBART(n_particles=8, n_trees=2, max_stages=1, ess_target=0.99)
    with pytest.raises(RuntimeError, match="did not reach beta=1"):
        model.fit(X, y, quietly=True)
    assert not model.is_fitted
    assert not model.trace
    assert model.log_evidence_ is None


def test_refit_and_update_replace_population():
    X, y = toy_data()
    model = SMCBART(n_particles=6, n_trees=2, betas=[0, 0.1, 1]).fit(X[:20], y[:20], quietly=True)
    model.update_fit(X[20:], y[20:], add_ndpost=9, quietly=True)
    assert model.data.n == 30
    assert len(model.trace) == 9
    assert model.beta_history_ == [0, 0.1, 1]
    model.fit(X, y, quietly=True)
    assert len(model.stage_diagnostics_) == 2
    assert model.get_params()["model_type"] == "SMCBART"


def test_posterior_chain_rebuilds_caches_and_uses_independent_default_kernel():
    from bart_playground import DefaultBART
    from bart_playground.samplers import DefaultSampler

    X, y = toy_data()
    smc = SMCBART(n_particles=6, n_trees=3, betas=[0, 0.2, 1]).fit(X, y, quietly=True)
    chain = smc.initialize_posterior_chain(random_state=1234)
    again = smc.initialize_posterior_chain(random_state=1234)
    assert isinstance(chain, DefaultBART)
    assert type(chain.sampler) is DefaultSampler
    assert chain.sampler.tol == 1
    assert chain.smc_particle_index_ == again.smc_particle_index_
    selected = smc.particles_[chain.smc_particle_index_]
    np.testing.assert_allclose(chain.trace[0].evaluate(), selected.evaluate(smc.data.X))
    for tree in chain.trace[0].trees:
        np.testing.assert_array_equal(tree.leaf_ids, tree.traverse_tree(smc.data.X))
        assert tree.n[0] == len(X)
        assert np.all(tree.n[tree.leaves] > 0)
    state = chain.trace[0]
    for _ in range(5):
        state = chain.sampler.one_iter(state, temp=1)
    assert np.all(np.isfinite(state.evaluate()))
    assert selected.cache is None
    assert all(tree.n is None for tree in selected.trees)
    assert chain.sampler.generator is not smc.sampler.generator
    assert not np.shares_memory(state.trees[0].leaf_vals, selected.trees[0].leaf_vals)


def test_posterior_chain_requires_fitted_population():
    with pytest.raises(ValueError, match="Fit SMCBART"):
        SMCBART().initialize_posterior_chain(1)


def test_posterior_population_keeps_every_particle_and_genuinely_updates():
    X, y = toy_data()
    smc = SMCBART(n_particles=5, n_trees=3, betas=[0, 0.2, 1]).fit(X, y, quietly=True)
    original = smc.posterior_f(X).copy()
    snapshots = []
    progress = []
    for sweep, population in smc.iter_posterior_populations(
            n_sweeps=4, burn_sweeps=2, random_state=123,
            progress_callback=lambda completed, total: progress.append((completed, total))):
        assert len(population) == 5
        snapshots.append(np.array([state.evaluate(X).copy() for state in population]))
    assert np.array(snapshots).shape == (4, 5, 30)
    assert progress == [(i, 6) for i in range(1, 7)]
    assert np.all(np.any(np.diff(snapshots, axis=0) != 0, axis=2))
    assert len(smc.posterior_population_) == 5
    assert sum(smc.population_sampler_.move_selected_counts.values()) == 6 * 5 * 3
    np.testing.assert_array_equal(original, smc.posterior_f(X))
    assert all(state.cache is None for state in smc.posterior_population_)
    repeated = [np.array([s.evaluate(X).copy() for s in p]) for _, p in
                smc.iter_posterior_populations(4, 2, random_state=123)]
    np.testing.assert_array_equal(snapshots, repeated)


def test_terminal_resampling_restores_equal_weight_population_without_aliasing():
    X, y = toy_data()
    smc = SMCBART(n_particles=4, n_trees=1, betas=[0, 1]).fit(X, y, quietly=True)
    smc.weights_ = np.array([0., 1., 0., 0.])
    _, states = next(smc.iter_posterior_populations(1, 0, random_state=1))
    np.testing.assert_array_equal(smc.posterior_resampling_indices_, [1, 1, 1, 1])
    assert len({id(state) for state in states}) == 4
    for left, right in zip(states, states[1:]):
        assert not np.shares_memory(left.trees[0].leaf_vals, right.trees[0].leaf_vals)
    assert len({float(state.trees[0].leaf_vals[state.trees[0].leaves][0]) for state in states}) > 1


@pytest.mark.parametrize("kwargs", [
    {"n_particles": 0}, {"n_particles": 2.5}, {"ndpost": 0}, {"n_trees": 0},
    {"n_mcmc_steps": 0}, {"ess_target": 1}, {"resample_threshold": 0},
    {"max_stages": 0}, {"tree_alpha": 1}, {"tree_beta": 0},
    {"fixed_eps_sigma2": -1}, {"f_k": np.nan}, {"eps_nu": 0},
    {"betas": [0, 0.5]}, {"betas": [0, 0.5, 0.5, 1]},
    {"betas": [0, np.nan, 1]}, {"betas": [0, 0.5, 1], "max_stages": 1},
    {"proposal_probs": {"grow": 1}}, {"proposal_probs": {"multi_grow": 1}},
    {"proposal_probs": {"change": -1, "swap": 2}},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SMCBART(**kwargs)


@pytest.mark.parametrize("X,y", [
    (np.zeros((0, 1)), np.zeros(0)), (np.zeros((2, 0)), np.zeros(2)),
    (np.zeros((3, 1)), np.zeros(2)), (np.zeros((2, 1)), np.array([np.nan, 1])),
    (np.array([[np.inf], [0]]), np.zeros(2)), (np.zeros(2), np.zeros(2)),
])
def test_invalid_data(X, y):
    with pytest.raises(ValueError, match="finite, nonempty"):
        SMCBART(n_particles=2, n_trees=1).fit(X, y, quietly=True)


def test_impossible_quick_decay_prior_fails_explicitly():
    with pytest.raises(RuntimeError, match="admissible prior tree"):
        SMCBART(n_particles=2, n_trees=1, quick_decay=True, tree_alpha=0.3,
                max_prior_attempts=5).fit(np.zeros((5, 1)), np.arange(5), quietly=True)
