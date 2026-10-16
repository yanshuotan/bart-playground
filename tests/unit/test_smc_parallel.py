"""Particle identities, outputs and cleanup for optional posterior processes."""

from copy import deepcopy
import multiprocessing as mp

import numpy as np
import pytest

from bart_playground import SMCBART
from bart_playground.smc import resolve_smc_workers, _transport_particle


@pytest.fixture
def fitted():
    rng = np.random.default_rng(719)
    X = rng.normal(size=(24, 2))
    y = X[:, 0] + rng.normal(size=len(X)) * .3
    model = SMCBART(n_particles=5, n_trees=3, betas=[0, .3, 1],
                    random_state=93).fit(X, y, quietly=True)
    return model, X[:7]


def assert_endpoints_equal(left, right):
    for a, b in zip(left, right):
        assert a.cache is None and b.cache is None
        for key in a.global_params:
            np.testing.assert_array_equal(a.global_params[key], b.global_params[key])
        for ta, tb in zip(a.trees, b.trees):
            assert ta.dataX is not None and tb.dataX is not None
            assert ta.n is None and tb.n is None
            for name in ("vars", "thresholds", "leaf_vals"):
                np.testing.assert_array_equal(getattr(ta, name), getattr(tb, name))


def test_serial_and_spawned_workers_are_bitwise_equal(fitted, tmp_path):
    model, X = fitted
    prediction_before = model.posterior_f(X).copy()
    annealing_rng = deepcopy(model.sampler.generator.bit_generator.state)
    reference = model.sample_posterior(X, 4, 2, random_state=345)
    final = model.posterior_population_
    counts = deepcopy(model.posterior_move_counts_)
    ancestry = model.posterior_resampling_indices_.copy()
    assert model.posterior_workers_ == 1
    assert model.posterior_sampling_complete_
    assert sum(counts["move_selected_counts"].values()) == 5*6*3
    assert reference["preds"].shape == (20, 7)
    assert reference["vars_histogram"].shape == (20, 2)
    # The final sweep's numeric summaries belong to the matching endpoint ID.
    for particle, state in enumerate(final):
        np.testing.assert_array_equal(reference["preds"][-5+particle],
            model.preprocessor.backtransform_y(state.evaluate(X)))
        np.testing.assert_array_equal(reference["vars_histogram"][-5+particle], state.vars_histogram)
        assert reference["sigmas"][-5+particle] == state.global_params["eps_sigma2"][0]
    for workers in (2, 3):  # Uneven grouping, independent of scheduler order.
        model.n_jobs = workers
        progress = []
        actual = model.sample_posterior(X, 4, 2, random_state=345, temp_dir=tmp_path,
            progress_callback=lambda done, total: progress.append((done, total)))
        assert progress == [(i, 5) for i in range(1, 6)]
        assert model.posterior_workers_ == workers
        assert model.posterior_sampling_complete_
        for name in reference:
            np.testing.assert_array_equal(actual[name], reference[name])
        assert_endpoints_equal(final, model.posterior_population_)
        assert model.posterior_move_counts_ == counts
        np.testing.assert_array_equal(model.posterior_resampling_indices_, ancestry)
        assert list(tmp_path.iterdir()) == []
    assert model.sampler.generator.bit_generator.state == annealing_rng
    np.testing.assert_array_equal(model.posterior_f(X), prediction_before)


def test_duplicate_ancestors_have_distinct_offspring_streams_and_burn_is_a_prefix(fitted):
    model, X = fitted
    model.weights_ = np.array([0., 1., 0., 0., 0.])
    all_draws = model.sample_posterior(X, 5, 0, random_state=15)
    final = model.posterior_population_
    tail = model.sample_posterior(X, 3, 2, random_state=15)
    np.testing.assert_array_equal(model.posterior_resampling_indices_, np.ones(5, dtype=int))
    for name in tail:
        np.testing.assert_array_equal(all_draws[name][10:], tail[name])
    assert_endpoints_equal(final, model.posterior_population_)
    assert len(np.unique(all_draws["preds"][:5, 0])) == 5
    assert not np.shares_memory(final[0].global_params["eps_sigma2"], final[1].global_params["eps_sigma2"])


def test_memmap_outputs_keep_layout_and_are_returned_without_copying(fitted, tmp_path):
    model, X = fitted
    model.n_jobs = 2
    specs = {"preds": ((10, 7), float), "sigmas": ((10,), float),
             "vars_histogram": ((10, 2), np.int32)}
    outputs = {name: np.lib.format.open_memmap(tmp_path/f"{name}.npy", mode="w+", shape=shape, dtype=dtype)
               for name, (shape, dtype) in specs.items()}
    returned = model.sample_posterior(X, 2, 0, random_state=3, outputs=outputs, temp_dir=tmp_path)
    assert returned is outputs
    for name in outputs:
        outputs[name].flush()
        np.testing.assert_array_equal(np.load(tmp_path/f"{name}.npy"), returned[name])
    assert not list(tmp_path.glob("bart-smc-*"))


def test_worker_failure_propagates_and_releases_shared_files(fitted, tmp_path):
    model, X = fitted
    model.n_jobs = 2
    before = {p.pid for p in mp.active_children()}
    model.data.y = model.data.y[:-1]  # Trigger a shape error inside spawned mutation.
    with pytest.raises(ValueError):
        model.sample_posterior(X, 2, 0, random_state=1, temp_dir=tmp_path)
    assert not model.posterior_sampling_complete_
    assert model.posterior_population_ == []
    assert list(tmp_path.iterdir()) == []
    assert {p.pid for p in mp.active_children()} == before


def test_callback_failure_cleans_up_and_default_serial_does_not_spawn(fitted, tmp_path, monkeypatch):
    model, X = fitted
    model.n_jobs = 2
    before = {p.pid for p in mp.active_children()}

    def stop(done, total):
        raise RuntimeError("stop collection")

    with pytest.raises(RuntimeError, match="stop collection"):
        model.sample_posterior(X, 2, 0, random_state=7, progress_callback=stop, temp_dir=tmp_path)
    assert not model.posterior_sampling_complete_
    assert model.posterior_population_ == []
    assert list(tmp_path.iterdir()) == []
    assert {p.pid for p in mp.active_children()} == before

    def no_processes(*args, **kwargs):
        raise AssertionError("serial collection must not construct a process pool")

    monkeypatch.setattr("bart_playground.smc.ProcessPoolExecutor", no_processes)
    model.n_jobs = 1
    model.sample_posterior(X, 1, 0, random_state=7)
    assert model.posterior_sampling_complete_


def test_transport_removes_training_references_without_changing_source(fitted):
    model, _ = fitted
    original = model.particles_[0]
    packed = _transport_particle(original)
    assert all(t.dataX is None for t in packed.trees)
    assert all(t.dataX is not None for t in original.trees)


@pytest.mark.parametrize("value", [0, 1.5, True, "2"])
def test_invalid_jobs_are_rejected_even_with_one_particle(value):
    with pytest.raises(ValueError, match="n_jobs"):
        SMCBART(n_particles=1, n_jobs=value)


def test_worker_resolution_matches_pt_and_does_not_cap_explicit_server_counts(monkeypatch):
    monkeypatch.setattr("bart_playground.bart._physical_core_count", lambda: 8)
    assert resolve_smc_workers(1, 50) == 1
    assert resolve_smc_workers(None, 50) == 50
    assert resolve_smc_workers(-1, 50) == 8
    assert resolve_smc_workers(-2, 50) == 7
    assert resolve_smc_workers(-99, 50) == 1
    assert resolve_smc_workers(32, 50) == 32
    assert resolve_smc_workers(99, 50) == 50
    assert SMCBART().get_params()["n_jobs"] == 1


def test_collection_requires_fit_and_valid_shapes(fitted):
    with pytest.raises(ValueError, match="Fit SMCBART"):
        SMCBART().sample_posterior(np.ones((2, 1)))
    model, X = fitted
    for kwargs in ({"n_sweeps": 0}, {"burn_sweeps": -1}, {"burn_sweeps": True}):
        with pytest.raises(ValueError):
            model.sample_posterior(X, **kwargs)
    for bad_X in (np.zeros((2, 3)), np.empty((0, 2)), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError, match="X must be"):
            model.sample_posterior(bad_X)
    with pytest.raises(ValueError, match="outputs"):
        model.sample_posterior(X, 2, outputs={"preds": np.zeros((10, 7), dtype=np.float32)})
