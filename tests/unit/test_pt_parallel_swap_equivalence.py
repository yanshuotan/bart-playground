"""The parallel PT swap path must reproduce the serial chain exactly."""

import numpy as np
import pytest

from bart_playground.bart import ParallelTemperingBART, _split_chains_across_workers


def test_parallel_local_moves_match_serial_before_swap():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    common = dict(
        ndpost=3, nskip=0, n_trees=6,
        temperatures=[1.0, 1.4, 2.0, 3.0],
        swap_interval=10, random_state=42, store_chain_traces=True,
    )
    serial = ParallelTemperingBART(n_jobs=1, **common).fit(X, y, quietly=True)
    threaded = ParallelTemperingBART(n_jobs=4, **common).fit(X, y, quietly=True)
    for serial_chain, threaded_chain in zip(serial.chain_traces, threaded.chain_traces):
        for serial_state, threaded_state in zip(serial_chain, threaded_chain):
            for serial_tree, threaded_tree in zip(serial_state.trees, threaded_state.trees):
                np.testing.assert_array_equal(serial_tree.leaf_vals, threaded_tree.leaf_vals)


@pytest.mark.parametrize("sampler_kind", ["default", "multi"])
@pytest.mark.parametrize("swap_sweeps", [None, 2])
@pytest.mark.parametrize("post_swap_repair_steps", [0, 1])
def test_parallel_swap_matches_serial(sampler_kind, swap_sweeps, post_swap_repair_steps):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    common = dict(
        ndpost=12,
        nskip=3,
        n_trees=6,
        temperatures=[1.0, 1.4, 2.0, 3.0],
        swap_interval=3,
        swap_sweeps=swap_sweeps,
        random_state=42,
        sampler_kind=sampler_kind,
        multi_tries=3,
        store_swap_diagnostics=True,
        post_swap_repair_steps=post_swap_repair_steps,
    )

    serial = ParallelTemperingBART(n_jobs=1, **common).fit(X, y, quietly=True)
    parallel = ParallelTemperingBART(n_jobs=4, **common).fit(X, y, quietly=True)

    np.testing.assert_array_equal(serial.swap_attempt_counts, parallel.swap_attempt_counts)
    np.testing.assert_array_equal(serial.swap_accept_counts, parallel.swap_accept_counts)
    assert serial.swap_diagnostics == parallel.swap_diagnostics
    assert len(serial.trace) == len(parallel.trace) == common["ndpost"]

    for serial_state, parallel_state in zip(serial.trace, parallel.trace):
        assert serial_state.global_params.keys() == parallel_state.global_params.keys()
        for key in serial_state.global_params:
            np.testing.assert_array_equal(
                serial_state.global_params[key], parallel_state.global_params[key]
            )
        for serial_tree, parallel_tree in zip(serial_state.trees, parallel_state.trees):
            for attribute in ("vars", "thresholds", "leaf_vals", "n", "leaf_ids"):
                np.testing.assert_array_equal(
                    getattr(serial_tree, attribute), getattr(parallel_tree, attribute)
                )

    for serial_sampler, parallel_sampler in zip(
        serial.chain_samplers, parallel.chain_samplers
    ):
        assert (
            serial_sampler.generator.bit_generator.state
            == parallel_sampler.generator.bit_generator.state
        )


def _assert_states_equal(state_a, state_b):
    assert state_a.global_params.keys() == state_b.global_params.keys()
    for key in state_a.global_params:
        np.testing.assert_array_equal(state_a.global_params[key], state_b.global_params[key])
    for tree_a, tree_b in zip(state_a.trees, state_b.trees):
        for attribute in ("vars", "thresholds", "leaf_vals"):
            np.testing.assert_array_equal(getattr(tree_a, attribute), getattr(tree_b, attribute))


def _toy_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(size=40)
    return X, y


def test_last_block_off_swap_boundary_records_final_draw():
    """A shorter run must be an exact prefix of a longer run with the same seed.

    nskip + ndpost = 17 is not a multiple of swap_interval = 5, so the final
    block (iterations 15-16) does not end on a swap boundary.
    """
    X, y = _toy_data()
    common = dict(
        nskip=3, n_trees=6, temperatures=[1.0, 1.4, 2.0, 3.0],
        swap_interval=5, random_state=42, store_chain_traces=True,
    )
    short = ParallelTemperingBART(ndpost=14, **common).fit(X, y, quietly=True)
    long = ParallelTemperingBART(ndpost=17, **common).fit(X, y, quietly=True)
    for state_short, state_long in zip(short.trace, long.trace):
        _assert_states_equal(state_short, state_long)
    for chain_short, chain_long in zip(short.chain_traces, long.chain_traces):
        for state_short, state_long in zip(chain_short, chain_long):
            _assert_states_equal(state_short, state_long)


def test_single_temperature_does_not_depend_on_swap_interval():
    """With one chain there are no swaps, so block size must not change the chain."""
    X, y = _toy_data()
    common = dict(ndpost=12, nskip=3, n_trees=6, temperatures=[1.0], random_state=42)
    blocked = ParallelTemperingBART(swap_interval=5, **common).fit(X, y, quietly=True)
    stepwise = ParallelTemperingBART(swap_interval=1, **common).fit(X, y, quietly=True)
    assert len(blocked.trace) == len(stepwise.trace) == common["ndpost"]
    for state_blocked, state_stepwise in zip(blocked.trace, stepwise.trace):
        _assert_states_equal(state_blocked, state_stepwise)


def test_parallel_matches_serial_off_swap_boundary():
    X, y = _toy_data()
    common = dict(
        ndpost=14, nskip=3, n_trees=6, temperatures=[1.0, 1.4, 2.0, 3.0],
        swap_interval=5, random_state=42, store_chain_traces=True,
        store_swap_diagnostics=True,
    )
    serial = ParallelTemperingBART(n_jobs=1, **common).fit(X, y, quietly=True)
    parallel = ParallelTemperingBART(n_jobs=4, **common).fit(X, y, quietly=True)
    assert serial.swap_diagnostics == parallel.swap_diagnostics
    for state_serial, state_parallel in zip(serial.trace, parallel.trace):
        _assert_states_equal(state_serial, state_parallel)
    for chain_serial, chain_parallel in zip(serial.chain_traces, parallel.chain_traces):
        for state_serial, state_parallel in zip(chain_serial, chain_parallel):
            _assert_states_equal(state_serial, state_parallel)


def test_cached_swap_loglik_is_bitwise_identical_to_uncached():
    """The per-swap-step SVD cache must reproduce the uncached likelihood exactly,
    including after the post-swap leaf refresh (tree structures unchanged)."""
    from bart_playground.bart import _leaf_basis_svd_for_sampler_state

    X, y = _toy_data()
    model = ParallelTemperingBART(
        ndpost=5, nskip=0, n_trees=6, temperatures=[1.0, 1.4, 2.0],
        swap_interval=5, random_state=42,
    ).fit(X, y, quietly=True)
    sampler = model.chain_samplers[0]
    state = sampler.get_init_state()
    for _ in range(10):
        state = sampler.one_iter(state, temp=1.0)

    svd = _leaf_basis_svd_for_sampler_state(model.sampler, state)
    for _ in range(2):
        for temp in (1.0, 1.4, 2.0):
            uncached = model._state_collapsed_loglik(state, temp)
            cached = model._state_collapsed_loglik(state, temp, leaf_basis_svd=svd)
            assert cached == uncached
        model._refresh_state_tempered_params(state, 1)


def test_cached_leaf_basis_refresh_is_bitwise_identical_to_uncached():
    """The post-swap leaf refresh must be identical with and without the cached basis.

    The n_jobs equivalence tests cannot catch a stale-cache bug here: the
    serial and worker paths share this cache, so both would be wrong together.
    This compares the cached refresh against one that rebuilds the basis.
    """
    from bart_playground.bart import _leaf_basis_and_svd_for_sampler_state

    X, y = _toy_data()
    model = ParallelTemperingBART(
        ndpost=5, nskip=0, n_trees=6, temperatures=[1.0, 1.4, 2.0],
        swap_interval=5, random_state=42,
    ).fit(X, y, quietly=True)
    chain_id = 1
    sampler = model.chain_samplers[chain_id]
    tree_ids = np.arange(model.chain_samplers[0].get_init_state().n_trees)

    state = model.chain_samplers[0].get_init_state()
    for _ in range(10):
        state = model.chain_samplers[0].one_iter(state, temp=1.0)

    leaf_basis, svd = _leaf_basis_and_svd_for_sampler_state(model.sampler, state)
    np.testing.assert_array_equal(leaf_basis, state.leaf_basis(tree_ids))

    # A refresh changes leaf values but no tree structure, so one basis has to
    # serve every refresh in a swap step. Replay the same RNG state both ways.
    for _round in range(3):
        rng_state = sampler.generator.bit_generator.state
        cached_state, uncached_state = state.copy(), state.copy()
        for target, basis in ((cached_state, leaf_basis), (uncached_state, None)):
            sampler.generator.bit_generator.state = rng_state
            model._refresh_state_tempered_params(target, chain_id, leaf_basis=basis)
        for tree_cached, tree_uncached in zip(cached_state.trees, uncached_state.trees):
            np.testing.assert_array_equal(tree_cached.leaf_vals, tree_uncached.leaf_vals)
        # The cached basis must still describe the refreshed state.
        np.testing.assert_array_equal(leaf_basis, cached_state.leaf_basis(tree_ids))
        state = cached_state

    # ...and the cached SVD must still reproduce the uncached likelihood after it.
    for temp in (1.0, 1.4, 2.0):
        assert model._state_collapsed_loglik(state, temp, leaf_basis_svd=svd) == (
            model._state_collapsed_loglik(state, temp)
        )


def test_parallel_matches_serial_with_many_accepted_swaps():
    """A close temperature ladder accepts most swaps, so states are displaced
    across several workers (cycles in the position permutation) before being
    moved back at the end of each swap step."""
    X, y = _toy_data()
    common = dict(
        ndpost=15, nskip=5, n_trees=6,
        temperatures=list(np.geomspace(1.0, 1.6, 7)),
        swap_interval=4, random_state=7, store_chain_traces=True,
        store_swap_diagnostics=True, post_swap_repair_steps=1,
    )
    serial = ParallelTemperingBART(n_jobs=1, **common).fit(X, y, quietly=True)
    parallel = ParallelTemperingBART(n_jobs=4, **common).fit(X, y, quietly=True)
    assert serial.swap_accept_counts.sum() > serial.swap_attempt_counts.sum() // 2
    assert serial.swap_diagnostics == parallel.swap_diagnostics
    np.testing.assert_array_equal(serial.swap_accept_counts, parallel.swap_accept_counts)
    for chain_serial, chain_parallel in zip(serial.chain_traces, parallel.chain_traces):
        assert len(chain_serial) == len(chain_parallel) == common["ndpost"]
        for state_serial, state_parallel in zip(chain_serial, chain_parallel):
            _assert_states_equal(state_serial, state_parallel)
    for serial_sampler, parallel_sampler in zip(serial.chain_samplers, parallel.chain_samplers):
        assert (
            serial_sampler.generator.bit_generator.state
            == parallel_sampler.generator.bit_generator.state
        )
        assert serial_sampler.move_accepted_counts == parallel_sampler.move_accepted_counts


def test_removed_backends_are_rejected():
    with pytest.raises(ValueError):
        ParallelTemperingBART(local_move_backend="joblib-loky")


def test_chain_split_is_contiguous_and_balanced():
    assert _split_chains_across_workers(7, 3) == [[0, 1, 2], [3, 4], [5, 6]]
    assert _split_chains_across_workers(6, 3) == [[0, 1], [2, 3], [4, 5]]
    assert _split_chains_across_workers(5, 1) == [[0, 1, 2, 3, 4]]
    assert _split_chains_across_workers(5, 5) == [[0], [1], [2], [3], [4]]
    with pytest.raises(ValueError):
        _split_chains_across_workers(3, 4)


def test_negative_n_jobs_resolves_against_physical_cores():
    from bart_playground.bart import _physical_core_count

    cores = _physical_core_count()
    assert cores >= 1
    temps = [1.0, 1.4, 2.0, 3.0]
    assert ParallelTemperingBART(
        n_jobs=-1, temperatures=temps
    )._effective_parallel_workers() == min(len(temps), cores)
    assert ParallelTemperingBART(
        n_jobs=-2, temperatures=temps
    )._effective_parallel_workers() == max(1, min(len(temps), cores - 1))
    # Never more workers than temperatures, and never fewer than one.
    assert ParallelTemperingBART(
        n_jobs=-99, temperatures=temps
    )._effective_parallel_workers() == 1


@pytest.mark.parametrize("n_jobs", [2, 3, 4, 7, -1])
def test_worker_count_does_not_change_results(n_jobs):
    """n_jobs only packs chains into processes, so results must not depend on it.

    With 7 temperatures and fewer workers, several chains share one process
    and the swap step addresses them by slot on a shared pipe.
    """
    X, y = _toy_data()
    common = dict(
        ndpost=15, nskip=5, n_trees=6,
        temperatures=list(np.geomspace(1.0, 1.6, 7)),
        swap_interval=4, random_state=7, store_chain_traces=True,
        store_swap_diagnostics=True, post_swap_repair_steps=1,
    )
    serial = ParallelTemperingBART(n_jobs=1, **common).fit(X, y, quietly=True)
    assert serial.swap_accept_counts.sum() > 0
    parallel = ParallelTemperingBART(n_jobs=n_jobs, **common).fit(X, y, quietly=True)

    params = parallel.get_params()
    workers = parallel._effective_parallel_workers()
    if n_jobs > 0:
        assert workers == n_jobs
    assert params["effective_parallel_workers"] == workers
    assert len(params["chains_per_worker"]) == workers
    assert sum(params["chains_per_worker"]) == 7

    assert serial.swap_diagnostics == parallel.swap_diagnostics
    np.testing.assert_array_equal(serial.swap_accept_counts, parallel.swap_accept_counts)
    np.testing.assert_array_equal(serial.swap_attempt_counts, parallel.swap_attempt_counts)
    for chain_serial, chain_parallel in zip(serial.chain_traces, parallel.chain_traces):
        for state_serial, state_parallel in zip(chain_serial, chain_parallel):
            _assert_states_equal(state_serial, state_parallel)
    for serial_sampler, parallel_sampler in zip(serial.chain_samplers, parallel.chain_samplers):
        assert (
            serial_sampler.generator.bit_generator.state
            == parallel_sampler.generator.bit_generator.state
        )
        assert serial_sampler.move_accepted_counts == parallel_sampler.move_accepted_counts
