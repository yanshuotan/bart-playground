"""`batched_swap_refresh`: serial and parallel agree, and with one sweep per
swap step it reduces to the default swap step exactly."""
import numpy as np
import pytest

from bart_playground.bart import ParallelTemperingBART


def _data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    return X, y


def _common(sampler_kind, swap_sweeps):
    return dict(
        collect_all_samplers=True, ndpost=12, nskip=3, n_trees=6,
        temperatures=[1.0, 1.4, 2.0, 3.0, 4.5], swap_interval=3, swap_sweeps=swap_sweeps,
        random_state=42, sampler_kind=sampler_kind, multi_tries=3, store_swap_diagnostics=True,
    )


def _assert_same_fit(a, b):
    np.testing.assert_array_equal(a.swap_attempt_counts, b.swap_attempt_counts)
    np.testing.assert_array_equal(a.swap_accept_counts, b.swap_accept_counts)
    assert a.swap_diagnostics == b.swap_diagnostics
    assert len(a.trace) == len(b.trace)
    for sa, sb in zip(a.trace, b.trace):
        for key in sa.global_params:
            np.testing.assert_array_equal(sa.global_params[key], sb.global_params[key])
        for ta, tb in zip(sa.trees, sb.trees):
            for attribute in ("vars", "thresholds", "leaf_vals"):
                np.testing.assert_array_equal(getattr(ta, attribute), getattr(tb, attribute))
    for sa, sb in zip(a.chain_samplers, b.chain_samplers):
        assert sa.generator.bit_generator.state == sb.generator.bit_generator.state


@pytest.mark.parametrize("sampler_kind", ["default", "multi"])
@pytest.mark.parametrize("swap_sweeps", [None, 2])
def test_batched_parallel_matches_serial(sampler_kind, swap_sweeps):
    X, y = _data()
    common = _common(sampler_kind, swap_sweeps)
    serial = ParallelTemperingBART(n_jobs=1, batched_swap_refresh=True, **common).fit(X, y, quietly=True)
    parallel = ParallelTemperingBART(n_jobs=3, batched_swap_refresh=True, **common).fit(X, y, quietly=True)
    assert serial.swap_accept_counts.sum() > 0
    _assert_same_fit(serial, parallel)


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_batched_with_one_sweep_equals_default(n_jobs):
    # With one sweep a state takes part in at most one swap, so the default step
    # refreshes exactly the states the batched step refreshes, from the same streams.
    X, y = _data()
    common = _common("default", 1)
    default = ParallelTemperingBART(n_jobs=n_jobs, **common).fit(X, y, quietly=True)
    batched = ParallelTemperingBART(n_jobs=n_jobs, batched_swap_refresh=True, **common).fit(X, y, quietly=True)
    assert default.swap_accept_counts.sum() > 0
    _assert_same_fit(default, batched)


def test_batched_differs_from_default_with_many_sweeps():
    X, y = _data()
    common = _common("default", None)
    default = ParallelTemperingBART(n_jobs=1, **common).fit(X, y, quietly=True)
    batched = ParallelTemperingBART(n_jobs=1, batched_swap_refresh=True, **common).fit(X, y, quietly=True)
    leaf_a = np.concatenate([t.leaf_vals[t.vars == -1] for t in default.trace[-1].trees])
    leaf_b = np.concatenate([t.leaf_vals[t.vars == -1] for t in batched.trace[-1].trees])
    assert not np.array_equal(leaf_a, leaf_b)
