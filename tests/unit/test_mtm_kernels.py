"""The fused MTMH scoring kernels must reproduce the reference path bit for bit."""
import numpy as np
import pytest

from bart_playground import MultiBART, moves
from bart_playground.samplers import mtmh_proposal_probs


def _friedman(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + rng.normal(size=n))
    return X, y


def _run(fast, X, y, **kwargs):
    old = moves.FAST_MTM_SCORING
    moves.FAST_MTM_SCORING = fast
    try:
        model = MultiBART(ndpost=kwargs.pop("ndpost", 60), nskip=0, n_trees=kwargs.pop("n_trees", 20),
                          proposal_probs=mtmh_proposal_probs, random_state=7, **kwargs)
        model.fit(X, y, quietly=True)
    finally:
        moves.FAST_MTM_SCORING = old
    return model


def _assert_same_chain(a, b):
    ta, tb = a.sampler.trace, b.sampler.trace
    assert len(ta) == len(tb)
    for sa, sb in zip(ta, tb):
        for tree_a, tree_b in zip(sa.trees, sb.trees):
            np.testing.assert_array_equal(tree_a.vars, tree_b.vars)
            np.testing.assert_array_equal(tree_a.thresholds, tree_b.thresholds)
            np.testing.assert_array_equal(tree_a.leaf_vals, tree_b.leaf_vals)
        assert sa.global_params.keys() == sb.global_params.keys()
        for key in sa.global_params:
            np.testing.assert_array_equal(sa.global_params[key], sb.global_params[key])
    assert a.sampler.move_accepted_counts == b.sampler.move_accepted_counts
    assert a.sampler.accepted_moves_logmh == b.sampler.accepted_moves_logmh


@pytest.mark.parametrize("kwargs", [
    {},
    {"temperature": 3.0},
    {"dirichlet_prior": True},
    {"multi_tries": [10, 5, 3]},
])
def test_fast_scoring_reproduces_reference_chain(kwargs):
    X, y = _friedman(300, 8, 1)
    ref = _run(False, X, y, **dict(kwargs))
    fast = _run(True, X, y, **dict(kwargs))
    _assert_same_chain(ref, fast)
