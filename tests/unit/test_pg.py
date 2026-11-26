import math
from functools import lru_cache

import pytest

import numpy as np
from bart_playground import PGBART
from bart_playground.params import Tree
from bart_playground.priors import ComprehensivePrior
from bart_playground.samplers import PGSampler
from bart_playground.util import Dataset
from bart_playground.pg_kernels import pg_tree_csmc, _leaf_term


ALPHA, BETA, SIGMA2 = 0.95, 1.0, 0.02


def _toy_problem():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(8, 2)).astype(np.float32)
    y = (np.where(X[:, 0] > 0.5, 0.3, -0.2) + 0.1 * rng.standard_normal(8)).astype(np.float32)
    thresholds = {v: np.quantile(X[:, v], [0.0, 0.3, 0.6]) for v in range(2)}
    return X, y, thresholds


def _sampler(cls, X, y, thresholds, seed, **kwargs):
    rng = np.random.default_rng(seed)
    prior = ComprehensivePrior(1, ALPHA, BETA, 2.0, 0.9, 3, "naive", rng, False, False, 1.0,
                               fixed_eps_sigma2=SIGMA2)
    sampler = cls(prior=prior, generator=rng, **kwargs)
    sampler.add_data(Dataset(X, y))
    sampler.add_thresholds(thresholds)
    return sampler


def _exact_leaf_count_posterior(X, y, thresholds, f_sigma2):
    """Posterior of the number of leaves of a single tree, by enumeration.

    The target is the truncated prior used by every sampler in the package
    (split probability ALPHA / (1 + d)^BETA, uniform variable, uniform
    threshold from the grid, no empty leaf) times the collapsed likelihood.
    """
    grids = [np.asarray(thresholds[v], dtype=np.float32) for v in range(X.shape[1])]

    @lru_cache(maxsize=None)
    def dist(rows, depth):
        rows_arr = np.array(rows)
        p_split = ALPHA / (1.0 + depth) ** BETA
        out = {1: (1.0 - p_split) * math.exp(_leaf_term(len(rows), float(np.sum(y[rows_arr], dtype=np.float64)),
                                                        SIGMA2, f_sigma2))}
        for v, grid in enumerate(grids):
            for tau in grid:
                go_left = X[rows_arr, v] <= tau
                if go_left.all() or not go_left.any():
                    continue
                left = dist(tuple(rows_arr[go_left]), depth + 1)
                right = dist(tuple(rows_arr[~go_left]), depth + 1)
                rule = p_split / len(grids) / len(grid)
                for a, wa in left.items():
                    for b, wb in right.items():
                        out[a + b] = out.get(a + b, 0.0) + rule * wa * wb
        return out

    d = dist(tuple(range(X.shape[0])), 0)
    total = sum(d.values())
    return {k: w / total for k, w in d.items()}


@pytest.mark.parametrize("proposal", ["prior", "optimal"])
def test_single_particle_returns_reference(proposal):
    X, y, thresholds = _toy_problem()
    sampler = _sampler(PGSampler, X, y, thresholds, seed=0, n_particles=1, proposal=proposal)
    tree = Tree.new(X)
    tree.split_leaf(0, 0, np.float32(thresholds[0][1]))
    out = pg_tree_csmc(
        sampler._XT, y, SIGMA2, 0.0625, math.log(ALPHA), BETA, False,
        np.empty(0), sampler._thr_vals, sampler._thr_start, sampler._thr_len,
        tree.vars.astype(np.int64), tree.thresholds.astype(np.float32), 1, 5000, 1,
        proposal == "optimal", sampler._XB,
    )
    vars_, thr, n, leaf_ids, selected, changed, _ = out
    assert selected == 0 and not changed
    np.testing.assert_array_equal(vars_, tree.vars)
    np.testing.assert_array_equal(n, tree.n)
    np.testing.assert_array_equal(leaf_ids, tree.leaf_ids)


@pytest.mark.parametrize("proposal,repeat", [("prior", False), ("optimal", False), ("optimal", True)])
def test_pg_matches_exact_tree_posterior(proposal, repeat):
    X, y, thresholds = _toy_problem()
    if repeat:
        # A repeated threshold, as quantile grids of discrete columns have.
        thresholds = {v: np.sort(np.append(g, g[1])) for v, g in thresholds.items()}
    sampler = _sampler(PGSampler, X, y, thresholds, seed=11, n_particles=5, proposal=proposal)
    exact = _exact_leaf_count_posterior(X, y, thresholds, sampler.tree_prior.f_sigma2)
    trace = sampler.run(20000, quietly=True)
    counts = np.bincount([t.trees[0].n_leaves for t in trace[1000:]], minlength=max(exact) + 1)
    freq = counts / counts.sum()
    for k, prob in exact.items():
        assert abs(freq[k] - prob) < 0.02, (k, freq[k], prob)
    assert counts[max(exact) + 1:].sum() == 0


@pytest.mark.parametrize("proposal", ["prior", "optimal"])
def test_pg_tree_caches_match_rebuilt_tree(proposal):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, 4))
    y = X[:, 0] * 3 + np.sin(6 * X[:, 1]) + 0.1 * rng.standard_normal(300)
    model = PGBART(ndpost=20, nskip=5, n_trees=10, random_state=0, n_particles=6, proposal=proposal)
    model.fit(X, y, quietly=True)
    state = model.trace[-1]
    total = np.zeros(X.shape[0], dtype=np.float32)
    for tree in state.trees:
        rebuilt = Tree.new(model.data.X)
        rebuilt.vars = tree.vars.copy()
        rebuilt.thresholds = tree.thresholds.copy()
        rebuilt.n = np.zeros(len(tree.vars), dtype=np.int32)
        rebuilt.n[0] = X.shape[0]
        assert rebuilt.update_n()
        np.testing.assert_array_equal(rebuilt.leaf_ids, tree.leaf_ids)
        np.testing.assert_array_equal(rebuilt.n, tree.n)
        total += tree.evaluate(model.data.X)
    np.testing.assert_allclose(state.evaluate(), total, rtol=1e-5, atol=1e-5)
    assert model.sampler.move_accepted_counts["pg"] > 0
    preds = model.predict(X)
    assert np.corrcoef(preds, y)[0, 1] > 0.9


def test_optimal_proposal_masses_match_enumeration():
    """At the root, the per-variable masses and Z equal a direct enumeration."""
    from bart_playground.pg_kernels import _hist_one, _split_log_ratios, _logsumexp
    X, y, thresholds = _toy_problem()
    sampler = _sampler(PGSampler, X, y, thresholds, seed=0, n_particles=2)
    f_sigma2 = 0.0625
    resid = y.astype(np.float64)
    N = len(y)
    buf = np.arange(N, dtype=np.int32)
    lt = _leaf_term(N, resid.sum(), SIGMA2, f_sigma2)
    cnt = np.empty(8, np.int64); sums = np.empty(8); out = np.empty(8)
    for v in range(X.shape[1]):
        nlen = int(sampler._thr_len[v])
        _hist_one(sampler._XB, v, buf, 0, N, resid, nlen, cnt, sums)
        _split_log_ratios(cnt, sums, 0, nlen, N, resid.sum(), lt, SIGMA2, f_sigma2, out)
        grid = sampler._thr_vals[sampler._thr_start[v]:sampler._thr_start[v] + nlen]
        for j, tau in enumerate(grid):
            go_left = X[:, v] <= tau
            if go_left.all() or not go_left.any():
                assert out[j] == -np.inf
                continue
            direct = (_leaf_term(go_left.sum(), resid[go_left].sum(), SIGMA2, f_sigma2)
                      + _leaf_term((~go_left).sum(), resid[~go_left].sum(), SIGMA2, f_sigma2) - lt)
            assert math.isclose(out[j], direct, rel_tol=1e-10, abs_tol=1e-10)
        assert np.isfinite(_logsumexp(out, nlen))
