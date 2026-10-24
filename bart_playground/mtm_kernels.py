"""Fused Numba kernels for scoring multiple-try (MTMH) candidates.

The Multi* moves score each candidate by simulating the modified tree
(`Tree.simulate_*`), which copies the per-row leaf ids and node counts, and
then calling `_single_tree_log_marginal_lkhd_numba`, which rescans all rows to
build per-leaf residual sums and recomputes the residual sum of squares. Only
the rows inside the modified subtree change, and the residual sum of squares
is shared by every candidate of one move, so the kernels below take the
current tree's per-leaf sums (`node_resid_sums`) and the shared sum of squares
once per move and redo only the rows that move.

Each kernel reproduces the original arithmetic exactly: the per-leaf sums are
float32 accumulations over rows in increasing row index (the order
`_get_resid_all` uses), and the marginal likelihood loops over nodes in index
order with the same expressions and types. Scores are therefore bit-identical
to the `simulate_*` + `calculate_simulated_likelihood` path, so a seeded chain
produces identical draws (checked in tests/unit/test_mtm_kernels.py).
"""
import math

import numpy as np
from numba import njit

from .params import _descendants_numba, _traverse_tree_single
from .priors import _get_resid_all, _trees_log_prior_numba


@njit(cache=True)
def residual_sum_sq(resids):
    """Same reduction as the first line of `_single_tree_log_marginal_lkhd_numba`."""
    return np.sum(resids ** 2)


@njit(cache=True)
def _row_index_numba(leaf_ids, n, residuals):
    """Rows reaching each node, in increasing row order, plus per-node residual sums.

    `rows[offsets[k]:offsets[k + 1]]` lists the rows whose leaf lies in the
    subtree of node k (k included); `n` (the tree's node counts) sizes the
    lists. The root's list would be every row, so it is left empty and
    `_row_range` stands in for it. Every candidate only touches the rows of
    one subtree, and visiting them in increasing order keeps each float32
    leaf sum identical to the full scan in `_get_resid_all`.
    """
    n_nodes = len(n)
    offsets = np.zeros(n_nodes + 1, np.int64)
    for k in range(1, n_nodes):
        offsets[k + 1] = offsets[k] + n[k]
    fill = offsets[:-1].copy()
    rows = np.empty(offsets[n_nodes], np.int32)
    for i in range(len(leaf_ids)):
        node = leaf_ids[i]
        while node != 0:
            rows[fill[node]] = i
            fill[node] += 1
            node = (node - 1) // 2
    for k in range(1, n_nodes):
        if fill[k] != offsets[k + 1]:
            raise ValueError("Node counts disagree with leaf ids.")
    sums, _ = _get_resid_all(leaf_ids, n_nodes, residuals)
    return offsets, rows, sums


@njit(cache=True)
def _row_range(offsets, start, n_rows):
    """Bounds of node `start`'s rows: all rows for the root, else its list."""
    if start == 0:
        return 0, n_rows
    return offsets[start], offsets[start + 1]


class RowIndex:
    """`_row_index_numba` for a tree as it stands."""

    __slots__ = ("offsets", "rows", "sums")

    def __init__(self, tree, residuals):
        self.offsets, self.rows, self.sums = _row_index_numba(tree.leaf_ids, tree.n, residuals)


@njit(cache=True)
def _marginal_from_sums(new_vars, new_n, sums, ls_resids, eps_sigma2, f_sigma2, temp):
    """`_single_tree_log_marginal_lkhd_numba` given per-leaf sums instead of rows.

    A node enters the original sum when some row lands in it, i.e. when it is a
    leaf with a positive count.
    """
    ls = ls_resids
    ridge_bias = 0.0
    logdet = 0.0
    tempered_eps_sigma2 = temp * eps_sigma2
    noise_ratio = tempered_eps_sigma2 / f_sigma2

    for node in range(len(new_n)):
        if new_vars[node] == -1 and new_n[node] > 0:
            logdet += math.log(new_n[node] / noise_ratio + 1.0)
            ls -= (sums[node] ** 2) / new_n[node]
            ridge_bias += (sums[node] ** 2) / (new_n[node] * (new_n[node] / noise_ratio + 1.0))

    return - (logdet + (ls + ridge_bias) / tempered_eps_sigma2) / 2


@njit(cache=True)
def _resplit_core(dataX, new_vars, new_thresholds, start, offsets, rows, n, sums,
                  residuals, ls_resids, eps_sigma2, f_sigma2, temp):
    """Score a tree whose split rules changed at `start` (structure unchanged).

    Mirrors `_update_n_and_leaf_id_numba_copy` from `start`, the validity loop of
    MultiChange / MultiSwap, and the marginal likelihood. Returns (valid, loglik);
    loglik is only meaningful when valid.
    """
    n_nodes = len(new_vars)
    new_n = n.copy()
    new_sums = sums.copy()
    desc = _descendants_numba(start, new_vars)
    for j in desc:
        new_n[j] = 0
        new_sums[j] = 0.0

    lo, hi = _row_range(offsets, start, dataX.shape[0])
    for r in range(lo, hi):
        i = r if start == 0 else rows[r]
        leaf = _traverse_tree_single(dataX[i], new_vars, new_thresholds, start, new_n)
        new_sums[leaf] += residuals[i]

    valid = True
    for i in range(start, n_nodes):
        if new_vars[i] != -2 and new_n[i] == 0:
            valid = False
            break
    if not valid:
        return False, 0.0
    return True, _marginal_from_sums(new_vars, new_n, new_sums, ls_resids, eps_sigma2, f_sigma2, temp)


@njit(cache=True)
def change_candidate(dataX, vars_, thresholds, offsets, rows, n, sums, node_id, var, threshold,
                     residuals, ls_resids, eps_sigma2, f_sigma2, temp):
    """`simulate_change_split` + validity check + likelihood, fused."""
    new_vars = vars_.copy()
    new_thresholds = thresholds.copy()
    new_vars[node_id] = var
    new_thresholds[node_id] = threshold
    return _resplit_core(dataX, new_vars, new_thresholds, node_id, offsets, rows, n, sums,
                         residuals, ls_resids, eps_sigma2, f_sigma2, temp)


@njit(cache=True)
def swap_candidate(dataX, vars_, thresholds, offsets, rows, n, sums, parent_id, child_id,
                   residuals, ls_resids, eps_sigma2, f_sigma2, temp):
    """`simulate_swap_split` + validity check + likelihood, fused."""
    new_vars = vars_.copy()
    new_thresholds = thresholds.copy()
    new_vars[child_id] = vars_[parent_id]
    new_vars[parent_id] = vars_[child_id]
    new_thresholds[child_id] = thresholds[parent_id]
    new_thresholds[parent_id] = thresholds[child_id]
    return _resplit_core(dataX, new_vars, new_thresholds, parent_id, offsets, rows, n, sums,
                         residuals, ls_resids, eps_sigma2, f_sigma2, temp)


@njit(cache=True)
def grow_candidate(dataX, vars_, offsets, rows, n, sums, node_id, var, threshold,
                   residuals, ls_resids, eps_sigma2, f_sigma2, temp, alpha, beta):
    """`simulate_split_leaf` + validity + likelihood + prior, fused.

    Returns (valid, loglik, logprior).
    """
    n_nodes = len(vars_)
    left_child = node_id * 2 + 1
    right_child = node_id * 2 + 2
    threshold = np.float32(threshold)

    max_needed_size = 2 * n_nodes if right_child >= n_nodes else n_nodes
    new_vars = np.full(max_needed_size, -2, dtype=np.int32)
    new_vars[:n_nodes] = vars_
    new_vars[node_id] = var
    new_vars[left_child] = -1
    new_vars[right_child] = -1

    new_n = np.zeros(max_needed_size, dtype=np.int32)
    new_n[:n_nodes] = n
    new_sums = np.zeros(max_needed_size, dtype=np.float32)
    new_sums[:n_nodes] = sums

    lo, hi = _row_range(offsets, node_id, dataX.shape[0])
    for r in range(lo, hi):
        i = r if node_id == 0 else rows[r]
        if dataX[i, var] <= threshold:
            new_n[left_child] += 1
            new_sums[left_child] += residuals[i]
        else:
            new_n[right_child] += 1
            new_sums[right_child] += residuals[i]

    if not (new_n[left_child] > 0 and new_n[right_child] > 0):
        return False, 0.0, 0.0
    loglik = _marginal_from_sums(new_vars, new_n, new_sums, ls_resids, eps_sigma2, f_sigma2, temp)
    return True, loglik, _trees_log_prior_numba(new_vars, alpha, beta)


@njit(cache=True)
def prune_candidate(vars_, offsets, rows, n, sums, node_id,
                    residuals, ls_resids, eps_sigma2, f_sigma2, temp, alpha, beta):
    """`simulate_prune_split` + likelihood + prior for a terminal split node, fused.

    Returns (loglik, logprior).
    """
    new_vars = vars_.copy()
    new_n = n.copy()
    new_sums = sums.copy()
    desc = _descendants_numba(node_id, vars_)
    for j in desc:
        new_vars[j] = -2
        new_n[j] = 0
    new_vars[node_id] = -1

    new_sums[node_id] = 0.0
    lo, hi = _row_range(offsets, node_id, len(residuals))
    for r in range(lo, hi):
        new_sums[node_id] += residuals[r if node_id == 0 else rows[r]]

    loglik = _marginal_from_sums(new_vars, new_n, new_sums, ls_resids, eps_sigma2, f_sigma2, temp)
    return loglik, _trees_log_prior_numba(new_vars, alpha, beta)


# ---------------------------------------------------------------------------
# Candidate loops, including the random draws.
#
# Numba's Generator methods share the bit generator with NumPy and reproduce
# NumPy's `integers` and `uniform` draw for draw, so the loops below consume
# the stream exactly as `fast_choice` / `fast_choice_with_weights` would. The
# subsequent selection among candidates stays in Python, because it relies on
# NumPy's pairwise summation.
# ---------------------------------------------------------------------------

_flat_threshold_cache = {}


def flat_thresholds(possible_thresholds):
    """`possible_thresholds` (dict var -> array) as one array plus offsets, cached."""
    key = id(possible_thresholds)
    hit = _flat_threshold_cache.get(key)
    if hit is not None and hit[0] is possible_thresholds:
        return hit[1], hit[2]
    arrays = [np.asarray(possible_thresholds[var]) for var in range(len(possible_thresholds))]
    flat = np.concatenate(arrays)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in arrays], out=offsets[1:])
    _flat_threshold_cache[key] = (possible_thresholds, flat, offsets)
    return flat, offsets


@njit(cache=True)
def _choice_index(rng, length):
    """Index drawn by `fast_choice` (no draw when there is a single option)."""
    if length == 1:
        return 0
    return rng.integers(0, length)


@njit(cache=True)
def _draw_var(rng, n_vars, s_cumsum):
    """Variable drawn by `fast_choice_with_weights(arange(p), s, s_cumsum)`;
    an empty `s_cumsum` stands for uniform weights (s is None)."""
    if len(s_cumsum) == 0 or n_vars == 1:
        return _choice_index(rng, n_vars)
    u = rng.uniform(0, s_cumsum[-1])
    return np.searchsorted(s_cumsum, u)


@njit(cache=True)
def _draw_threshold(rng, var, flat_thr, thr_offsets):
    start = thr_offsets[var]
    return flat_thr[start + _choice_index(rng, thr_offsets[var + 1] - start)]


@njit(cache=True)
def change_candidates(rng, n_samples, dataX, vars_, thresholds, offsets, rows, n, sums, split_nodes,
                      s_cumsum, flat_thr, thr_offsets, residuals, ls_resids, eps_sigma2, f_sigma2, temp):
    """MultiChange candidate loop. Returns node ids, vars, thresholds, log weights."""
    nodes = np.empty(n_samples, np.int64)
    var_ids = np.empty(n_samples, np.int64)
    thr_out = np.empty(n_samples, flat_thr.dtype)
    log_w = np.empty(n_samples, np.float64)
    n_vars = dataX.shape[1]
    for c in range(n_samples):
        node_id = split_nodes[_choice_index(rng, len(split_nodes))]
        var = _draw_var(rng, n_vars, s_cumsum)
        threshold = _draw_threshold(rng, var, flat_thr, thr_offsets)
        valid, loglik = change_candidate(dataX, vars_, thresholds, offsets, rows, n, sums, node_id, var,
                                         threshold, residuals, ls_resids, eps_sigma2, f_sigma2, temp)
        nodes[c] = node_id
        var_ids[c] = var
        thr_out[c] = threshold
        log_w[c] = 0.5 * loglik if valid else -np.inf
    return nodes, var_ids, thr_out, log_w


@njit(cache=True)
def grow_candidates(rng, n_samples, dataX, vars_, offsets, rows, n, sums, leaves,
                    s_cumsum, flat_thr, thr_offsets, residuals, ls_resids, eps_sigma2, f_sigma2, temp,
                    alpha, beta):
    """MultiGrow / reverse-MultiPrune candidate loop. Returns node ids, vars, thresholds, log weights."""
    nodes = np.empty(n_samples, np.int64)
    var_ids = np.empty(n_samples, np.int64)
    thr_out = np.empty(n_samples, flat_thr.dtype)
    log_w = np.empty(n_samples, np.float64)
    n_vars = dataX.shape[1]
    for c in range(n_samples):
        node_id = leaves[_choice_index(rng, len(leaves))]
        var = _draw_var(rng, n_vars, s_cumsum)
        threshold = _draw_threshold(rng, var, flat_thr, thr_offsets)
        valid, loglik, logprior = grow_candidate(dataX, vars_, offsets, rows, n, sums, node_id, var, threshold,
                                                 residuals, ls_resids, eps_sigma2, f_sigma2, temp,
                                                 alpha, beta)
        nodes[c] = node_id
        var_ids[c] = var
        thr_out[c] = threshold
        log_w[c] = 0.5 * (loglik + logprior) if valid else -np.inf
    return nodes, var_ids, thr_out, log_w


@njit(cache=True)
def swap_candidates(rng, n_samples, dataX, vars_, thresholds, offsets, rows, n, sums, pairs,
                    residuals, ls_resids, eps_sigma2, f_sigma2, temp):
    """MultiSwap candidate loop over `pairs` (k x 2). Returns pair indices, log weights."""
    picks = np.empty(n_samples, np.int64)
    log_w = np.empty(n_samples, np.float64)
    for c in range(n_samples):
        k = _choice_index(rng, pairs.shape[0])
        valid, loglik = swap_candidate(dataX, vars_, thresholds, offsets, rows, n, sums, pairs[k, 0], pairs[k, 1],
                                       residuals, ls_resids, eps_sigma2, f_sigma2, temp)
        picks[c] = k
        log_w[c] = 0.5 * loglik if valid else -np.inf
    return picks, log_w
