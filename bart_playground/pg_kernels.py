"""Conditional SMC kernel for the Particle Gibbs tree update (PG-BART).

Lakshminarayanan, Roy and Teh (2015, AISTATS) replace the local
grow/prune/change/swap proposals of the backfitting sweep by a conditional SMC
run that proposes a complete tree for the current residual. A tree is built
top-down: nodes are expanded one at a time in breadth-first order, and each
expansion decides whether the node stops or splits and, if it splits, on which
rule. Breadth-first order is increasing heap index, so the old tree's
decisions form a fixed sequence that the reference particle (slot 0) replays.

Target. The tree prior is the one the Default sampler targets in this
package: a node at depth d splits with probability alpha / (1 + d)^beta (or
alpha^d with quick decay), the split variable is drawn from the split
probabilities s (uniform without the Dirichlet prior), the threshold
uniformly from that variable's grid, and every leaf must hold at least one
training row. This is the prior of Chipman et al. truncated to trees without
empty leaves, not the "valid split" prior of the PG paper, so PG and Default
chains have the same posterior.

Proposals and weights. Two proposals for a node's decision are available.

The bootstrap proposal of the 2015 paper draws it from the untruncated prior.
A split that leaves a child empty gets weight zero, which is the truncation
indicator; any other split multiplies the weight by the ratio of the
children's collapsed marginal likelihoods to the parent's, and a stop leaves it
unchanged (nodes still pending count as leaves, as in the paper).

The one-step optimal proposal of Lakshminarayanan, Roy and Teh (2013) draws
the decision from its conditional given the data: stop with mass 1 - p_d and
the split (v, tau) with mass p_d s_v / |grid_v| times the same likelihood
ratio, zero for splits that leave a child empty. The weight is multiplied by
the normalising constant Z of these masses, which does not depend on the
decision drawn, and the reference particle gets the same factor at the node it
replays. Z depends only on the node, so it is computed once per node, with the
per-variable masses it is made of; drawing a split then needs one pass over
the chosen variable. Each pass works on the binned covariates XB, where
XB[i, v] is the first grid index j with X[i, v] <= grid_v[j], so the row
goes left of threshold j exactly when XB[i, v] <= j; one pass over a node's
rows fills the histograms of all variables at once. Thresholds may repeat
(quantile grids of discrete columns): a repeated value keeps its share of the
prior and its splits are the same partition.

Resampling is multinomial after every stage and keeps slot 0 on the reference
path. When no particle has a pending node, the returned tree is drawn from the
final weights over all particles, the reference included (Andrieu et al.,
2010).

Each node's training rows are a contiguous segment of one index buffer, and
particles share nodes, so copying a particle at resampling copies only its
list of node references. Particles that make the same split of a shared node
also share its children.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _leaf_term(n, resid_sum, sigma2_t, f_sigma2):
    """One leaf's share of the collapsed log marginal likelihood.

    Equals that leaf's terms in `_single_tree_log_marginal_lkhd_numba`, which
    also subtracts sum(resid ** 2) / (2 sigma2_t); that term is the same for
    every tree and cancels in the weights.
    """
    return (-0.5 * math.log(1.0 + n * f_sigma2 / sigma2_t)
            + 0.5 * resid_sum * resid_sum / (sigma2_t * (n + sigma2_t / f_sigma2)))


@njit(cache=True)
def _split_prob(depth, log_alpha, beta, quick_decay):
    # Same parameterisation as `_trees_log_prior_numba`.
    if quick_decay:
        return math.exp(log_alpha * depth)
    return math.exp(log_alpha - beta * math.log(1.0 + depth))


@njit(cache=True)
def _logsumexp(a, n):
    m = -np.inf
    for j in range(n):
        if a[j] > m:
            m = a[j]
    if m == -np.inf:
        return m
    t = 0.0
    for j in range(n):
        t += math.exp(a[j] - m)
    return m + math.log(t)


@njit(cache=True)
def _hist_all(XB, buf, st, n, resid, hoff, H, cnt, sums):
    """Row counts and residual sums of one node per (variable, bin).

    Variable v's bins are cnt[hoff[v]:hoff[v] + len_v + 1]; one pass over the
    node's rows, reading each row's bins contiguously.
    """
    for b in range(H):
        cnt[b] = 0
        sums[b] = 0.0
    p = XB.shape[1]
    for q in range(st, st + n):
        i = buf[q]
        r = resid[i]
        for v in range(p):
            b = hoff[v] + XB[i, v]
            cnt[b] += 1
            sums[b] += r


@njit(cache=True)
def _hist_one(XB, v, buf, st, n, resid, nlen, cnt, sums):
    """As `_hist_all` for variable v alone, its bins starting at 0."""
    for b in range(nlen + 1):
        cnt[b] = 0
        sums[b] = 0.0
    for q in range(st, st + n):
        i = buf[q]
        b = XB[i, v]
        cnt[b] += 1
        sums[b] += resid[i]


@njit(cache=True)
def _split_log_ratios(cnt, sums, off, nlen, n, total, lt_parent, sigma2_t, f_sigma2, out):
    """Log likelihood ratio of every split of a node on one variable.

    Reads that variable's bins cnt[off:off + nlen + 1] (and sums); out[j] is
    the ratio of the split at its j-th threshold, -inf when a child would be
    empty.
    """
    n_left = 0
    s_left = 0.0
    for j in range(nlen):
        n_left += cnt[off + j]
        s_left += sums[off + j]
        if n_left == 0 or n_left == n:
            out[j] = -np.inf
        else:
            out[j] = (_leaf_term(n_left, s_left, sigma2_t, f_sigma2)
                      + _leaf_term(n - n_left, total - s_left, sigma2_t, f_sigma2) - lt_parent)


@njit(cache=True)
def _grow_1d(a, size):
    b = np.empty(size, a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=True)
def _grow_2d(a, size):
    b = np.zeros((a.shape[0], size), a.dtype)
    b[:, :a.shape[1]] = a
    return b


@njit(cache=True)
def _grow_rows(a, size):
    b = np.empty((size, a.shape[1]), a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=True)
def pg_tree_csmc(XT, resid, sigma2_t, f_sigma2, log_alpha, beta, quick_decay,
                 var_cumsum, thr_vals, thr_start, thr_len,
                 ref_vars, ref_thr, n_particles, max_stages, seed, optimal, XB):
    """Draw one tree from the conditional SMC kernel.

    Parameters
    ----------
    XT : (p, N) array
        Training covariates, one row per variable.
    resid : (N,) array
        Residual the tree is fitted to.
    sigma2_t : float
        Noise variance times the temperature.
    f_sigma2 : float
        Prior variance of a leaf value.
    var_cumsum : (p,) or (0,) array
        Cumulative split probabilities s; empty for uniform variables.
    thr_vals, thr_start, thr_len : arrays
        Threshold grids, variable v's grid being
        thr_vals[thr_start[v]:thr_start[v] + thr_len[v]].
    ref_vars, ref_thr : arrays
        The old tree in heap layout (-1 leaf, -2 absent).
    n_particles : int
        Number of particles C, the reference included.
    max_stages : int
        Cap on the number of stages; nodes still pending then become leaves.
    seed : int
        Seed of Numba's random stream for this call.
    optimal : bool
        Use the one-step optimal proposal instead of the prior.
    XB : (N, p) int32 array
        Binned covariates (see the module docstring); may be empty unless
        optimal.

    Returns
    -------
    vars, thresholds, n, leaf_ids : arrays
        The drawn tree in heap layout with its node counts and the leaf of
        every training row.
    selected : int
        Slot of the drawn particle (0 is the reference).
    changed : bool
        Whether the drawn tree differs from the old one.
    n_stages : int
        Stages run.
    """
    np.random.seed(seed)
    p, N = XT.shape
    C = n_particles
    L = ref_vars.shape[0]

    # Node pool, shared by all particles. Node k's rows are
    # buf[nd_start[k]:nd_start[k] + nd_n[k]].
    cap_nodes = 64
    nd_start = np.empty(cap_nodes, np.int64)
    nd_n = np.empty(cap_nodes, np.int64)
    nd_heap = np.empty(cap_nodes, np.int64)
    nd_depth = np.empty(cap_nodes, np.int64)
    nd_lt = np.empty(cap_nodes, np.float64)
    nd_sum = np.empty(cap_nodes, np.float64)
    # First split made of each node (its left child, -1 if none), so that
    # particles making the same split share the children.
    nd_kid = np.full(cap_nodes, -1, np.int64)
    nd_kid_v = np.empty(cap_nodes, np.int64)
    nd_kid_t = np.empty(cap_nodes, thr_vals.dtype)
    # Optimal proposal, per node: log Z (nan until computed) and the log
    # masses of stopping (column 0) and of splitting on each variable.
    nd_logz = np.full(cap_nodes, np.nan)
    nd_mass = np.empty((cap_nodes, p + 1), np.float64)
    buf = np.empty(4 * N, np.int32)
    root_sum = 0.0
    for i in range(N):
        buf[i] = i
        root_sum += resid[i]
    buf_top = N
    nd_start[0] = 0
    nd_n[0] = N
    nd_heap[0] = 0
    nd_depth[0] = 0
    nd_lt[0] = _leaf_term(N, root_sum, sigma2_t, f_sigma2)
    nd_sum[0] = root_sum
    n_nodes = 1

    if optimal:
        if XB.shape[0] != N or XB.shape[1] != p:
            raise ValueError("optimal proposal needs XB of shape (N, p)")
        log_s = np.empty(p, np.float64)
        prev = 0.0
        for v in range(p):
            if thr_len[v] == 0:
                log_s[v] = -np.inf
            elif var_cumsum.shape[0] == 0:
                log_s[v] = -math.log(p) - math.log(thr_len[v])
            else:
                mass = var_cumsum[v] - prev
                log_s[v] = (math.log(mass / var_cumsum[-1]) - math.log(thr_len[v])
                            if mass > 0.0 else -np.inf)
            if var_cumsum.shape[0] != 0:
                prev = var_cumsum[v]
        hoff = np.empty(p, np.int64)
        H = 0
        max_len = 1
        for v in range(p):
            hoff[v] = H
            H += thr_len[v] + 1
            max_len = max(max_len, thr_len[v] + 1)
        h_cnt = np.empty(H, np.int64)
        h_sum = np.empty(H, np.float64)
        ratios = np.empty(max_len, np.float64)

    # Particles: their nodes in creation (= breadth-first) order and the
    # decision taken at each (0 pending, 1 leaf, 2 split).
    K = 16
    P_node = np.zeros((C, K), np.int64)
    P_state = np.zeros((C, K), np.int8)
    P_var = np.zeros((C, K), np.int64)
    P_thr = np.zeros((C, K), thr_vals.dtype)
    P_cnt = np.ones(C, np.int64)
    P_head = np.zeros(C, np.int64)
    dead = np.zeros(C, np.bool_)
    logw = np.zeros(C, np.float64)

    n_stages = 0
    for _ in range(max_stages):
        for c in range(C):
            if dead[c] or P_head[c] >= P_cnt[c]:
                continue
            j = P_head[c]
            P_head[c] = j + 1
            node = P_node[c, j]
            h = nd_heap[node]
            split = False
            v = 0
            tau = thr_vals[0]
            if optimal:
                st = nd_start[node]
                n = nd_n[node]
                if np.isnan(nd_logz[node]):
                    ps = _split_prob(nd_depth[node], log_alpha, beta, quick_decay)
                    nd_mass[node, 0] = math.log(1.0 - ps) if ps < 1.0 else -np.inf
                    log_ps = math.log(ps) if ps > 0.0 else -np.inf
                    if log_ps > -np.inf and n >= 2:
                        _hist_all(XB, buf, st, n, resid, hoff, H, h_cnt, h_sum)
                    for vv in range(p):
                        if log_s[vv] == -np.inf or log_ps == -np.inf or n < 2:
                            nd_mass[node, vv + 1] = -np.inf
                            continue
                        _split_log_ratios(h_cnt, h_sum, hoff[vv], thr_len[vv], n, nd_sum[node],
                                          nd_lt[node], sigma2_t, f_sigma2, ratios)
                        nd_mass[node, vv + 1] = log_ps + log_s[vv] + _logsumexp(ratios, thr_len[vv])
                    nd_logz[node] = _logsumexp(nd_mass[node], p + 1)
                logz = nd_logz[node]
                if logz == -np.inf:
                    if c == 0:
                        raise ValueError("reference tree has zero proposal mass")
                    dead[c] = True
                    logw[c] = -np.inf
                    continue
                logw[c] += logz
            if c == 0:
                if h < L and ref_vars[h] >= 0:
                    split = True
                    v = ref_vars[h]
                    tau = ref_thr[h]
            elif optimal:
                # Pick stop or a variable by its mass, then a threshold of it.
                u = np.random.random()
                acc = 0.0
                pick = p
                for a in range(p + 1):
                    acc += math.exp(nd_mass[node, a] - logz)
                    if u < acc:
                        pick = a
                        break
                while pick > 0 and nd_mass[node, pick] == -np.inf:
                    pick -= 1  # rounding at the top end
                if pick > 0:
                    split = True
                    v = pick - 1
                    nlen = thr_len[v]
                    _hist_one(XB, v, buf, st, n, resid, nlen, h_cnt, h_sum)
                    _split_log_ratios(h_cnt, h_sum, 0, nlen, n, nd_sum[node], nd_lt[node],
                                      sigma2_t, f_sigma2, ratios)
                    m = -np.inf
                    for t in range(nlen):
                        m = max(m, ratios[t])
                    tot = 0.0
                    for t in range(nlen):
                        tot += math.exp(ratios[t] - m)
                    u = np.random.random() * tot
                    acc = 0.0
                    jj = -1
                    for t in range(nlen):
                        if ratios[t] > -np.inf:
                            jj = t
                            acc += math.exp(ratios[t] - m)
                            if u < acc:
                                break
                    tau = thr_vals[thr_start[v] + jj]
            elif np.random.random() < _split_prob(nd_depth[node], log_alpha, beta, quick_decay):
                split = True
                if var_cumsum.shape[0] == 0:
                    v = np.random.randint(0, p)
                else:
                    v = np.searchsorted(var_cumsum, np.random.random() * var_cumsum[-1])
                tau = thr_vals[thr_start[v] + np.random.randint(0, thr_len[v])]
            if not split:
                P_state[c, j] = 1
                continue

            if nd_kid[node] >= 0 and nd_kid_v[node] == v and nd_kid_t[node] == tau:
                # Another particle already made this split of this node.
                left = nd_kid[node]
                right = left + 1
            else:
                st = nd_start[node]
                n = nd_n[node]
                row = XT[v]
                n_left = 0
                for q in range(st, st + n):
                    if row[buf[q]] <= tau:
                        n_left += 1
                if n_left == 0 or n_left == n:
                    if c == 0:
                        raise ValueError("reference tree has an empty leaf")
                    dead[c] = True
                    logw[c] = -np.inf
                    continue

                if buf_top + n > buf.shape[0]:
                    buf = _grow_1d(buf, max(2 * buf.shape[0], buf_top + n))
                if n_nodes + 2 > cap_nodes:
                    cap_nodes *= 2
                    nd_start = _grow_1d(nd_start, cap_nodes)
                    nd_n = _grow_1d(nd_n, cap_nodes)
                    nd_heap = _grow_1d(nd_heap, cap_nodes)
                    nd_depth = _grow_1d(nd_depth, cap_nodes)
                    nd_lt = _grow_1d(nd_lt, cap_nodes)
                    old_cap = nd_logz.shape[0]
                    nd_sum = _grow_1d(nd_sum, cap_nodes)
                    nd_kid = _grow_1d(nd_kid, cap_nodes)
                    nd_kid[old_cap:] = -1
                    nd_kid_v = _grow_1d(nd_kid_v, cap_nodes)
                    nd_kid_t = _grow_1d(nd_kid_t, cap_nodes)
                    nd_logz = _grow_1d(nd_logz, cap_nodes)
                    nd_logz[old_cap:] = np.nan
                    nd_mass = _grow_rows(nd_mass, cap_nodes)
                li = buf_top
                ri = buf_top + n_left
                sum_left = 0.0
                sum_right = 0.0
                for q in range(st, st + n):
                    i = buf[q]
                    if row[i] <= tau:
                        buf[li] = i
                        li += 1
                        sum_left += resid[i]
                    else:
                        buf[ri] = i
                        ri += 1
                        sum_right += resid[i]
                left = n_nodes
                right = n_nodes + 1
                nd_start[left] = buf_top
                nd_n[left] = n_left
                nd_start[right] = buf_top + n_left
                nd_n[right] = n - n_left
                buf_top += n
                nd_heap[left] = 2 * h + 1
                nd_heap[right] = 2 * h + 2
                nd_depth[left] = nd_depth[node] + 1
                nd_depth[right] = nd_depth[node] + 1
                nd_lt[left] = _leaf_term(n_left, sum_left, sigma2_t, f_sigma2)
                nd_lt[right] = _leaf_term(n - n_left, sum_right, sigma2_t, f_sigma2)
                nd_sum[left] = sum_left
                nd_sum[right] = sum_right
                n_nodes += 2
                if nd_kid[node] < 0:
                    nd_kid[node] = left
                    nd_kid_v[node] = v
                    nd_kid_t[node] = tau

            if not optimal:
                logw[c] += nd_lt[left] + nd_lt[right] - nd_lt[node]

            P_state[c, j] = 2
            P_var[c, j] = v
            P_thr[c, j] = tau
            if P_cnt[c] + 2 > K:
                K *= 2
                P_node = _grow_2d(P_node, K)
                P_state = _grow_2d(P_state, K)
                P_var = _grow_2d(P_var, K)
                P_thr = _grow_2d(P_thr, K)
            P_node[c, P_cnt[c]] = left
            P_node[c, P_cnt[c] + 1] = right
            P_cnt[c] += 2
        n_stages += 1

        done = True
        for c in range(C):
            if not dead[c] and P_head[c] < P_cnt[c]:
                done = False
                break
        if done:
            break

        # Conditional multinomial resampling: slot 0 keeps the reference.
        w = np.exp(logw - logw.max())
        cw = np.cumsum(w)
        anc = np.zeros(C, np.int64)
        for c in range(1, C):
            anc[c] = min(np.searchsorted(cw, np.random.random() * cw[-1], side="right"), C - 1)
        new_node = np.zeros_like(P_node)
        new_state = np.zeros_like(P_state)
        new_var = np.zeros_like(P_var)
        new_thr = np.zeros_like(P_thr)
        new_cnt = np.empty_like(P_cnt)
        new_head = np.empty_like(P_head)
        new_dead = np.empty_like(dead)
        for c in range(C):
            a = anc[c]
            cnt = P_cnt[a]
            new_node[c, :cnt] = P_node[a, :cnt]
            new_state[c, :cnt] = P_state[a, :cnt]
            new_var[c, :cnt] = P_var[a, :cnt]
            new_thr[c, :cnt] = P_thr[a, :cnt]
            new_cnt[c] = cnt
            new_head[c] = P_head[a]
            new_dead[c] = dead[a]
        P_node = new_node
        P_state = new_state
        P_var = new_var
        P_thr = new_thr
        P_cnt = new_cnt
        P_head = new_head
        dead = new_dead
        logw[:] = 0.0

    w = np.exp(logw - logw.max())
    cw = np.cumsum(w)
    k = min(np.searchsorted(cw, np.random.random() * cw[-1], side="right"), C - 1)

    max_h = 0
    for j in range(P_cnt[k]):
        max_h = max(max_h, nd_heap[P_node[k, j]])
    size = 8
    while size < max_h + 1:
        size *= 2
    out_vars = np.full(size, -2, np.int64)
    out_thr = np.full(size, np.nan, thr_vals.dtype)
    out_n = np.zeros(size, np.int32)
    leaf_ids = np.empty(N, np.int32)
    changed = False
    for j in range(P_cnt[k]):
        node = P_node[k, j]
        h = nd_heap[node]
        out_n[h] = nd_n[node]
        if P_state[k, j] == 2:
            out_vars[h] = P_var[k, j]
            out_thr[h] = P_thr[k, j]
            if not (h < L and ref_vars[h] == P_var[k, j] and ref_thr[h] == P_thr[k, j]):
                changed = True
        else:
            out_vars[h] = -1
            for q in range(nd_start[node], nd_start[node] + nd_n[node]):
                leaf_ids[buf[q]] = h
            if h < L and ref_vars[h] >= 0:
                changed = True
    return out_vars, out_thr, out_n, leaf_ids, k, changed, n_stages
