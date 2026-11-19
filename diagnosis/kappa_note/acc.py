"""Acceptance rates of Default grow/prune/change/swap, iterations 3001-5000 of one chain per setting (run 0)."""
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ[k] = "1"
S = sys.argv[1]
src = open(S + "/kgbm.py").read().split("def one(")[0]
exec(src)                                   # builds `jobs` exactly as kgbm.py
import experiment_fixed100 as exp
from bart_playground import DefaultBART
sel = [j for j in jobs if j[2] == 0 and j[0] == "main" and j[1].startswith("dart")]
def run(g, name, r, X, y):
    m = DefaultBART(ndpost=3000, nskip=0, n_trees=100, tol=1, proposal_probs=exp.default_proposal_probs,
                    random_state=777, dirichlet_prior=name.startswith("dart"), s_alpha=1.0)
    m.fit(X, y, quietly=True)
    sm = m.sampler
    a0, p0 = dict(sm.move_accepted_counts), dict(sm.move_success_counts)
    sm.continue_run(2000, last_state=m.trace[-1], quietly=True)
    acc = {k: (sm.move_accepted_counts[k] - a0[k]) / max(1, sm.move_success_counts[k] - p0[k]) for k in a0}
    leaves = np.mean([t.n_leaves if hasattr(t, "n_leaves") else np.sum(np.asarray(t.vars) == -1) for t in sm.trace[-1].trees]) if hasattr(sm, "trace") else np.nan
    return g, name, acc
res = Parallel(n_jobs=8)(delayed(run)(*j) for j in sel)
pickle.dump(res, open(S + "/acc_dart.pkl", "wb"))
for g, name, acc in res: print(f"{g:5s} {name:26s} " + " ".join(f"{k}={v:.3f}" for k, v in acc.items()), flush=True)
