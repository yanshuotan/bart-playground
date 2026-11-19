"""Controlled noise experiments: real y, real y + extra noise, and semi-synthetic y = f_hat(X) + noise.

Each config: 3 paired runs (store train splits 0-2, store fixed test panel) x 4 Default chains x 10k.
"""
import os, sys, json, pickle, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ[k] = "1"
import numpy as np
S = sys.argv[1]; sys.path[:0] = [S + "/bp", S + "/bp/diagnosis"]
import experiment_fixed100 as exp
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
STORE = os.environ.get("PAPER_STORE", "paper_store_20261028")
OUT = S + "/mixexp"
D = pickle.load(open(S + "/datasets.pkl", "rb"))
TAG = {"seoul_bike": "SeoulBike", "abalone": "Abalone", "airfoil": "Airfoil"}
NU0 = {"seoul_bike": 0.042, "abalone": 0.383, "airfoil": 0.025}   # 1-R^2 of the real data (long Default chains)
PLAN = {"seoul_bike": [("real", None), ("noise", 0.17), ("noise", 0.4), ("synth", 0.042), ("synth", 0.17), ("synth", 0.4)],
        "abalone": [("real", None), ("synth", 0.38), ("synth", 0.1), ("synth", 0.04), ("synth", 0.01)],
        "airfoil": [("real", None), ("noise", 0.1), ("noise", 0.4), ("synth", 0.025), ("synth", 0.1)]}
RUNS = [0, 1, 2]
def ix(t, r, k):
    return np.loadtxt(f"{STORE}/fixed100_{t}/indices/fixed100_{t}__run{r:03d}__{k}.csv", delimiter=",", comments="#").astype(int).ravel()

tasks, meta = [], {}
for name, plan in PLAN.items():
    X, y = D[name]; t = TAG[name]
    fhat = HistGradientBoostingRegressor(max_iter=500, random_state=0).fit(X, y).predict(X)
    for kind, nu in plan:
        cfg = f"{name}__{kind}" + (f"_{nu:g}" if nu else "")
        for r in RUNS:
            tr, te = ix(t, r, "train_idx"), ix(t, r, "fixed_test_idx")
            rng = np.random.default_rng([7, r, int(1000 * (nu or 0)), len(kind)])
            if kind == "real":
                ytr, f_te = y[tr], None
            elif kind == "noise":
                s2 = (nu * y[tr].var() - NU0[name] * y[tr].var()) / (1 - nu)
                ytr, f_te = y[tr] + rng.normal(0, np.sqrt(s2), len(tr)), None
            else:
                s2 = nu / (1 - nu) * fhat[tr].var()
                ytr, f_te = fhat[tr] + rng.normal(0, np.sqrt(s2), len(tr)), fhat[te]
            yte = y[te] if f_te is None else f_te
            meta[(cfg, r)] = dict(n=len(tr), var_ytr=float(ytr.var()), range_ytr=float(np.ptp(ytr)), truth=f_te is not None,
                                  y_test=yte.tolist())
            for c in range(4):
                tasks.append(dict(chain_id=c, chain_seed=500_000 + 2024 + r * 1000 + c, run_id=r, X_train=X[tr], y_train=ytr,
                                  X_test_fixed=X[te], y_test_fixed=yte, tmp_dir=f"{OUT}/{cfg}"))
os.makedirs(OUT, exist_ok=True)
pickle.dump(meta, open(OUT + "/meta.pkl", "wb"))
print(len(tasks), "chains", flush=True)
# large-n chains first so the tail of the schedule is short
tasks.sort(key=lambda d: -len(d["y_train"]))
def go(task):
    t0 = time.time()
    exp.run_long_default_chain_streaming(**task, ndpost=10_000, nskip=0, n_trees=100,
                                         proposal_probs_default=exp.default_proposal_probs, store_every=1,
                                         chunk_size=2000, store_preds=True, dirichlet_prior=False, s_alpha=1.0)
    print(f"done {task['tmp_dir'].split('/')[-1]} run{task['run_id']} chain{task['chain_id']} {time.time()-t0:.0f}s", flush=True)
Parallel(n_jobs=8, verbose=0)(delayed(go)(t) for t in tasks)
print("ALL DONE", flush=True)
