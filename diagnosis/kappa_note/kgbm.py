"""1-R^2 before any MCMC: 5-fold cross-validated HistGradientBoosting residual variance over var(y_train)."""
import sys, glob, os, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
S = sys.argv[1]; sys.path[:0] = [S + "/bp", S + "/bp/diagnosis"]
from fixed100_support import DATASET_CONFIGS, _load_generator_dataset
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from joblib import Parallel, delayed
STORE = os.environ.get("PAPER_STORE", "paper_store_20261028")
D = pickle.load(open(S + "/datasets.pkl", "rb"))
if "cpu_act" not in D:
    from fixed100_support import load_dataset; D["cpu_act"] = load_dataset("cpu_act")
for p_ in (20, 100, 200):
    from fixed100_support import repo_nested_friedman1_generator
    D[f"dart{p_}"] = repo_nested_friedman1_generator(2000, p_, 42, 1.0)
def idx(p): return np.loadtxt(p, delimiter=",", comments="#").astype(int).ravel()
jobs = []   # (group, name, run, X, y)
FT = {"FriedmanSparseDir_p20": "FriedmanSparseDirP20K5", "FriedmanSparseDir_p100": "FriedmanSparseDir", "FriedmanSparseDir_p200": "FriedmanSparseDirP200K5"}
TAGS = {"abalone": "Abalone", "friedman": "Friedman", "concrete": "Concrete", "calhousing": "CalHousing_subsample5000",
        "airfoil": "Airfoil", "ccpp": "CCPP", "cpu_act": "CPUAct", "superconduct": "Superconduct_subsample5000", "seoul_bike": "SeoulBike", "dart20": "FriedmanSparseDir_p20", "dart100": "FriedmanSparseDir_p100", "dart200": "FriedmanSparseDir_p200"}
for name, t in TAGS.items():
    X, y = D[name]
    for r in range(5):
        tr = idx(f"{STORE}/fixed100_{t}/indices/fixed100_{FT.get(t, t)}__run{r:03d}__train_idx.csv"); jobs.append(("main", name, r, X[tr], y[tr]))
cal = fetch_california_housing(return_X_y=True); fr = {}
for d in sorted(glob.glob(STORE + "/store_scaling/scal_*")):
    tag = os.path.basename(d)
    if tag.startswith("scal_CCPP"): X, y = D["ccpp"]
    elif tag.startswith("scal_CalHousing"): X, y = cal
    else:
        sd = int(tag[-3:]) / 100
        if sd not in fr: fr[sd] = _load_generator_dataset({**DATASET_CONFIGS["friedman"], "n_samples": 20_100, "noise": sd}, "friedman1")
        X, y = fr[sd]
    for r in range(5):
        tr = idx(f"{d}/indices/{tag}__run{r:03d}__train_idx.csv"); jobs.append(("sweep", tag, r, X[tr], y[tr]))
# controlled experiments: rebuild the training responses exactly as mixexp.py does
NU0 = {"seoul_bike": 0.042, "abalone": 0.383, "airfoil": 0.025}
PLAN = {"seoul_bike": [("real", None), ("noise", 0.17), ("noise", 0.4), ("synth", 0.042), ("synth", 0.17), ("synth", 0.4)],
        "abalone": [("real", None), ("synth", 0.38), ("synth", 0.1), ("synth", 0.04), ("synth", 0.01)],
        "airfoil": [("real", None), ("noise", 0.1), ("noise", 0.4), ("synth", 0.025), ("synth", 0.1)]}
for name, plan in PLAN.items():
    X, y = D[name]; t = TAGS[name]
    fhat = HistGradientBoostingRegressor(max_iter=500, random_state=0).fit(X, y).predict(X)
    for kind, nu in plan:
        cfg = f"{name}__{kind}" + (f"_{nu:g}" if nu else "")
        for r in range(3):
            tr = idx(f"{STORE}/fixed100_{t}/indices/fixed100_{FT.get(t, t)}__run{r:03d}__train_idx.csv")
            rng = np.random.default_rng([7, r, int(1000 * (nu or 0)), len(kind)])
            if kind == "real": ytr = y[tr]
            elif kind == "noise": ytr = y[tr] + rng.normal(0, np.sqrt((nu * y[tr].var() - NU0[name] * y[tr].var()) / (1 - nu)), len(tr))
            else: ytr = fhat[tr] + rng.normal(0, np.sqrt(nu / (1 - nu) * fhat[tr].var()), len(tr))
            jobs.append(("ctl", cfg, r, X[tr], ytr))
def one(g, n, r, X, y):
    p = cross_val_predict(HistGradientBoostingRegressor(max_iter=500, random_state=0), X, y, cv=KFold(5, shuffle=True, random_state=0))
    return g, n, r, len(y), np.mean((y - p) ** 2) / y.var()
res = Parallel(n_jobs=8)(delayed(one)(*j) for j in jobs)
out = {}
for g, n, r, nn, nu in res: out.setdefault((g, n), []).append((nn, nu))
out = {k: (np.mean([a for a, b in v]), np.mean([b for a, b in v])) for k, v in out.items()}
pickle.dump(out, open(S + "/kgbm.pkl", "wb")); print(len(out), "settings")
