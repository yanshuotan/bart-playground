"""Cross-dataset: noise fraction, n, R-hat, chain disagreement on the response scale, baselines."""
import os, sys, pickle, re, numpy as np, arviz as az, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = [sys.argv[1] + "/bp", sys.argv[1] + "/bp/diagnosis"]
from fixed100_support import load_dataset
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
STORE = os.environ.get("PAPER_STORE", "paper_store_20261028")
D = pickle.load(open(sys.argv[1] + "/datasets.pkl", "rb"))
if "cpu_act" not in D: D["cpu_act"] = load_dataset("cpu_act")
from fixed100_support import DATASET_CONFIGS, _load_generator_dataset
for p_ in (20, 100, 200):
    from fixed100_support import repo_nested_friedman1_generator
    D[f"dart{p_}"] = repo_nested_friedman1_generator(2000, p_, 42, 1.0)
FT = {"FriedmanSparseDir_p20": "FriedmanSparseDirP20K5", "FriedmanSparseDir_p100": "FriedmanSparseDir", "FriedmanSparseDir_p200": "FriedmanSparseDirP200K5"}
TAGS = {"abalone": "Abalone", "friedman": "Friedman", "concrete": "Concrete", "calhousing": "CalHousing_subsample5000",
        "airfoil": "Airfoil", "ccpp": "CCPP", "cpu_act": "CPUAct", "superconduct": "Superconduct_subsample5000",
        "seoul_bike": "SeoulBike", "dart20": "FriedmanSparseDir_p20", "dart100": "FriedmanSparseDir_p100", "dart200": "FriedmanSparseDir_p200"}
def rd(path):
    with open(path) as f:
        shape = tuple(int(s) for s in re.findall(r"\d+", f.readline()))
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)
def ix(t, r, k): return np.loadtxt(f"{STORE}/fixed100_{t}/indices/fixed100_{FT.get(t, t)}__run{r:03d}__{k}.csv", delimiter=",", comments="#").astype(int).ravel()
rows = []
for name, t in TAGS.items():
    X, y = D[name]
    acc = []
    for r in range(5):
        tr, te = ix(t, r, "train_idx"), ix(t, r, "fixed_test_idx")
        rng = y[tr].max() - y[tr].min()
        base = f"{STORE}/fixed100_{t}"
        res = {}
        for m in ["default", "default_long", "mtmh_pt"]:
            P = rd(f"{base}/preds/fixed100_{FT.get(t, t)}__run{r:03d}__{m}__preds.csv")      # chains, 100, draws
            Sg = rd(f"{base}/sigmas/fixed100_{FT.get(t, t)}__run{r:03d}__{m}__sigmas.csv")[..., 0]
            h = 3000                                                                 # paper: drop first 30%, keep 7,000
            P, Sg = P[:, :, h:], Sg[:, h:]
            sig = np.sqrt(Sg.mean()) * rng                                          # raw-scale noise sd
            rh = np.array([az.rhat(P[:, i, :]) for i in range(P.shape[1])])
            cm = P.mean(2)                                                          # chain means
            res[m] = dict(rhat=np.median(rh), sig=sig, between=np.median(cm.std(0, ddof=1)) / sig,
                          within=np.median(P.std(2).mean(0)) / sig,
                          rmse=np.sqrt(np.mean((P.mean((0, 2)) - y[te]) ** 2)),
                          chain_rmse_spread=np.ptp([np.sqrt(np.mean((cm[c] - y[te]) ** 2)) for c in range(4)]))
        gb = HistGradientBoostingRegressor(max_iter=500, random_state=0).fit(X[tr], y[tr])
        rf = RandomForestRegressor(300, n_jobs=8, random_state=0).fit(X[tr], y[tr])
        assert np.allclose(y[te], np.loadtxt(f"{base}/subsample_y_test/fixed100_{FT.get(t, t)}__run{r:03d}__subsample_y_test.csv", comments="#"))
        acc.append(dict(n=len(tr), d=X.shape[1], sdy=y[tr].std(), nu=(res["default_long"]["sig"] / y[tr].std()) ** 2,
                        **{f"{m}_{k}": v for m in res for k, v in res[m].items()},
                        gb=np.sqrt(np.mean((gb.predict(X[te]) - y[te]) ** 2)),
                        rf=np.sqrt(np.mean((rf.predict(X[te]) - y[te]) ** 2))))
    row = {k: np.mean([a[k] for a in acc]) for k in acc[0]}
    row["name"] = name
    rows.append(row); print(name, "done", flush=True)
pickle.dump(rows, open(sys.argv[1] + "/xds_rows.pkl", "wb"))
rows.sort(key=lambda r: r["default_long_rhat"])
print(f"{'setting':13s} {'n':>5s} {'d':>3s} {'1-R2':>6s} {'n/nu':>7s} | {'Rh10k':>6s} {'Rh1e7':>6s} {'RhMP':>6s} | {'B/sig':>6s} {'W/sig':>6s} {'B/W':>5s} | {'BART':>7s} {'GBM':>7s} {'RF':>7s} {'sdY':>7s}")
for r in rows:
    print(f"{r['name']:13s} {r['n']:5.0f} {r['d']:3.0f} {r['nu']:6.3f} {r['n']/r['nu']:7.0f} | {r['default_rhat']:6.3f} {r['default_long_rhat']:6.3f} {r['mtmh_pt_rhat']:6.3f} | "
          f"{r['default_between']:6.3f} {r['default_within']:6.3f} {r['default_between']/r['default_within']:5.2f} | {r['default_long_rmse']:7.3g} {r['gb']:7.3g} {r['rf']:7.3g} {r['sdy']:7.3g}")
