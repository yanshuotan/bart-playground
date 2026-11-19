import os, sys, re, numpy as np, pickle
S = sys.argv[1]
STORE = os.environ.get("PAPER_STORE", "paper_store_20261028")
D = pickle.load(open(S + "/datasets.pkl", "rb"))
TAGS = {"abalone": "Abalone", "friedman": "Friedman", "concrete": "Concrete", "calhousing": "CalHousing_subsample5000",
        "airfoil": "Airfoil", "ccpp": "CCPP", "superconduct": "Superconduct_subsample5000", "seoul_bike": "SeoulBike"}
def rd(p):
    with open(p) as f: shape = tuple(int(s) for s in re.findall(r"\d+", f.readline()))
    return np.loadtxt(p, delimiter=",", comments="#").reshape(shape)
print("sigma (raw scale) by decile of the 1e7 Default chains, mean over 5 runs x 4 chains, relative to decile 1; and test RMSE of each decile's mean")
out = {}
for name, t in TAGS.items():
    X, y = D[name]; sig, rm = [], []
    for r in range(5):
        tr = np.loadtxt(f"{STORE}/fixed100_{t}/indices/fixed100_{t}__run{r:03d}__train_idx.csv", delimiter=",", comments="#").astype(int).ravel()
        te = np.loadtxt(f"{STORE}/fixed100_{t}/indices/fixed100_{t}__run{r:03d}__fixed_test_idx.csv", delimiter=",", comments="#").astype(int).ravel()
        Sg = np.sqrt(rd(f"{STORE}/fixed100_{t}/sigmas/fixed100_{t}__run{r:03d}__default_long__sigmas.csv")[..., 0]) * np.ptp(y[tr])
        P = rd(f"{STORE}/fixed100_{t}/preds/fixed100_{t}__run{r:03d}__default_long__preds.csv")
        sig.append(Sg.reshape(4, 10, -1).mean(2))                     # chains x deciles
        rm.append(np.sqrt(((P.reshape(4, 100, 10, -1).mean(3) - y[te][None, :, None]) ** 2).mean(1)))
    sig, rm = np.concatenate(sig), np.concatenate(rm)
    out[name] = (sig, rm)
    s = sig.mean(0); r_ = rm.mean(0)
    print(f"{name:13s} sigma: " + " ".join(f"{v/s[0]:.3f}" for v in s) + "  | RMSE: " + " ".join(f"{v/r_[0]:.3f}" for v in r_), flush=True)
pickle.dump(out, open(S + "/drift.pkl", "wb"))
