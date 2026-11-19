import sys, pickle, re, glob, os, numpy as np, arviz as az, warnings
warnings.filterwarnings("ignore")
S = sys.argv[1]; sys.path[:0] = [S + "/bp", S + "/bp/diagnosis"]
from fixed100_support import DATASET_CONFIGS, _load_generator_dataset
from sklearn.datasets import fetch_california_housing
from scipy.stats import spearmanr
ROOT = os.environ.get("PAPER_STORE", "paper_store_20261028") + "/store_scaling"
D = pickle.load(open(S + "/datasets.pkl", "rb"))
def rd(p):
    with open(p) as f: shape = tuple(int(s) for s in re.findall(r"\d+", f.readline()))
    return np.loadtxt(p, delimiter=",", comments="#").reshape(shape)
src = {}
def source(tag):
    if tag.startswith("scal_CCPP"): return D["ccpp"]
    if tag.startswith("scal_CalHousing"):
        if "cal" not in src:
            X, y = fetch_california_housing(return_X_y=True); src["cal"] = (X, y.astype(float))
        return src["cal"]
    sd = int(tag[-3:]) / 100
    if sd not in src:
        src[sd] = _load_generator_dataset({**DATASET_CONFIGS["friedman"], "n_samples": 20_100, "noise": sd}, "friedman1")
    return src[sd]
rows = []
for d in sorted(glob.glob(ROOT + "/scal_*")):
    tag = os.path.basename(d); X, y = source(tag); acc = []
    for r in range(5):
        tr = np.loadtxt(f"{d}/indices/{tag}__run{r:03d}__train_idx.csv", delimiter=",", comments="#").astype(int).ravel()
        P = rd(f"{d}/preds/{tag}__run{r:03d}__default__preds.csv")[:, :, 3000:]
        Sg = rd(glob.glob(f"{d}/sigmas/{tag}__run{r:03d}__default__sigmas.csv")[0])[..., 0][:, 3000:]
        nu = Sg.mean() * (y[tr].max() - y[tr].min()) ** 2 / y[tr].var()
        rh = np.median([az.rhat(P[:, i, :]) for i in range(P.shape[1])])
        acc.append((len(tr), nu, rh))
    n, nu, rh = np.mean(acc, 0); rows.append(dict(name=tag, n=n, nu=nu, rh=rh, group=tag.split("_")[1]))
    print(f"{tag:28s} n={n:6.0f} 1-R2={nu:.4f} n/nu={n/nu:9.0f} Rhat={rh:.3f}", flush=True)
main = pickle.load(open(S + "/xds_rows.pkl", "rb"))
for m in main: rows.append(dict(name=m["name"], n=m["n"], nu=m["nu"], rh=m["default_rhat"], group="main"))
pickle.dump(rows, open(S + "/master_rows.pkl", "wb"))
x = np.array([r["n"] / r["nu"] for r in rows]); yv = np.array([r["rh"] for r in rows]); nn = np.array([r["n"] for r in rows])
print("Spearman Rhat vs n/nu:", spearmanr(x, yv)[0], " vs n:", spearmanr(nn, yv)[0], " vs 1/nu:", spearmanr([1/r["nu"] for r in rows], yv)[0])
b = np.polyfit(np.log(x), np.log(yv - 1), 1); print("log(Rhat-1) = %.2f log(n/nu) + %.2f" % tuple(b))
