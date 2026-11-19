import sys, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
S, OUT = sys.argv[1], sys.argv[2]
main = pickle.load(open(S + "/xds_rows.pkl", "rb")); sweep = [r for r in pickle.load(open(S + "/master_rows.pkl", "rb")) if r["group"] != "main"]
ctl = pickle.load(open(S + "/mixana_rows.pkl", "rb"))
LAB = {"abalone": "Abalone", "friedman": "Friedman", "concrete": "Concrete", "calhousing": "CalHousing", "airfoil": "Airfoil",
       "ccpp": "CCPP", "cpu_act": "CPU act", "superconduct": "Superconduct", "seoul_bike": "SeoulBike"}
# every point: Default, 4 chains x 10k, median pointwise R-hat on the second half; 1-R^2 from the same draws
pts = {"Paper settings": [(r["n"] / (r["default_sig"] / r["sdy"]) ** 2, r["default_rhat"], LAB[r["name"]]) for r in main],
       "Size / noise sweeps": [(r["n"] / r["nu"], r["rh"], None) for r in sweep],
       "Real y + added noise": [(r["n"] / r["nu"], r["rh"], None) for r in ctl if "__synth" not in r["cfg"]],
       "Semi-synthetic, known f": [(r["n"] / r["nu"], r["rh"], None) for r in ctl if "__synth" in r["cfg"]]}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]; MK = ["o", "s", "^", "D"]
allx = np.concatenate([[p[0] for p in v] for v in pts.values()]); ally = np.concatenate([[p[1] for p in v] for v in pts.values()])
rho = spearmanr(allx, ally)[0]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#555", "ytick.color": "#555"})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]
for (k, v), c, m in zip(pts.items(), C, MK):
    a.scatter([p[0] for p in v], [p[1] for p in v], s=46 if k == "Paper settings" else 34, c=c, marker=m, label=k,
              edgecolors="white", linewidths=1.2, zorder=3 if k == "Paper settings" else 2)
for x, y, l in pts["Paper settings"]:
    off = {"CalHousing": (-62, 3), "Friedman": (5, -10), "CCPP": (5, 4)}.get(l, (5, -3))
    a.annotate(l, (x, y), xytext=off, textcoords="offset points", fontsize=8, color="#444")
a.set_xscale("log"); a.grid(True, color="#e6e6e6", lw=0.6, zorder=0)
a.set_xlabel(r"$n\,/\,(1-R^2)$   (training size $\div$ noise fraction)"); a.set_ylabel(r"median pointwise $\hat R$ (Default, $10^4$ its)")
a.set_title(f"(a) One quantity orders mixing difficulty (45 settings, Spearman ρ = {rho:.2f})", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8.5, loc="upper left")
b = ax[1]
for ds, c in zip(["seoul_bike", "airfoil", "abalone"], C):
    for kind, ls, mk in [("real", "-", "o"), ("synth", "--", "D")]:
        rr = [r for r in ctl if r["cfg"].startswith(ds) and (("synth" in r["cfg"]) == (kind == "synth"))]
        nom = {"seoul_bike": 0.042, "airfoil": 0.025, "abalone": 0.383}[ds]
        xs = [float(r["cfg"].split("_")[-1]) if "_" in r["cfg"].split("__")[1] else nom for r in rr]
        o = np.argsort(xs); xs = np.array(xs)[o]; ys = np.array([r["rh"] for r in rr])[o]
        b.plot(xs, ys, ls=ls, marker=mk, color=c, lw=2, ms=6, mec="white", mew=1.2,
               label=f"{LAB[ds]}: {'real y (+ noise)' if kind == 'real' else 'f̂(X) + Gaussian noise'}")
b.set_xscale("log"); b.grid(True, color="#e6e6e6", lw=0.6)
b.set_xlabel(r"noise fraction $1-R^2$ (design value)"); b.set_ylabel(r"median pointwise $\hat R$")
b.set_title("(b) Same covariates, only the noise level changes", fontsize=10, loc="left")
b.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "/mixing_vs_information.png", dpi=180)
print("rho", rho, len(allx))
