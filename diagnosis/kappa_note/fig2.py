import sys, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S, OUT = sys.argv[1], sys.argv[2]
T = pickle.load(open(S + "/table48.pkl", "rb")); kg = pickle.load(open(S + "/kgbm.pkl", "rb")); acc = pickle.load(open(S + "/acc_all.pkl", "rb"))
LAB = {"abalone": "Abalone", "friedman": "Friedman", "concrete": "Concrete", "calhousing": "CalHousing", "airfoil": "Airfoil",
       "ccpp": "CCPP", "cpu_act": "CPU act", "superconduct": "Superconduct", "seoul_bike": "SeoulBike"}
def grp(t):
    if t["group"] == "main" and t["name"].startswith("dart"): return "dart"
    if t["group"] != "ctl": return t["group"]
    return "synth" if "__synth" in t["name"] else "realnoise"
G = [("main", "Paper settings", "#2a78d6", "o"), ("dart", "Paper settings, DART", "#e87ba4", "P"), ("sweep", "Size and noise sweeps", "#eb6834", "s"),
     ("realnoise", "Observed y, with added noise", "#1baf7a", "^"), ("synth", "Semi-synthetic y", "#6250d6", "D")]
kap = lambda t: t["n"] / t["nug"]
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "xtick.color": "#444", "ytick.color": "#444"})
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
for g, lab, c, m in G:
    tt = [t for t in T if grp(t) == g]
    a.scatter([kap(t) for t in tt], [t["rh"] - 1 for t in tt], c=c, marker=m, s=40 if g == "main" else (60 if g == "dart" else 30), label=lab,
              edgecolors="white", linewidths=1.1, zorder=3 if g == "main" else 2)
off = {"CalHousing": (7, -1), "Airfoil": (-36, 3), "Friedman": (6, -12), "Concrete": (-52, -9), "Superconduct": (6, 3), "SeoulBike": (6, -9)}
for t in T:
    if grp(t) == "main": a.annotate(LAB[t["name"]], (kap(t), t["rh"] - 1), xytext=off.get(LAB[t["name"]], (6, -3)), textcoords="offset points", fontsize=7.5, color="#333")
x = np.array([kap(t) for t in T]); y = np.array([t["rh"] - 1 for t in T]); b = np.polyfit(np.log(x), np.log(y), 1)
xx = np.geomspace(x.min(), x.max(), 50); a.plot(xx, np.exp(np.polyval(b, np.log(xx))), color="#999", lw=1, ls="--", zorder=1)
a.set_xscale("log"); a.set_yscale("log"); a.grid(True, which="major", color="#e8e8e8", lw=0.6, zorder=0)
a.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]); a.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
a.set_xlabel(r"$\kappa = n/(1-R^2_{\mathrm{CV}})$"); a.set_ylabel(r"median pointwise $\hat R - 1$")
a.set_title("(a) Default, four chains of $10^4$ iterations, 48 settings", fontsize=9.5, loc="left")
a.legend(frameon=False, fontsize=8, loc="lower right")
b_ = ax[1]
for g, lab, c, m in G:
    rr = [(kg[(gg, n)][0] / kg[(gg, n)][1], A["change"]) for gg, n, A in acc
          if not n.endswith("__real") and (gg == "main" and n.startswith("dart") == (g == "dart") if g in ("main", "dart") else gg == "ctl" and (("__synth" in n) == (g == "synth")))]
    if g == "sweep" or not rr: continue
    b_.scatter([r[0] for r in rr], [r[1] for r in rr], c=c, marker=m, s=60 if g == "dart" else 36, edgecolors="white", linewidths=1.1, label=lab)
b_.set_xscale("log"); b_.grid(True, color="#e8e8e8", lw=0.6)
b_.set_xlabel(r"$\kappa = n/(1-R^2_{\mathrm{CV}})$"); b_.set_ylabel("acceptance rate of change moves")
b_.set_title("(b) Default, iterations 3,001–5,000, run 0, 25 settings", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(OUT + "/rhat_vs_kappa.png", dpi=200); print(b)
