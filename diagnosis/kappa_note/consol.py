import sys, pickle, numpy as np
from scipy.stats import spearmanr, kendalltau
S = sys.argv[1]
main = pickle.load(open(S + "/xds_rows.pkl", "rb")); sweep = [r for r in pickle.load(open(S + "/master_rows.pkl", "rb")) if r["group"] != "main"]
ctl = pickle.load(open(S + "/mixana_rows.pkl", "rb")); kg = pickle.load(open(S + "/kgbm.pkl", "rb"))
T = []
for r in main: T.append(dict(group="main", name=r["name"], n=r["n"], nu=(r["default_sig"] / r["sdy"]) ** 2, nug=kg[("main", r["name"])][1],
                             rh=r["default_rhat"], rhl=r["default_long_rhat"], nul=r["nu"]))
for r in sweep: T.append(dict(group="sweep", name=r["name"], n=r["n"], nu=r["nu"], nug=kg[("sweep", r["name"])][1], rh=r["rh"]))
for r in ctl: T.append(dict(group="ctl", name=r["cfg"], n=r["n"], nu=r["nu"], nug=kg[("ctl", r["cfg"])][1], rh=r["rh"]))
pickle.dump(T, open(S + "/table48.pkl", "wb"))
rh = np.array([t["rh"] for t in T]); n = np.array([t["n"] for t in T])
nu = np.array([t["nu"] for t in T]); nug = np.array([t["nug"] for t in T])
for lab, x in [("n/nu (posterior sigma)", n / nu), ("n/nu (CV GBM)", n / nug), ("n", n), ("1/nu (posterior)", 1 / nu), ("1/nu (GBM)", 1 / nug)]:
    print(f"all settings, 10k R-hat vs {lab:24s} Spearman {spearmanr(x, rh)[0]:.2f}  Kendall {kendalltau(x, rh)[0]:.2f}")
for g in ["main", "sweep", "ctl"]:
    m = np.array([t["group"] == g for t in T])
    print(g, m.sum(), "Spearman n/nu_gbm %.2f  n %.2f" % (spearmanr((n / nug)[m], rh[m])[0], spearmanr(n[m], rh[m])[0]))
M = [t for t in T if t["group"] == "main"]
rl = [t["rhl"] for t in M]
for lab, x in [("n/nu long", [t["n"] / t["nul"] for t in M]), ("n/nu gbm", [t["n"] / t["nug"] for t in M]), ("n", [t["n"] for t in M])]:
    print(f"9 paper settings, 1e7 R-hat vs {lab:10s} Spearman {spearmanr(x, rl)[0]:.2f}")
b = np.polyfit(np.log(n / nug), np.log(rh - 1), 1); print("fit log(Rhat-1) on log(n/nu_gbm): slope %.2f" % b[0])
print("\nname n nu_post nu_gbm kappa_gbm rh rhl")
for t in sorted(M, key=lambda t: t["n"] / t["nug"]):
    print(f"{t['name']:13s} {t['n']:5.0f} {t['nu']:.3f} {t['nul']:.3f} {t['nug']:.3f} {t['n']/t['nug']:8.0f} {t['rh']:.3f} {t['rhl']:.3f}")
print()
for t in T:
    if t["group"] == "ctl": print(f"{t['name']:26s} nu_post {t['nu']:.3f} nu_gbm {t['nug']:.3f} rh {t['rh']:.3f}")
