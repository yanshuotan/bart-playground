"""Reproduce Table `tab:kappa` and the Spearman correlations in Appendix `app:kappa`.

kappa = n / (1 - R^2_CV), where 1 - R^2_CV is the 5-fold cross-validated
HistGradientBoosting MSE over var(y_train), averaged over the five runs
(computed by kgbm.py -> kgbm.pkl). The three R-hat columns are the five-run
means of the median pointwise R-hat as reported in the paper's appendix table
(tab:appendix-all-methods).

    python kappa_appendix.py [path/to/kgbm.pkl]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

KG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("kgbm.pkl")
kg = pickle.load(open(KG, "rb"))

# name in kgbm.pkl -> (label, Default, Default thinned long record, MTMH+PT)
RHAT = {
    "abalone": ("Abalone", 1.097, 1.000, 1.013),
    "concrete": ("Concrete", 1.173, 1.003, 1.032),
    "dart200": ("DART d=200", 1.184, 1.000, 1.015),
    "dart100": ("DART d=100", 1.173, 1.000, 1.021),
    "dart20": ("DART d=20", 1.199, 1.000, 1.018),
    "friedman": ("Friedman", 1.233, 1.002, 1.028),
    "airfoil": ("Airfoil", 1.298, 1.041, 1.056),
    "calhousing": ("CalHousing", 1.251, 1.018, 1.072),
    "superconduct": ("Superconduct", 1.441, 1.234, 1.142),
    "seoul_bike": ("SeoulBike", 1.466, 1.318, 1.278),
    "ccpp": ("CCPP", 1.439, 1.126, 1.213),
    "cpu_act": ("CPU activity", 1.518, 1.236, 1.183),
}

rows = []
for name, (label, rd, rl, rp) in RHAT.items():
    n, nu = kg[("main", name)]
    rows.append((label, n, nu, n / nu, rd, rl, rp))
rows.sort(key=lambda r: r[3])

print(f"{'Setting':13s} {'n':>6s} {'1-R2cv':>7s} {'kappa':>8s} {'Default':>8s} {'thinned':>8s} {'MTMH+PT':>8s}")
for label, n, nu, k, rd, rl, rp in rows:
    print(f"{label:13s} {n:6.0f} {nu:7.3f} {k:8.0f} {rd:8.3f} {rl:8.3f} {rp:8.3f}")

n = np.array([r[1] for r in rows]); nu = np.array([r[2] for r in rows]); kappa = n / nu
print("\nSpearman correlation with median R-hat (12 settings)")
print(f"{'':10s} {'Default':>8s} {'thinned':>8s} {'MTMH+PT':>8s}")
for lab, x in [("kappa", kappa), ("n", n), ("1/(1-R2)", 1 / nu)]:
    vals = [spearmanr(x, [r[c] for r in rows])[0] for c in (4, 5, 6)]
    print(f"{lab:10s} " + " ".join(f"{v:8.2f}" for v in vals))

hard = [r[0] for r in rows if r[3] > 2.5e4]
both = [r[0] for r in rows if r[5] > 1.1 and r[6] > 1.1]
print(f"\nkappa > 2.5e4: {hard}\nthinned and MTMH+PT both > 1.1: {both}")
