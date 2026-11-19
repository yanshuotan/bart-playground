"""Mixing figures for the SeoulBike run-0 long chains, drawn as in section3_long_budget.py."""
import os, sys, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, S + "/bp/diagnosis/paper")
from core import worst_directions
CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
B = os.environ.get("PAPER_STORE", "paper_store_20261028") + "/fixed100_SeoulBike/preds/fixed100_SeoulBike__run000__"
def rd(p):
    with open(p) as f: shape = tuple(int(s) for s in re.findall(r"\d+", f.readline()))
    return np.loadtxt(p, delimiter=",", comments="#").reshape(shape)
z = np.load(S + "/longpt/seoulbike_run0_mtmh_pt_summary.npz")
burn = 3000
rows = [("Default, $10^7$ iterations, stride 1,000", rd(B + "default_long__preds.csv").transpose(0, 2, 1)[:, burn:], 1000),
        ("MTMH+PT, $10^6$ iterations, stride 100", z["preds"][:, burn:].astype(float), 100)]
fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), layout="constrained")
P = "abcd"
for r, (label, draws, stride) in enumerate(rows):
    res = worst_directions(draws, n_directions=2); pr = res["projected"]; ev = res["eigenvalues"]; half = int(res["fit_draws"])
    tax, sax = axes[r]
    it = (np.arange(burn, burn + pr.shape[1]) + 1) * stride
    for c in range(4):
        tax.plot(it, pr[c, :, 0], color=CHAIN_COLORS[c], lw=0.5, alpha=0.8, label=f"chain {c + 1}")
    tax.axvline(it[half], color="0.35", ls=(0, (4, 2)), lw=1.0)
    tax.set(title=f"({P[2*r]}) {label}, worst direction", xlabel="iteration", ylabel="projected prediction")
    tax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if r == 0: tax.legend(ncol=4, fontsize=7.5, loc="upper center")
    co = pr[:, half:, :]; idx = np.linspace(0, co.shape[1] - 1, 1500, dtype=int)
    for c in range(4):
        sax.scatter(co[c, idx, 0], co[c, idx, 1], s=7, alpha=0.18, color=CHAIN_COLORS[c], edgecolors="none")
        sax.scatter(*co[c].mean(0), marker="X", s=90, color=CHAIN_COLORS[c], edgecolor="black", linewidth=0.6, zorder=3)
    sax.set(title=f"({P[2*r+1]}) {label}, held half", xlabel=f"direction 1 ($\\lambda_1$={ev[0]:.3g})",
            ylabel=f"direction 2 ($\\lambda_2$={ev[1]:.3g})")
    for a in (tax, sax): a.grid(alpha=0.2)
    print(label, "lambda", ev)
fig.savefig(OUT + "/long_pt_worst2.png", dpi=200); plt.close(fig)

# traces of the collapsed log posterior, sigma and total splits over the whole run
L = z["logpost"].sum(2); sg = z["sigma"]; H = z["splits"].sum(2); it = (np.arange(10000) + 1) * 100
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), layout="constrained")
for a, (v, lab) in zip(axes, [(L, "collapsed log posterior ($T=1$)"), (sg, r"$\sigma$ (response scale)"), (H, "total splits")]):
    for c in range(4):
        sm = np.convolve(v[c], np.ones(20) / 20, mode="valid")
        a.plot(it[:len(v[c])], v[c], color=CHAIN_COLORS[c], lw=0.3, alpha=0.25)
        a.plot(it[19:], sm, color=CHAIN_COLORS[c], lw=1.0, label=f"chain {c + 1}")
    a.axvline(burn * 100, color="0.35", ls=(0, (4, 2)), lw=1.0)
    a.set(xlabel="iteration", title=lab); a.grid(alpha=0.2); a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axes[0].set_ylim(np.quantile(L, 0.01), L.max() + 20)
axes[0].legend(ncol=2, fontsize=7.5, loc="lower right")
fig.savefig(OUT + "/long_pt_traces.png", dpi=200); plt.close(fig)
