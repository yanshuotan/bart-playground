import sys, glob, os, pickle, numpy as np, arviz as az, warnings
warnings.filterwarnings("ignore")
S = sys.argv[1]; OUT = S + "/mixexp"
meta = pickle.load(open(OUT + "/meta.pkl", "rb"))
rows = []
for cfg in sorted({k[0] for k in meta}):
    acc = []
    for r in sorted({k[1] for k in meta if k[0] == cfg}):
        m = meta[(cfg, r)]; dirs = [f"{OUT}/{cfg}/run{r:03d}_chain{c:03d}" for c in range(4)]
        if not all(os.path.exists(d + "/vars_histogram_total.csv") for d in dirs): continue
        P = np.stack([np.loadtxt(d + "/preds_rows.csv", delimiter=",").T for d in dirs])[:, :, 3000:]   # chains,100,draws
        Sg = np.stack([np.loadtxt(d + "/sigmas_rows.csv", delimiter=",").ravel() for d in dirs])
        sig = np.sqrt(Sg[:, 3000:].mean()) * m["range_ytr"]
        nu = sig ** 2 / m["var_ytr"]
        rh = np.median([az.rhat(P[:, i, :]) for i in range(100)])
        cm = P.mean(2); yt = np.array(m["y_test"])
        acc.append(dict(n=m["n"], nu=nu, rh=rh, B=np.median(cm.std(0, ddof=1)) / sig, W=np.median(P.std(2).mean(0)) / sig,
                        drift=np.sqrt(Sg[:, 9000:].mean() / Sg[:, 5000:6000].mean()),
                        err=np.sqrt(np.mean((P.mean((0, 2)) - yt) ** 2)) / sig))
    if not acc: continue
    a = {k: np.mean([x[k] for x in acc]) for k in acc[0]}; a["cfg"] = cfg; a["runs"] = len(acc); rows.append(a)
pickle.dump(rows, open(S + "/mixana_rows.pkl", "wb"))
print(f"{'config':28s} runs {'n':>5s} {'1-R2':>6s} {'n/nu':>8s} {'Rhat':>6s} {'B/sig':>6s} {'W/sig':>6s} {'sig drift':>9s} {'err/sig':>7s}")
for a in rows:
    print(f"{a['cfg']:28s} {a['runs']:4d} {a['n']:5.0f} {a['nu']:6.3f} {a['n']/a['nu']:8.0f} {a['rh']:6.3f} {a['B']:6.3f} {a['W']:6.3f} {a['drift']:9.4f} {a['err']:7.3f}")
