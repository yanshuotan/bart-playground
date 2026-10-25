"""Compare speed-up variants with the baseline short chains on mixing and prediction.

Usage: python hpc_opt/compare_variants.py DATASET OUT_DIR baseline=/path variant=/path ...
Each path is a store root holding <tag>/preds etc. (the baseline is share_pt).

Per store, method and run (post-burn draws, 4 chains):
  - pointwise rank R-hat over the 100 test points (median, q90);
  - relative RMSE (/ naive train-mean RMSE) and relative CRPS (/ climatology),
    computed as in diagnosis/paper/section6_methods.py;
  - pooled R-hat against the baseline: the variant's chains and the baseline's
    chains together as one 8-chain sample. If both sample the same posterior
    this matches the within-store values; a systematic shift would inflate it;
  - common PCA: PCs of all stores' pooled draws of that run, with the mean and
    sd of each store's PC1/PC2 scores, plus a figure for run 0;
  - J, mean swap acceptance and wall time per chain where recorded.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnosis"))
sys.path.insert(0, str(ROOT / "diagnosis" / "paper"))

from core import load_with_shape, pointwise_rhat  # noqa: E402
from section6_methods import PREDICTIVE_NOISE_SEED, predictive_baselines, predictive_metrics  # noqa: E402

TAGS = {"concrete": "fixed100_Concrete", "seoul_bike": "fixed100_SeoulBike", "abalone": "fixed100_Abalone",
        "airfoil": "fixed100_Airfoil", "calhousing": "fixed100_CalHousing_subsample5000", "ccpp": "fixed100_CCPP"}
METHODS = ("default_pt", "mtmh_pt")
BURN = 3000


def _file(store, tag, folder, run, suffix):
    name = f"{tag}__run{run:03d}__{suffix}__preds.csv" if folder == "preds" else f"{tag}__run{run:03d}__{suffix}.csv"
    return store / tag / folder / name


def load_run(store, tag, run, method):
    preds = _file(store, tag, "preds", run, method)
    if not preds.exists():
        return None
    draws = load_with_shape(preds).transpose(0, 2, 1)[:, BURN:, :]
    sig = load_with_shape(_file(store, tag, "sigmas", run, f"{method}__sigmas"))
    sig = np.asarray(sig, float).reshape(sig.shape[0], -1)[:, BURN:]
    rates = _file(store, tag, "swap_accept_rates", run, f"{method}__swap_accept_rates")
    rates = load_with_shape(rates) if rates.exists() else None
    wall = None
    wpath = store / tag / "wall_times" / f"{tag}__run{run:03d}__wall_times.csv"
    if wpath.exists():
        frame = pd.read_csv(wpath)
        wall = frame.loc[frame["method"] == method, "wall_s"].to_numpy()
    return {"draws": draws, "sigmas": sig, "rates": rates, "wall": wall}


def main():
    dataset, out = sys.argv[1], Path(sys.argv[2])
    stores = dict(arg.split("=", 1) for arg in sys.argv[3:])
    stores = {k: Path(v) for k, v in stores.items()}
    out.mkdir(parents=True, exist_ok=True)
    tag = TAGS[dataset]

    import experiment_fixed100 as exp
    from fixed100_support import load_dataset
    X, y = load_dataset(dataset)
    splits = exp.make_fixed100_splits(X, y, n_runs=5)

    rows, pca_rows, figure_data = [], [], {}
    for run in range(5):
        y_train = np.asarray(splits[run]["y_train"], float)
        truth = np.asarray(splits[run]["y_test_fixed"], float)
        base_scores = predictive_baselines(y_train, truth)
        y_scale = float(y_train.max() - y_train.min())
        for method in METHODS:
            loaded = {name: load_run(path, tag, run, method) for name, path in stores.items()}
            loaded = {k: v for k, v in loaded.items() if v is not None}
            if not loaded:
                continue
            baseline = loaded.get("baseline")
            for name, data in loaded.items():
                rhat = pointwise_rhat(data["draws"])
                rmse, crps = predictive_metrics(data["draws"], data["sigmas"], truth, y_scale, PREDICTIVE_NOISE_SEED)
                row = {"store": name, "method": method, "run": run,
                       "rhat_median": float(np.nanmedian(rhat)), "rhat_q90": float(np.nanquantile(rhat, 0.9)),
                       "rel_rmse": rmse / base_scores["naive_rmse"],
                       "rel_crps": crps / base_scores["climatology_crps"]}
                if baseline is not None and name != "baseline":
                    pooled = pointwise_rhat(np.concatenate([baseline["draws"], data["draws"]], axis=0))
                    row["pooled_rhat_median"] = float(np.nanmedian(pooled))
                    row["pooled_rhat_q90"] = float(np.nanquantile(pooled, 0.9))
                    # |difference of posterior means| over its standard error, with each
                    # store's error estimated from the spread of its chain means.
                    mb, mv = baseline["draws"].mean(axis=(0, 1)), data["draws"].mean(axis=(0, 1))
                    se = np.sqrt(baseline["draws"].mean(axis=1).var(axis=0, ddof=1) / baseline["draws"].shape[0]
                                 + data["draws"].mean(axis=1).var(axis=0, ddof=1) / data["draws"].shape[0])
                    row["mean_shift_z_median"] = float(np.median(np.abs(mv - mb) / se))
                if data["rates"] is not None:
                    rates = np.asarray(data["rates"], float)
                    row["J"] = rates.shape[-1] + 1
                    row["swap_rate_mean"] = float(rates.mean())
                    row["swap_rate_min"] = float(rates.mean(axis=0).min())
                if data["wall"] is not None and len(data["wall"]):
                    row["wall_s_mean"] = float(np.mean(data["wall"]))
                rows.append(row)

            # Common PCA over every store's draws of this run.
            names = list(loaded)
            pooled = np.concatenate([loaded[n]["draws"].reshape(-1, loaded[n]["draws"].shape[-1]) for n in names])
            centre = pooled.mean(axis=0)
            _, _, vt = np.linalg.svd(pooled - centre, full_matrices=False)
            for n in names:
                scores = (loaded[n]["draws"] - centre) @ vt[:2].T  # (chain, draw, 2)
                for chain in range(scores.shape[0]):
                    pca_rows.append({"store": n, "method": method, "run": run, "chain": chain,
                                     "pc1_mean": float(scores[chain, :, 0].mean()), "pc1_sd": float(scores[chain, :, 0].std()),
                                     "pc2_mean": float(scores[chain, :, 1].mean()), "pc2_sd": float(scores[chain, :, 1].std())})
                if run == 0:
                    figure_data[(method, n)] = scores[:, ::20, :]

    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"{dataset}_variant_runs.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(out / f"{dataset}_variant_pca.csv", index=False)
    summary = frame.groupby(["method", "store"]).agg(["mean", "std"]).drop(columns="run")
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.dropna(axis=1, how="all")
    summary.to_csv(out / f"{dataset}_variant_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summary.round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(stores)
    fig, axes = plt.subplots(len(METHODS), len(names), figsize=(3.2 * len(names), 3.0 * len(METHODS)),
                             sharex="row", sharey="row", squeeze=False)
    for i, method in enumerate(METHODS):
        for j, n in enumerate(names):
            ax = axes[i, j]
            scores = figure_data.get((method, n))
            if scores is not None:
                for chain in range(scores.shape[0]):
                    ax.scatter(scores[chain, :, 0], scores[chain, :, 1], s=2, alpha=0.4, label=f"chain {chain}")
            ax.set_title(f"{method} / {n}", fontsize=9)
    axes[0, 0].legend(markerscale=4, fontsize=7)
    fig.suptitle(f"{dataset} run 0: common PCA of post-burn predictions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{dataset}_variant_pca_run0.png", dpi=130)


if __name__ == "__main__":
    main()
