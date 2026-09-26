#!/usr/bin/env python
"""Mixing diagnostics for the stored long (`default_long`) chains.

Section 3 of the paper diagnoses the short Default chains; this script asks the
prior question about the references those diagnostics lean on: have the long
chains themselves mixed, and are their stored draws enough?

Per dataset and run it reports, on the post-burn long draws:

    (a) the worst between/within direction and the four chains projected on it
    (b) cross-chain pointwise rank-split R-hat in rolling windows
    (c) within-chain segment rank-split R-hat (four adjacent segments as
        dependent pseudo-chains), i.e. local stability inside one chain
    (d) the four chains on their own PCA axes, with centroids
    (e) prefix curves: R-hat and bulk ESS recomputed on the first x% of the
        draws, which is what says whether more draws would still move them
    (f) autocorrelation of the worst-direction projection per chain

Conventions follow `diagnosis/paper/diagnosis.py`: windows and segments count
*stored* draws, so a window of 1,000 is 1,000 × long_store_every original
iterations (100 for Abalone, CalHousing, CCPP and p100; 1,000 for Concrete,
Friedman and SeoulBike).

This file is self-contained and imports nothing from the other diagnosis
scripts. Run from the repository root:

    .venv/Scripts/python diagnosis/longchain/long_chain_mixing.py
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.linalg import eigh
from sklearn.decomposition import PCA

CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
# A stored long draw is long_store_every original iterations; read from metadata.
RHAT_THRESHOLD = 1.01
ESS_PER_CHAIN_TARGET = 100.0  # 400 total for four chains, the usual rule of thumb


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_with_shape(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if "original_shape=" not in header:
        raise ValueError(f"Missing original_shape header: {path}")
    shape = ast.literal_eval(header.split("original_shape=")[-1])
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)


def long_predictions(store_root: Path, dataset: str, run: int, burn: int) -> np.ndarray:
    """(chain, draw, test_point) post-burn long-chain predictions."""
    tag = dataset_tag(store_root, dataset)
    path = store_root / dataset / "preds" / f"{tag}__run{run:03d}__default_long__preds.csv"
    draws = load_with_shape(path).transpose(0, 2, 1)
    return draws[:, burn:, :]


def dataset_tag(store_root: Path, dataset: str) -> str:
    """File-name tag of a store directory (p100 stores files as ...SparseDir)."""
    metadata = sorted((store_root / dataset / "metadata").glob("*__dataset_metadata.csv"))
    if not metadata:
        raise FileNotFoundError(f"No dataset metadata in {store_root / dataset}")
    return metadata[0].name.split("__dataset_metadata")[0]


def long_runs(store_root: Path, dataset: str) -> list[int]:
    tag = dataset_tag(store_root, dataset)
    runs = []
    for path in (store_root / dataset / "rmses").glob(f"{tag}__run*__default_long__rmses.csv"):
        runs.append(int(re.search(r"__run(\d+)__", path.name).group(1)))
    return sorted(runs)


def store_every(store_root: Path, dataset: str, run: int) -> int:
    tag = dataset_tag(store_root, dataset)
    path = store_root / dataset / "metadata" / f"{tag}__run{run:03d}__default_long_metadata.csv"
    frame = pd.read_csv(path)
    return int(frame.loc[frame["key"] == "long_store_every", "value"].iloc[0])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def pointwise_rhat(draws: np.ndarray) -> np.ndarray:
    data = xr.DataArray(draws, dims=("chain", "draw", "test_point"), name="prediction")
    return np.asarray(az.rhat(data, method="rank")["prediction"].values)


def pointwise_ess(draws: np.ndarray, method: str = "bulk") -> np.ndarray:
    data = xr.DataArray(draws, dims=("chain", "draw", "test_point"), name="prediction")
    return np.asarray(az.ess(data, method=method)["prediction"].values)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.nanmedian(values)),
        "q25": float(np.nanquantile(values, 0.25)),
        "q75": float(np.nanquantile(values, 0.75)),
        "q90": float(np.nanquantile(values, 0.90)),
        "maximum": float(np.nanmax(values)),
        "fraction_above_1_01": float(np.nanmean(values > RHAT_THRESHOLD)),
    }


def rolling_cross_rhat(draws: np.ndarray, window: int, step: int) -> pd.DataFrame:
    rows = []
    for start in range(0, draws.shape[1] - window + 1, step):
        rows.append({"window_start": start, "window_end": start + window,
                     **summarize(pointwise_rhat(draws[:, start:start + window]))})
    return pd.DataFrame(rows)


def rolling_within_rhat(draws: np.ndarray, segment_length: int, n_segments: int, step: int) -> pd.DataFrame:
    rows = []
    block_length = segment_length * n_segments
    for start in range(0, draws.shape[1] - block_length + 1, step):
        block = draws[:, start:start + block_length]
        segments = block.reshape(draws.shape[0], n_segments, segment_length, draws.shape[2])
        pseudo = segments.transpose(1, 2, 0, 3)
        data = xr.DataArray(pseudo, dims=("chain", "draw", "original_chain", "test_point"),
                            name="prediction")
        values = np.asarray(az.rhat(data, method="rank")["prediction"].values)
        for chain_id, chain_values in enumerate(values):
            rows.append({"original_chain": chain_id, "window_start": start,
                         "window_end": start + block_length, **summarize(chain_values)})
    return pd.DataFrame(rows)


def prefix_convergence(draws: np.ndarray, fractions: tuple[float, ...]) -> pd.DataFrame:
    """R-hat and ESS on growing prefixes: does adding draws still change them?"""
    rows = []
    total = draws.shape[1]
    for fraction in fractions:
        n = max(int(round(fraction * total)), 20)
        if n > total:
            continue
        prefix = draws[:, :n]
        rhat = pointwise_rhat(prefix)
        bulk = pointwise_ess(prefix, "bulk")
        tail = pointwise_ess(prefix, "tail")
        rows.append({
            "fraction": fraction,
            "draws_per_chain": n,
            "rhat_median": float(np.nanmedian(rhat)),
            "rhat_q90": float(np.nanquantile(rhat, 0.90)),
            "rhat_max": float(np.nanmax(rhat)),
            "fraction_above_1_01": float(np.nanmean(rhat > RHAT_THRESHOLD)),
            "ess_bulk_min": float(np.nanmin(bulk)),
            "ess_bulk_median": float(np.nanmedian(bulk)),
            "ess_tail_min": float(np.nanmin(tail)),
            "ess_bulk_per_draw": float(np.nanmedian(bulk) / (n * draws.shape[0])),
        })
    return pd.DataFrame(rows)


def worst_direction(draws: np.ndarray, ridge_fraction: float) -> dict[str, object]:
    m, n, p = draws.shape
    means = draws.mean(axis=1)
    grand = means.mean(axis=0)
    within = np.zeros((p, p), dtype=float)
    for chain in draws:
        centered = chain - chain.mean(axis=0)
        within += centered.T @ centered / (n - 1)
    within /= m
    offsets = means - grand
    between = n * offsets.T @ offsets / (m - 1)
    ridge = max(ridge_fraction * np.trace(within) / p, ridge_fraction)
    eigenvalues, eigenvectors = eigh(between / n, within + ridge * np.eye(p))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    direction = eigenvectors[:, order[0]]
    direction /= np.linalg.norm(direction)
    projected = np.einsum("mnp,p->mn", draws, direction)
    data = xr.DataArray(projected, dims=("chain", "draw"), name="projection")
    return {
        "lambda_max": float(eigenvalues[0]),
        "eigen_gap": float(eigenvalues[0] - eigenvalues[1]),
        "projected_rhat": float(az.rhat(data, method="rank")["projection"].values),
        "projected_ess_bulk": float(az.ess(data, method="bulk")["projection"].values),
        "projected": projected,
    }


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - series.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return np.zeros(max_lag + 1)
    full = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    return full[: max_lag + 1] / denominator


def mean_pairwise_centroid_distance(draws: np.ndarray) -> float:
    centers = draws.mean(axis=1)
    distances = [np.linalg.norm(centers[i] - centers[j])
                 for i in range(len(centers)) for j in range(i + 1, len(centers))]
    return float(np.mean(distances))


def between_within_ratio(draws: np.ndarray) -> float:
    """Mean squared distance between chain centroids over mean within-chain squared radius."""
    centers = draws.mean(axis=1)
    between = np.mean([np.sum((centers[i] - centers[j]) ** 2)
                       for i in range(len(centers)) for j in range(i + 1, len(centers))])
    within = np.mean([np.mean(np.sum((chain - chain.mean(axis=0)) ** 2, axis=1)) for chain in draws])
    return float(between / within) if within > 0 else float("nan")


def spaced_indices(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_run(draws, worst, cross, within, prefix, burn, plot_draws, max_lag, title) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.2), layout="constrained")
    trace_ax, cross_ax, within_ax, pca_ax, prefix_ax, acf_ax = axes.flat

    projected = np.asarray(worst["projected"])
    iterations = np.arange(burn, burn + projected.shape[1])
    for chain_id in range(projected.shape[0]):
        trace_ax.plot(iterations, projected[chain_id], color=CHAIN_COLORS[chain_id],
                      linewidth=0.5, alpha=0.8, label=f"chain {chain_id + 1}")
    trace_ax.set(title=f"(a) Worst between/within direction (R-hat {worst['projected_rhat']:.4f})",
                 xlabel="stored draw", ylabel="projected prediction")
    trace_ax.legend(ncol=2, fontsize=8)

    x = cross["window_end"].to_numpy()
    cross_ax.plot(x, cross["median"], color="#245f9e", linewidth=1.8, label="median")
    cross_ax.fill_between(x, cross["q25"], cross["q75"], color="#245f9e", alpha=0.2, label="test-point IQR")
    cross_ax.plot(x, cross["maximum"], color="#245f9e", linewidth=0.9, linestyle=":", label="max")
    cross_ax.axhline(RHAT_THRESHOLD, color="#555555", linestyle="--", linewidth=1, label="1.01")
    cross_ax.set(title="(b) Cross-chain pointwise rank-split R-hat", xlabel="window end (stored draw)",
                 ylabel="R-hat")
    cross_ax.legend(fontsize=8)

    for chain_id, frame in within.groupby("original_chain"):
        frame = frame.sort_values("window_end")
        color = CHAIN_COLORS[int(chain_id)]
        within_ax.plot(frame["window_end"], frame["median"], color=color, linewidth=1.4,
                       label=f"chain {int(chain_id) + 1}")
        within_ax.fill_between(frame["window_end"], frame["q25"], frame["q75"], color=color, alpha=0.08)
    within_ax.axhline(RHAT_THRESHOLD, color="#555555", linestyle="--", linewidth=1, label="1.01")
    within_ax.set(title="(c) Within-chain segment rank-split R-hat", xlabel="window end (stored draw)",
                  ylabel="R-hat")
    within_ax.legend(ncol=3, fontsize=8)

    flat = draws.reshape(-1, draws.shape[-1])
    pca = PCA(n_components=2, random_state=0).fit(flat)
    coords = pca.transform(flat).reshape(draws.shape[0], draws.shape[1], 2)
    idx = spaced_indices(coords.shape[1], plot_draws)
    for chain_id in range(coords.shape[0]):
        color = CHAIN_COLORS[chain_id]
        pca_ax.scatter(coords[chain_id, idx, 0], coords[chain_id, idx, 1], s=8, alpha=0.15,
                       color=color, edgecolors="none", label=f"chain {chain_id + 1}")
        center = coords[chain_id].mean(axis=0)
        pca_ax.scatter(center[0], center[1], marker="X", s=90, color=color, edgecolor="black",
                       linewidth=0.6)
    pca_ax.set(title="(d) Long chains on their own PCA axes",
               xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
               ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    pca_ax.legend(ncol=2, fontsize=8)

    prefix_ax.plot(prefix["draws_per_chain"], prefix["rhat_max"], color="#b5391f", marker="o",
                   markersize=3.5, label="max R-hat")
    prefix_ax.plot(prefix["draws_per_chain"], prefix["rhat_median"], color="#245f9e", marker="o",
                   markersize=3.5, label="median R-hat")
    prefix_ax.axhline(RHAT_THRESHOLD, color="#555555", linestyle="--", linewidth=1, label="1.01")
    prefix_ax.set(title="(e) Prefix convergence", xlabel="draws per chain used", ylabel="R-hat")
    ess_ax = prefix_ax.twinx()
    ess_ax.plot(prefix["draws_per_chain"], prefix["ess_bulk_min"], color="#2ca02c", marker="s",
                markersize=3.5, linestyle="--", label="min bulk ESS")
    ess_ax.set_ylabel("bulk ESS (min over test points)")
    handles = prefix_ax.get_legend_handles_labels()[0] + ess_ax.get_legend_handles_labels()[0]
    labels = prefix_ax.get_legend_handles_labels()[1] + ess_ax.get_legend_handles_labels()[1]
    prefix_ax.legend(handles, labels, fontsize=8, loc="center right")

    lags = np.arange(max_lag + 1)
    for chain_id in range(projected.shape[0]):
        acf_ax.plot(lags, autocorrelation(projected[chain_id], max_lag), color=CHAIN_COLORS[chain_id],
                    linewidth=1.2, label=f"chain {chain_id + 1}")
    acf_ax.axhline(0.0, color="#555555", linewidth=0.8)
    acf_ax.axhline(0.05, color="#999999", linestyle=":", linewidth=0.8)
    acf_ax.set(title="(f) Autocorrelation of the worst-direction projection",
               xlabel="lag (stored draws)", ylabel="ACF")
    acf_ax.legend(ncol=2, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=15)
    return fig


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------

def analyse_run(args, dataset: str, run: int) -> dict[str, object]:
    stride = store_every(args.store_root, dataset, run)
    draws = long_predictions(args.store_root, dataset, run, args.long_burn)
    n_chains, n_draws, n_points = draws.shape

    worst = worst_direction(draws, args.ridge_fraction)
    cross = rolling_cross_rhat(draws, args.window, args.step)
    within = rolling_within_rhat(draws, args.segment_length, args.n_segments, args.step)
    prefix = prefix_convergence(draws, tuple(args.prefix_fractions))

    rhat = pointwise_rhat(draws)
    bulk = pointwise_ess(draws, "bulk")
    tail = pointwise_ess(draws, "tail")
    overall = summarize(rhat)

    fig = plot_run(draws, worst, cross, within, prefix, args.long_burn, args.plot_draws,
                   args.max_lag, f"{dataset} run {run:03d} — long chains "
                                 f"({n_draws} stored draws/chain, stride {stride})")
    figure_path = args.figure_dir / f"{dataset}_run{run:03d}_long_mixing.png"
    fig.savefig(figure_path, dpi=args.dpi)
    plt.close(fig)

    row = {
        "dataset": dataset,
        "run": run,
        "store_every": stride,
        "chains": n_chains,
        "stored_draws_per_chain": n_draws,
        "iterations_per_chain": n_draws * stride,
        "rhat_median": overall["median"],
        "rhat_q90": overall["q90"],
        "rhat_max": overall["maximum"],
        "fraction_above_1_01": overall["fraction_above_1_01"],
        "ess_bulk_min": float(np.nanmin(bulk)),
        "ess_bulk_median": float(np.nanmedian(bulk)),
        "ess_tail_min": float(np.nanmin(tail)),
        "ess_bulk_min_per_chain": float(np.nanmin(bulk)) / n_chains,
        "worst_lambda_max": worst["lambda_max"],
        "worst_projected_rhat": worst["projected_rhat"],
        "worst_projected_ess_bulk": worst["projected_ess_bulk"],
        "cross_rhat_median": float(cross["median"].median()),
        "cross_rhat_max": float(cross["maximum"].max()),
        "within_rhat_median": float(within["median"].median()),
        "within_rhat_max": float(within["maximum"].max()),
        "centroid_distance": mean_pairwise_centroid_distance(draws),
        "between_within_ratio": between_within_ratio(draws),
        "prefix_rhat_max_last": float(prefix["rhat_max"].iloc[-1]),
        "prefix_rhat_max_half": float(prefix.loc[prefix["fraction"] <= 0.5, "rhat_max"].iloc[-1]),
    }
    row["mixed"] = bool(row["rhat_max"] < RHAT_THRESHOLD
                        and row["worst_projected_rhat"] < RHAT_THRESHOLD
                        and row["ess_bulk_min_per_chain"] >= ESS_PER_CHAIN_TARGET)
    return {
        "row": row,
        "cross": cross.assign(dataset=dataset, run=run),
        "within": within.assign(dataset=dataset, run=run),
        "prefix": prefix.assign(dataset=dataset, run=run),
        "figure": figure_path.name,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_summary(path: Path, args, runs: pd.DataFrame) -> None:
    lines = [
        "# Long-chain mixing diagnosis",
        "",
        "Are the stored `default_long` chains themselves mixed, and are their draws enough?",
        "The short-chain diagnostics in `diagnosis/paper` use them as the reference, so this",
        "asks the prior question. Windows, segments and burn-in all count *stored* draws;",
        "multiply by `store_every` for original iterations.",
        "",
        "## Settings",
        "",
    ]
    lines += [f"- `{k}`: {v}" for k, v in {
        "long_burn": args.long_burn, "window": args.window, "step": args.step,
        "segment_length": args.segment_length, "n_segments": args.n_segments,
        "prefix_fractions": list(args.prefix_fractions), "ridge_fraction": args.ridge_fraction,
        "rhat_threshold": RHAT_THRESHOLD, "ess_per_chain_target": ESS_PER_CHAIN_TARGET,
    }.items()]
    lines += [
        "",
        "`mixed` is max pointwise R-hat < 1.01 **and** worst-direction R-hat < 1.01 **and**",
        f"min bulk ESS ≥ {ESS_PER_CHAIN_TARGET:g} per chain. It is a screening rule, not a proof.",
        "",
        "## Per run",
        "",
    ]
    view = runs[["dataset", "run", "stored_draws_per_chain", "iterations_per_chain", "rhat_max",
                 "worst_projected_rhat", "ess_bulk_min", "ess_tail_min", "between_within_ratio",
                 "mixed"]].copy()
    for column in ("rhat_max", "worst_projected_rhat", "between_within_ratio"):
        view[column] = view[column].map(lambda v: f"{v:.4f}")
    for column in ("ess_bulk_min", "ess_tail_min"):
        view[column] = view[column].map(lambda v: f"{v:.0f}")
    lines.append(markdown_table(view))

    lines += ["", "## Per dataset (mean over runs)", ""]
    grouped = runs.groupby("dataset", sort=False).agg(
        runs=("run", "count"),
        stored_draws=("stored_draws_per_chain", "max"),
        iterations=("iterations_per_chain", "max"),
        rhat_max=("rhat_max", "max"),
        worst_rhat_max=("worst_projected_rhat", "max"),
        ess_bulk_min=("ess_bulk_min", "min"),
        mixed=("mixed", "all"),
    ).reset_index()
    for column in ("rhat_max", "worst_rhat_max"):
        grouped[column] = grouped[column].map(lambda v: f"{v:.4f}")
    grouped["ess_bulk_min"] = grouped["ess_bulk_min"].map(lambda v: f"{v:.0f}")
    lines.append(markdown_table(grouped))
    lines += ["", "Figures: one per run in `figures/`, panels (a)-(f) as described in the script docstring.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    diagnosis_root = script.parent.parent
    parser = argparse.ArgumentParser(description="Mixing diagnostics for the stored long chains.")
    parser.add_argument("--store-root", type=Path, default=diagnosis_root / "store")
    parser.add_argument("--out-dir", type=Path, default=script.parent)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Default: every store directory that has default_long results.")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="Default: every run with a long chain in each dataset.")
    parser.add_argument("--long-burn", type=int, default=30, help="Stored draws dropped (paper uses 30).")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=1000)
    parser.add_argument("--n-segments", type=int, default=4)
    parser.add_argument("--prefix-fractions", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--max-lag", type=int, default=100)
    parser.add_argument("--plot-draws", type=int, default=1500)
    parser.add_argument("--ridge-fraction", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--jobs", type=int, default=4, help="Runs analysed in parallel.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.store_root = args.store_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.figure_dir = args.out_dir / "figures"
    args.table_dir = args.out_dir / "tables"
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or sorted(
        d.name for d in args.store_root.iterdir()
        if d.is_dir() and d.name.startswith("fixed100_") and long_runs(args.store_root, d.name)
    )
    jobs = []
    for dataset in datasets:
        available = long_runs(args.store_root, dataset)
        for run in (args.runs if args.runs is not None else available):
            if run in available:
                jobs.append((dataset, run))
    print(f"[long-mixing] {len(jobs)} runs: " + ", ".join(f"{d}/{r:03d}" for d, r in jobs), flush=True)

    results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(analyse_run)(args, dataset, run) for dataset, run in jobs
    )

    runs = pd.DataFrame([r["row"] for r in results])
    runs.to_csv(args.table_dir / "long_run_metrics.csv", index=False)
    pd.concat([r["cross"] for r in results]).to_csv(args.table_dir / "long_cross_rhat.csv", index=False)
    pd.concat([r["within"] for r in results]).to_csv(args.table_dir / "long_within_rhat.csv", index=False)
    pd.concat([r["prefix"] for r in results]).to_csv(args.table_dir / "long_prefix_convergence.csv", index=False)
    write_summary(args.out_dir / "long_chain_mixing_summary.md", args, runs)

    print("\n" + (args.out_dir / "long_chain_mixing_summary.md").read_text(encoding="utf-8"))
    not_mixed = runs.loc[~runs["mixed"], ["dataset", "run", "rhat_max", "ess_bulk_min"]]
    if len(not_mixed):
        print("Runs that did not pass the screening rule:")
        print(not_mixed.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
