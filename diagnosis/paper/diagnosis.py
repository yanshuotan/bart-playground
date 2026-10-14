#!/usr/bin/env python
"""Paper-facing Default-sampler diagnostics for the three fixed-100 datasets.

This file is intentionally self-contained. It does not import any of the
scripts in ``diagnosis/exploratory``.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import eigh


DATASETS = ("fixed100_Abalone", "fixed100_Concrete", "fixed100_Friedman")
RUNS = tuple(range(5))
CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def load_with_shape(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if "original_shape=" not in header:
        raise ValueError(f"Missing original_shape header: {path}")
    shape = ast.literal_eval(header.split("original_shape=")[-1])
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)


def dataset_tag(store_root: Path, dataset: str) -> str:
    """File-name prefix inside a store directory.

    Usually the directory name, but the sparse variants differ: files in
    fixed100_FriedmanSparseDir_p20 are named fixed100_FriedmanSparseDirP20K5__*.
    """
    preds = sorted((store_root / dataset / "preds").glob("*__preds.csv"))
    if not preds:
        raise FileNotFoundError(f"No predictions in {store_root / dataset / 'preds'}")
    return preds[0].name.split("__run")[0]


def prediction_path(store_root: Path, dataset: str, run: int, method: str) -> Path:
    tag = dataset_tag(store_root, dataset)
    return store_root / dataset / "preds" / f"{tag}__run{run:03d}__{method}__preds.csv"


def predictions(store_root: Path, dataset: str, run: int, method: str, burn: int = 0) -> np.ndarray:
    raw = load_with_shape(prediction_path(store_root, dataset, run, method))
    draws = raw.transpose(0, 2, 1)
    return draws[:, burn:, :] if burn else draws


def pointwise_rhat(draws: np.ndarray) -> np.ndarray:
    data = xr.DataArray(draws, dims=("chain", "draw", "test_point"), name="prediction")
    return np.asarray(az.rhat(data, method="rank")["prediction"].values)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.nanmedian(values)),
        "q25": float(np.nanquantile(values, 0.25)),
        "q75": float(np.nanquantile(values, 0.75)),
        "q90": float(np.nanquantile(values, 0.90)),
        "maximum": float(np.nanmax(values)),
    }


def rolling_cross_rhat(draws: np.ndarray, window: int, step: int) -> pd.DataFrame:
    rows = []
    for start in range(0, draws.shape[1] - window + 1, step):
        rows.append({"window_start": start, "window_end": start + window,
                     **summarize(pointwise_rhat(draws[:, start:start + window]))})
    return pd.DataFrame(rows)


def rolling_within_rhat(
    draws: np.ndarray,
    segment_length: int,
    n_segments: int,
    step: int,
) -> pd.DataFrame:
    rows = []
    block_length = segment_length * n_segments
    for start in range(0, draws.shape[1] - block_length + 1, step):
        block = draws[:, start:start + block_length]
        segments = block.reshape(draws.shape[0], n_segments, segment_length, draws.shape[2])
        pseudo = segments.transpose(1, 2, 0, 3)
        data = xr.DataArray(
            pseudo,
            dims=("chain", "draw", "original_chain", "test_point"),
            name="prediction",
        )
        values = np.asarray(az.rhat(data, method="rank")["prediction"].values)
        for chain_id, chain_values in enumerate(values):
            rows.append({
                "original_chain": chain_id,
                "window_start": start,
                "window_end": start + block_length,
                **summarize(chain_values),
            })
    return pd.DataFrame(rows)


def worst_direction(draws: np.ndarray, ridge_fraction: float) -> dict[str, np.ndarray | float]:
    m, n, p = draws.shape
    half = n // 2
    if half < 2 or n - half < 2:
        raise ValueError(f"Need at least four draws per chain to cross-fit; got {n}")
    fit = draws[:, :half, :]
    held = draws[:, half:, :]
    means = fit.mean(axis=1)
    grand = means.mean(axis=0)
    within = np.zeros((p, p), dtype=float)
    for chain in fit:
        centered = chain - chain.mean(axis=0)
        within += centered.T @ centered / (half - 1)
    within /= m
    offsets = means - grand
    ridge = max(ridge_fraction * np.trace(within) / p, ridge_fraction)
    eigenvalues, eigenvectors = eigh(
        offsets.T @ offsets / (m - 1), within + ridge * np.eye(p)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    directions = eigenvectors[:, order[:2]].copy()
    for column in range(directions.shape[1]):
        directions[:, column] /= np.linalg.norm(directions[:, column])
        anchor = int(np.argmax(np.abs(directions[:, column])))
        if directions[anchor, column] < 0:
            directions[:, column] *= -1
    projected_2d = np.einsum("mnp,pk->mnk", draws - grand[None, None, :], directions)
    scored_2d = np.einsum("mnp,pk->mnk", held - grand[None, None, :], directions)
    projected_rhats = []
    for column in range(scored_2d.shape[2]):
        data = xr.DataArray(scored_2d[:, :, column], dims=("chain", "draw"), name="projection")
        projected_rhats.append(float(az.rhat(data, method="rank")["projection"].values))
    return {
        "lambda_max": float(eigenvalues[0]),
        "lambda_second": float(eigenvalues[1]),
        "eigen_gap": float(eigenvalues[0] - eigenvalues[1]),
        "projected_rhat": projected_rhats[0],
        "second_projected_rhat": projected_rhats[1],
        "directions": directions,
        "fit_draws": half,
        "projected": projected_2d[:, :, 0],
        "projected_2d": projected_2d,
    }


def mean_pairwise_centroid_distance(draws: np.ndarray) -> float:
    centers = draws.mean(axis=1)
    distances = [
        np.linalg.norm(centers[left] - centers[right])
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
    ]
    return float(np.mean(distances))


def spaced_indices(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


def plot_diagnosis(
    short_all: np.ndarray,
    short_post_burn: np.ndarray,
    worst: dict[str, np.ndarray | float],
    cross: pd.DataFrame,
    within: pd.DataFrame,
    short_burn: int,
    plot_draws: int,
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), layout="constrained")
    trace_ax, direction_ax, cross_ax, within_ax = axes.flat
    projected = np.asarray(worst["projected"])
    iterations = np.arange(short_burn, short_all.shape[1])
    for chain_id in range(projected.shape[0]):
        trace_ax.plot(iterations, projected[chain_id], color=CHAIN_COLORS[chain_id],
                      linewidth=0.65, alpha=0.8, label=f"chain {chain_id + 1}")
    trace_ax.axvline(short_burn + int(worst["fit_draws"]), color="0.35",
                     linestyle=(0, (4, 2)), linewidth=1.0)
    trace_ax.set(title="(a) Maximum between/within direction", xlabel="iteration",
                 ylabel="projected prediction")
    trace_ax.legend(ncol=2, fontsize=8)

    x = cross["window_end"].to_numpy()
    cross_ax.plot(x, cross["median"], color="#245f9e", linewidth=1.8, label="median")
    cross_ax.fill_between(x, cross["q25"], cross["q75"], color="#245f9e", alpha=0.2,
                          label="test-point IQR")
    cross_ax.axhline(1.01, color="#555555", linestyle="--", linewidth=1, label="1.01")
    cross_ax.set(title="(c) Cross-chain pointwise rank-split R-hat", xlabel="window end", ylabel="R-hat")
    cross_ax.legend(fontsize=8)

    for chain_id, frame in within.groupby("original_chain"):
        frame = frame.sort_values("window_end")
        x = frame["window_end"].to_numpy()
        color = CHAIN_COLORS[int(chain_id)]
        within_ax.plot(x, frame["median"], color=color, linewidth=1.5,
                       label=f"chain {int(chain_id) + 1}")
        within_ax.fill_between(x, frame["q25"], frame["q75"], color=color, alpha=0.08)
    within_ax.axhline(1.01, color="#555555", linestyle="--", linewidth=1, label="1.01")
    within_ax.set(title="(d) Within-chain segment rank-split R-hat", xlabel="window end", ylabel="R-hat")
    within_ax.legend(ncol=3, fontsize=8)

    short_coords = np.asarray(worst["projected_2d"])
    panel_title = "(b) Top two between/within directions"
    xlabel = f"direction 1 ($\\lambda_1$={float(worst['lambda_max']):.3g})"
    ylabel = f"direction 2 ($\\lambda_2$={float(worst['lambda_second']):.3g})"
    short_idx = spaced_indices(short_coords.shape[1], plot_draws)
    for chain_id in range(short_coords.shape[0]):
        color = CHAIN_COLORS[chain_id]
        direction_ax.scatter(short_coords[chain_id, short_idx, 0], short_coords[chain_id, short_idx, 1],
                             s=10, alpha=0.18, color=color, edgecolors="none",
                             label=f"default chain {chain_id + 1}")
        center = short_coords[chain_id].mean(axis=0)
        direction_ax.scatter(center[0], center[1], marker="X", s=85, color=color,
                             edgecolor="black", linewidth=0.6)
    direction_ax.set(
        title=panel_title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    direction_ax.legend(ncol=2, fontsize=7.5)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    return fig


def aggregate_dataset_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in run_frame.columns if column not in {"dataset", "run"}]
    rows = []
    for dataset, frame in run_frame.groupby("dataset", sort=False):
        row: dict[str, float | str | int] = {"dataset": dataset, "n_runs": len(frame)}
        for metric in metrics:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_sd"] = frame[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(path: Path, settings: dict[str, int | float], dataset_summary: pd.DataFrame) -> None:
    lines = ["# Default-sampler diagnosis summary", "", "All values summarize five runs per dataset.", "",
             "## Settings", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in settings.items())
    lines.extend([
        "",
        "`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.",
        "",
    ])
    lines.extend(["", "## Dataset-level results", ""])
    columns = [
        "dataset", "worst_projected_rhat_mean", "cross_rhat_median_mean",
        "within_rhat_median_mean", "short_centroid_distance_mean",
        "long_rhat_median_mean",
    ]
    view = dataset_summary[columns].copy()
    for column in columns[1:]:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    lines.append(markdown_table(view))
    lines.extend(["", "The long chains are empirical references; their own R-hat summaries qualify comparisons with the production-length chains.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    diagnosis_root = script.parent.parent
    parser = argparse.ArgumentParser(description="Run paper-facing Default diagnostics.")
    parser.add_argument("--store-root", type=Path, default=diagnosis_root / "store")
    parser.add_argument("--table-dir", type=Path, default=script.parent / "tables")
    parser.add_argument("--figure-dir", type=Path, default=script.parent / "figures" / "diagnosis")
    parser.add_argument("--summary-path", type=Path, default=script.parent / "diagnosis_summary.md")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--runs", nargs="+", type=int, default=list(RUNS))
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=1000)
    parser.add_argument("--n-segments", type=int, default=4)
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=3000)
    parser.add_argument("--plot-draws", type=int, default=750)
    parser.add_argument("--ridge-fraction", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    summary_path = args.summary_path.resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    run_rows, cross_frames, within_frames = [], [], []
    for dataset in args.datasets:
        for run in args.runs:
            print(f"[diagnosis] {dataset} run {run:03d}", flush=True)
            short_all = predictions(args.store_root, dataset, run, "default", 0)
            short_post_burn = short_all[:, args.short_burn:, :]
            long_all = predictions(args.store_root, dataset, run, "default_long", 0)
            long_post_burn = long_all[:, args.long_burn:, :]

            worst = worst_direction(short_all[:, args.short_burn:, :], args.ridge_fraction)
            cross = rolling_cross_rhat(short_all, args.window, args.step)
            within = rolling_within_rhat(short_all, args.segment_length, args.n_segments, args.step)
            long_values = pointwise_rhat(long_post_burn)
            long_summary = summarize(long_values)

            cross_frames.append(cross.assign(dataset=dataset, run=run))
            within_frames.append(within.assign(dataset=dataset, run=run))
            run_rows.append({
                "dataset": dataset,
                "run": run,
                "worst_lambda_max": worst["lambda_max"],
                "worst_lambda_second": worst["lambda_second"],
                "worst_eigen_gap": worst["eigen_gap"],
                "worst_projected_rhat": worst["projected_rhat"],
                "second_projected_rhat": worst["second_projected_rhat"],
                "cross_rhat_median": cross["median"].median(),
                "cross_rhat_q90": cross["q90"].median(),
                "within_rhat_median": within["median"].median(),
                "within_rhat_q90": within["q90"].median(),
                "short_centroid_distance": mean_pairwise_centroid_distance(short_post_burn),
                "long_rhat_median": long_summary["median"],
                "long_rhat_q90": long_summary["q90"],
            })

            fig = plot_diagnosis(
                short_all, short_post_burn, worst, cross, within,
                args.short_burn, args.plot_draws,
                f"{dataset} | run {run:03d} | Default sampler diagnosis",
            )
            fig.savefig(figure_dir / f"{dataset}_run{run:03d}_default_diagnosis.png",
                        dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    run_frame = pd.DataFrame(run_rows)
    dataset_frame = aggregate_dataset_summary(run_frame)
    run_frame.to_csv(table_dir / "diagnosis_run_summary.csv", index=False)
    dataset_frame.to_csv(table_dir / "diagnosis_dataset_summary.csv", index=False)
    pd.concat(cross_frames, ignore_index=True).to_csv(table_dir / "diagnosis_cross_chain_rhat_windows.csv", index=False)
    pd.concat(within_frames, ignore_index=True).to_csv(table_dir / "diagnosis_within_chain_rhat_windows.csv", index=False)

    settings = {
        "runs_per_dataset": len(args.runs),
        "cross_chain_window": args.window,
        "rolling_step": args.step,
        "segment_length": args.segment_length,
        "segments_per_block": args.n_segments,
        "short_burn": args.short_burn,
        "long_burn": args.long_burn,
    }
    write_summary(summary_path, settings, dataset_frame)
    print(f"[diagnosis] tables: {table_dir}", flush=True)
    print(f"[diagnosis] figures: {figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
