#!/usr/bin/env python
"""Paper-facing Default-sampler diagnostics for the three fixed-100 datasets.

This file is intentionally self-contained. It does not import any of the
scripts in ``diagnosis/exploratory``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (dataset_tag, load_with_shape, markdown_table,
                  pointwise_rhat, summarize_rhat, worst_direction_summary)
import numpy as np
import pandas as pd


DATASETS = ("fixed100_Abalone", "fixed100_Concrete", "fixed100_Friedman")
RUNS = tuple(range(5))
CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def prediction_path(store_root: Path, dataset: str, run: int, method: str) -> Path:
    tag = dataset_tag(store_root, dataset)
    return store_root / dataset / "preds" / f"{tag}__run{run:03d}__{method}__preds.csv"


def predictions(store_root: Path, dataset: str, run: int, method: str, burn: int = 0) -> np.ndarray:
    raw = load_with_shape(prediction_path(store_root, dataset, run, method))
    draws = raw.transpose(0, 2, 1)
    return draws[:, burn:, :] if burn else draws


def spaced_indices(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


def plot_diagnosis(
    short_all: np.ndarray,
    short_post_burn: np.ndarray,
    worst: dict[str, np.ndarray | float],
    short_burn: int,
    plot_draws: int,
    title: str = "",
) -> plt.Figure:
    fig, axd = plt.subplot_mosaic([["a", "b"]], figsize=(13.2, 4.6), layout="constrained")
    trace_ax, direction_ax = axd["a"], axd["b"]
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

    # the fitted first half is left out of the scatter; the trace shows both halves
    short_coords = np.asarray(worst["projected_2d"])[:, int(worst["fit_draws"]):, :]
    panel_title = "(b) Top two between/within directions, held half"
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
    for ax in axd.values():
        ax.grid(alpha=0.2)
    if title:
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
        "dataset", "worst_lambda_max_mean", "cross_rhat_median_mean",
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

    run_rows = []
    for dataset in args.datasets:
        for run in args.runs:
            print(f"[diagnosis] {dataset} run {run:03d}", flush=True)
            short_all = predictions(args.store_root, dataset, run, "default", 0)
            short_post_burn = short_all[:, args.short_burn:, :]
            long_all = predictions(args.store_root, dataset, run, "default_long", 0)
            long_post_burn = long_all[:, args.long_burn:, :]

            worst = worst_direction_summary(short_all[:, args.short_burn:, :], args.ridge_fraction)
            short_summary = summarize_rhat(pointwise_rhat(short_post_burn))
            long_summary = summarize_rhat(pointwise_rhat(long_post_burn))

            run_rows.append({
                "dataset": dataset,
                "run": run,
                "worst_lambda_max": worst["lambda_max"],
                "worst_lambda_second": worst["lambda_second"],
                "worst_eigen_gap": worst["eigen_gap"],
                "cross_rhat_median": short_summary["median"],
                "cross_rhat_q90": short_summary["q90"],
                "long_rhat_median": long_summary["median"],
                "long_rhat_q90": long_summary["q90"],
            })

            # No title: the paper caption names the dataset and run.
            fig = plot_diagnosis(short_all, short_post_burn, worst,
                                 args.short_burn, args.plot_draws)
            fig.savefig(figure_dir / f"{dataset}_run{run:03d}_default_diagnosis.png",
                        dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    run_frame = pd.DataFrame(run_rows)
    dataset_frame = aggregate_dataset_summary(run_frame)
    run_frame.to_csv(table_dir / "diagnosis_run_summary.csv", index=False)
    dataset_frame.to_csv(table_dir / "diagnosis_dataset_summary.csv", index=False)

    settings = {
        "runs_per_dataset": len(args.runs),
        "short_burn": args.short_burn,
        "long_burn": args.long_burn,
    }
    write_summary(summary_path, settings, dataset_frame)
    print(f"[diagnosis] tables: {table_dir}", flush=True)
    print(f"[diagnosis] figures: {figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
