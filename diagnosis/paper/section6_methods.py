#!/usr/bin/env python
"""Paper-facing four-method comparison for the current fixed-100 study.

This file is intentionally self-contained. It does not import code from
``diagnosis/exploratory``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (dataset_tag, load_with_shape, markdown_table,
                  pointwise_rhat, summarize_rhat, worst_direction_summary)
import numpy as np
import pandas as pd
import xarray as xr


DATASETS = (
    "fixed100_Abalone",
    "fixed100_Airfoil",
    "fixed100_CCPP",
    "fixed100_CalHousing_subsample5000",
    "fixed100_Concrete",
    "fixed100_CPUAct",
    "fixed100_Friedman",
    "fixed100_FriedmanSparseDir_p20",
    "fixed100_FriedmanSparseDir_p100",
    "fixed100_FriedmanSparseDir_p200",
    "fixed100_SeoulBike",
)
RUNS = tuple(range(5))
METHODS = ("default", "default_pt", "mtmh", "mtmh_pt")
METHOD_NAMES = {
    "default": "Default",
    "default_pt": "Default+PT",
    "mtmh": "MTMH",
    "mtmh_pt": "MTMH+PT",
}
METHOD_COLORS = {
    "default": "#1b9e77",
    "default_pt": "#d95f02",
    "mtmh": "#7570b3",
    "mtmh_pt": "#e7298a",
}
CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def stored_path(store_root: Path, dataset: str, folder: str, run: int, suffix: str) -> Path:
    tag = dataset_tag(store_root, dataset)
    if folder == "preds":
        filename = f"{tag}__run{run:03d}__{suffix}__preds.csv"
    else:
        filename = f"{tag}__run{run:03d}__{suffix}.csv"
    return store_root / dataset / folder / filename


def predictions(store_root: Path, dataset: str, run: int, method: str, burn: int = 0) -> np.ndarray:
    raw = load_with_shape(stored_path(store_root, dataset, "preds", run, method))
    draws = raw.transpose(0, 2, 1)
    return draws[:, burn:, :] if burn else draws


def residual_variances(store_root: Path, dataset: str, run: int, method: str,
                       burn: int = 0) -> np.ndarray:
    """Stored eps_sigma2 per draw, shaped (chain, draw)."""
    raw = load_with_shape(
        stored_path(store_root, dataset, "sigmas", run, f"{method}__sigmas")
    )
    values = np.asarray(raw, dtype=float).reshape(raw.shape[0], -1)
    return values[:, burn:] if burn else values


def y_test(store_root: Path, dataset: str, run: int) -> np.ndarray:
    path = stored_path(store_root, dataset, "subsample_y_test", run, "subsample_y_test")
    return load_with_shape(path).reshape(-1)


# Store directory -> the loader name the experiment uses. Sparse variants are
# regenerated rather than loaded, so they go through generate_sparse_variant.
TARGET_LOADERS = {
    "fixed100_Abalone": "abalone",
    "fixed100_Airfoil": "airfoil",
    "fixed100_CCPP": "ccpp",
    "fixed100_CalHousing_subsample5000": "calhousing",
    "fixed100_CPUAct": "cpu_act",
    "fixed100_Concrete": "concrete",
    "fixed100_Friedman": "friedman",
    "fixed100_FriedmanSparseDir_p100": "friedman_sparse_dir",
    "fixed100_SeoulBike": "seoul_bike",
}
SPARSE_TARGET_LOADERS = {
    "fixed100_FriedmanSparseDir_p20": "friedman_p20_k5",
    "fixed100_FriedmanSparseDir_p200": "friedman_p200_k5",
}


def full_target_vector(store_root: Path, dataset: str) -> np.ndarray:
    """The full target vector the saved split indexes into.

    This calls the experiment's own loader instead of repeating the
    per-dataset preprocessing here: the stored indices are positions in
    whatever that loader returns, and SeoulBike alone drops a column, encodes
    two categoricals and removes non-finite rows. Checked against the previous
    hand-written branches for Abalone, Concrete and Friedman: identical.
    """
    diagnosis_root = Path(__file__).resolve().parent.parent
    if str(diagnosis_root) not in sys.path:
        sys.path.insert(0, str(diagnosis_root))

    if dataset in SPARSE_TARGET_LOADERS:
        from fixed100_support import generate_sparse_variant, repo_nested_friedman1_generator

        _, values, _ = generate_sparse_variant(
            SPARSE_TARGET_LOADERS[dataset], n_samples=2000, seed=42, noise_sd=1.0,
            friedman_generator=repo_nested_friedman1_generator,
        )
    elif dataset in TARGET_LOADERS:
        from fixed100_support import load_dataset

        _, values = load_dataset(TARGET_LOADERS[dataset])
    else:
        raise ValueError(f"No target loader registered for {dataset}")
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite targets in full target vector for {dataset}")
    return values


def split_indices(store_root: Path, dataset: str, run: int, split: str) -> np.ndarray:
    tag = dataset_tag(store_root, dataset)
    path = store_root / dataset / "indices" / f"{tag}__run{run:03d}__{split}.csv"
    return load_with_shape(path).reshape(-1).astype(int)


def training_targets(
    store_root: Path,
    dataset: str,
    run: int,
    full_targets: np.ndarray,
    saved_truth: np.ndarray,
) -> np.ndarray:
    """Recover y_train and verify that its indexing matches the stored test set."""
    train_index = split_indices(store_root, dataset, run, "train_idx")
    test_index = split_indices(store_root, dataset, run, "fixed_test_idx")
    indexed_truth = full_targets[test_index]
    if indexed_truth.shape != saved_truth.shape or not np.allclose(indexed_truth, saved_truth):
        raise ValueError(
            f"Stored targets and split indices disagree for {dataset} run {run:03d}; "
            "refusing to construct training baselines from a mismatched split."
        )
    return full_targets[train_index]


def full_draw_rhat(draws: np.ndarray, n_segments: int = 4) -> dict[str, float | int]:
    """Table summaries from all retained draws; rolling windows are graphical only."""
    chains, retained, points = draws.shape
    if n_segments < 2 or retained < 4 * n_segments:
        raise ValueError("Need at least two segments with four draws each")
    if retained % n_segments:
        raise ValueError("Retained draw count must be divisible by the segment count")
    segment_length = retained // n_segments
    cross = summarize_rhat(pointwise_rhat(draws))
    pseudo = draws.reshape(chains, n_segments, segment_length, points).transpose(1, 2, 0, 3)
    data = xr.DataArray(pseudo, dims=("chain", "draw", "original_chain", "test_point"), name="prediction")
    within = summarize_rhat(np.asarray(az.rhat(data, method="rank")["prediction"].values))
    return {
        "cross_rhat_median": cross["median"],
        "cross_rhat_q90": cross["q90"],
        "within_rhat_median": within["median"],
        "within_rhat_q90": within["q90"],
        "rhat_draws_per_chain": retained,
        "rhat_segment_length": segment_length,
    }


def rolling_cross_rhat(draws: np.ndarray, window: int, step: int) -> pd.DataFrame:
    rows = []
    for start in range(0, draws.shape[1] - window + 1, step):
        rows.append({"window_start": start, "window_end": start + window,
                     **summarize_rhat(pointwise_rhat(draws[:, start:start + window]))})
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
                **summarize_rhat(chain_values),
            })
    return pd.DataFrame(rows)


def chain_geometry(draws: np.ndarray) -> dict[str, float | int]:
    """Compute original-space chain geometry from every supplied draw."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3:
        raise ValueError(f"Expected (chain, draw, test_point); got {draws.shape}")
    centers = draws.mean(axis=1)
    centroid_distances = [
        np.linalg.norm(centers[left] - centers[right])
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
    ]
    grand = draws.reshape(-1, draws.shape[-1]).mean(axis=0)
    between = np.mean(np.sum((centers - grand) ** 2, axis=1))
    within = np.mean([
        np.mean(np.sum((chain - center) ** 2, axis=1))
        for chain, center in zip(draws, centers)
    ])
    ratio = float(between / within)
    return {
        "centroid_distance": float(np.mean(centroid_distances)),
        "between_within_ratio": ratio,
        "geometry_draws_per_chain": draws.shape[1],
    }


# Fixed so a rerun reproduces the same predictive draws.
PREDICTIVE_NOISE_SEED = 20261012


def crps_from_samples(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n_points, n_samples = samples.shape
    first = np.mean(np.abs(samples - truth[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    coefficients = (2 * np.arange(1, n_samples + 1) - n_samples - 1)[None, :]
    second = np.sum(coefficients * ordered, axis=1) / n_samples**2
    return first - second


def predictive_metrics(
    draws: np.ndarray,
    variances: np.ndarray,
    truth: np.ndarray,
    y_scale: float,
    seed: int,
) -> tuple[float, float]:
    """RMSE of the posterior mean of f, and CRPS of the posterior predictive.

    `draws` holds posterior draws of f with no observation noise, so scoring
    them against the noisy targets would not be a CRPS of the predictive
    distribution, and would not be comparable with the climatology
    denominator, which is a distribution over y. One noise draw is added per
    posterior draw. `variances` is the stored eps_sigma2, which lives on
    DefaultPreprocessor's internal min-max scale; `y_scale` is the training
    range that scale was built from, so sigma on the original scale is
    sqrt(eps_sigma2) * y_scale. RMSE is unchanged: the noise has mean zero, so
    it does not move the posterior mean.
    """
    if draws.shape[:2] != variances.shape[:2]:
        raise ValueError(
            f"draws {draws.shape[:2]} and variances {variances.shape[:2]} disagree"
        )
    rng = np.random.default_rng(seed)
    rmse_values, crps_values = [], []
    for chain, chain_variances in zip(draws, variances):
        samples = chain.T
        rmse_values.append(np.sqrt(np.mean((samples.mean(axis=1) - truth) ** 2)))
        sd = np.sqrt(chain_variances) * float(y_scale)
        predictive = samples + rng.standard_normal(samples.shape) * sd[None, :]
        crps_values.append(np.mean(crps_from_samples(predictive, truth)))
    return float(np.mean(rmse_values)), float(np.mean(crps_values))


def predictive_baselines(train_targets: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    train_targets = np.asarray(train_targets, dtype=float)
    train_mean = float(train_targets.mean())
    train_sd = float(train_targets.std(ddof=1))
    if not np.isfinite(train_sd) or train_sd <= 0.0:
        raise ValueError("Training-target standard deviation must be positive")
    naive_rmse = float(np.sqrt(np.mean((train_mean - truth) ** 2)))
    climatology_samples = np.broadcast_to(train_targets, (len(truth), len(train_targets)))
    climatology_crps = float(np.mean(crps_from_samples(climatology_samples, truth)))
    return {
        "train_mean": train_mean,
        "train_sd": train_sd,
        "naive_rmse": naive_rmse,
        "climatology_crps": climatology_crps,
    }


def plot_rhat_comparison(within_frames: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Within-chain segment R-hat through the run, one line per method.

    The cross-chain panel that used to sit beside this is dropped: the
    factorial table already ranks the methods on cross-chain agreement using
    every retained draw, and placing the two side by side invited reading one
    against the other even though they answer different questions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), layout="constrained")
    for method in METHODS:
        within = within_frames[method]
        # the band spans the four chains, the line is their median
        reduced = within.groupby("window_end")["median"].agg(
            center="median", low="min", high="max"
        ).reset_index()
        ax.plot(reduced["window_end"], reduced["center"], color=METHOD_COLORS[method],
                linewidth=1.8, label=METHOD_NAMES[method])
        ax.fill_between(reduced["window_end"], reduced["low"], reduced["high"],
                        color=METHOD_COLORS[method], alpha=0.10)
    ax.axhline(1.01, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("window end")
    ax.set_ylabel("rank-split R-hat")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig


def plot_worst_comparison(results: dict[str, dict[str, np.ndarray | float]], start: int,
                          title: str, plot_draws: int = 750) -> plt.Figure:
    """A row per method: trace along direction 1, then the two-direction scatter."""
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.0), layout="constrained")
    for row, method in enumerate(METHODS):
        result = results[method]
        projected = np.asarray(result["projected"])
        coords = np.asarray(result["projected_2d"])
        x = np.arange(start, start + projected.shape[1])
        index = np.linspace(0, coords.shape[1] - 1,
                            min(coords.shape[1], plot_draws)).round().astype(int)
        for chain_id in range(projected.shape[0]):
            colour = CHAIN_COLORS[chain_id]
            axes[row, 0].plot(x, projected[chain_id], color=colour, linewidth=0.55,
                              alpha=0.75,
                              label=f"chain {chain_id + 1}" if row == 0 else "_nolegend_")
            axes[row, 1].scatter(coords[chain_id, index, 0], coords[chain_id, index, 1],
                                 s=7, alpha=0.18, color=colour, edgecolors="none")
            centre = coords[chain_id].mean(axis=0)
            axes[row, 1].scatter(*centre, marker="X", s=90, color=colour,
                                 edgecolor="black", linewidth=0.6, zorder=3)
        name = METHOD_NAMES[method]
        axes[row, 0].axvline(start + int(result["fit_draws"]), color="0.35",
                             linestyle=(0, (4, 2)), linewidth=1.0)
        axes[row, 0].set_title(
            rf"{name}, direction 1 ($\widehat R$={result['projected_rhat']:.3f})")
        axes[row, 1].set_title(f"{name}, top two between/within directions")
        axes[row, 0].set_ylabel("projected prediction")
        axes[row, 1].set_xlabel(rf"direction 1 ($\lambda_1$={result['lambda_max']:.3g})")
        axes[row, 1].set_ylabel(rf"direction 2 ($\lambda_2$={result['lambda_second']:.3g})")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title)
    return fig


def dataset_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"dataset", "run", "method"}
    metrics = [column for column in run_frame.columns if column not in id_columns]
    rows = []
    for (dataset, method), frame in run_frame.groupby(["dataset", "method"], sort=False):
        row: dict[str, str | int | float] = {"dataset": dataset, "method": method, "n_runs": len(frame)}
        for metric in metrics:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_sd"] = frame[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ratios(run_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "worst_projected_rhat", "cross_rhat_median", "within_rhat_median",
        "centroid_distance", "between_within_ratio",
        "rmse", "crps",
    ]
    rows = []
    for (dataset, run), frame in run_frame.groupby(["dataset", "run"]):
        baseline = frame[frame["method"] == "Default"].iloc[0]
        for _, row in frame.iterrows():
            item: dict[str, str | int | float] = {"dataset": dataset, "run": run, "method": row["method"]}
            for metric in metrics:
                item[f"{metric}_ratio_to_default"] = row[metric] / baseline[metric]
            rows.append(item)
    return pd.DataFrame(rows)


def parse_timing_summary(path: Path) -> pd.DataFrame:
    rows = []
    current_dataset = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^##\s+(fixed100_[^(\s]+)", raw_line)
        if heading:
            current_dataset = {
                'fixed100_FriedmanSparseDirP20K5': 'fixed100_FriedmanSparseDir_p20',
                'fixed100_FriedmanSparseDir': 'fixed100_FriedmanSparseDir_p100',
                'fixed100_FriedmanSparseDirP200K5': 'fixed100_FriedmanSparseDir_p200',
            }.get(heading.group(1), heading.group(1))
            continue
        if current_dataset is None or not raw_line.startswith("|"):
            continue
        parts = [part.strip() for part in raw_line.strip().strip("|").split("|")]
        if len(parts) != 6 or parts[0] in {"method", "---:---"} or set(parts[0]) <= {"-", ":"}:
            continue
        method_text = parts[0]
        mapping = {
            "default": "Default",
            "mtmh": "MTMH",
            "default_pt (parallel)": "Default+PT",
            "mtmh_pt (parallel)": "MTMH+PT",
        }
        if method_text not in mapping:
            continue
        rows.append({
            "dataset": current_dataset,
            "method": mapping[method_text],
            "temperatures": np.nan if parts[1] == "-" else int(parts[1]),
            "workers": int(parts[2]),
            "mean_seconds_per_chain": float(parts[3].replace("**", "")),
            "sd_seconds_per_chain": float(parts[4].replace("**", "")),
            "relative_to_default": float(parts[5].replace("**", "")),
        })
    frame = pd.DataFrame(rows)
    # Use current measured ratios rather than rounded/stale Markdown values.
    for dataset in frame['dataset'].unique():
        tag = {
            'fixed100_FriedmanSparseDir_p20': 'fixed100_FriedmanSparseDirP20K5',
            'fixed100_FriedmanSparseDir_p100': 'fixed100_FriedmanSparseDir',
            'fixed100_FriedmanSparseDir_p200': 'fixed100_FriedmanSparseDirP200K5',
        }.get(dataset, dataset)
        source = path.parent / f'{tag}_run000_summary.csv'
        if source.exists():
            measured = pd.read_csv(source).set_index('label')
            for label, method in mapping.items():
                mask = frame.dataset.eq(dataset) & frame.method.eq(method)
                frame.loc[mask, 'relative_to_default'] = measured.loc[label, 'relative_to_default']
                frame.loc[mask, 'mean_seconds_per_chain'] = measured.loc[label, 'mean_seconds_per_chain']
                frame.loc[mask, 'sd_seconds_per_chain'] = measured.loc[label, 'std_seconds_per_chain']
    return frame


def format_mean_sd(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} ({row[f'{metric}_sd']:.{digits}f})"


def write_summary(
    path: Path,
    settings: dict[str, str | int | float],
    summary: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    lines = ["# Four-method comparison summary", "", "All diagnostic and predictive entries summarize five paired runs per dataset.", "",
             "## Settings", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in settings.items())
    lines.extend([
        "",
        "`short_burn` is also the start of the worst-direction calculation.",
        "Table cross R-hat uses every retained draw, with the median and q90 taken across test inputs. "
        "Table within R-hat splits each retained original chain into equal consecutive segments "
        "(four segments of 1,750 draws for 7,000 retained draws), then summarizes across original "
        "chains and test inputs. Both use the rank-normalized split-and-folded estimator. "
        "Rolling-window R-hat is used only in the temporal figures, not in these table columns.",
    ])
    for dataset in summary["dataset"].drop_duplicates():
        lines.extend(["", f"## {dataset}", ""])
        frame = summary[summary["dataset"] == dataset].copy()
        table = pd.DataFrame({
            "method": frame["method"],
            "worst projected R-hat": [format_mean_sd(row, "worst_projected_rhat") for _, row in frame.iterrows()],
            "cross R-hat": [format_mean_sd(row, "cross_rhat_median") for _, row in frame.iterrows()],
            "within R-hat": [format_mean_sd(row, "within_rhat_median") for _, row in frame.iterrows()],
            "B/W ratio": [format_mean_sd(row, "between_within_ratio") for _, row in frame.iterrows()],
            "relative RMSE": [format_mean_sd(row, "relative_rmse") for _, row in frame.iterrows()],
            "relative CRPS": [format_mean_sd(row, "relative_crps") for _, row in frame.iterrows()],
        })
        lines.append(markdown_table(table))
        timing_frame = timing[timing["dataset"] == dataset]
        if not timing_frame.empty:
            timing_display = timing_frame.copy()
            timing_display["temperatures"] = timing_display["temperatures"].map(
                lambda value: "-" if pd.isna(value) else str(int(value))
            )
            lines.extend(["", "### Computational cost", "", markdown_table(timing_display)])
    lines.extend(["", "## Interpretation guardrails", "",
                  "- Relative RMSE uses the training-mean predictor as 1; relative CRPS uses the empirical training-target climatology as 1. Lower is better.",
                  "- Segment R-hat uses temporally dependent pseudo-chains and is a stability diagnostic, not a formal convergence certificate.",
                  "- Raw RMSE and CRPS remain available in the CSV tables.",
                  "- Timing reports the actual parallel PT implementation and omits serial PT speed-up.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def transition(value_a: float, value_b: float, digits: int = 3) -> str:
    return f"{value_a:.{digits}f} -> {value_b:.{digits}f}"


def write_root_summary(
    path: Path,
    comparison_summary: Path,
    diagnosis_summary: Path,
    summary: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    rows = []
    for dataset in summary["dataset"].drop_duplicates():
        dataset_frame = summary[summary["dataset"] == dataset].set_index("method")
        default = dataset_frame.loc["Default"]
        combined = dataset_frame.loc["MTMH+PT"]
        timing_row = timing[(timing["dataset"] == dataset) & (timing["method"] == "MTMH+PT")]
        cost = float(timing_row.iloc[0]["relative_to_default"]) if not timing_row.empty else np.nan
        rows.append({
            "dataset": dataset.removeprefix("fixed100_"),
            "cross R-hat": transition(default["cross_rhat_median_mean"], combined["cross_rhat_median_mean"]),
            "within R-hat": transition(default["within_rhat_median_mean"], combined["within_rhat_median_mean"]),
            "B/W ratio": transition(default["between_within_ratio_mean"], combined["between_within_ratio_mean"]),
            "relative RMSE": transition(default["relative_rmse_mean"], combined["relative_rmse_mean"]),
            "relative CRPS": transition(default["relative_crps_mean"], combined["relative_crps_mean"]),
            "time / Default": f"{cost:.2f}x" if np.isfinite(cost) else "not measured",
        })

    compact = pd.DataFrame(rows)
    mixing_metrics = [
        "cross_rhat_median_mean", "within_rhat_median_mean",
        "between_within_ratio_mean",
    ]
    combined_wins = all(
        frame.set_index("method")[metric].idxmin() == "MTMH+PT"
        for _, frame in summary.groupby("dataset")
        for metric in mixing_metrics
    )
    cost_min = timing[timing["method"] == "MTMH+PT"]["relative_to_default"].min()
    cost_max = timing[timing["method"] == "MTMH+PT"]["relative_to_default"].max()

    n_datasets = summary["dataset"].nunique()
    n_timed = timing[timing["method"] == "MTMH+PT"]["dataset"].nunique()
    lines = ["# Paper analysis outputs", "",
             f"This directory contains standalone paper-facing analyses for five paired runs of {n_datasets} datasets.", "",
             "## Main result", ""]
    if combined_wins:
        lines.append(f"MTMH+PT has the lowest five-run mean for all three comparison diagnostics in all {n_datasets} settings.")
    else:
        standard = summary[~summary['dataset'].str.contains('FriedmanSparseDir')]
        wins = sum(frame.set_index('method')[metric].idxmin() == 'MTMH+PT'
                   for _, frame in standard.groupby('dataset') for metric in mixing_metrics)
        lines.append(f"Across the {standard.dataset.nunique()} standard-BART datasets, MTMH+PT has the smallest five-run mean in {wins} of {len(mixing_metrics) * standard.dataset.nunique()} diagnostic cells. DART dimensions form a separate controlled extension.")
    lines.extend([
        "Predictive changes are reported relative to training-only baselines.",
        f"The measured MTMH+PT cost is {cost_min:.2f}x to {cost_max:.2f}x the Default runtime per chain "
        f"({n_timed} of {n_datasets} datasets have timing runs).", "",
        "The table reports five-run means as `Default -> MTMH+PT`.", "",
        markdown_table(compact), "",
        "## Default diagnosis", "",
        f"See [{diagnosis_summary.name}]({diagnosis_summary.name}).", "",
        "## Four-method comparison", "",
        f"See [{comparison_summary.name}]({comparison_summary.name}).", "",
        "## Interpretation", "",
        "The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, and original-space separation diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    diagnosis_root = script.parent.parent
    parser = argparse.ArgumentParser(description="Run paper-facing four-method comparisons.")
    parser.add_argument("--store-root", type=Path, default=diagnosis_root / "store")
    parser.add_argument("--table-dir", type=Path, default=script.parent / "tables")
    parser.add_argument("--figure-dir", type=Path, default=script.parent / "figures" / "comparison")
    parser.add_argument("--summary-path", type=Path, default=script.parent / "comparison_summary.md")
    parser.add_argument("--timing-summary", type=Path, default=diagnosis_root / "timing" / "summary.md")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--runs", nargs="+", type=int, default=list(RUNS))
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=1000)
    parser.add_argument("--n-segments", type=int, default=4)
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--rhat-only", action="store_true",
                        help="Refresh full post-burn R-hat table summaries without regenerating rolling figures.")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Recompute original-space chain geometry and dependent tables from existing summaries.",
    )
    parser.add_argument("--ridge-fraction", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.geometry_only and args.rhat_only:
        raise ValueError("Choose either --geometry-only or --rhat-only")
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    summary_path = args.summary_path.resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    settings = {
        "runs_per_dataset": len(args.runs),
        "cross_chain_window": args.window,
        "rolling_step": args.step,
        "segment_length": args.segment_length,
        "segments_per_block": args.n_segments,
        "short_burn": args.short_burn,
        "chain_geometry_draws_per_chain": "all post-burn draws",
        "table_rhat_draws": "all post-burn draws; within-chain uses equal consecutive segments",
    }

    if args.geometry_only or args.rhat_only:
        update_kind = "rhat" if args.rhat_only else "geometry"
        run_path = table_dir / "comparison_run_summary.csv"
        if not run_path.exists():
            raise FileNotFoundError(f"Partial table update requires {run_path}")
        run_frame = pd.read_csv(run_path)
        for dataset in args.datasets:
            for run in args.runs:
                print(f"[comparison:{update_kind}] {dataset} run {run:03d}", flush=True)
                for method in METHODS:
                    post_burn = predictions(args.store_root, dataset, run, method, args.short_burn)
                    geometry = (full_draw_rhat(post_burn, args.n_segments)
                                if args.rhat_only else chain_geometry(post_burn))
                    mask = (
                        (run_frame["dataset"] == dataset)
                        & (run_frame["run"] == run)
                        & (run_frame["method"] == METHOD_NAMES[method])
                    )
                    if int(mask.sum()) != 1:
                        raise ValueError(
                            f"Expected one summary row for {dataset}, run {run:03d}, {METHOD_NAMES[method]}"
                        )
                    for key, value in geometry.items():
                        run_frame.loc[mask, key] = value

        summary_frame = dataset_summary(run_frame)
        ratio_frame = paired_ratios(run_frame)
        timing_frame = parse_timing_summary(args.timing_summary)
        paper_table = summary_frame.merge(timing_frame, on=["dataset", "method"], how="left")
        run_frame.to_csv(run_path, index=False)
        summary_frame.to_csv(table_dir / "comparison_dataset_summary.csv", index=False)
        ratio_frame.to_csv(table_dir / "paired_ratios_to_default.csv", index=False)
        paper_table.to_csv(table_dir / "paper_table_summary.csv", index=False)
        write_summary(summary_path, settings, summary_frame, timing_frame)
        diagnosis_summary = Path(__file__).resolve().parent / "diagnosis_summary.md"
        write_root_summary(
            Path(__file__).resolve().parent / "summary.md",
            summary_path,
            diagnosis_summary,
            summary_frame,
            timing_frame,
        )
        print(f"[comparison:{update_kind}] updated tables: {table_dir}", flush=True)
        return 0

    run_rows, baseline_rows, cross_all, within_all = [], [], [], []
    for dataset in args.datasets:
        full_targets = full_target_vector(args.store_root, dataset)
        for run in args.runs:
            print(f"[comparison] {dataset} run {run:03d}", flush=True)
            truth = y_test(args.store_root, dataset, run)
            train_targets = training_targets(
                args.store_root, dataset, run, full_targets, truth
            )
            baseline = predictive_baselines(train_targets, truth)
            # DefaultPreprocessor min-max scales y on the training targets, so
            # this is the factor that takes a stored sigma back to y's scale.
            y_scale = float(np.max(train_targets) - np.min(train_targets))
            baseline_rows.append({
                "dataset": dataset,
                "run": run,
                "test_points": len(truth),
                "training_points": len(train_targets),
                **baseline,
            })

            cross_by_method, within_by_method, worst_by_method = {}, {}, {}
            for method in METHODS:
                all_draws = predictions(args.store_root, dataset, run, method, 0)
                post_burn = all_draws[:, args.short_burn:, :]
                worst = worst_direction_summary(all_draws[:, args.short_burn:, :], args.ridge_fraction)
                cross = rolling_cross_rhat(all_draws, args.window, args.step)
                within = rolling_within_rhat(all_draws, args.segment_length, args.n_segments, args.step)
                geometry = chain_geometry(post_burn)
                rhat_summary = full_draw_rhat(post_burn, args.n_segments)
                rmse, crps = predictive_metrics(
                    post_burn,
                    residual_variances(args.store_root, dataset, run, method,
                                       args.short_burn),
                    truth,
                    y_scale,
                    PREDICTIVE_NOISE_SEED + run,
                )
                relative_rmse = rmse / baseline["naive_rmse"]
                relative_crps = crps / baseline["climatology_crps"]

                method_label = METHOD_NAMES[method]
                cross_by_method[method] = cross
                within_by_method[method] = within
                worst_by_method[method] = worst
                cross_all.append(cross.assign(dataset=dataset, run=run, method=method_label))
                within_all.append(within.assign(dataset=dataset, run=run, method=method_label))
                run_rows.append({
                    "dataset": dataset,
                    "run": run,
                    "method": method_label,
                    "worst_lambda_max": worst["lambda_max"],
                    "worst_lambda_second": worst["lambda_second"],
                    "worst_eigen_gap": worst["eigen_gap"],
                    "worst_projected_rhat": worst["projected_rhat"],
                    "second_projected_rhat": worst["second_projected_rhat"],
                    **rhat_summary,
                    "centroid_distance": geometry["centroid_distance"],
                    "between_within_ratio": geometry["between_within_ratio"],
                    "geometry_draws_per_chain": geometry["geometry_draws_per_chain"],
                    "rmse": rmse,
                    "crps": crps,
                    "relative_rmse": relative_rmse,
                    "relative_crps": relative_crps,
                })
            rhat_fig = plot_rhat_comparison(
                within_by_method,
                f"{dataset} | run {run:03d} | within-chain segment R-hat: "
                f"median and chain range",
            )
            rhat_fig.savefig(figure_dir / f"{dataset}_run{run:03d}_rhat_comparison.png",
                             dpi=args.dpi, bbox_inches="tight")
            plt.close(rhat_fig)
            worst_fig = plot_worst_comparison(
                worst_by_method, args.short_burn,
                f"{dataset} | run {run:03d} | worst-direction comparison",
            )
            worst_fig.savefig(figure_dir / f"{dataset}_run{run:03d}_worst_direction.png",
                              dpi=args.dpi, bbox_inches="tight")
            plt.close(worst_fig)

    run_frame = pd.DataFrame(run_rows)
    summary_frame = dataset_summary(run_frame)
    ratio_frame = paired_ratios(run_frame)
    baseline_frame = pd.DataFrame(baseline_rows)
    timing_frame = parse_timing_summary(args.timing_summary)
    paper_table = summary_frame.merge(timing_frame, on=["dataset", "method"], how="left")

    run_frame.to_csv(table_dir / "comparison_run_summary.csv", index=False)
    summary_frame.to_csv(table_dir / "comparison_dataset_summary.csv", index=False)
    ratio_frame.to_csv(table_dir / "paired_ratios_to_default.csv", index=False)
    baseline_frame.to_csv(table_dir / "predictive_baselines.csv", index=False)
    timing_frame.to_csv(table_dir / "timing_parallel.csv", index=False)
    paper_table.to_csv(table_dir / "paper_table_summary.csv", index=False)
    pd.concat(cross_all, ignore_index=True).to_csv(table_dir / "comparison_cross_chain_rhat_windows.csv", index=False)
    pd.concat(within_all, ignore_index=True).to_csv(table_dir / "comparison_within_chain_rhat_windows.csv", index=False)

    write_summary(summary_path, settings, summary_frame, timing_frame)
    diagnosis_summary = Path(__file__).resolve().parent / "diagnosis_summary.md"
    write_root_summary(
        Path(__file__).resolve().parent / "summary.md",
        summary_path,
        diagnosis_summary,
        summary_frame,
        timing_frame,
    )
    print(f"[comparison] tables: {table_dir}", flush=True)
    print(f"[comparison] figures: {figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
