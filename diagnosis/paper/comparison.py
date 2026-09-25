#!/usr/bin/env python
"""Paper-facing four-method comparison for the three fixed-100 datasets.

This file is intentionally self-contained. It does not import code from
``diagnosis/exploratory``.
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
from scipy.linalg import eigh
from scipy.stats import gaussian_kde


DATASETS = ("fixed100_Abalone", "fixed100_Concrete", "fixed100_Friedman")
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


def load_with_shape(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if "original_shape=" not in header:
        raise ValueError(f"Missing original_shape header: {path}")
    shape = ast.literal_eval(header.split("original_shape=")[-1])
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)


def stored_path(store_root: Path, dataset: str, folder: str, run: int, suffix: str) -> Path:
    if folder == "preds":
        filename = f"{dataset}__run{run:03d}__{suffix}__preds.csv"
    else:
        filename = f"{dataset}__run{run:03d}__{suffix}.csv"
    return store_root / dataset / folder / filename


def predictions(store_root: Path, dataset: str, run: int, method: str, burn: int = 0) -> np.ndarray:
    raw = load_with_shape(stored_path(store_root, dataset, "preds", run, method))
    draws = raw.transpose(0, 2, 1)
    return draws[:, burn:, :] if burn else draws


def y_test(store_root: Path, dataset: str, run: int) -> np.ndarray:
    path = stored_path(store_root, dataset, "subsample_y_test", run, "subsample_y_test")
    return load_with_shape(path).reshape(-1)


def full_target_vector(store_root: Path, dataset: str) -> np.ndarray:
    """Load/regenerate the full target vector used to create the saved split."""
    if dataset == "fixed100_Abalone":
        path = store_root / "uci_cache" / "abalone__targets.csv"
        values = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    elif dataset == "fixed100_Concrete":
        path = store_root / "uci_cache" / "concrete__targets.csv"
        values = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    elif dataset == "fixed100_Friedman":
        rng = np.random.default_rng(42)
        features = rng.uniform(0.0, 1.0, size=(2000, 10))
        values = (
            10.0 * np.sin(np.pi * features[:, 0] * features[:, 1])
            + 20.0 * (features[:, 2] - 0.5) ** 2
            + 10.0 * features[:, 3]
            + 5.0 * features[:, 4]
            + rng.normal(0.0, 1.0, size=2000)
        )
    else:
        raise ValueError(f"No self-contained target loader for {dataset}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite targets in full target vector for {dataset}")
    return values


def split_indices(store_root: Path, dataset: str, run: int, split: str) -> np.ndarray:
    path = store_root / dataset / "indices" / f"{dataset}__run{run:03d}__{split}.csv"
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


def pointwise_rhat(draws: np.ndarray) -> np.ndarray:
    data = xr.DataArray(draws, dims=("chain", "draw", "test_point"), name="prediction")
    return np.asarray(az.rhat(data, method="rank")["prediction"].values)


def summarize_rhat(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.nanmedian(values)),
        "q25": float(np.nanquantile(values, 0.25)),
        "q75": float(np.nanquantile(values, 0.75)),
        "q90": float(np.nanquantile(values, 0.90)),
        "maximum": float(np.nanmax(values)),
        "fraction_above_1_01": float(np.nanmean(values > 1.01)),
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


def worst_direction(draws: np.ndarray, ridge_fraction: float) -> dict[str, np.ndarray | float]:
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
        "projected": projected,
    }


def chain_separation(draws: np.ndarray, max_draws: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n_use = min(max_draws, draws.shape[1])
    sampled = []
    for chain in draws:
        indices = np.sort(rng.choice(len(chain), n_use, replace=False)) if n_use < len(chain) else np.arange(len(chain))
        sampled.append(chain[indices])
    sampled = np.stack(sampled)
    centers = sampled.mean(axis=1)
    centroid_distances = [
        np.linalg.norm(centers[left] - centers[right])
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
    ]
    grand = sampled.reshape(-1, sampled.shape[-1]).mean(axis=0)
    between = np.mean(np.sum((centers - grand) ** 2, axis=1))
    within = np.mean([
        np.mean(np.sum((chain - center) ** 2, axis=1))
        for chain, center in zip(sampled, centers)
    ])
    return {
        "centroid_distance": float(np.mean(centroid_distances)),
        "between_within_ratio": float(between / within),
        "separation_draws_per_chain": n_use,
    }


def euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_sq = np.sum(left * left, axis=1, keepdims=True)
    right_sq = np.sum(right * right, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0))


def energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    cross = euclidean_distances(left, right).mean()
    left_dist = euclidean_distances(left, left)
    right_dist = euclidean_distances(right, right)
    left_within = (left_dist.sum() - np.trace(left_dist)) / (len(left) * (len(left) - 1))
    right_within = (right_dist.sum() - np.trace(right_dist)) / (len(right) * (len(right) - 1))
    return float(2.0 * cross - left_within - right_within)


def chain_pair_energy(
    short: np.ndarray,
    long: np.ndarray,
    draws_per_chain: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Average energy distance over all short-chain/long-chain pairs.

    Each chain is sampled independently and without replacement before forming
    the Cartesian product. With four short and four long chains, the run-level
    estimand is therefore the mean of 16 chain-pair energy distances.
    """
    if short.shape[0] != 4 or long.shape[0] != 4:
        raise ValueError(
            "Chain-pair energy distance requires four short and four long chains; "
            f"received {short.shape[0]} and {long.shape[0]}."
        )
    if short.shape[1] < draws_per_chain or long.shape[1] < draws_per_chain:
        raise ValueError(
            f"Energy distance requested {draws_per_chain} draws per chain, but the "
            f"available post-burn lengths are {short.shape[1]} and {long.shape[1]}."
        )

    rng = np.random.default_rng(seed)
    short_sample = np.stack([
        chain[rng.choice(len(chain), draws_per_chain, replace=False)]
        for chain in short
    ])
    long_sample = np.stack([
        chain[rng.choice(len(chain), draws_per_chain, replace=False)]
        for chain in long
    ])
    pair_values = np.asarray([
        energy_distance(short_sample[short_chain], long_sample[long_chain])
        for short_chain in range(short_sample.shape[0])
        for long_chain in range(long_sample.shape[0])
    ])
    return float(pair_values.mean()), float(pair_values.std(ddof=1)), pair_values


def crps_from_samples(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n_points, n_samples = samples.shape
    first = np.mean(np.abs(samples - truth[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    coefficients = (2 * np.arange(1, n_samples + 1) - n_samples - 1)[None, :]
    second = np.sum(coefficients * ordered, axis=1) / n_samples**2
    return first - second


def predictive_metrics(draws: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    rmse_values, crps_values = [], []
    for chain in draws:
        samples = chain.T
        rmse_values.append(np.sqrt(np.mean((samples.mean(axis=1) - truth) ** 2)))
        crps_values.append(np.mean(crps_from_samples(samples, truth)))
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


def plot_rhat_comparison(cross_frames: dict[str, pd.DataFrame], within_frames: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), layout="constrained")
    for method in METHODS:
        cross = cross_frames[method].sort_values("window_end")
        axes[0].plot(cross["window_end"], cross["median"], color=METHOD_COLORS[method],
                     linewidth=1.8, label=METHOD_NAMES[method])
        axes[0].fill_between(cross["window_end"], cross["q25"], cross["q75"],
                             color=METHOD_COLORS[method], alpha=0.10)

        within = within_frames[method]
        reduced = within.groupby("window_end")["median"].agg(
            center="median", low="min", high="max"
        ).reset_index()
        axes[1].plot(reduced["window_end"], reduced["center"], color=METHOD_COLORS[method],
                     linewidth=1.8, label=METHOD_NAMES[method])
        axes[1].fill_between(reduced["window_end"], reduced["low"], reduced["high"],
                             color=METHOD_COLORS[method], alpha=0.10)
    axes[0].set_title("Cross-chain pointwise R-hat: median and test-point IQR")
    axes[1].set_title("Within-chain segment R-hat: median and chain range")
    for ax in axes:
        ax.axhline(1.01, color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel("window end")
        ax.set_ylabel("rank-split R-hat")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig


def plot_worst_comparison(results: dict[str, dict[str, np.ndarray | float]], start: int, title: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.0), layout="constrained")
    for row, method in enumerate(METHODS):
        result = results[method]
        projected = np.asarray(result["projected"])
        x = np.arange(start, start + projected.shape[1])
        for chain_id in range(projected.shape[0]):
            axes[row, 0].plot(x, projected[chain_id], color=CHAIN_COLORS[chain_id],
                              linewidth=0.55, alpha=0.75,
                              label=f"chain {chain_id + 1}" if row == 0 else "_nolegend_")
            values = projected[chain_id]
            try:
                density = gaussian_kde(values)
                grid = np.linspace(values.min(), values.max(), 250)
                axes[row, 1].plot(grid, density(grid), color=CHAIN_COLORS[chain_id], linewidth=1.5)
            except np.linalg.LinAlgError:
                pass
        axes[row, 0].set_title(
            f"{METHOD_NAMES[method]} trace | lambda={result['lambda_max']:.3f}, R-hat={result['projected_rhat']:.3f}"
        )
        axes[row, 1].set_title(f"{METHOD_NAMES[method]} density")
        axes[row, 0].set_ylabel("projection")
        axes[row, 1].set_ylabel("density")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("iteration")
    axes[-1, 1].set_xlabel("worst-direction projection")
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
        "centroid_distance", "between_within_ratio", "energy_distance",
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
            current_dataset = heading.group(1)
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
    return pd.DataFrame(rows)


def format_mean_sd(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} ({row[f'{metric}_sd']:.{digits}f})"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


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
        "`short_burn` is also the start of the worst-direction calculation. `long_burn` counts stored long-chain draws; because the long chains were saved after downsampling, its effective burn-in in original iterations is `long_burn × long_store_every`.",
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
            "scaled energy": [format_mean_sd(row, "scaled_energy_distance") for _, row in frame.iterrows()],
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
                  f"- Energy distance is averaged over all {settings['energy_chain_pairs']} short-chain/long-chain pairs, using {settings['energy_draws_per_chain']:,} randomly sampled draws from each chain.",
                  "- The chain-pair mean is divided by training-target SD times sqrt(number of test points); the within-run SD across the 16 pairs is retained in the numerical tables.",
                  "- Raw RMSE, CRPS, and energy distance remain available in the CSV tables.",
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
            "projected R-hat": transition(default["worst_projected_rhat_mean"], combined["worst_projected_rhat_mean"]),
            "cross R-hat": transition(default["cross_rhat_median_mean"], combined["cross_rhat_median_mean"]),
            "within R-hat": transition(default["within_rhat_median_mean"], combined["within_rhat_median_mean"]),
            "B/W ratio": transition(default["between_within_ratio_mean"], combined["between_within_ratio_mean"]),
            "scaled energy": transition(default["scaled_energy_distance_mean"], combined["scaled_energy_distance_mean"]),
            "relative RMSE": transition(default["relative_rmse_mean"], combined["relative_rmse_mean"]),
            "relative CRPS": transition(default["relative_crps_mean"], combined["relative_crps_mean"]),
            "time / Default": f"{cost:.2f}x",
        })

    compact = pd.DataFrame(rows)
    mixing_metrics = [
        "worst_projected_rhat_mean", "cross_rhat_median_mean", "within_rhat_median_mean",
        "between_within_ratio_mean", "scaled_energy_distance_mean",
    ]
    combined_wins = all(
        frame.set_index("method")[metric].idxmin() == "MTMH+PT"
        for _, frame in summary.groupby("dataset")
        for metric in mixing_metrics
    )
    cost_min = timing[timing["method"] == "MTMH+PT"]["relative_to_default"].min()
    cost_max = timing[timing["method"] == "MTMH+PT"]["relative_to_default"].max()

    lines = ["# Paper analysis outputs", "",
             "This directory contains standalone paper-facing analyses for five paired runs of three datasets.", "",
             "## Main result", ""]
    if combined_wins:
        lines.append("MTMH+PT has the lowest five-run mean for all five reported mixing diagnostics in all three datasets.")
    lines.extend([
        "Predictive changes are reported relative to training-only baselines.",
        f"The measured MTMH+PT cost is {cost_min:.2f}x to {cost_max:.2f}x the Default runtime per chain.", "",
        "The table reports five-run means as `Default -> MTMH+PT`.", "",
        markdown_table(compact), "",
        "## Default diagnosis", "",
        f"See [{diagnosis_summary.name}]({diagnosis_summary.name}).", "",
        "## Four-method comparison", "",
        f"See [{comparison_summary.name}]({comparison_summary.name}).", "",
        "## Interpretation", "",
        "The results support improved mixing through agreement across worst-direction, pointwise cross-chain, within-chain segment, original-space separation, and scaled energy-distance diagnostics. Relative RMSE and CRPS make predictive performance comparable across datasets. The timing values quantify the associated computational cost. Segment R-hat is a stability diagnostic based on dependent pseudo-chains, and the long Default run is an empirical reference for energy distance.", "",
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
    parser.add_argument("--long-burn", type=int, default=30)
    parser.add_argument("--separation-draws", type=int, default=1500)
    parser.add_argument("--energy-draws", type=int, default=1000)
    parser.add_argument(
        "--energy-only",
        action="store_true",
        help="Recompute chain-pair energy metrics and dependent summaries from existing tables.",
    )
    parser.add_argument("--ridge-fraction", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        "long_burn": args.long_burn,
        "chain_separation_draws_per_chain": args.separation_draws,
        "energy_chain_pairs": "4 x 4",
        "energy_draws_per_chain": args.energy_draws,
    }

    if args.energy_only:
        run_path = table_dir / "comparison_run_summary.csv"
        baseline_path = table_dir / "predictive_and_energy_baselines.csv"
        if not run_path.exists() or not baseline_path.exists():
            raise FileNotFoundError(
                "--energy-only requires existing comparison_run_summary.csv and "
                "predictive_and_energy_baselines.csv tables."
            )
        run_frame = pd.read_csv(run_path)
        baseline_frame = pd.read_csv(baseline_path)
        pair_rows = []
        for dataset_index, dataset in enumerate(args.datasets):
            for run in args.runs:
                print(f"[energy] {dataset} run {run:03d}", flush=True)
                baseline_match = baseline_frame[
                    (baseline_frame["dataset"] == dataset) & (baseline_frame["run"] == run)
                ]
                if len(baseline_match) != 1:
                    raise ValueError(f"Missing unique energy scale for {dataset} run {run:03d}")
                energy_scale = float(baseline_match.iloc[0]["energy_scale_train_sd_sqrt_p"])
                long_draws = predictions(args.store_root, dataset, run, "default_long", args.long_burn)
                for method_index, method in enumerate(METHODS):
                    post_burn = predictions(
                        args.store_root, dataset, run, method, args.short_burn
                    )
                    energy, pair_sd, pair_values = chain_pair_energy(
                        post_burn,
                        long_draws,
                        args.energy_draws,
                        seed=4040 + 10000 * dataset_index + 100 * run + method_index,
                    )
                    method_label = METHOD_NAMES[method]
                    mask = (
                        (run_frame["dataset"] == dataset)
                        & (run_frame["run"] == run)
                        & (run_frame["method"] == method_label)
                    )
                    if mask.sum() != 1:
                        raise ValueError(
                            f"Missing unique run-summary row for {dataset} run {run:03d} {method_label}"
                        )
                    run_frame.loc[mask, "energy_distance"] = energy
                    run_frame.loc[mask, "energy_pair_sd"] = pair_sd
                    run_frame.loc[mask, "scaled_energy_distance"] = energy / energy_scale
                    run_frame.loc[mask, "scaled_energy_pair_sd"] = pair_sd / energy_scale
                    for pair_index, value in enumerate(pair_values):
                        pair_rows.append({
                            "dataset": dataset,
                            "run": run,
                            "method": method_label,
                            "short_chain": pair_index // 4,
                            "long_chain": pair_index % 4,
                            "energy_distance": value,
                            "scaled_energy_distance": value / energy_scale,
                            "draws_per_chain": args.energy_draws,
                        })

        summary_frame = dataset_summary(run_frame)
        ratio_frame = paired_ratios(run_frame)
        timing_frame = parse_timing_summary(args.timing_summary)
        paper_table = summary_frame.merge(timing_frame, on=["dataset", "method"], how="left")
        run_frame.to_csv(run_path, index=False)
        summary_frame.to_csv(table_dir / "comparison_dataset_summary.csv", index=False)
        ratio_frame.to_csv(table_dir / "paired_ratios_to_default.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(table_dir / "comparison_energy_chain_pairs.csv", index=False)
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
        print(f"[energy] updated tables: {table_dir}", flush=True)
        return 0

    run_rows, baseline_rows, pair_rows, cross_all, within_all = [], [], [], [], []
    for dataset_index, dataset in enumerate(args.datasets):
        full_targets = full_target_vector(args.store_root, dataset)
        for run in args.runs:
            print(f"[comparison] {dataset} run {run:03d}", flush=True)
            truth = y_test(args.store_root, dataset, run)
            train_targets = training_targets(
                args.store_root, dataset, run, full_targets, truth
            )
            baseline = predictive_baselines(train_targets, truth)
            energy_scale = baseline["train_sd"] * np.sqrt(len(truth))
            long_draws = predictions(args.store_root, dataset, run, "default_long", args.long_burn)
            baseline_rows.append({
                "dataset": dataset,
                "run": run,
                "test_points": len(truth),
                "training_points": len(train_targets),
                **baseline,
                "energy_scale_train_sd_sqrt_p": energy_scale,
            })

            cross_by_method, within_by_method, worst_by_method = {}, {}, {}
            for method_index, method in enumerate(METHODS):
                all_draws = predictions(args.store_root, dataset, run, method, 0)
                post_burn = all_draws[:, args.short_burn:, :]
                worst = worst_direction(all_draws[:, args.short_burn:, :], args.ridge_fraction)
                cross = rolling_cross_rhat(all_draws, args.window, args.step)
                within = rolling_within_rhat(all_draws, args.segment_length, args.n_segments, args.step)
                separation = chain_separation(
                    post_burn, args.separation_draws,
                    seed=2028 + 1000 * dataset_index + run,
                )
                energy, energy_pair_sd, energy_pair_values = chain_pair_energy(
                    post_burn, long_draws, args.energy_draws,
                    seed=4040 + 10000 * dataset_index + 100 * run + method_index,
                )
                rmse, crps = predictive_metrics(post_burn, truth)
                scaled_energy = energy / energy_scale
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
                    "worst_eigen_gap": worst["eigen_gap"],
                    "worst_projected_rhat": worst["projected_rhat"],
                    "cross_rhat_median": cross["median"].median(),
                    "cross_rhat_q90": cross["q90"].median(),
                    "cross_fraction_above_1_01": cross["fraction_above_1_01"].median(),
                    "within_rhat_median": within["median"].median(),
                    "within_rhat_q90": within["q90"].median(),
                    "within_fraction_above_1_01": within["fraction_above_1_01"].median(),
                    "centroid_distance": separation["centroid_distance"],
                    "between_within_ratio": separation["between_within_ratio"],
                    "energy_distance": energy,
                    "energy_pair_sd": energy_pair_sd,
                    "scaled_energy_distance": scaled_energy,
                    "scaled_energy_pair_sd": energy_pair_sd / energy_scale,
                    "rmse": rmse,
                    "crps": crps,
                    "relative_rmse": relative_rmse,
                    "relative_crps": relative_crps,
                })
                for pair_index, value in enumerate(energy_pair_values):
                    pair_rows.append({
                        "dataset": dataset,
                        "run": run,
                        "method": method_label,
                        "short_chain": pair_index // 4,
                        "long_chain": pair_index % 4,
                        "energy_distance": value,
                        "scaled_energy_distance": value / energy_scale,
                        "draws_per_chain": args.energy_draws,
                    })

            rhat_fig = plot_rhat_comparison(
                cross_by_method, within_by_method,
                f"{dataset} | run {run:03d} | pointwise R-hat comparison",
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
    baseline_frame.to_csv(table_dir / "predictive_and_energy_baselines.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(table_dir / "comparison_energy_chain_pairs.csv", index=False)
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
