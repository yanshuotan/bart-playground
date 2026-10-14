#!/usr/bin/env python
"""Run all fixed-100 diagnostics on a named stored dataset.

Example
-------
python run_all_analyses.py fixed100_Abalone

The positional argument is a dataset/store name. The script finds the
corresponding directory under ``diagnosis/store`` and saves
every table and figure under ``exploratory/results/<dataset name>/``.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import eigh
from scipy.stats import gaussian_kde


LONG_METHOD = "default_long"
SHORT_METHOD_ORDER = ("default", "default_pt", "mtmh", "mtmh_pt")
METHOD_NAMES = {
    "default": "Default",
    "default_pt": "Default PT",
    "mtmh": "MTMH",
    "mtmh_pt": "MTMH + PT",
    "default_long": "Default long",
}
METHOD_COLORS = {
    "default": "#1b9e77",
    "default_pt": "#d95f02",
    "mtmh": "#7570b3",
    "mtmh_pt": "#e7298a",
    "default_long": "#222222",
}
CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

ALL_GROUPS = (
    "long",
    "separation",
    "predictive",
    "rhat",
    "ess",
)


@dataclass
class Config:
    dataset_name: str
    data_tag: str
    store_dir: Path
    output_dir: Path
    long_burn: int = 10
    short_burn: int = 500
    window: int = 1000
    step: int = 100
    segment_length: int = 1000
    n_segments: int = 4
    plot_draws: int = 1000
    separation_draws_per_chain: int = 1500
    separation_seed: int = 2028
    eigen_projection_start: int = 3000
    plot_chain_seed: int = 2029

    @property
    def segment_block(self) -> int:
        return self.segment_length * self.n_segments


class Store:
    def __init__(self, config: Config):
        self.cfg = config

    def path(self, metric: str, run_id: int, method: str) -> Path:
        name = f"{self.cfg.data_tag}__run{run_id:03d}__{method}__{metric}.csv"
        return self.cfg.store_dir / metric / name

    def split_path(self, folder: str, run_id: int, suffix: str) -> Path:
        name = f"{self.cfg.data_tag}__run{run_id:03d}__{suffix}.csv"
        return self.cfg.store_dir / folder / name

    def has(self, metric: str, run_id: int, method: str) -> bool:
        return self.path(metric, run_id, method).exists()

    def load(self, metric: str, run_id: int, method: str) -> np.ndarray:
        path = self.path(metric, run_id, method)
        raw, shape = load_with_shape(path)
        return np.asarray(raw).reshape(shape)

    def predictions(self, run_id: int, method: str, burn: int = 0) -> np.ndarray:
        """Return predictions as (original_chain, draw, test_point)."""
        array = self.load("preds", run_id, method)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D predictions in {self.path('preds', run_id, method)}; got {array.shape}")
        result = array.transpose(0, 2, 1)
        return result[:, burn:, :] if burn else result

    def sigma2(self, run_id: int, method: str, burn: int = 0) -> np.ndarray:
        array = np.squeeze(self.load("sigmas", run_id, method))
        if array.ndim != 2:
            raise ValueError(f"Expected sigma^2 as (chain, draw); got {array.shape}")
        return array[:, burn:] if burn else array

    def scalar_metric(self, metric: str, run_id: int, method: str, burn: int = 0) -> np.ndarray:
        array = np.squeeze(self.load(metric, run_id, method))
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"Expected {metric} as (chain, draw); got {array.shape}")
        return array[:, burn:] if burn else array

    def y_test(self, run_id: int) -> np.ndarray | None:
        path = self.split_path("subsample_y_test", run_id, "subsample_y_test")
        if not path.exists():
            return None
        raw, shape = load_with_shape(path)
        return np.asarray(raw).reshape(shape).reshape(-1)

    def runs(self, method: str, metric: str = "preds") -> list[int]:
        folder = self.cfg.store_dir / metric
        if not folder.exists():
            return []
        pattern = re.compile(rf"^{re.escape(self.cfg.data_tag)}__run(\d+)__{re.escape(method)}__{re.escape(metric)}\.csv$")
        result = []
        for path in folder.glob(f"{self.cfg.data_tag}__run*__{method}__{metric}.csv"):
            match = pattern.match(path.name)
            if match:
                result.append(int(match.group(1)))
        return sorted(set(result))


def load_with_shape(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if "original_shape=" not in header:
        raise ValueError(f"Missing original_shape in header: {path}")
    shape = ast.literal_eval(header.split("original_shape=")[-1])
    data = np.loadtxt(path, delimiter=",", comments="#")
    return data, tuple(shape)


def locate_store(dataset_name: str, store_root: Path) -> Path | None:
    """Find a store directory by dataset name."""
    requested = Path(dataset_name).name
    direct = store_root / requested
    if direct.is_dir():
        return direct.resolve()

    matches = [
        folder
        for folder in store_root.rglob("*") if folder.is_dir() and folder.name.casefold() == requested.casefold()
    ] if store_root.exists() else []
    return matches[0].resolve() if len(matches) == 1 else None


def method_name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def method_shades(method: str, count: int) -> list[tuple[float, float, float]]:
    base = np.asarray(mcolors.to_rgb(method_color(method)))
    if count <= 1:
        return [tuple(base)]
    return [tuple(base * (0.4 + 0.6 * i / (count - 1)) + (1 - (0.4 + 0.6 * i / (count - 1)))) for i in range(count)]


def sampled_chain_id(cfg: Config, run_id: int, method: str, chain_ids: Iterable[int]) -> int:
    """Select one reproducible chain for crowded multi-method rolling plots."""
    available = np.asarray(sorted(int(chain_id) for chain_id in chain_ids), dtype=int)
    if available.size == 0:
        raise ValueError(f"No chains available for {method}, run {run_id:03d}.")
    method_order = (LONG_METHOD, *SHORT_METHOD_ORDER)
    method_index = method_order.index(method)
    rng = np.random.default_rng(
        np.random.SeedSequence([cfg.plot_chain_seed, int(run_id), method_index])
    )
    return int(rng.choice(available))


def short_methods_with_data(store: Store) -> list[str]:
    return [method for method in SHORT_METHOD_ORDER if store.runs(method)]


def short_run_union(store: Store, methods: Iterable[str]) -> list[int]:
    runs: set[int] = set()
    for method in methods:
        runs.update(store.runs(method))
    return sorted(runs)


def print_metric(name: str) -> None:
    print(f"\n[metric] {name}", flush=True)


def save_figure(fig: plt.Figure, cfg: Config, filename: str) -> None:
    path = cfg.output_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def save_table(rows: list[dict] | pd.DataFrame, cfg: Config, filename: str) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = cfg.output_dir / filename
    frame.to_csv(path, index=False)
    print(f"  saved {path.name}", flush=True)
    return frame


def finish_method_legend(fig: plt.Figure, ax: plt.Axes, ncol: int = 4, bottom: float = 0.10) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=ncol, frameon=False)
    fig.tight_layout(rect=[0, bottom, 1, 1])


def available_methods_for_run(store: Store, methods: Iterable[str], run_id: int, metric: str = "preds") -> list[str]:
    return [method for method in methods if store.has(metric, run_id, method)]


def crps_from_samples(samples: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n_points, n_samples = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    coeffs = (2 * np.arange(1, n_samples + 1) - n_samples - 1)[None, :]
    term2 = np.sum(coeffs * ordered, axis=1) / n_samples**2
    return term1 - term2


def chain_separation_metrics(
    draws: np.ndarray,
    max_draws_per_chain: int,
    seed: int,
) -> dict[str, float | int]:
    """Measure chain separation in the original prediction coordinates.

    ``draws`` has shape (chain, draw, test point).  The same number of
    post-burn-in draws is sampled from each chain before calculating the
    average pairwise distance between chain centroids and the ratio of
    between-chain to within-chain squared dispersion.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3:
        raise ValueError(f"Expected (chain, draw, test point); got {draws.shape}")

    rng = np.random.default_rng(seed)
    n_use = min(max_draws_per_chain, draws.shape[1])
    sampled = []
    for chain in draws:
        if n_use < len(chain):
            indices = np.sort(rng.choice(len(chain), size=n_use, replace=False))
            sampled.append(chain[indices])
        else:
            sampled.append(chain)
    sampled_draws = np.stack(sampled)

    centroids = sampled_draws.mean(axis=1)
    pairwise_distances = [
        float(np.linalg.norm(centroids[left] - centroids[right]))
        for left in range(len(centroids))
        for right in range(left + 1, len(centroids))
    ]
    mean_centroid_distance = float(np.mean(pairwise_distances)) if pairwise_distances else np.nan

    grand_centroid = sampled_draws.reshape(-1, sampled_draws.shape[-1]).mean(axis=0)
    between = float(np.mean(np.sum((centroids - grand_centroid) ** 2, axis=1)))
    within = float(np.mean([
        np.mean(np.sum((chain - centroid) ** 2, axis=1))
        for chain, centroid in zip(sampled_draws, centroids)
    ]))

    return {
        "draws_per_chain": n_use,
        "n_test_points": sampled_draws.shape[-1],
        "mean_centroid_distance": mean_centroid_distance,
        "between_within_ratio": between / within if within > 0 else np.nan,
    }


def multivariate_rhat(draws: np.ndarray, ridge_fraction: float = 1e-8) -> float:
    draws = np.asarray(draws, dtype=float)
    m, n, p = draws.shape
    if m < 2 or n < 2:
        return np.nan
    means = draws.mean(axis=1)
    grand_mean = means.mean(axis=0)
    within = np.zeros((p, p))
    for chain in draws:
        centered = chain - chain.mean(axis=0)
        within += centered.T @ centered / (n - 1)
    within /= m
    offsets = means - grand_mean
    between = n * (offsets.T @ offsets) / (m - 1)
    ridge = ridge_fraction * np.trace(within) / p
    within_regularized = within + max(ridge, ridge_fraction) * np.eye(p)
    chol = np.linalg.cholesky(within_regularized)
    whitened = np.linalg.solve(chol, between / n)
    whitened = np.linalg.solve(chol, whitened.T).T
    lambda_max = max(0.0, float(np.linalg.eigvalsh(whitened).max()))
    return float(np.sqrt((n - 1) / n + ((m + 1) / m) * lambda_max))


def panel_axes(count: int, width: float = 6.2, height: float = 4.8) -> tuple[plt.Figure, np.ndarray]:
    cols = min(2, max(1, count))
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, axes


def analyze_long_behavior(store: Store, cfg: Config, long_runs: list[int]) -> None:
    print_metric("long-chain scalar traces")
    for run_id in long_runs:
        print(f"  run {run_id:03d}", flush=True)
        traces: list[tuple[str, np.ndarray]] = []
        for metric, label in (("sigmas", "sigma^2"), ("rmses", "RMSE")):
            if store.has(metric, run_id, LONG_METHOD):
                traces.append((label, store.scalar_metric(metric, run_id, LONG_METHOD, cfg.long_burn)))
        if not traces:
            continue
        fig, axes = plt.subplots(len(traces), 1, figsize=(9, 3.2 * len(traces)), sharex=True, squeeze=False)
        for ax, (label, values) in zip(axes[:, 0], traces):
            for chain_id in range(values.shape[0]):
                ax.plot(values[chain_id], color=CHAIN_COLORS[chain_id % len(CHAIN_COLORS)],
                        linewidth=1, label=f"chain {chain_id}")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.legend(ncol=min(4, values.shape[0]), fontsize=8)
        axes[-1, 0].set_xlabel("sample index")
        fig.suptitle(f"Run {run_id:03d}: long-chain scalar traces")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_figure(fig, cfg, f"r{run_id:03d}_long_trace.png")


def analyze_chain_separation(
    store: Store,
    cfg: Config,
    short_methods: list[str],
    short_runs: list[int],
) -> None:
    print_metric("original prediction-space chain separation")
    rows = []
    for run_id in short_runs:
        for method in available_methods_for_run(store, short_methods, run_id):
            draws = store.predictions(run_id, method, cfg.short_burn)
            metrics = chain_separation_metrics(
                draws,
                max_draws_per_chain=cfg.separation_draws_per_chain,
                seed=cfg.separation_seed + run_id,
            )
            rows.append({
                "run": run_id,
                "method": method_name(method),
                "n_chains": draws.shape[0],
                **metrics,
            })
    save_table(rows, cfg, "chain_separation.csv")


def analyze_predictive(store: Store, cfg: Config, short_methods: list[str], short_runs: list[int], long_runs: list[int]) -> None:
    print_metric("RMSE and CRPS")
    rows = []
    methods = ([LONG_METHOD] if long_runs else []) + short_methods
    for run_id in sorted(set(short_runs) | set(long_runs)):
        y_true = store.y_test(run_id)
        if y_true is None:
            print(f"  run {run_id:03d}: no subsample_y_test; skipped", flush=True)
            continue
        for method in available_methods_for_run(store, methods, run_id):
            burn = cfg.long_burn if method == LONG_METHOD else cfg.short_burn
            draws = store.predictions(run_id, method, burn)
            for chain_id in range(draws.shape[0]):
                samples = draws[chain_id].T
                mean_prediction = samples.mean(axis=1)
                rows.append({
                    "run": run_id,
                    "method": method_name(method),
                    "chain": chain_id,
                    "rmse": float(np.sqrt(np.mean((mean_prediction - y_true) ** 2))),
                    "crps": float(np.mean(crps_from_samples(samples, y_true))),
                })
    frame = save_table(rows, cfg, "predictive_metrics_by_chain.csv")
    if not frame.empty:
        summary = frame.groupby(["run", "method"], as_index=False)[["rmse", "crps"]].agg(["mean", "std"])
        summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
        save_table(summary.reset_index(drop=True), cfg, "predictive_metrics.csv")


def plot_method_panels(frame: pd.DataFrame, metrics: dict[str, str], cfg: Config, filename_key: str,
                       title: str, ylabel: str, chainwise: bool = False, x: str = "window_start") -> None:
    if frame.empty:
        return
    for run_id, run_frame in frame.groupby("run", sort=True):
        fig, axes = panel_axes(len(metrics), width=6.4, height=4.2)
        for ax, (metric, label) in zip(axes.flat, metrics.items()):
            for method_label, method_frame in run_frame.groupby("method", sort=False):
                method_key = next((key for key, value in METHOD_NAMES.items() if value == method_label), method_label)
                if chainwise:
                    chain_id = sampled_chain_id(cfg, int(run_id), method_key, method_frame["original_chain"].unique())
                    chain_frame = method_frame[method_frame["original_chain"] == chain_id].sort_values(x)
                    ax.plot(chain_frame[x], chain_frame[metric], color=method_color(method_key),
                            linewidth=1.7, label=f"{method_label} | chain {chain_id}")
                else:
                    method_frame = method_frame.sort_values(x)
                    ax.plot(method_frame[x], method_frame[metric], color=method_color(method_key),
                            linewidth=2, label=method_label)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel(x.replace("_", " "))
        first_ax = next(ax for ax in axes.flat if ax.get_visible())
        handles, labels = first_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
        fig.suptitle(f"Run {int(run_id):03d}: {title}")
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        save_figure(fig, cfg, f"r{int(run_id):03d}_{filename_key}.png")


def plot_rhat_bands(frame: pd.DataFrame, cfg: Config, filename_key: str, title: str,
                    ylabel: str, chainwise: bool = False, x: str = "window_start") -> None:
    """Plot mean +/- SD and median/IQR summaries across prediction dimensions."""
    if frame.empty:
        return
    for run_id, run_frame in frame.groupby("run", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
        for method_label, method_frame in run_frame.groupby("method", sort=False):
            method_key = next((key for key, value in METHOD_NAMES.items() if value == method_label), method_label)
            legend_label = method_label
            if chainwise:
                chain_id = sampled_chain_id(
                    cfg, int(run_id), method_key, method_frame["original_chain"].unique()
                )
                method_frame = method_frame[method_frame["original_chain"] == chain_id]
                legend_label = f"{method_label} | chain {chain_id}"
            method_frame = method_frame.sort_values(x)
            x_values = method_frame[x].to_numpy()
            color = method_color(method_key)

            mean = method_frame["mean"].to_numpy()
            std = method_frame["std"].to_numpy()
            axes[0].plot(x_values, mean, color=color, linewidth=2, label=legend_label)
            axes[0].fill_between(x_values, mean - std, mean + std, color=color, alpha=0.18)

            median = method_frame["median"].to_numpy()
            q25 = method_frame["q25"].to_numpy()
            q75 = method_frame["q75"].to_numpy()
            axes[1].plot(x_values, median, color=color, linewidth=2, label=legend_label)
            axes[1].fill_between(x_values, q25, q75, color=color, alpha=0.18)

        axes[0].set_title("Mean ± SD")
        axes[1].set_title("Median with IQR")
        for ax in axes:
            ax.set_xlabel(x.replace("_", " "))
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
        fig.suptitle(f"Run {int(run_id):03d}: {title}")
        fig.tight_layout(rect=[0, 0.12, 1, 0.92])
        save_figure(fig, cfg, f"r{int(run_id):03d}_{filename_key}.png")


def analyze_rhat(store: Store, cfg: Config, short_methods: list[str], short_runs: list[int], long_runs: list[int]) -> None:
    print_metric("rolling R-hat and MPSRF")
    long_rows = []
    for run_id in long_runs:
        draws = store.predictions(run_id, LONG_METHOD, 0)
        if draws.shape[0] < 2:
            continue
        for start in range(0, draws.shape[1] - cfg.window + 1, cfg.step):
            long_rows.append({"run": run_id, "window_start": start,
                              "multivariate_rhat": multivariate_rhat(draws[:, start:start + cfg.window])})
    long_frame = pd.DataFrame(long_rows)
    reference = None
    if not long_frame.empty:
        save_table(long_frame, cfg, "rhat_long_mpsrf.csv")
        reference = float(long_frame["multivariate_rhat"].median())
        fig, ax = plt.subplots(figsize=(9, 5))
        for run_id, run_frame in long_frame.groupby("run"):
            ax.plot(run_frame["window_start"], run_frame["multivariate_rhat"], label=f"Run {run_id:03d}")
        ax.axhline(reference, color="black", linestyle="--", linewidth=1, label=f"median = {reference:.4f}")
        ax.set(title="Default long: rolling MPSRF", xlabel="window start", ylabel="MPSRF")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        save_figure(fig, cfg, "rhat_long_mpsrf.png")

    if not short_runs or not short_methods:
        return

    mpsrf_rows, point_rows = [], []
    for run_id in short_runs:
        for method in available_methods_for_run(store, short_methods, run_id):
            draws = store.predictions(run_id, method, 0)
            if draws.shape[0] >= 2:
                for start in range(0, draws.shape[1] - cfg.window + 1, cfg.step):
                    window = draws[:, start:start + cfg.window]
                    mpsrf_rows.append({"run": run_id, "method": method_name(method), "window_start": start,
                                        "multivariate_rhat": multivariate_rhat(window)})
                    data = xr.DataArray(window, dims=("chain", "draw", "test_point"), name="prediction")
                    values = az.rhat(data, method="rank")["prediction"].values
                    point_rows.append({"run": run_id, "method": method_name(method), "window_start": start,
                                       "mean": np.nanmean(values), "std": np.nanstd(values, ddof=1),
                                       "median": np.nanmedian(values), "q25": np.nanquantile(values, 0.25),
                                       "q75": np.nanquantile(values, 0.75)})

    mpsrf_frame = save_table(mpsrf_rows, cfg, "rhat_cross_mpsrf.csv")
    if not mpsrf_frame.empty:
        for run_id, run_frame in mpsrf_frame.groupby("run"):
            fig, ax = plt.subplots(figsize=(9, 5))
            for method_label, method_frame in run_frame.groupby("method", sort=False):
                key = next((k for k, v in METHOD_NAMES.items() if v == method_label), method_label)
                ax.plot(method_frame["window_start"], method_frame["multivariate_rhat"],
                        color=method_color(key), linewidth=2, label=method_label)
            if reference is not None:
                ax.axhline(reference, color="black", linestyle="--", linewidth=1,
                           label=f"long median = {reference:.4f}")
            ax.set(title=f"Run {run_id:03d}: cross-chain MPSRF", xlabel="window start", ylabel="MPSRF")
            ax.grid(alpha=0.25)
            finish_method_legend(fig, ax, ncol=5, bottom=0.12)
            save_figure(fig, cfg, f"r{run_id:03d}_rhat_mpsrf.png")
    point_frame = save_table(point_rows, cfg, "rhat_cross_pred.csv")
    plot_rhat_bands(point_frame, cfg, "rhat_pred",
                    "cross-chain rank-split R-hat across test points", "rank-split R-hat")

    within_rows, within_mpsrf_rows = [], []
    for run_id in short_runs:
        for method in available_methods_for_run(store, short_methods, run_id):
            draws = store.predictions(run_id, method, 0)
            for start in range(0, draws.shape[1] - cfg.segment_block + 1, cfg.step):
                block = draws[:, start:start + cfg.segment_block]
                segments = block.reshape(draws.shape[0], cfg.n_segments, cfg.segment_length, draws.shape[2])
                pseudo = segments.transpose(1, 2, 0, 3)
                data = xr.DataArray(pseudo, dims=("chain", "draw", "original_chain", "test_point"), name="prediction")
                values = az.rhat(data, method="rank")["prediction"].values
                for chain_id in range(draws.shape[0]):
                    chain_values = values[chain_id]
                    within_rows.append({"run": run_id, "method": method_name(method), "original_chain": chain_id,
                                        "window_end": start + cfg.segment_block,
                                        "mean": np.nanmean(chain_values), "std": np.nanstd(chain_values, ddof=1),
                                        "median": np.nanmedian(chain_values),
                                        "q25": np.nanquantile(chain_values, 0.25),
                                        "q75": np.nanquantile(chain_values, 0.75)})
                    within_mpsrf_rows.append({"run": run_id, "method": method_name(method), "original_chain": chain_id,
                                               "window_end": start + cfg.segment_block,
                                               "mpsrf": multivariate_rhat(segments[chain_id])})
    within_frame = save_table(within_rows, cfg, "rhat_within_pred.csv")
    plot_rhat_bands(within_frame, cfg, "wrhat_pred",
                    "within-chain segment R-hat across test points", "rank-split R-hat",
                    chainwise=True, x="window_end")
    within_mpsrf_frame = save_table(within_mpsrf_rows, cfg, "rhat_within_mpsrf.csv")
    plot_method_panels(within_mpsrf_frame, {"mpsrf": "100D MPSRF"}, cfg, "wrhat_mpsrf",
                       "within-chain segment MPSRF", "MPSRF", chainwise=True, x="window_end")


def worst_direction_projection(draws: np.ndarray, ridge_fraction: float = 1e-8) -> dict[str, np.ndarray | float]:
    """Project (chain, draw, feature) samples onto the MPSRF worst direction."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3 or not np.all(np.isfinite(draws)):
        raise ValueError("Expected finite draws with shape (chain, draw, feature).")
    m, n, p = draws.shape
    if m < 2 or n < 2:
        raise ValueError("At least two chains and two draws per chain are required.")

    means = draws.mean(axis=1)
    grand_mean = means.mean(axis=0)
    centered = draws - means[:, None, :]
    within = np.einsum("mnp,mnq->pq", centered, centered, optimize=True) / (m * (n - 1))
    offsets = means - grand_mean
    between_over_n = (offsets.T @ offsets) / (m - 1)
    ridge = ridge_fraction * np.trace(within) / p
    within_regularized = within + max(float(ridge), ridge_fraction) * np.eye(p)

    eigenvalues, eigenvectors = eigh(between_over_n, within_regularized, check_finite=True)
    directions = eigenvectors[:, -2:][:, ::-1].copy()
    for column in range(directions.shape[1]):
        directions[:, column] /= np.linalg.norm(directions[:, column])
        anchor = int(np.argmax(np.abs(directions[:, column])))
        if directions[anchor, column] < 0:
            directions[:, column] *= -1

    lambda_max = max(0.0, float(eigenvalues[-1]))
    lambda_second = max(0.0, float(eigenvalues[-2])) if p > 1 else np.nan
    projected_2d = np.einsum("mnp,pk->mnk", draws - grand_mean, directions)
    projected_rhats = []
    for column in range(projected_2d.shape[2]):
        projected_data = xr.DataArray(projected_2d[:, :, column], dims=("chain", "draw"), name="projection")
        projected_rhats.append(float(az.rhat(projected_data, method="rank")["projection"].values))
    mpsrf = np.sqrt((n - 1) / n + ((m + 1) / m) * lambda_max)
    return {
        "direction": directions[:, 0],
        "directions": directions,
        "projected": projected_2d[:, :, 0],
        "projected_2d": projected_2d,
        "lambda_max": lambda_max,
        "lambda_second": lambda_second,
        "eigen_gap": lambda_max - lambda_second,
        "mpsrf": float(mpsrf),
        "projected_rhat": projected_rhats[0],
        "second_projected_rhat": projected_rhats[1],
    }


def analyze_worst_direction(store: Store, cfg: Config, short_methods: list[str], short_runs: list[int]) -> None:
    print_metric(f"post-{cfg.eigen_projection_start} MPSRF worst-direction projection")
    summary_rows, direction_rows = [], []

    for run_id in short_runs:
        methods = available_methods_for_run(store, short_methods, run_id)
        if not methods:
            continue
        fig, axes = plt.subplots(len(methods), 2, figsize=(14, 2.8 * len(methods)), squeeze=False)

        for row_id, method in enumerate(methods):
            all_draws = store.predictions(run_id, method, 0)
            if all_draws.shape[1] <= cfg.eigen_projection_start:
                raise ValueError(
                    f"Run {run_id:03d}, {method} has only {all_draws.shape[1]} iterations."
                )
            draws = all_draws[:, cfg.eigen_projection_start:, :]
            result = worst_direction_projection(draws)
            summary_rows.append({
                "run": run_id,
                "method": method_name(method),
                "lambda_max": result["lambda_max"],
                "lambda_second": result["lambda_second"],
                "eigen_gap": result["eigen_gap"],
                "mpsrf": result["mpsrf"],
                "projected_rhat": result["projected_rhat"],
                "second_projected_rhat": result["second_projected_rhat"],
            })
            for direction_id in range(result["directions"].shape[1]):
                for point_id, coefficient in enumerate(result["directions"][:, direction_id]):
                    direction_rows.append({
                        "run": run_id,
                        "method": method_name(method),
                        "direction": direction_id + 1,
                        "test_point": point_id,
                        "coefficient": coefficient,
                    })

            trace_ax, density_ax = axes[row_id]
            projected = result["projected"]
            iterations = np.arange(cfg.eigen_projection_start, all_draws.shape[1])
            density_grid = np.linspace(np.min(projected), np.max(projected), 300)
            for chain_id in range(projected.shape[0]):
                color = CHAIN_COLORS[chain_id % len(CHAIN_COLORS)]
                label = f"chain {chain_id}"
                trace_ax.plot(iterations, projected[chain_id], color=color, linewidth=0.7,
                              alpha=0.75, label=label)
                if np.ptp(projected[chain_id]) > 0:
                    density = gaussian_kde(projected[chain_id])
                    density_ax.plot(density_grid, density(density_grid), color=color,
                                    linewidth=1.8, label=label)

            label = method_name(method)
            trace_ax.set_title(
                f"{label}: trace | λmax={result['lambda_max']:.4g}, R-hat={result['projected_rhat']:.4f}"
            )
            density_ax.set_title(f"{label}: density")
            trace_ax.set(xlabel="iteration", ylabel="worst-direction projection")
            density_ax.set(xlabel="worst-direction projection", ylabel="density")
            trace_ax.grid(alpha=0.2)
            density_ax.grid(alpha=0.2)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False)
        fig.suptitle(f"Run {run_id:03d}: post-{cfg.eigen_projection_start} MPSRF worst direction")
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        save_figure(fig, cfg, f"r{run_id:03d}_worst_direction.png")

    summary = save_table(summary_rows, cfg, "worst_direction_summary.csv")
    save_table(direction_rows, cfg, "worst_direction_eigenvectors.csv")
    if summary.empty:
        return

    baseline = method_name("default")
    lambda_pivot = summary.pivot(index="run", columns="method", values="lambda_max")
    comparison_rows = []
    for method in short_methods:
        label = method_name(method)
        method_frame = summary[summary["method"] == label]
        if method_frame.empty:
            continue
        lambda_values = method_frame["lambda_max"].to_numpy()
        rhat_values = method_frame["projected_rhat"].to_numpy()
        paired_delta = (lambda_pivot[label] - lambda_pivot[baseline]).dropna()
        comparison_rows.append({
            "method": label,
            "lambda_median": np.median(lambda_values),
            "lambda_q25": np.quantile(lambda_values, 0.25),
            "lambda_q75": np.quantile(lambda_values, 0.75),
            "projected_rhat_median": np.median(rhat_values),
            "projected_rhat_q25": np.quantile(rhat_values, 0.25),
            "projected_rhat_q75": np.quantile(rhat_values, 0.75),
            "median_lambda_delta_vs_default": np.median(paired_delta),
        })
    comparison = save_table(comparison_rows, cfg, "worst_direction_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(comparison))
    labels = comparison["method"].tolist()
    for ax, metric, q25, q75, title in (
        (axes[0], "lambda_median", "lambda_q25", "lambda_q75", "Maximum generalized eigenvalue"),
        (axes[1], "projected_rhat_median", "projected_rhat_q25", "projected_rhat_q75", "Projected rank-split R-hat"),
    ):
        center = comparison[metric].to_numpy()
        lower = comparison[q25].to_numpy()
        upper = comparison[q75].to_numpy()
        ax.errorbar(positions, center, yerr=np.vstack((center - lower, upper - center)),
                    fmt="o", color="#333333", ecolor="#777777", capsize=5, markersize=7)
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_title(f"{title}: median with IQR")
        ax.grid(axis="y", alpha=0.25)
    axes[1].axhline(1.01, color="black", linestyle="--", linewidth=1, label="1.01")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, cfg, "worst_direction_comparison.png")


def analyze_ess(store: Store, cfg: Config, short_methods: list[str], short_runs: list[int]) -> None:
    print_metric("per-chain rolling relative ESS")
    pred_rows = []
    for run_id in short_runs:
        for method in available_methods_for_run(store, short_methods, run_id):
            draws = store.predictions(run_id, method, 0)
            for start in range(0, draws.shape[1] - cfg.window + 1, cfg.step):
                for chain_id in range(draws.shape[0]):
                    data = xr.DataArray(draws[chain_id:chain_id + 1, start:start + cfg.window],
                                        dims=("chain", "draw", "test_point"), name="prediction")
                    bulk = az.ess(data, method="bulk")["prediction"].values / cfg.window
                    tail = az.ess(data, method="tail")["prediction"].values / cfg.window
                    pred_rows.append({"run": run_id, "method": method_name(method), "original_chain": chain_id,
                                      "window_start": start, "bulk_median": np.nanmedian(bulk),
                                      "bulk_q05": np.nanquantile(bulk, 0.05), "tail_median": np.nanmedian(tail),
                                      "tail_q05": np.nanquantile(tail, 0.05)})
    pred_frame = save_table(pred_rows, cfg, "ess_pred.csv")
    plot_method_panels(pred_frame, {"bulk_median": "Bulk relative ESS: median", "bulk_q05": "Bulk relative ESS: 5% quantile",
                                    "tail_median": "Tail relative ESS: median", "tail_q05": "Tail relative ESS: 5% quantile"},
                       cfg, "ess_pred", "per-chain relative ESS across test points", "ESS / window length", chainwise=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every fixed-100 diagnostic for a named stored dataset.")
    parser.add_argument("dataset", help="Dataset/store name, e.g. fixed100_Abalone")
    parser.add_argument("--output-root", type=Path, help="Parent output directory (default: results beside this script)")
    parser.add_argument("--store-root", type=Path, help="Store root override")
    parser.add_argument("--only", help=f"Comma-separated groups; choices: {', '.join(ALL_GROUPS)}")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    fixed_root = script_dir.parent
    store_root = (args.store_root or fixed_root / "store").resolve()
    dataset_name = Path(args.dataset).name
    store_dir = locate_store(dataset_name, store_root)
    output_root = (args.output_root or script_dir / "results").resolve()
    output_dir = output_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)

    if store_dir is None:
        print("Unable to find long-chain or short-chain data.")
        return 1

    cfg = Config(dataset_name, store_dir.name, store_dir, output_dir, window=args.window, step=args.step)
    store = Store(cfg)
    long_runs = store.runs(LONG_METHOD)
    short_methods = short_methods_with_data(store)
    short_runs = short_run_union(store, short_methods)

    if not long_runs and not short_runs:
        print("Unable to find long-chain or short-chain data.")
        return 1
    if not long_runs:
        print("No long-chain data found; skipping long-chain analyses.")
    if not short_runs:
        print("No short-chain data found; skipping short-chain analyses.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {dataset_name}")
    print(f"Data tag: {cfg.data_tag}")
    print(f"Store: {store_dir}")
    print(f"Output: {output_dir}")
    print(f"Long runs: {long_runs}")
    print(f"Short methods: {short_methods}")
    print(f"Short runs: {short_runs}")

    groups = set(ALL_GROUPS)
    if args.only:
        groups = {item.strip() for item in args.only.split(",") if item.strip()}
        unknown = groups - set(ALL_GROUPS)
        if unknown:
            raise ValueError(f"Unknown analysis group(s): {sorted(unknown)}")

    if long_runs and "long" in groups:
        analyze_long_behavior(store, cfg, long_runs)
    if short_runs and "separation" in groups:
        analyze_chain_separation(store, cfg, short_methods, short_runs)
    if (short_runs or long_runs) and "predictive" in groups:
        analyze_predictive(store, cfg, short_methods, short_runs, long_runs)
    if (short_runs or long_runs) and "rhat" in groups:
        analyze_rhat(store, cfg, short_methods, short_runs, long_runs)
        if short_runs:
            analyze_worst_direction(store, cfg, short_methods, short_runs)
    if short_runs and "ess" in groups:
        analyze_ess(store, cfg, short_methods, short_runs)

    print(f"\nAnalysis complete. Results saved to: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
