#!/usr/bin/env python
"""Build the default-sampler diagnostic figure proposed for paper Section 3.

The four panels use only the default sampler:

1. traces along the maximum between-chain / within-chain prediction direction;
2. rolling pointwise rank-split R-hat, summarized across fixed test inputs;
3. rolling within-chain segment R-hat for each original chain;
4. default short-chain predictions projected onto PCA axes fitted to long
   default-chain predictions.

The script reads the stored prediction arrays directly.  It also writes the
cross-chain R-hat, within-chain segment R-hat, and PCA centroid values used in
the figure so that the paper plot can be audited without reading pixels.

Example
-------
python diagnosis/exploratory/plot_section3_default_diagnostics.py fixed100_Abalone

To generate the paper figure directly:

python diagnosis/exploratory/plot_section3_default_diagnostics.py fixed100_Abalone ^
  --run-id 0 ^
  --figure-output C:/Users/ztykk/Phd/BART/paper/figures/outline_example/section3_default_diagnostic.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

from run_all_analyses import (
    CHAIN_COLORS,
    Config,
    LONG_METHOD,
    Store,
    locate_store,
    worst_direction_projection,
)


def rolling_pointwise_rhat(
    draws: np.ndarray,
    window: int,
    step: int,
) -> pd.DataFrame:
    """Compute rank-split R-hat at every test input and summarize each window."""
    rows: list[dict[str, float | int]] = []
    for start in range(0, draws.shape[1] - window + 1, step):
        block = draws[:, start : start + window]
        data = xr.DataArray(
            block,
            dims=("chain", "draw", "test_point"),
            name="prediction",
        )
        values = np.asarray(az.rhat(data, method="rank")["prediction"].values)
        rows.append(
            {
                "window_start": start,
                "window_end": start + window,
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanquantile(values, 0.25)),
                "q75": float(np.nanquantile(values, 0.75)),
                "q90": float(np.nanquantile(values, 0.90)),
                "maximum": float(np.nanmax(values)),
                "fraction_above_1_01": float(np.nanmean(values > 1.01)),
            }
        )
    return pd.DataFrame(rows)


def rolling_within_chain_rhat(
    draws: np.ndarray,
    segment_length: int,
    n_segments: int,
    step: int,
) -> pd.DataFrame:
    """Compare consecutive segments within each original chain using R-hat."""
    rows: list[dict[str, float | int]] = []
    block_length = segment_length * n_segments
    for start in range(0, draws.shape[1] - block_length + 1, step):
        block = draws[:, start : start + block_length]
        segments = block.reshape(
            draws.shape[0],
            n_segments,
            segment_length,
            draws.shape[2],
        )
        pseudo_chains = segments.transpose(1, 2, 0, 3)
        data = xr.DataArray(
            pseudo_chains,
            dims=("chain", "draw", "original_chain", "test_point"),
            name="prediction",
        )
        values = np.asarray(az.rhat(data, method="rank")["prediction"].values)
        for chain_id, chain_values in enumerate(values):
            rows.append(
                {
                    "original_chain": chain_id,
                    "window_start": start,
                    "window_end": start + block_length,
                    "median": float(np.nanmedian(chain_values)),
                    "q25": float(np.nanquantile(chain_values, 0.25)),
                    "q75": float(np.nanquantile(chain_values, 0.75)),
                    "q90": float(np.nanquantile(chain_values, 0.90)),
                    "maximum": float(np.nanmax(chain_values)),
                    "fraction_above_1_01": float(np.nanmean(chain_values > 1.01)),
                }
            )
    return pd.DataFrame(rows)


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


def build_figure(
    short_draws: np.ndarray,
    short_pca_draws: np.ndarray,
    long_draws: np.ndarray,
    projection_start: int,
    ridge_fraction: float,
    window: int,
    step: int,
    segment_length: int,
    n_segments: int,
    plot_draws: int,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if short_draws.shape[1] <= projection_start:
        raise ValueError(
            f"projection-start={projection_start} requires more than "
            f"{short_draws.shape[1]} short-chain draws."
        )

    projection_result = worst_direction_projection(
        short_draws[:, projection_start:, :],
        ridge_fraction=ridge_fraction,
    )
    projected = np.asarray(projection_result["projected"])
    iterations = np.arange(projection_start, short_draws.shape[1])

    rhat_frame = rolling_pointwise_rhat(short_draws, window=window, step=step)
    within_rhat_frame = rolling_within_chain_rhat(
        short_draws,
        segment_length=segment_length,
        n_segments=n_segments,
        step=step,
    )

    pca = PCA(n_components=2, random_state=0)
    pca.fit(long_draws.reshape(-1, long_draws.shape[-1]))
    long_coords = pca.transform(long_draws.reshape(-1, long_draws.shape[-1])).reshape(
        long_draws.shape[0], long_draws.shape[1], 2
    )
    short_coords = pca.transform(
        short_pca_draws.reshape(-1, short_pca_draws.shape[-1])
    ).reshape(
        short_pca_draws.shape[0], short_pca_draws.shape[1], 2
    )
    centroid_rows: list[dict[str, float | int | str]] = []
    for source, coords in (("long_default", long_coords), ("default", short_coords)):
        for chain_id in range(coords.shape[0]):
            center = coords[chain_id].mean(axis=0)
            centroid_rows.append(
                {
                    "source": source,
                    "chain": chain_id,
                    "pc1": float(center[0]),
                    "pc2": float(center[1]),
                }
            )
    centroid_frame = pd.DataFrame(centroid_rows)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), layout="constrained")
    trace_ax, rhat_ax, within_ax, pca_ax = axes.flat

    for chain_id in range(projected.shape[0]):
        color = CHAIN_COLORS[chain_id % len(CHAIN_COLORS)]
        trace_ax.plot(
            iterations,
            projected[chain_id],
            color=color,
            linewidth=0.65,
            alpha=0.8,
            label=f"chain {chain_id + 1}",
        )
    trace_ax.set(
        title="(a) Default chains: maximum between/within direction",
        xlabel="iteration",
        ylabel="projected prediction",
    )
    trace_ax.legend(ncol=2, fontsize=8)

    for chain_id, chain_frame in within_rhat_frame.groupby("original_chain"):
        chain_frame = chain_frame.sort_values("window_end")
        color = CHAIN_COLORS[chain_id % len(CHAIN_COLORS)]
        x_values = chain_frame["window_end"].to_numpy()
        within_ax.plot(
            x_values,
            chain_frame["median"].to_numpy(),
            color=color,
            linewidth=1.5,
            label=f"chain {chain_id + 1}",
        )
        within_ax.fill_between(
            x_values,
            chain_frame["q25"].to_numpy(),
            chain_frame["q75"].to_numpy(),
            color=color,
            alpha=0.08,
        )
    within_ax.axhline(1.01, color="#555555", linestyle="--", linewidth=1.0, label="1.01")
    within_ax.set(
        title="(c) Within-chain segment rank-split R-hat",
        xlabel="window end",
        ylabel="R-hat",
    )
    within_ax.legend(ncol=3, fontsize=8)

    x = rhat_frame["window_end"].to_numpy()
    median = rhat_frame["median"].to_numpy()
    q25 = rhat_frame["q25"].to_numpy()
    q75 = rhat_frame["q75"].to_numpy()
    rhat_ax.plot(x, median, color="#245f9e", linewidth=1.8, label="median")
    rhat_ax.fill_between(x, q25, q75, color="#245f9e", alpha=0.2, label="test-point IQR")
    rhat_ax.axhline(1.01, color="#555555", linestyle="--", linewidth=1.0, label="1.01")
    rhat_ax.set(
        title="(b) Rolling pointwise rank-split R-hat",
        xlabel="window end",
        ylabel="R-hat",
    )
    rhat_ax.legend(fontsize=8)

    long_index = evenly_spaced_indices(long_coords.shape[1], plot_draws)
    for chain_id in range(long_coords.shape[0]):
        pca_ax.scatter(
            long_coords[chain_id, long_index, 0],
            long_coords[chain_id, long_index, 1],
            s=8,
            alpha=0.07,
            color="#4a4a4a",
            edgecolors="none",
            label="long default reference" if chain_id == 0 else "_nolegend_",
        )

    short_index = evenly_spaced_indices(short_coords.shape[1], plot_draws)
    for chain_id in range(short_coords.shape[0]):
        color = CHAIN_COLORS[chain_id % len(CHAIN_COLORS)]
        pca_ax.scatter(
            short_coords[chain_id, short_index, 0],
            short_coords[chain_id, short_index, 1],
            s=10,
            alpha=0.18,
            color=color,
            edgecolors="none",
            label=f"default chain {chain_id + 1}",
        )
        center = short_coords[chain_id].mean(axis=0)
        pca_ax.scatter(
            center[0],
            center[1],
            marker="X",
            s=85,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
    pca_ax.set(
        title="(d) Default short chains on long-chain PCA axes",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    pca_ax.legend(ncol=2, fontsize=7.5)

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    return fig, rhat_frame, within_rhat_frame, centroid_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Section 3 diagnostics for the default BART sampler."
    )
    parser.add_argument("dataset", help="Stored dataset name, e.g. fixed100_Abalone")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--projection-start", type=int, default=3000)
    parser.add_argument("--short-burn", type=int, default=500)
    parser.add_argument("--long-burn", type=int, default=10)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=1000)
    parser.add_argument("--n-segments", type=int, default=4)
    parser.add_argument("--plot-draws", type=int, default=1000)
    parser.add_argument("--ridge-fraction", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    diagnosis_dir = script_dir.parent
    store_root = (args.store_root or diagnosis_dir / "store").resolve()
    store_dir = locate_store(args.dataset, store_root)
    if store_dir is None:
        raise FileNotFoundError(f"Unable to locate stored dataset: {args.dataset}")

    dataset_name = Path(args.dataset).name
    output_root = (args.output_root or script_dir / "results").resolve()
    output_dir = output_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        dataset_name=dataset_name,
        data_tag=store_dir.name,
        store_dir=store_dir,
        output_dir=output_dir,
    )
    store = Store(cfg)
    if not store.has("preds", args.run_id, "default"):
        raise FileNotFoundError(f"No default predictions for run {args.run_id:03d}.")
    if not store.has("preds", args.run_id, LONG_METHOD):
        raise FileNotFoundError(f"No long default predictions for run {args.run_id:03d}.")

    short_all = store.predictions(args.run_id, "default", burn=0)
    short_for_pca = store.predictions(args.run_id, "default", burn=args.short_burn)
    long_for_pca = store.predictions(args.run_id, LONG_METHOD, burn=args.long_burn)

    figure, rhat_frame, within_rhat_frame, centroid_frame = build_figure(
        short_draws=short_all,
        short_pca_draws=short_for_pca,
        long_draws=long_for_pca,
        projection_start=args.projection_start,
        ridge_fraction=args.ridge_fraction,
        window=args.window,
        step=args.step,
        segment_length=args.segment_length,
        n_segments=args.n_segments,
        plot_draws=args.plot_draws,
    )

    stem = f"section3_default_diagnostics_r{args.run_id:03d}"
    figure_output = (
        args.figure_output.resolve()
        if args.figure_output
        else output_dir / f"{stem}.png"
    )
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    rhat_frame.to_csv(output_dir / f"{stem}_rhat.csv", index=False)
    within_rhat_frame.to_csv(
        output_dir / f"{stem}_within_chain_rhat.csv",
        index=False,
    )
    centroid_frame.to_csv(output_dir / f"{stem}_pca_centroids.csv", index=False)

    print(f"Figure: {figure_output}")
    print(f"Supporting tables: {output_dir / stem}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
