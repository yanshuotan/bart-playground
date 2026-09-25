#!/usr/bin/env python
"""Compare sampler autocorrelation on PC1 axes fitted to long default chains.

Within each paired run, PCA is fitted once to the pooled long-default
prediction draws.  Default, Default+PT, MTMH, and MTMH+PT are then projected
onto that common PC1.  The output figure contains only this diagnostic:

1. median PC1 ACF with an interquartile band across runs and chains;
2. paired run-level median integrated autocorrelation times.

Chain-level ACF, IAT, and ESS-proxy values are saved as CSV files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from plot_section3_default_diagnostics import initial_positive_tau, normalized_acf
from run_all_analyses import (
    Config,
    LONG_METHOD,
    METHOD_COLORS,
    METHOD_NAMES,
    SHORT_METHOD_ORDER,
    Store,
    locate_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four BART samplers using long-chain PC1 autocorrelation."
    )
    parser.add_argument("dataset", help="Stored dataset name, e.g. fixed100_Abalone")
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--analysis-start", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=10)
    parser.add_argument("--plot-max-lag", type=int, default=500)
    parser.add_argument("--iat-max-lag", type=int, default=3000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def common_runs(store: Store) -> list[int]:
    run_sets = [set(store.runs(LONG_METHOD))]
    run_sets.extend(set(store.runs(method)) for method in SHORT_METHOD_ORDER)
    return sorted(set.intersection(*run_sets)) if run_sets else []


def analyze(
    store: Store,
    runs: list[int],
    analysis_start: int,
    long_burn: int,
    plot_max_lag: int,
    iat_max_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chain_rows: list[dict[str, float | int | str]] = []
    acf_rows: list[dict[str, float | int | str]] = []
    calculation_max_lag = max(plot_max_lag, iat_max_lag)

    for run_id in runs:
        long_draws = store.predictions(run_id, LONG_METHOD, burn=long_burn)
        pca = PCA(n_components=2, random_state=0).fit(
            long_draws.reshape(-1, long_draws.shape[-1])
        )

        for method in SHORT_METHOD_ORDER:
            draws = store.predictions(run_id, method, burn=analysis_start)
            coordinates = pca.transform(
                draws.reshape(-1, draws.shape[-1])
            ).reshape(draws.shape[0], draws.shape[1], 2)
            pc1 = coordinates[:, :, 0]

            for chain_id, values in enumerate(pc1):
                acf = normalized_acf(values, calculation_max_lag)
                iat = initial_positive_tau(acf[: iat_max_lag + 1])
                chain_rows.append(
                    {
                        "run": run_id,
                        "method": METHOD_NAMES[method],
                        "chain": chain_id,
                        "draws": values.size,
                        "iat": iat,
                        "ess_proxy": values.size / iat,
                        "acf_lag_10": acf[10],
                        "acf_lag_50": acf[50],
                        "acf_lag_100": acf[100],
                        "acf_lag_200": acf[200],
                        "acf_lag_500": acf[500],
                    }
                )
                acf_rows.extend(
                    {
                        "run": run_id,
                        "method": METHOD_NAMES[method],
                        "chain": chain_id,
                        "lag": lag,
                        "acf": float(acf[lag]),
                    }
                    for lag in range(plot_max_lag + 1)
                )

    return pd.DataFrame(chain_rows), pd.DataFrame(acf_rows)


def summarize(chain_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    baseline = (
        chain_frame[chain_frame["method"] == METHOD_NAMES["default"]]
        .groupby("run")["iat"]
        .median()
    )
    for method in SHORT_METHOD_ORDER:
        label = METHOD_NAMES[method]
        values = chain_frame[chain_frame["method"] == label]
        run_medians = values.groupby("run")["iat"].median()
        ratios = (run_medians / baseline).dropna()
        rows.append(
            {
                "method": label,
                "iat_median": values["iat"].median(),
                "iat_q25": values["iat"].quantile(0.25),
                "iat_q75": values["iat"].quantile(0.75),
                "ess_proxy_median": values["ess_proxy"].median(),
                "paired_run_ratio_median": ratios.median(),
                "paired_run_wins_vs_default": int((ratios < 1).sum()),
                "paired_runs": int(ratios.size),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(
    chain_frame: pd.DataFrame,
    acf_frame: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    acf_ax, iat_ax = axes

    for method in SHORT_METHOD_ORDER:
        label = METHOD_NAMES[method]
        color = METHOD_COLORS[method]
        values = acf_frame[acf_frame["method"] == label]
        by_lag = values.groupby("lag")["acf"]
        median = by_lag.median()
        q25 = by_lag.quantile(0.25)
        q75 = by_lag.quantile(0.75)
        lags = median.index.to_numpy()
        acf_ax.plot(lags, median.to_numpy(), color=color, linewidth=1.8, label=label)
        acf_ax.fill_between(
            lags,
            q25.to_numpy(),
            q75.to_numpy(),
            color=color,
            alpha=0.12,
        )
    acf_ax.axhline(0, color="#777777", linewidth=0.8)
    acf_ax.set(
        title="PC1 autocorrelation: median with chain/run IQR",
        xlabel="lag",
        ylabel="autocorrelation",
    )
    acf_ax.legend(fontsize=8)
    acf_ax.grid(alpha=0.2)

    labels = [METHOD_NAMES[method] for method in SHORT_METHOD_ORDER]
    positions = np.arange(len(labels))
    run_frame = (
        chain_frame.groupby(["run", "method"], as_index=False)["iat"]
        .median()
        .pivot(index="run", columns="method", values="iat")
    )
    for run_id, row in run_frame.iterrows():
        values = np.asarray([row[label] for label in labels], dtype=float)
        iat_ax.plot(positions, values, color="#aaaaaa", linewidth=0.9, alpha=0.75)
    for position, method in enumerate(SHORT_METHOD_ORDER):
        label = METHOD_NAMES[method]
        values = run_frame[label].dropna().to_numpy()
        iat_ax.scatter(
            np.full(values.size, position),
            values,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            s=45,
            zorder=3,
        )
    iat_ax.set_xticks(positions, labels, rotation=18, ha="right")
    iat_ax.set_yscale("log")
    iat_ax.set(
        title="Paired run median PC1 integrated autocorrelation time",
        ylabel="IAT (log scale)",
    )
    iat_ax.grid(axis="y", alpha=0.2)
    return fig


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
    store = Store(Config(dataset_name, store_dir.name, store_dir, output_dir))
    runs = common_runs(store)
    if not runs:
        raise ValueError("No run has long-default and all four short-method predictions.")

    chain_frame, acf_frame = analyze(
        store=store,
        runs=runs,
        analysis_start=args.analysis_start,
        long_burn=args.long_burn,
        plot_max_lag=args.plot_max_lag,
        iat_max_lag=args.iat_max_lag,
    )
    summary_frame = summarize(chain_frame)

    stem = "pc1_autocorrelation_comparison"
    chain_frame.to_csv(output_dir / f"{stem}_by_chain.csv", index=False)
    acf_frame.to_csv(output_dir / f"{stem}_by_lag.csv", index=False)
    summary_frame.to_csv(output_dir / f"{stem}_summary.csv", index=False)

    figure_output = (
        args.figure_output.resolve()
        if args.figure_output
        else output_dir / f"{stem}.png"
    )
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_comparison(chain_frame, acf_frame)
    figure.savefig(figure_output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"Runs: {runs}")
    print(summary_frame.to_string(index=False))
    print(f"Figure: {figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
