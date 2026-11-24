#!/usr/bin/env python3
"""Frequentist coverage of the latent function over independent Friedman training sets.

Reads the output of ``coverage_replicates.py`` (``store_coverage/cov_Friedman``).
Each replicate has a fresh 1,425-row Friedman #1 training set, the 100 inputs
of the main Friedman panel with the true f stored as the test response, and
four chains of 10,000 iterations for each of Default, MTMH, Default+PT and
MTMH+PT.

For each replicate and method, after dropping 3,000 warm-up draws, a central
90% interval is formed at every input from each chain alone and from the
equal-weight mixture of the four chains, as in ``section7_costs.py``. The
indicator that the interval contains f, averaged over replicates, is the
frequentist coverage at that input; averaged over inputs as well, it is the
overall coverage.

Writes to the table directory:

* ``coverage_replicates_runs.csv``: one row per replicate and method;
* ``coverage_replicates_inputs.csv``: coverage at each input, per method;
* ``coverage_replicates_summary.csv``: one row per method;

and ``coverage_replicates_inputs.png`` (per-input coverage, sorted) to the
figure directory. Per-replicate results are cached under
``cache/section4_coverage_replicates``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path[:0] = [str(HERE), str(DIAG)]
from core import ResultCache, pointwise_rhat  # noqa: E402

CACHE_VERSION = 1
TAG = "cov_Friedman"
METHODS = ("default", "mtmh", "default_pt", "mtmh_pt")
METHOD_NAMES = {"default": "Default", "mtmh": "MTMH", "default_pt": "Default+PT", "mtmh_pt": "MTMH+PT"}
STYLE = {"Default": "#1f77b4", "MTMH": "#2ca02c", "Default+PT": "#d62728", "MTMH+PT": "#9467bd"}


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-dir", type=Path, default=DIAG / "store_coverage")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--level", type=float, default=0.90)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(supplied)


def read_stored(path: Path) -> np.ndarray:
    """A store CSV with its ``# original_shape=(...)`` header."""
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline()
    shape = tuple(int(v) for v in header.split("original_shape=")[-1].strip(" ()\n").split(",") if v.strip())
    values = pd.read_csv(path, comment="#", header=None, dtype=np.float64, engine="c").to_numpy()
    return values.reshape(shape)


def files(root: Path, replicate: int, method: str) -> tuple[Path, Path]:
    run = f"{TAG}__run{replicate:03d}"
    return (root / "preds" / f"{run}__{method}__preds.csv",
            root / "subsample_y_test" / f"{run}__subsample_y_test.csv")


def score(preds_path: Path, truth_path: Path, burn: int, level: float) -> dict:
    draws = read_stored(preds_path).transpose(0, 2, 1)[:, burn:, :]   # (chain, draw, input)
    truth = read_stored(truth_path).ravel()
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail], axis=1)            # (chain, input)
    flat = draws.reshape(-1, draws.shape[-1])
    pooled_lo, pooled_hi = np.quantile(flat, [tail, 1.0 - tail], axis=0)
    chain_means = draws.mean(axis=1)
    return {
        "inside_single": ((lo <= truth) & (truth <= hi)).astype(np.int8),          # (chain, input)
        "inside_pooled": ((pooled_lo <= truth) & (truth <= pooled_hi)).astype(np.int8),
        "width_single": float(np.median(hi - lo)),
        "width_pooled": float(np.median(pooled_hi - pooled_lo)),
        "rhat_median": float(np.median(pointwise_rhat(draws))),
        "rmse_f_single": float(np.sqrt(((chain_means - truth) ** 2).mean())),
        "rmse_f_pooled": float(np.sqrt(((chain_means.mean(axis=0) - truth) ** 2).mean())),
    }


def figure(inputs: pd.DataFrame, path: Path, level: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, column, title in zip(axes, ("cover_single", "cover_pooled"),
                                 ("Single chain", "Four-chain mixture")):
        for name, part in inputs.groupby("method", sort=False):
            values = np.sort(part[column].to_numpy())
            ax.plot(np.arange(1, len(values) + 1), values, color=STYLE[name], lw=1.6, label=name)
        ax.axhline(level, color="0.35", ls=(0, (4, 2)), lw=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Input, sorted by coverage")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"Coverage of central {int(round(100 * level))}% interval")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    root = args.store_dir / TAG
    replicates = sorted({int(p.name.split("__run")[1][:3]) for p in (root / "preds").glob(f"{TAG}__run*__preds.csv")})
    units = [(r, m) for r in replicates for m in METHODS if files(root, r, m)[0].is_file()]
    print(f"[coverage] {len(replicates)} replicates, {len(units)} replicate-method pairs", flush=True)

    cache = ResultCache("section4_coverage_replicates",
                        {"version": CACHE_VERSION, "burn": args.burn, "level": args.level},
                        enabled=not args.no_cache)
    results, pending = {}, []
    for unit in units:
        inputs = list(files(root, *unit))
        cached = cache.get(unit, inputs)
        if cached is None:
            pending.append((unit, inputs))
        else:
            results[unit] = cached
    if pending:
        print(f"[coverage] scoring {len(pending)} pairs on {args.n_jobs} workers", flush=True)
        scored = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(score)(inputs[0], inputs[1], args.burn, args.level) for _, inputs in pending)
        for (unit, inputs), payload in zip(pending, scored):
            cache.put(unit, inputs, payload)
            results[unit] = payload
    print(f"[coverage] cache: {cache.hits} reused, {cache.misses} computed", flush=True)

    # complete replicates only, so every method averages the same training sets
    complete = [r for r in replicates if all((r, m) in results for m in METHODS)]
    rows = []
    for r in complete:
        for m in METHODS:
            res = results[(r, m)]
            per_chain = res["inside_single"].mean(axis=1)
            rows.append(dict(replicate=r, method=METHOD_NAMES[m],
                             cover_single=float(per_chain.mean()), cover_single_min_chain=float(per_chain.min()),
                             cover_pooled=float(res["inside_pooled"].mean()),
                             **{k: res[k] for k in ("width_single", "width_pooled", "rhat_median",
                                                    "rmse_f_single", "rmse_f_pooled")}))
    runs = pd.DataFrame(rows)

    inputs = []
    for m in METHODS:
        single = np.stack([results[(r, m)]["inside_single"] for r in complete])   # (rep, chain, input)
        pooled = np.stack([results[(r, m)]["inside_pooled"] for r in complete])   # (rep, input)
        for j in range(single.shape[-1]):
            inputs.append(dict(method=METHOD_NAMES[m], input=j,
                               cover_single=float(single[:, :, j].mean()),
                               cover_pooled=float(pooled[:, j].mean())))
    inputs = pd.DataFrame(inputs)

    summary = []
    n = len(complete)
    for name, part in runs.groupby("method", sort=False):
        per_input = inputs[inputs.method == name]
        summary.append(dict(
            method=name, replicates=n,
            cover_single=part.cover_single.mean(), cover_single_se=part.cover_single.std(ddof=1) / np.sqrt(n),
            cover_pooled=part.cover_pooled.mean(), cover_pooled_se=part.cover_pooled.std(ddof=1) / np.sqrt(n),
            width_ratio=part.width_single.mean() / part.width_pooled.mean(),
            width_pooled=part.width_pooled.mean(),
            rhat_median=part.rhat_median.mean(),
            replicate_cover_single_min=part.cover_single.min(),
            replicate_cover_single_q10=part.cover_single.quantile(0.1),
            replicate_cover_pooled_min=part.cover_pooled.min(),
            input_cover_single_min=per_input.cover_single.min(),
            input_cover_single_q10=per_input.cover_single.quantile(0.1),
            input_cover_single_below_080=(per_input.cover_single < 0.80).mean(),
            input_cover_pooled_min=per_input.cover_pooled.min(),
            input_cover_pooled_q10=per_input.cover_pooled.quantile(0.1),
            input_cover_pooled_below_080=(per_input.cover_pooled < 0.80).mean(),
            rmse_f_single=part.rmse_f_single.mean(), rmse_f_pooled=part.rmse_f_pooled.mean(),
        ))
    summary = pd.DataFrame(summary)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.table_dir / "coverage_replicates_runs.csv", index=False)
    inputs.to_csv(args.table_dir / "coverage_replicates_inputs.csv", index=False)
    summary.to_csv(args.table_dir / "coverage_replicates_summary.csv", index=False)
    figure(inputs, args.figure_dir / "coverage_replicates_inputs.png", args.level)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(4).T.to_string())
    print(f"[coverage] wrote coverage_replicates_*.csv to {args.table_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
