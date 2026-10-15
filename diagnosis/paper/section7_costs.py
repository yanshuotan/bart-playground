#!/usr/bin/env python3
"""What poor mixing costs: pooling, error decomposition, coverage and ESS.

Four questions evaluated from the stored draws:

* **Pooling.** Single-chain and pooled estimators are scored side by side.
  Between-chain variance describes the empirical finite mixture.
* **Error decomposition.** Writing a chain's error as
  :math:`\\bar f_c-y=(\\bar f-y)+(\\bar f_c-\\bar f)` and averaging over chains
  splits a single chain's mean squared error into a part every chain shares and
  a part that varies among finite chain-specific posterior-mean estimates,
  including finite Monte Carlo variation.
* **Containment of the latent function.** On the Friedman designs :math:`f` is
  known, so the containment rate of single-chain and pooled credible intervals
  is measured over 100 fixed inputs on one dataset realization per design,
  with paired training splits.
* **Efficiency.** Bulk and tail ESS per test input, summarised by median and
  10th percentile, and divided by the measured cost ratio. Two conventions are
  reported side by side: each chain measured on its own and then averaged over
  chains, and all four chains passed to one call. Rank normalization uses the
  draws supplied to each calculation. Joint ESS incorporates serial dependence
  and between-chain differences. Relative ESS divides by S for the single-chain
  calculation and C*S for the joint calculation. Cost-normalized columns charge
  one chain or four sequential chains, respectively. R-hat accompanies these
  diagnostics, with separate columns screened on pointwise agreement.

Predictions are stored on the original response scale.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG))
from core import load_with_shape  # noqa: E402
from section6_methods import (full_target_vector, predictions,  # noqa: E402
                              residual_variances, split_indices)
from fixed100_support import (  # noqa: E402
    DATASET_CONFIGS,
    SPECS,
    _load_generator_dataset,
    generate_sparse_variant,
    load_dataset,
)
from run_fixed100 import repo_nested_friedman1_generator  # noqa: E402

METHODS = ["default", "mtmh", "default_pt", "mtmh_pt"]
METHOD_NAMES = {"default": "Default", "default_pt": "Default+PT",
                "mtmh": "MTMH", "mtmh_pt": "MTMH+PT",
                "default_long": "Default (long)"}
TIMING_LABEL = {"default": "default", "mtmh": "mtmh",
                "default_pt": "default_pt (parallel)",
                "mtmh_pt": "mtmh_pt (parallel)", "default_long": "default"}
# store folder -> the tag the timing summaries use
TIMING_TAG = {
    "fixed100_Abalone": "fixed100_Abalone",
    "fixed100_Airfoil": "fixed100_Airfoil",
    "fixed100_CCPP": "fixed100_CCPP",
    "fixed100_CPUAct": "fixed100_CPUAct",
    "fixed100_CalHousing_subsample5000": "fixed100_CalHousing_subsample5000",
    "fixed100_Concrete": "fixed100_Concrete",
    "fixed100_Friedman": "fixed100_Friedman",
    "fixed100_FriedmanSparseDir_p20": "fixed100_FriedmanSparseDirP20K5",
    "fixed100_FriedmanSparseDir_p100": "fixed100_FriedmanSparseDir",
    "fixed100_FriedmanSparseDir_p200": "fixed100_FriedmanSparseDirP200K5",
    "fixed100_SeoulBike": "fixed100_SeoulBike",
}
DISPLAY = {
    "fixed100_Abalone": "Abalone", "fixed100_Airfoil": "Airfoil",
    "fixed100_CCPP": "CCPP", "fixed100_CPUAct": "CPU activity",
    "fixed100_CalHousing_subsample5000": "CalHousing",
    "fixed100_Concrete": "Concrete", "fixed100_Friedman": "Friedman",
    "fixed100_FriedmanSparseDir_p20": "DART $d=20$",
    "fixed100_FriedmanSparseDir_p100": "DART $d=100$",
    "fixed100_FriedmanSparseDir_p200": "DART $d=200$",
    "fixed100_SeoulBike": "SeoulBike",
}
# store folder -> how to rebuild the design so that the noise-free signal is known
SYNTHETIC = {
    "fixed100_Friedman": ("config", "friedman"),
    "fixed100_FriedmanSparseDir_p20": ("spec", "friedman_p20_k5"),
    # the d=100 variant is configured like a named dataset, not as a sparse spec
    "fixed100_FriedmanSparseDir_p100": ("config", "friedman_sparse_dir"),
    "fixed100_FriedmanSparseDir_p200": ("spec", "friedman_p200_k5"),
}


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--timing-dir", type=Path, default=DIAG / "timing")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(TIMING_TAG))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--interval", type=float, default=0.9,
                        help="Central credible level for the containment rate.")
    parser.add_argument("--long", action="store_true",
                        help="Also score the stored 10^7 Default chains.")
    parser.add_argument("--reuse-ess", action="store_true",
                        help="Reuse existing ESS counts for identical retained draws; refresh other statistics.")
    return parser.parse_args(supplied)


def cost_ratios(timing_dir: Path, dataset: str) -> dict[str, float]:
    path = timing_dir / f"{TIMING_TAG[dataset]}_run000_summary.csv"
    rows = {r["label"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    ratios = {m: float(rows[TIMING_LABEL[m]]["relative_to_default"])
              for m in METHOD_NAMES if TIMING_LABEL[m] in rows}
    # the long chains run 10^7 iterations at Default's per-iteration cost, so
    # they cost a thousand production runs rather than one
    ratios["default_long"] = 1000.0 * ratios["default"]
    return ratios


def targets(dataset: str, store_root: Path, run: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Observed responses at the 100 fixed inputs, and the true f when known."""
    index = next((store_root / dataset / "indices").glob(
        f"*run{run:03d}__fixed_test_idx.csv"))
    test_idx = load_with_shape(index).astype(int).ravel()

    if dataset in SYNTHETIC:
        kind, name = SYNTHETIC[dataset]
        if kind == "config":
            config = DATASET_CONFIGS[name]
            _, y = _load_generator_dataset(config, "friedman1")
            _, signal = _load_generator_dataset({**config, "noise": 0.0}, "friedman1")
        else:
            _, y, _ = generate_sparse_variant(
                name, n_samples=2000, seed=42, noise_sd=1.0,
                friedman_generator=repo_nested_friedman1_generator)
            _, signal, _ = generate_sparse_variant(
                name, n_samples=2000, seed=42, noise_sd=0.0,
                friedman_generator=repo_nested_friedman1_generator)
        return y[test_idx], signal[test_idx]

    source = next(k for k, v in DATASET_CONFIGS.items()
                  if v.get("dataset_tag") == dataset)
    _, y = load_dataset(source)
    return y[test_idx], None


def stored_draws(store_root: Path, dataset: str, run: int, method: str,
                 burn: int) -> np.ndarray | None:
    hits = sorted((store_root / dataset / "preds").glob(
        f"*__run{run:03d}__{method}__preds.csv"))
    if not hits:
        return None
    return load_with_shape(hits[0]).transpose(0, 2, 1)[:, burn:, :]


def check_alignment(store_root: Path, dataset: str, run: int, method: str,
                    draws: np.ndarray, y: np.ndarray) -> None:
    """The responses must be the ones the stored draws were scored against.

    Rebuilding a design or reading a test index wrongly would line the draws up
    with the wrong rows and would not otherwise announce itself, so the
    per-draw RMSE is recomputed here and compared with the stored one.
    """
    hits = sorted((store_root / dataset / "rmses").glob(
        f"*__run{run:03d}__{method}__rmses.csv"))
    if not hits:
        return
    stored = np.asarray(load_with_shape(hits[0]), dtype=float)
    stored = stored.reshape(stored.shape[0], -1)[:, -draws.shape[1]:]
    recomputed = np.sqrt(((draws - y[None, None, :]) ** 2).mean(axis=2))
    gap = float(np.max(np.abs(recomputed - stored)))
    if gap > 1e-6:
        raise ValueError(
            f"{dataset} run {run} {method}: recomputed RMSE departs from the "
            f"stored RMSE by {gap:.3e}, so the responses are not aligned with "
            f"the stored draws")


def score(draws: np.ndarray, y: np.ndarray, truth: np.ndarray | None,
          level: float, cached: dict | None = None) -> dict[str, float]:
    """Pooling, decomposition, containment and ESS for one method's draws."""
    chain_means = draws.mean(axis=1)                    # (chain, test)
    pooled_mean = chain_means.mean(axis=0)              # (test)

    # the cross term cancels once the chains are averaged, so a single chain's
    # mean squared error splits into a shared part and a chain-specific part
    single_mse = float(((chain_means - y[None, :]) ** 2).mean())
    pooled_mse = float(((pooled_mean - y[None, :]) ** 2).mean())
    chain_mse = float(((chain_means - pooled_mean[None, :]) ** 2).mean())

    # Exact variance decomposition of the finite, equally weighted mixture.
    within = draws.var(axis=1, ddof=0).mean(axis=0)
    between = chain_means.var(axis=0, ddof=0)
    total = within + between
    np.testing.assert_allclose(total, draws.reshape(-1, draws.shape[-1]).var(axis=0),
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(single_mse, pooled_mse + chain_mse,
                               rtol=1e-10, atol=1e-12)
    share = np.divide(between, total, out=np.zeros_like(total), where=total > 0)

    out = {
        "rmse_single": float(np.sqrt(single_mse)),
        "rmse_single_mean_chainwise": float(np.sqrt(((chain_means-y)**2).mean(axis=1)).mean()),
        "rmse_pooled": float(np.sqrt(pooled_mse)),
        "mse_single": single_mse,
        "mse_pooled": pooled_mse,
        "mse_chain_specific": chain_mse,
        "chain_share_of_mse": chain_mse / single_mse if single_mse else np.nan,
        "between_share_median": float(np.median(share)),
        "between_share_q90": float(np.quantile(share, 0.9)),
        "within_variance_mean": float(within.mean()),
        "between_variance_mean": float(between.mean()),
        "pooled_variance_mean": float(total.mean()),
    }

    tail = (1.0 - level) / 2.0
    edges = [tail, 1.0 - tail]
    single_lo, single_hi = np.quantile(draws, edges, axis=1)          # (chain, test)
    flat = draws.reshape(-1, draws.shape[-1])
    pooled_lo, pooled_hi = np.quantile(flat, edges, axis=0)           # (test)
    out["width_single"] = float(np.median(single_hi - single_lo))
    out["width_pooled"] = float(np.median(pooled_hi - pooled_lo))
    if truth is not None:
        inside = (single_lo <= truth[None, :]) & (truth[None, :] <= single_hi)
        out["cover_f_single"] = float(inside.mean())
        out["cover_f_pooled"] = float(((pooled_lo <= truth) & (truth <= pooled_hi)).mean())
        out["rmse_f_single"] = float(np.sqrt(((chain_means - truth[None, :]) ** 2).mean()))
        out["rmse_f_pooled"] = float(np.sqrt(((pooled_mean - truth) ** 2).mean()))

    # One chain is the baseline a practitioner may actually run, so ESS is
    # measured within each chain on its own and then averaged over chains.
    # Laying each (chain, input) pair out as its own series gives those values
    # in one call; arviz splits a lone chain in half, which is the usual
    # split-ESS.
    #
    # The "multi" variant computes joint four-chain diagnostics. Rank
    # normalization uses S draws for a single chain and C*S for the joint
    # calculation; relative ESS uses the corresponding supplied draw count.
    chains, retained, points = draws.shape
    if cached is not None:
        if int(cached['draws']) != retained:
            raise ValueError('Cached ESS uses a different retained draw count')
        out.update({key: value for key, value in cached.items()
                    if key.startswith(('ess_', 'rel_ess_'))
                    and 'pooled' not in key and 'pooling' not in key})
        missing = [k for k in ('ess_bulk_multi_median', 'ess_bulk_single_median')
                   if k not in out]
        if missing:
            raise ValueError(f'Cached ESS lacks {missing}; rerun without --reuse-ess')
        together = xr.DataArray(draws, dims=("chain", "draw", "point"), name="prediction")
        rhat = az.rhat(together, method='rank')['prediction'].values
        out.update(rhat_median=float(np.median(rhat)), rhat_q90=float(np.quantile(rhat, .9)),
                   agreement_screen_pass=float(np.all(np.isfinite(rhat)) and np.max(rhat) <= 1.01))
        return out
    alone = xr.DataArray(draws.transpose(1, 0, 2).reshape(1, -1, chains * points),
                         dims=("chain", "draw", "point"), name="prediction")
    together = xr.DataArray(draws, dims=("chain", "draw", "point"), name="prediction")
    for kind in ("bulk", "tail"):
        single = az.ess(alone, method=kind)["prediction"].values.reshape(chains, points)
        single = single.mean(axis=0)
        multi = az.ess(together, method=kind)["prediction"].values
        # Relative ESS divides by the draw count supplied to each calculation.
        # Storage stride and computational cost are recorded separately.
        for name, values, supplied in (("single", single, retained),
                                       ("multi", multi, chains * retained)):
            out[f"ess_{kind}_{name}_median"] = float(np.median(values))
            out[f"ess_{kind}_{name}_q10"] = float(np.quantile(values, 0.1))
            out[f"rel_ess_{kind}_{name}_median"] = float(np.median(values / supplied))
            out[f"rel_ess_{kind}_{name}_q10"] = float(np.quantile(values / supplied, 0.1))
    rhat = az.rhat(together, method="rank")["prediction"].values
    out["rhat_median"] = float(np.median(rhat))
    out["rhat_q90"] = float(np.quantile(rhat, 0.9))
    out["agreement_screen_pass"] = float(np.all(np.isfinite(rhat)) and np.max(rhat) <= 1.01)
    return out


def costs_main(args) -> int:
    args.table_dir.mkdir(parents=True, exist_ok=True)
    wanted = METHODS + (["default_long"] if args.long else [])
    cache = {}
    if args.reuse_ess:
        previous = pd.read_csv(args.table_dir / 'consequences_run_level.csv')
        cache = {(r['dataset'], int(r['run']), r['method']): r for r in previous.to_dict('records')}

    rows = []
    for dataset in args.datasets:
        ratios = cost_ratios(args.timing_dir, dataset)
        for run in args.runs:
            y, truth = targets(dataset, args.store_root, run)
            for method in wanted:
                draws = stored_draws(args.store_root, dataset, run, method, args.burn)
                if draws is None:
                    continue
                check_alignment(args.store_root, dataset, run, method, draws, y)
                row = {"dataset": dataset, "run": run,
                       "method": METHOD_NAMES[method], "draws": draws.shape[1],
                       "cost_ratio": ratios.get(method, np.nan)}
                previous = cache.get((dataset, run, METHOD_NAMES[method]))
                if args.reuse_ess and previous is None:
                    raise ValueError(f'Missing ESS cache for {dataset} {run} {method}')
                row.update(score(draws, y, truth, args.interval, cached=previous))
                for kind in ("bulk", "tail"):
                    for scope in ("single", "multi"):
                        # A four-chain estimate costs four sequential chains.
                        charged = row["cost_ratio"] * (draws.shape[0]
                                                       if scope == "multi" else 1)
                        row[f"ess_{kind}_{scope}_per_cost"] = (
                            row[f"ess_{kind}_{scope}_median"] / charged
                            if charged else np.nan)
                        row[f"ess_{kind}_{scope}_per_cost_screened"] = (
                            row[f"ess_{kind}_{scope}_per_cost"]
                            if row["agreement_screen_pass"] else np.nan)
                rows.append(row)
                del draws
            print(f"[consequences] {dataset} run {run:03d}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.table_dir / "consequences_run_level.csv", index=False)
    summary = frame.drop(columns=["run"]).groupby(["dataset", "method"], sort=False).mean()
    summary.reset_index().to_csv(args.table_dir / "consequences_summary.csv", index=False)

    pd.set_option("display.width", 220)
    order = [m for m in METHOD_NAMES.values() if m in set(frame.method)]
    for label, columns in (
        ("chain-specific share of a single chain's MSE", ["chain_share_of_mse"]),
        ("between-chain share of empirical pooled latent variance, median over inputs",
         ["between_share_median"]),
        ("RMSE, single chain then pooled", ["rmse_single", "rmse_pooled"]),
        ("relative bulk ESS, divided by the draws supplied: "
         "one chain averaged over chains, then all four chains in one call",
         ["rel_ess_bulk_single_median", "rel_ess_bulk_multi_median"]),
        ("bulk ESS: raw counts, then per unit of Default-relative cost",
         ["ess_bulk_single_median", "ess_bulk_multi_median",
          "ess_bulk_single_per_cost", "ess_bulk_multi_per_cost"]),
    ):
        print()
        print(label)
        for column in columns:
            wide = summary.reset_index().pivot(index="dataset", columns="method",
                                               values=column)
            wide.index = [DISPLAY[d] for d in wide.index]
            print(f"  {column}")
            print(wide[order].round(4).to_string())

    covered = summary.reset_index().dropna(subset=["cover_f_single"])
    if len(covered):
        print()
        print(f"containment of the true f at the {args.interval:.0%} level, "
              "single chain then pooled")
        for column in ("cover_f_single", "cover_f_pooled", "width_single", "width_pooled"):
            wide = covered.pivot(index="dataset", columns="method", values=column)
            wide.index = [DISPLAY[d] for d in wide.index]
            print(f"  {column}")
            print(wide[order].round(3).to_string())
    print()
    print(f"tables: {args.table_dir}")
    return 0




def scalar_summary(values):
    """Bulk/tail ESS of whatever chains are supplied, with its own denominator.

    A run-level row supplies all four chains, so its `ess_*` is the four-chain
    value and `rel_ess_*` divides by C*S; a chain-level row supplies one chain
    and divides by S. The relative figure divides diagnostic ESS by the
    number of draws supplied to that calculation.
    """
    data = xr.DataArray(values, dims=('chain', 'draw'), name='quantity')
    rhat_data = data
    if values.shape[0] == 1:
        half = values.shape[1] // 2
        rhat_data = xr.DataArray(np.stack([values[0, :half], values[0, -half:]]),
                                 dims=('chain', 'draw'), name='quantity')
    result = {'rhat': float(az.rhat(rhat_data, method='rank')['quantity']),
              'ess_bulk': float(az.ess(data, method='bulk')['quantity']),
              'ess_tail': float(az.ess(data, method='tail')['quantity']),
              'draws_supplied': int(values.size),
              'pooled_mean': float(values.mean()),
              'pooled_q05': float(np.quantile(values, .05)),
              'pooled_q95': float(np.quantile(values, .95))}
    for kind in ('bulk', 'tail'):
        result[f'rel_ess_{kind}'] = result[f'ess_{kind}'] / values.size
    result['agreement_screen_pass'] = np.isfinite(result['rhat']) and result['rhat'] <= 1.01
    return result

def scalar_main(args):
    rows, chains = [], []
    out = args.table_dir; out.mkdir(exist_ok=True)
    figs = args.table_dir.parent/'figures'/'scalar_quantities'; figs.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        targets = full_target_vector(args.store_root, dataset)
        for run in args.runs:
            train = targets[split_indices(args.store_root, dataset, run, 'train_idx')]
            scale = float(np.ptp(train))
            fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout='constrained') if run == 0 else (None, None)
            for column, method in enumerate(['default', 'mtmh', 'default_pt', 'mtmh_pt', 'default_long']):
                if args.plots_only and column == 4:
                    continue
                draws = stored_draws(args.store_root, dataset, run, method, 0)
                if draws is None:
                    raise FileNotFoundError(f'{dataset} run {run}: missing {method}')
                variances = residual_variances(args.store_root, dataset, run, method)
                if variances.shape != draws.shape[:2]:
                    raise ValueError('prediction / variance draw alignment mismatch')
                quantities = {'panel_mean_f': draws.mean(axis=2),
                              'noise_variance_y': variances * scale**2}
                for qindex, (quantity, full) in enumerate(quantities.items()):
                    values = full[:, args.burn:]
                    base = {'dataset': dataset, 'run': run, 'method': METHOD_NAMES[method],
                            'quantity': quantity, 'draws_per_chain': values.shape[1],
                            'store_stride': 1000 if method == 'default_long' else 1}
                    per_chain = [scalar_summary(series[None, :]) for series in values]
                    summary = scalar_summary(values)
                    # the same quantity under the other convention, so the two
                    # sit in one row rather than in two tables
                    for key in ('ess_bulk', 'ess_tail', 'rel_ess_bulk', 'rel_ess_tail'):
                        summary[f'{key}_single_mean'] = float(
                            np.mean([d[key] for d in per_chain]))
                    rows.append({**base, **summary})
                    for c, diagnostics in enumerate(per_chain):
                        chains.append({**base, 'chain': c, **diagnostics})
                    if fig is not None and column < 4:
                        ax = axes[qindex, column]
                        stride = max(1, full.shape[1]//1500)
                        for c, series in enumerate(full):
                            ax.plot(np.arange(0, len(series), stride), series[::stride], lw=.5, alpha=.7, label=f'chain {c}')
                        ax.axvline(args.burn, color='black', ls='--', lw=.8)
                        ax.set_title(f'{METHOD_NAMES[method]} | R-hat {rows[-1]["rhat"]:.3f}')
                        ax.set_ylabel(quantity); ax.set_xlabel('stored draw')
                        if quantity == 'noise_variance_y':
                            ax.set_yscale('log')
                            ax.set_ylabel('noise variance (outcome scale; log axis)')
                del draws
            if fig is not None:
                axes[0, 0].legend(fontsize=6)
                fig.suptitle(f'{dataset}, run 0; dashed line ends warm-up')
                fig.savefig(figs/f'{dataset}_run000.png', dpi=150); plt.close(fig)
            print(f'[scalar] {dataset} run {run}', flush=True)
    if args.plots_only:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'scalar_quantities_run_level.csv', index=False)
    pd.DataFrame(chains).to_csv(out/'scalar_quantities_chain_level.csv', index=False)
    reported = ['rhat', 'ess_bulk', 'ess_tail', 'rel_ess_bulk', 'rel_ess_tail',
                'ess_bulk_single_mean', 'ess_tail_single_mean',
                'rel_ess_bulk_single_mean', 'rel_ess_tail_single_mean',
                'pooled_mean', 'agreement_screen_pass']
    frame.groupby(['dataset', 'method', 'quantity'], sort=False)[reported].agg(['mean','min','max']).to_csv(out/'scalar_quantities_summary.csv')


def parse_scalar_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panel-average and noise-variance diagnostics")
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(TIMING_TAG))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--plots-only", action="store_true",
                        help="Refresh run-0 figures without overwriting diagnostic tables")
    return parser.parse_args(supplied)


def main(supplied: list[str] | None = None) -> int:
    """`costs` scores the stored draws; `scalar` diagnoses the two scalar series."""
    argv = list(sys.argv[1:] if supplied is None else supplied)
    part = argv[0] if argv and not argv[0].startswith("-") else "costs"
    rest = argv[1:] if argv and not argv[0].startswith("-") else argv
    if part == "scalar":
        scalar_main(parse_scalar_args(rest))
        return 0
    if part != "costs":
        raise SystemExit(f"unknown part {part!r}; expected 'costs' or 'scalar'")
    return costs_main(parse_args(rest))


if __name__ == "__main__":
    raise SystemExit(main())
