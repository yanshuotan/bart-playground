#!/usr/bin/env python3
"""What poor mixing costs: pooling, error decomposition, coverage and ESS.

Four questions, all answered from the stored draws without resampling:

* **Pooling.** Running several chains and averaging is what practitioners do.
  For each method the single-chain and pooled estimators are scored side by
  side, and the between-chain share of predictive variance at each test input
  says what fraction of the spread one chain misses.
* **Error decomposition.** Writing a chain's error as
  :math:`\\bar f_c-y=(\\bar f-y)+(\\bar f_c-\\bar f)` and averaging over chains
  splits a single chain's mean squared error into a part every chain shares and
  a part that depends on which chain was run. The second part is the cost of
  poor mixing that RMSE can see; if it is small, that is why RMSE moves so
  little.
* **Coverage of the latent function.** On the Friedman designs :math:`f` is
  known, so the containment rate of single-chain and pooled credible intervals
  is measurable. The 100 inputs are not independent replications, so this is a
  containment rate over a fixed input set, not a calibrated coverage estimate.
* **Efficiency.** Bulk and tail ESS per test input, summarised by median and
  10th percentile, and divided by the measured cost ratio. ESS per unit of work
  is only meaningful where the chains agree, so the agreement summary is
  reported in the same row.

Predictions are stored on the original scale of the response; the noise
variance is stored on the internal scale and is not needed here.
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

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(DIAG / "longchain"))
from chains import load_with_shape  # noqa: E402
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
          level: float) -> dict[str, float]:
    """Pooling, decomposition, containment and ESS for one method's draws."""
    chain_means = draws.mean(axis=1)                    # (chain, test)
    pooled_mean = chain_means.mean(axis=0)              # (test)

    # the cross term cancels once the chains are averaged, so a single chain's
    # mean squared error splits into a shared part and a chain-specific part
    single_mse = float(((chain_means - y[None, :]) ** 2).mean())
    pooled_mse = float(((pooled_mean - y[None, :]) ** 2).mean())
    chain_mse = float(((chain_means - pooled_mean[None, :]) ** 2).mean())

    within = draws.var(axis=1, ddof=1).mean(axis=0)     # (test)
    between = chain_means.var(axis=0, ddof=1)           # (test)
    share = between / (between + within)

    out = {
        "rmse_single": float(np.sqrt(single_mse)),
        "rmse_pooled": float(np.sqrt(pooled_mse)),
        "mse_single": single_mse,
        "mse_pooled": pooled_mse,
        "mse_chain_specific": chain_mse,
        "chain_share_of_mse": chain_mse / single_mse if single_mse else np.nan,
        "between_share_median": float(np.median(share)),
        "between_share_q90": float(np.quantile(share, 0.9)),
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
    # measured within each chain on its own and averaged over chains, and
    # separately on all chains together. Laying each (chain, input) pair out as
    # its own series gives the single-chain values in one call; arviz splits a
    # lone chain in half, which is the usual split-ESS.
    chains, retained, points = draws.shape
    alone = xr.DataArray(draws.transpose(1, 0, 2).reshape(1, -1, chains * points),
                         dims=("chain", "draw", "point"), name="prediction")
    together = xr.DataArray(draws, dims=("chain", "draw", "point"), name="prediction")
    for kind in ("bulk", "tail"):
        single = az.ess(alone, method=kind)["prediction"].values.reshape(chains, points)
        single = single.mean(axis=0)
        pooled = az.ess(together, method=kind)["prediction"].values
        # Relative ESS is the fraction of stored draws that count as
        # independent. Dividing by the number of draws removes the run length,
        # so a short run and a thinned long run are on the same footing: it
        # measures how redundant the draws are, not how many there are.
        for name, values, total in (("single", single, retained),
                                    ("pooled", pooled, chains * retained)):
            out[f"ess_{kind}_{name}_median"] = float(np.median(values))
            out[f"ess_{kind}_{name}_q10"] = float(np.quantile(values, 0.1))
            out[f"rel_ess_{kind}_{name}_median"] = float(np.median(values / total))
            out[f"rel_ess_{kind}_{name}_q10"] = float(np.quantile(values / total, 0.1))
        # four agreeing chains would give four times one chain's ESS; the
        # shortfall is what disagreement costs
        out[f"ess_{kind}_pooling_gain"] = float(np.median(pooled / (chains * single)))
    rhat = az.rhat(together, method="rank")["prediction"].values
    out["rhat_median"] = float(np.median(rhat))
    return out


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    wanted = METHODS + (["default_long"] if args.long else [])

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
                row.update(score(draws, y, truth, args.interval))
                for kind in ("bulk", "tail"):
                    for scope in ("single", "pooled"):
                        row[f"ess_{kind}_{scope}_per_cost"] = (
                            row[f"ess_{kind}_{scope}_median"] / row["cost_ratio"]
                            if row["cost_ratio"] else np.nan)
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
        ("between-chain share of predictive variance, median over inputs",
         ["between_share_median"]),
        ("RMSE, single chain then pooled", ["rmse_single", "rmse_pooled"]),
        ("relative bulk ESS, the independent fraction of the stored draws: "
         "one chain, then four chains pooled",
         ["rel_ess_bulk_single_median", "rel_ess_bulk_pooled_median"]),
        ("bulk ESS: raw counts, per unit of Default-relative cost, and the "
         "share of the fourfold pooling gain that disagreement leaves",
         ["ess_bulk_single_median", "ess_bulk_pooled_median",
          "ess_bulk_single_per_cost", "ess_bulk_pooled_per_cost",
          "ess_bulk_pooling_gain"]),
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


if __name__ == "__main__":
    raise SystemExit(main())
