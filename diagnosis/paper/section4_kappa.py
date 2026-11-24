#!/usr/bin/env python3
"""The size and noise scale kappa = n / (1 - R^2) against Default agreement.

Three groups of settings, each summarized by its median pointwise R-hat of four
Default chains of 10,000 iterations (3,000 warm-up draws dropped) and averaged
over runs:

* ``main``: the paper settings (``section7_methods.DATASETS``), five runs; the
  Default, thinned-long and MTMH+PT R-hat are read from the tables of
  ``section7_methods.py`` and ``long_tail_tables.py``;
* ``sweep``: the training-size and noise sweep of ``scaling_sweep.py``
  (``store_scaling``), five runs;
* ``controlled``: the responses of ``controlled_noise.py`` (``store_controlled``),
  every run stored there.

Two estimates of the noise fraction 1 - R^2 are formed on each training split:
the five-fold cross-validated mean squared error of histogram gradient boosting
(500 iterations, otherwise scikit-learn defaults) divided by var(y_train),
available before sampling; and the posterior estimate sigma_bar^2 / var(y_train)
from the retained Default draws. Cross-validated values are cached per
(group, setting, run) under ``cache/section4_kappa``.

Writes ``kappa_run_level.csv``, ``kappa_settings.csv`` and
``kappa_statistics.csv`` to the table directory.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path[:0] = [str(HERE), str(DIAG)]
from core import ResultCache, load_with_shape, pointwise_rhat  # noqa: E402
from section7_costs import DISPLAY  # noqa: E402
from section7_methods import (DATASETS, SPARSE_TARGET_LOADERS, TARGET_LOADERS,  # noqa: E402
                              split_indices, y_test)

CACHE_VERSION = 1
BURN = 3000
CV_SETTINGS = {"model": "HistGradientBoostingRegressor(max_iter=500, random_state=0)",
               "folds": "KFold(5, shuffle=True, random_state=0)"}


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--scaling-store", type=Path, default=DIAG / "store_scaling")
    parser.add_argument("--controlled-store", type=Path, default=DIAG / "store_controlled")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--groups", nargs="+", default=["main", "sweep", "controlled"],
                        choices=["main", "sweep", "controlled"])
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(supplied)


# ------------------------------------------------------------------ designs
def main_design(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Full (X, y) of a paper setting, as the experiment's own loader returns it."""
    from fixed100_support import generate_sparse_variant, load_dataset, repo_nested_friedman1_generator

    if dataset in SPARSE_TARGET_LOADERS:
        X, y, _ = generate_sparse_variant(SPARSE_TARGET_LOADERS[dataset], n_samples=2000, seed=42,
                                          noise_sd=1.0, friedman_generator=repo_nested_friedman1_generator)
    else:
        X, y = load_dataset(TARGET_LOADERS[dataset])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)


def sweep_source(tag: str) -> tuple[np.ndarray, np.ndarray]:
    import scaling_sweep

    source = {"CCPP": "ccpp", "CalHousing": "calhousing", "Friedman": "friedman"}[tag.split("_")[1]]
    noise = int(tag.split("_sd")[1]) / 100 if "_sd" in tag else 1.0
    return scaling_sweep.load_source(SimpleNamespace(source=source, n_generated=20_100), noise)


def controlled_source(tag: str) -> tuple[np.ndarray, np.ndarray]:
    from fixed100_support import load_dataset

    source = {"SeoulBike": "seoul_bike", "Airfoil": "airfoil", "Abalone": "abalone",
              "CPUAct": "cpu_act"}[tag.split("_")[1]]
    X, y = load_dataset(source)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)


def stored(root: Path, tag: str, folder: str, run: int, name: str) -> Path:
    return root / tag / folder / f"{tag}__run{run:03d}__{name}.csv"


# ------------------------------------------------------------------ per-run quantities
def cv_noise_fraction(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    fitted = cross_val_predict(HistGradientBoostingRegressor(max_iter=500, random_state=0), X, y,
                               cv=KFold(5, shuffle=True, random_state=0))
    return float(np.mean((y - fitted) ** 2) / y.var())


def posterior_noise_fraction(sigma2_path: Path, y_train: np.ndarray) -> float:
    """sigma_bar^2 / var(y_train) from the internal-scale noise variance draws."""
    sigma2 = np.asarray(load_with_shape(sigma2_path), dtype=float)
    sigma2 = sigma2.reshape(sigma2.shape[0], -1)[:, BURN:]
    return float(sigma2.mean() * np.ptp(y_train) ** 2 / y_train.var())


def median_rhat(preds_path: Path) -> float:
    draws = load_with_shape(preds_path).transpose(0, 2, 1)[:, BURN:, :]
    return float(np.median(pointwise_rhat(draws)))


def run_units(args) -> list[dict]:
    """Every (group, setting, run) with the paths it reads."""
    units = []
    for dataset in args.datasets:
        for run in range(5):
            units.append(dict(group="main", setting=dataset, run=run, store=args.store_root,
                              tag=dataset, sigma_name="default__sigmas", preds_name=None))
    for tag in sorted(p.name for p in args.scaling_store.glob("scal_*") if p.is_dir()):
        for run in range(5):
            units.append(dict(group="sweep", setting=tag, run=run, store=args.scaling_store, tag=tag,
                              sigma_name="default__sigmas", preds_name="default__preds"))
    for tag in sorted(p.name for p in args.controlled_store.glob("ctl_*") if p.is_dir()):
        runs = sorted(int(p.name.split("__run")[1][:3]) for p in
                      (args.controlled_store / tag / "preds").glob("*__default__preds.csv"))
        for run in runs:
            units.append(dict(group="controlled", setting=tag, run=run, store=args.controlled_store,
                              tag=tag, sigma_name="default__sigmas", preds_name="default__preds"))
    return [unit for unit in units if unit["group"] in args.groups]


def unit_inputs(unit) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Training indices, training response and the files the CV value depends on."""
    if unit["group"] == "main":
        from core import dataset_tag

        tag = dataset_tag(unit["store"], unit["setting"])
        path = unit["store"] / unit["setting"] / "indices" / f"{tag}__run{unit['run']:03d}__train_idx.csv"
        return load_with_shape(path).astype(int).ravel(), None, [path]
    path = stored(unit["store"], unit["tag"], "indices", unit["run"], "train_idx")
    files = [path]
    y_train = None
    if unit["group"] == "controlled":
        y_path = stored(unit["store"], unit["tag"], "subsample_y_train", unit["run"], "subsample_y_train")
        y_train = load_with_shape(y_path).reshape(-1)
        files.append(y_path)
    return load_with_shape(path).astype(int).ravel(), y_train, files


def cv_job(X, y, key):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return key, cv_noise_fraction(X, y)


# ------------------------------------------------------------------ main
def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache("section4_kappa", {"version": CACHE_VERSION, **CV_SETTINGS},
                        enabled=not args.no_cache)
    units = run_units(args)
    designs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def design(unit):
        name = unit["setting"] if unit["group"] == "main" else unit["tag"]
        if name not in designs:
            if unit["group"] == "main":
                designs[name] = main_design(name)
            elif unit["group"] == "sweep":
                designs[name] = sweep_source(name)
            else:
                designs[name] = controlled_source(name)
        return designs[name]

    rows, pending = [], []
    for unit in units:
        X, y = design(unit)
        train, y_override, files = unit_inputs(unit)
        y_train = y[train] if y_override is None else y_override
        if unit["group"] == "main":
            # the rebuilt response must match the stored test panel
            test = split_indices(unit["store"], unit["setting"], unit["run"], "fixed_test_idx")
            if not np.allclose(y[test], y_test(unit["store"], unit["setting"], unit["run"])):
                raise ValueError(f"{unit['setting']} run {unit['run']}: rebuilt response misaligned")
        if unit["group"] == "main":
            from core import dataset_tag
            tag = dataset_tag(unit["store"], unit["setting"])
            sigma_path = unit["store"] / unit["setting"] / "sigmas" / f"{tag}__run{unit['run']:03d}__default__sigmas.csv"
        else:
            sigma_path = stored(unit["store"], unit["tag"], "sigmas", unit["run"], unit["sigma_name"])
        row = dict(group=unit["group"], setting=unit["setting"], run=unit["run"], n=len(train),
                   nu_post=posterior_noise_fraction(sigma_path, y_train))
        if unit["preds_name"]:
            row["rhat_default"] = median_rhat(stored(unit["store"], unit["tag"], "preds", unit["run"], unit["preds_name"]))
        key = (unit["group"], unit["setting"], f"run{unit['run']:03d}")
        cached = cache.get(key, files)
        if cached is None:
            pending.append((X[train], y_train, key, files))
        else:
            row["nu_cv"] = cached
        rows.append(row)
        print(f"[kappa] {unit['group']:10s} {unit['setting']:40s} run {unit['run']}", flush=True)

    if pending:
        print(f"[kappa] {len(pending)} cross-validated fits on {args.n_jobs} workers", flush=True)
        results = dict(Parallel(n_jobs=args.n_jobs)(delayed(cv_job)(X, y, key) for X, y, key, _ in pending))
        for X, y, key, files in pending:
            cache.put(key, files, results[key])
        for row in rows:
            key = (row["group"], row["setting"], f"run{row['run']:03d}")
            if key in results:
                row["nu_cv"] = results[key]
    print(f"[kappa] cache: {cache.hits} reused, {cache.misses} computed", flush=True)

    run_frame = pd.DataFrame(rows)
    run_frame.to_csv(args.table_dir / "kappa_run_level.csv", index=False)
    settings = run_frame.groupby(["group", "setting"], sort=False).mean(numeric_only=True).reset_index()
    settings = settings.drop(columns="run")
    settings["kappa_cv"] = settings.n / settings.nu_cv
    settings["kappa_post"] = settings.n / settings.nu_post

    # paper settings: R-hat from the method tables
    methods = pd.read_csv(args.table_dir / "comparison_dataset_summary.csv")
    tails = pd.concat([pd.read_csv(args.table_dir / f) for f in
                       ("long_tail_as_method_summary.csv", "dart_tail_as_method_summary.csv")])
    pick = lambda frame, col, value, field: frame[frame[col] == value].set_index("dataset")[field]
    main = settings.group == "main"
    settings.loc[main, "rhat_default"] = settings.loc[main, "setting"].map(
        pick(methods, "method", "Default", "cross_rhat_median_mean"))
    settings.loc[main, "rhat_mtmh_pt"] = settings.loc[main, "setting"].map(
        pick(methods, "method", "MTMH+PT", "cross_rhat_median_mean"))
    settings.loc[main, "rhat_thinned"] = settings.loc[main, "setting"].map(
        pick(tails, "method", "default_long", "cross_rhat_median"))
    settings["label"] = [DISPLAY.get(s, s) for s in settings.setting]

    # controlled settings: response design
    for idx in settings.index[settings.group == "controlled"]:
        tag = settings.at[idx, "setting"]
        meta = pd.read_csv(stored(args.controlled_store, tag, "metadata", 0, "default_metadata"),
                           header=None, index_col=0).iloc[:, 0]
        settings.at[idx, "response"] = meta["response"]
        settings.at[idx, "target_noise_fraction"] = pd.to_numeric(meta.get("target_noise_fraction"), errors="coerce")
    settings.to_csv(args.table_dir / "kappa_settings.csv", index=False)

    statistics(settings).to_csv(args.table_dir / "kappa_statistics.csv", index=False)
    print(f"[kappa] wrote kappa_run_level.csv, kappa_settings.csv, kappa_statistics.csv to {args.table_dir}", flush=True)
    return 0


def ols(columns: list[np.ndarray], response: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack([*columns, np.ones_like(response)])
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    resid = response - design @ beta
    dof = len(response) - design.shape[1]
    cov = resid @ resid / dof * np.linalg.inv(design.T @ design)
    r2 = 1 - resid @ resid / np.sum((response - response.mean()) ** 2)
    return beta, np.sqrt(np.diag(cov)), float(r2)


def statistics(settings: pd.DataFrame) -> pd.DataFrame:
    out = []
    add = lambda scope, name, value: out.append(dict(scope=scope, statistic=name, value=value))
    every = settings.dropna(subset=["rhat_default"])
    add("all", "settings", len(every))
    for group, part in every.groupby("group"):
        add("all", f"settings_{group}", len(part))
    rhat = every.rhat_default.to_numpy()
    for estimate in ("cv", "post"):
        nu = every[f"nu_{estimate}"].to_numpy()
        n = every.n.to_numpy()
        add("all", f"spearman_kappa_{estimate}", spearmanr(n / nu, rhat)[0])
        add("all", f"spearman_inverse_noise_{estimate}", spearmanr(1 / nu, rhat)[0])
        y = np.log(rhat - 1)
        beta, se, r2 = ols([np.log(n), np.log(1 / nu)], y)
        add("all", f"ols_{estimate}_a", beta[0]); add("all", f"ols_{estimate}_a_se", se[0])
        add("all", f"ols_{estimate}_b", beta[1]); add("all", f"ols_{estimate}_b_se", se[1])
        add("all", f"ols_{estimate}_r2", r2)
        add("all", f"ols_{estimate}_kappa_only_r2", ols([np.log(n / nu)], y)[2])
    add("all", "spearman_n", spearmanr(every.n, rhat)[0])
    add("all", "ols_log_n_only_r2", ols([np.log(every.n.to_numpy())], np.log(rhat - 1))[2])
    main = settings[settings.group == "main"]
    for column in ("rhat_default", "rhat_thinned", "rhat_mtmh_pt"):
        for name, x in (("kappa_cv", main.n / main.nu_cv), ("n", main.n), ("inverse_noise_cv", 1 / main.nu_cv)):
            add("main", f"spearman_{name}_{column}", spearmanr(x, main[column])[0])
    return pd.DataFrame(out)


if __name__ == "__main__":
    raise SystemExit(main())
