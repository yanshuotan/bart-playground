#!/usr/bin/env python3
"""Default-sampler agreement when only the response noise changes.

The training inputs, training size and prediction panel stay those of runs 0-4
of the main experiment on SeoulBike, Airfoil and Abalone; only the training
response changes. Three responses are used:

* ``observed``: the observed training response;
* ``added``: the observed response plus independent Gaussian noise, scaled so
  that the noise fraction 1 - R^2 reaches a target, taking the observed
  response's own noise fraction to be ``nu0``;
* ``semi``: a semi-synthetic response f_hat(X) + Gaussian noise at the target
  noise fraction, where f_hat is a histogram gradient boosting fit (500
  iterations) to the full dataset. Its prediction panel is scored against
  f_hat, which is the true regression function of this response.

``nu0`` is the posterior noise fraction of the stored long Default chains,
sigma_bar^2 / var(y_train) averaged over runs 0-4, rounded to three decimals.

Each configuration runs four Default chains of 10,000 iterations per run and is
stored like ``scaling_sweep.py`` under ``<store-dir>/ctl_<Source>_<kind>[_nfNNNN]``.
Run 0 of every configuration also gets a separate Default fit (random state
777) whose move acceptance over iterations 3,001-5,000 is recorded.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment_fixed100 as exp  # noqa: E402
from experiment_fixed100 import (  # noqa: E402
    _load_2d_or_empty,
    _save_numeric_csv,
    _write_key_value_csv,
    run_long_default_chain_streaming,
)
from fixed100_support import load_dataset  # noqa: E402

SOURCES = {"seoul_bike": "SeoulBike", "airfoil": "Airfoil", "abalone": "Abalone"}
# (kind, target noise fraction); "nu0" is the observed response's own level
PLAN = {
    "seoul_bike": [("observed", None), ("added", 0.17), ("added", 0.4),
                   ("semi", "nu0"), ("semi", 0.17), ("semi", 0.4)],
    "airfoil": [("observed", None), ("added", 0.1), ("added", 0.4),
                ("semi", "nu0"), ("semi", 0.1)],
    "abalone": [("observed", None), ("semi", "nu0"), ("semi", 0.1),
                ("semi", 0.04), ("semi", 0.01)],
}
RUNS = (0, 1, 2, 3, 4)
BURN = 3000
# seed tuple of the response noise: (7, run, int(1000 * target), KIND_CODE[kind])
KIND_CODE = {"observed": 4, "added": 5, "semi": 5}
CHAIN_SEED_OFFSET = 500_000
ACCEPTANCE_SEED = 777


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-store", type=Path, default=HERE / "store",
                        help="Store of the main experiment: splits and long Default chains.")
    parser.add_argument("--store-dir", type=Path, default=HERE / "store_controlled")
    parser.add_argument("--sources", nargs="+", default=list(PLAN), choices=list(PLAN))
    parser.add_argument("--runs", nargs="+", type=int, default=list(RUNS))
    parser.add_argument("--ndpost", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build every response and print the plan without sampling.")
    return parser.parse_args(supplied)


def read_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline()
    shape = tuple(int(s) for s in header.split("original_shape=")[-1].strip(" ()\n").split(",") if s.strip())
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)


def split(main_store: Path, source: str, run: int, kind: str) -> np.ndarray:
    tag = f"fixed100_{SOURCES[source]}"
    return read_csv(main_store / tag / "indices" / f"{tag}__run{run:03d}__{kind}.csv").astype(int).ravel()


def observed_noise_fraction(main_store: Path, source: str, y: np.ndarray) -> float:
    """Posterior noise fraction of the long Default chains, runs 0-4."""
    tag = f"fixed100_{SOURCES[source]}"
    values = []
    for run in range(5):
        train = split(main_store, source, run, "train_idx")
        sigma2 = read_csv(main_store / tag / "sigmas" / f"{tag}__run{run:03d}__default_long__sigmas.csv")
        sigma2 = np.asarray(sigma2, dtype=float).reshape(sigma2.shape[0], -1)[:, BURN:]
        scale = np.ptp(y[train])
        values.append(sigma2.mean() * scale ** 2 / y[train].var())
    return round(float(np.mean(values)), 3)


def config_tag(source: str, kind: str, target: float | None) -> str:
    tag = f"ctl_{SOURCES[source]}_{kind}"
    return tag if target is None else f"{tag}_nf{int(round(1000 * target)):04d}"


def build_configs(args: argparse.Namespace) -> list[dict]:
    """Every (configuration, run) with its training and test responses."""
    out = []
    for source in args.sources:
        X, y = load_dataset(source)
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        nu0 = observed_noise_fraction(args.main_store, source, y)
        from sklearn.ensemble import HistGradientBoostingRegressor
        f_hat = HistGradientBoostingRegressor(max_iter=500, random_state=0).fit(X, y).predict(X)
        for kind, target in PLAN[source]:
            target = nu0 if target == "nu0" else target
            tag = config_tag(source, kind, target)
            for run in args.runs:
                train = split(args.main_store, source, run, "train_idx")
                test = split(args.main_store, source, run, "fixed_test_idx")
                rng = np.random.default_rng([7, run, int(1000 * (target or 0)), KIND_CODE[kind]])
                if kind == "observed":
                    y_train, y_test = y[train], y[test]
                elif kind == "added":
                    variance = (target - nu0) * y[train].var() / (1 - target)
                    y_train, y_test = y[train] + rng.normal(0, np.sqrt(variance), len(train)), y[test]
                else:
                    variance = target / (1 - target) * f_hat[train].var()
                    y_train, y_test = f_hat[train] + rng.normal(0, np.sqrt(variance), len(train)), f_hat[test]
                out.append(dict(source=source, kind=kind, target=target, nu0=nu0, tag=tag, run=run,
                                train=train, test=test, X_train=X[train], X_test=X[test],
                                y_train=y_train, y_test=y_test))
    return out


def finished(args, cfg) -> bool:
    root = args.store_dir / cfg["tag"]
    return (root / "preds" / f"{cfg['tag']}__run{cfg['run']:03d}__default__preds.csv").is_file()


def chain_task(args, cfg, chain_id):
    tmp_root = args.store_dir / cfg["tag"] / "_stream_tmp" / f"run{cfg['run']:03d}"
    return run_long_default_chain_streaming(
        chain_id=chain_id,
        chain_seed=CHAIN_SEED_OFFSET + exp.GLOBAL_BASE_CHAIN_SEED + 1000 * cfg["run"] + chain_id,
        run_id=cfg["run"], X_train=cfg["X_train"], y_train=cfg["y_train"],
        X_test_fixed=cfg["X_test"], y_test_fixed=cfg["y_test"],
        ndpost=args.ndpost, nskip=0, n_trees=args.n_trees,
        proposal_probs_default=exp.default_proposal_probs, store_every=1,
        chunk_size=args.chunk_size, tmp_dir=tmp_root, store_preds=True,
        dirichlet_prior=False, s_alpha=1.0,
    )


def acceptance_task(args, cfg):
    """Move acceptance of one Default fit over iterations 3,001-5,000."""
    from bart_playground import DefaultBART

    model = DefaultBART(ndpost=3000, nskip=0, n_trees=args.n_trees, tol=1,
                        proposal_probs=exp.default_proposal_probs, random_state=ACCEPTANCE_SEED,
                        dirichlet_prior=False, s_alpha=1.0)
    model.fit(cfg["X_train"], cfg["y_train"], quietly=True)
    sampler = model.sampler
    accepted0, proposed0 = dict(sampler.move_accepted_counts), dict(sampler.move_success_counts)
    sampler.continue_run(2000, last_state=model.trace[-1], quietly=True)
    rates = {move: (sampler.move_accepted_counts[move] - accepted0[move])
             / max(1, sampler.move_success_counts[move] - proposed0[move]) for move in accepted0}
    _write_key_value_csv(
        args.store_dir / cfg["tag"] / "metadata" / f"{cfg['tag']}__run{cfg['run']:03d}__default_acceptance.csv",
        {"iterations": "3001-5000", "random_state": ACCEPTANCE_SEED,
         **{f"accept_{move}": rate for move, rate in rates.items()}},
    )
    return cfg["tag"]


def save_run(args, cfg) -> None:
    root, tag, run = args.store_dir / cfg["tag"], cfg["tag"], cfg["run"]
    tmp_root = root / "_stream_tmp" / f"run{run:03d}"
    sigmas, rmses, preds = [], [], []
    for chain_id in range(args.n_chains):
        chain_dir = tmp_root / f"run{run:03d}_chain{chain_id:03d}"
        sigmas.append(_load_2d_or_empty(chain_dir / "sigmas_rows.csv", 1).reshape(-1, 1))
        rmses.append(_load_2d_or_empty(chain_dir / "rmses_rows.csv", 1).reshape(-1))
        preds.append(_load_2d_or_empty(chain_dir / "preds_rows.csv", len(cfg["test"])).T)
    stacked = {"sigmas": np.asarray(sigmas), "rmses": np.asarray(rmses), "preds": np.asarray(preds)}
    for kind, values in stacked.items():
        _save_numeric_csv(root / kind / f"{tag}__run{run:03d}__default__{kind}.csv", values)
    _save_numeric_csv(root / "indices" / f"{tag}__run{run:03d}__train_idx.csv", cfg["train"])
    _save_numeric_csv(root / "indices" / f"{tag}__run{run:03d}__fixed_test_idx.csv", cfg["test"])
    _save_numeric_csv(root / "subsample_y_train" / f"{tag}__run{run:03d}__subsample_y_train.csv", cfg["y_train"])
    _save_numeric_csv(root / "subsample_y_test" / f"{tag}__run{run:03d}__subsample_y_test.csv", cfg["y_test"])
    _write_key_value_csv(
        root / "metadata" / f"{tag}__run{run:03d}__default_metadata.csv",
        {"phase": "controlled", "source": cfg["source"], "response": cfg["kind"],
         "target_noise_fraction": "" if cfg["target"] is None else cfg["target"],
         "observed_noise_fraction_nu0": cfg["nu0"], "run_id": run,
         "n_train": len(cfg["train"]), "n_test": len(cfg["test"]), "ndpost": args.ndpost,
         "n_chains": args.n_chains, "n_trees": args.n_trees,
         "chain_seed": f"{CHAIN_SEED_OFFSET} + {exp.GLOBAL_BASE_CHAIN_SEED} + 1000*run + chain",
         "test_target": "f_hat(X_test)" if cfg["kind"] == "semi" else "observed y_test",
         "preds_shape": str(stacked["preds"].shape)},
    )
    shutil.rmtree(tmp_root, ignore_errors=True)


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    configs = build_configs(args)
    todo = [cfg for cfg in configs if not finished(args, cfg)]
    print(f"[controlled] {len(configs)} configuration-runs, {len(todo)} to run", flush=True)
    for cfg in configs:
        print(f"  {cfg['tag']:32s} run {cfg['run']} n={len(cfg['train'])} target={cfg['target']} "
              f"nu0={cfg['nu0']} var(y_train)={cfg['y_train'].var():.4g}", flush=True)
    if args.dry_run:
        return 0
    # largest training sets first so the end of the schedule is short
    tasks = [(cfg, chain_id) for cfg in sorted(todo, key=lambda c: -len(c["train"]))
             for chain_id in range(args.n_chains)]
    acceptance = [cfg for cfg in configs if cfg["run"] == 0 and not (
        args.store_dir / cfg["tag"] / "metadata" / f"{cfg['tag']}__run000__default_acceptance.csv").is_file()]
    jobs = [delayed(chain_task)(args, cfg, chain_id) for cfg, chain_id in tasks]
    jobs += [delayed(acceptance_task)(args, cfg) for cfg in acceptance]
    print(f"[controlled] {len(tasks)} chains and {len(acceptance)} acceptance fits on {args.n_jobs} workers", flush=True)
    for done in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
        label = done if isinstance(done, str) else Path(done["tmp_dir"]).name
        print(f"[controlled] done {label}", flush=True)
    for cfg in todo:
        save_run(args, cfg)
        print(f"[controlled] {cfg['tag']} run {cfg['run']:03d} saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
