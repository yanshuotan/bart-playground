"""Data loading and sparse-data helpers for fixed-100 runs."""

from __future__ import annotations

import csv
import inspect
import json
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from ucimlrepo import fetch_ucirepo

from bart_playground.DataGenerator import DataGenerator

import experiment_fixed100 as exp

# Re-exported from experiment_fixed100, which owns them. Keeping a second set
# of literals here is what let the two drift apart unnoticed.
GLOBAL_FIXED_TEST_SEED = exp.GLOBAL_FIXED_TEST_SEED
GLOBAL_BASE_TRAIN_SEED = exp.GLOBAL_BASE_TRAIN_SEED
GLOBAL_BASE_CHAIN_SEED = exp.GLOBAL_BASE_CHAIN_SEED


DATASET_CONFIGS = {
    "abalone": {
        "dataset_tag": "fixed100_Abalone",
        "uci_id": 1,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "categorical_columns": "auto",
        "target_column": None,
    },
    "concrete": {
        "dataset_tag": "fixed100_Concrete",
        "uci_id": 165,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "target_column": None,
    },
    "friedman": {
        "dataset_tag": "fixed100_Friedman",
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "n_samples": 2000,
        "n_features": 10,
        "noise": 1.0,
        "seed": 42,
    },
    "friedman_sparse_dir": {
        "dataset_tag": "fixed100_FriedmanSparseDir",
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "n_samples": 2000,
        "n_features": 100,
        "noise": 1.0,
        "seed": 42,
        "dirichlet_prior": True,
        "s_alpha": 1.0,
    },

    "ccpp": {
        "dataset_tag": "fixed100_CCPP",
        "uci_id": 294,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "categorical_columns": [],
        "target_column": None,
    },


    "seoul_bike": {
        "dataset_tag": "fixed100_SeoulBike",
        "uci_id": 560,
        # Matches the stored SeoulBike long chain (see its
        # default_long_metadata.csv).
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "datetime_columns": ["Date"],
        "categorical_columns": "auto",
        "target_column": "Rented Bike Count",
    },

    "cpu_act": {
        "dataset_tag": "fixed100_CPUAct",
        # OpenML version 8 of cpu_act; 8192 rows, 21 numeric features, target usr.
        "openml_id": 44132,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "categorical_columns": [],
        "target_column": None,
    },

    "airfoil": {
        "dataset_tag": "fixed100_Airfoil",
        # OpenML airfoil_self_noise version 1; 1503 rows, 5 numeric features,
        # target pressure (scaled sound pressure level in dB).
        "openml_id": 43919,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "categorical_columns": [],
        "target_column": None,
    },

    "calhousing": {
        "dataset_tag": "fixed100_CalHousing_subsample5000",
        "subsample_n": 5000,
        "subsample_seed": 42,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
    },

    "superconduct": {
        # 21,263 rows and 81 numeric predictors of critical temperature, with
        # no missing values and no categorical columns. Subsampled to the same
        # 5,000 rows as CalHousing so that the two differ in predictor count
        # and not in sample size.
        "dataset_tag": "fixed100_Superconduct_subsample5000",
        "uci_id": 464,
        "subsample_n": 5000,
        "subsample_seed": 42,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "categorical_columns": [],
        "target_column": None,
    },
}


GENERATOR_SCENARIOS = {
    "friedman": "friedman1",
    "friedman_sparse_dir": "friedman1",
}


def _load_generator_dataset(cfg: dict, scenario: str):
    gen_kwargs = {
        "n_samples": int(cfg.get("n_samples", 2000)),
        "n_features": int(cfg.get("n_features", 10)),
        "random_seed": int(cfg.get("seed", 42)),
    }
    if "snr" in cfg:
        gen_kwargs["snr"] = float(cfg["snr"])
    else:
        gen_kwargs["noise"] = float(cfg.get("noise", cfg.get("noise_std", 1.0)))
    generator = DataGenerator(**gen_kwargs)
    X, y = generator.generate(scenario)
    return X.astype(float), np.asarray(y).reshape(-1).astype(float)


def _safe_column_names(df) -> list[str]:
    return [str(c) for c in getattr(df, "columns", [])]


def _select_target_and_features(features, targets, *, target_column=None):
    """Select a numeric target whether UCI stores it in targets or features."""
    X_df = features.copy()
    targets_df = (
        targets.copy() if hasattr(targets, "copy") else pd.DataFrame(targets)
    )

    if isinstance(target_column, str):
        if targets_df is not None and target_column in _safe_column_names(targets_df):
            y = targets_df[target_column].to_numpy()
            consumed = target_column
        elif target_column in _safe_column_names(X_df):
            y = X_df[target_column].to_numpy()
            X_df = X_df.drop(columns=[target_column])
            consumed = None
        else:
            raise ValueError(
                f"target_column={target_column!r} not found. "
                f"target columns={_safe_column_names(targets_df)}, "
                f"feature columns={_safe_column_names(X_df)}"
            )
    elif target_column is not None:
        if targets_df is None or targets_df.shape[1] == 0:
            raise ValueError("Integer target_column requested, but targets are empty")
        y = targets_df.iloc[:, int(target_column)].to_numpy()
        consumed = _safe_column_names(targets_df)[int(target_column)]
    else:
        if targets_df is None or targets_df.shape[1] == 0:
            raise ValueError("Dataset has no target")
        if targets_df.shape[1] != 1:
            raise ValueError(
                f"Multiple target columns found {list(targets_df.columns)}; "
                "set target_column in DATASET_CONFIGS"
            )
        y = targets_df.iloc[:, 0].to_numpy()
        consumed = _safe_column_names(targets_df)[0]

    # UCI sometimes files a genuine predictor under "targets" (SeoulBike's
    # Functioning Day). Anything there that is not the target is a feature.
    for col in _safe_column_names(targets_df):
        if col != consumed:
            X_df[col] = targets_df[col].to_numpy()

    return X_df, np.asarray(y).reshape(-1).astype(float)


def _preprocess_features(features, *, drop_columns=None, categorical_columns=None,
                         datetime_columns=None):
    """Drop nothing by default; dates become a day index, categories one-hot."""
    X_df = features.copy()
    for col in list(drop_columns or []):
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    # A date is a timestamp, not a category: days since the earliest row keeps
    # the ordering in one column instead of one dummy per distinct date.
    for col in list(datetime_columns or []):
        if col not in X_df.columns:
            continue
        parsed = pd.to_datetime(X_df[col], dayfirst=True)
        X_df[col] = (parsed - parsed.min()).dt.days.astype(float)

    if categorical_columns == "auto":
        cat_cols = list(
            X_df.select_dtypes(include=["object", "category", "bool"]).columns
        )
    else:
        cat_cols = [
            c for c in list(categorical_columns or []) if c in X_df.columns
        ]

    if cat_cols:
        # Keep every category, matching the previous long-only runner exactly.
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    return X_df.to_numpy(dtype=float), list(X_df.columns), cat_cols


def _clean_finite_rows(X, y):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask], int((~mask).sum())


UCI_CACHE_DIR = Path(__file__).resolve().parent / "store" / "uci_cache"
OPENML_CACHE_DIR = Path(__file__).resolve().parent / "store" / "openml_cache"


def _fetch_uci_frames(name: str, uci_id: int):
    """Raw UCI features/targets, from a local cache when one exists.

    Cluster compute nodes generally have no outbound network, so the first call
    on a machine that does have one writes the frames to disk and later calls
    read them back. Populate the cache on a login node before submitting a
    batch job.
    """
    features_path = UCI_CACHE_DIR / f"{name}__features.csv"
    targets_path = UCI_CACHE_DIR / f"{name}__targets.csv"
    if features_path.is_file() and targets_path.is_file():
        return pd.read_csv(features_path), pd.read_csv(targets_path)

    ds = fetch_ucirepo(id=uci_id)
    features = ds.data.features.copy()
    targets = ds.data.targets.copy()
    UCI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)
    return features, targets


def _fetch_openml_frames(name: str, data_id: int):
    """Raw OpenML features/target, cached next to the UCI frames.

    Fetching is by numeric data_id, never by name: OpenML keeps several
    versions of a dataset under one name and they do not all have the same
    columns. Same offline story as `_fetch_uci_frames`: populate the cache on a
    login node before submitting a batch job.
    """
    features_path = OPENML_CACHE_DIR / f"{name}__features.csv"
    targets_path = OPENML_CACHE_DIR / f"{name}__targets.csv"
    if features_path.is_file() and targets_path.is_file():
        return pd.read_csv(features_path), pd.read_csv(targets_path)

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True)
    features = bunch.data.copy()
    targets = bunch.target.to_frame() if hasattr(bunch.target, "to_frame") else pd.DataFrame(bunch.target)
    OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)
    return features, targets


def load_dataset(name: str):
    cfg = DATASET_CONFIGS[name]

    if name in GENERATOR_SCENARIOS:
        return _load_generator_dataset(cfg, GENERATOR_SCENARIOS[name])

    if name == "calhousing":
        from sklearn.datasets import fetch_california_housing

        X, y = fetch_california_housing(return_X_y=True)
        rng = np.random.default_rng(cfg["subsample_seed"])
        # Sorting is required to reproduce the earlier California long-run
        # loader and therefore its fixed-test indices exactly.
        idx = np.sort(
            rng.choice(len(X), size=cfg["subsample_n"], replace=False)
        )
        X = X[idx].astype(float)
        y = np.asarray(y[idx]).reshape(-1).astype(float)
        X, y, n_removed = _clean_finite_rows(X, y)
        print(
            f"[LOAD] {name}: X={X.shape}, y={y.shape}, "
            f"removed_nonfinite_rows={n_removed}",
            flush=True,
        )
        return X, y

    if "openml_id" in cfg:
        features, targets = _fetch_openml_frames(name, cfg["openml_id"])
    else:
        features, targets = _fetch_uci_frames(name, cfg["uci_id"])
    features, y = _select_target_and_features(
        features,
        targets,
        target_column=cfg.get("target_column"),
    )
    X, feature_names, cat_cols = _preprocess_features(
        features,
        drop_columns=cfg.get("drop_columns", []),
        categorical_columns=cfg.get("categorical_columns", "auto"),
        datetime_columns=cfg.get("datetime_columns", []),
    )
    X, y, n_removed = _clean_finite_rows(X, y)
    X, y = _apply_subsample(name, cfg, X, y)
    print(
        f"[LOAD] {name}: X={X.shape}, y={y.shape}, "
        f"removed_nonfinite_rows={n_removed}, "
        f"categorical_columns_encoded={cat_cols}, "
        f"features={feature_names}",
        flush=True,
    )

    return X, y


def _apply_subsample(name: str, cfg: dict, X, y):
    """Keep a fixed random subset of the rows when the config asks for one.

    Rows are sorted after selection so the subset, and therefore the fixed test
    indices drawn from it, depend only on the seed and not on draw order.
    """
    size = cfg.get("subsample_n")
    if size is None:
        return X, y
    if size > X.shape[0]:
        raise ValueError(
            f"{name}: subsample_n={size} exceeds the {X.shape[0]} available rows")
    rng = np.random.default_rng(cfg["subsample_seed"])
    idx = np.sort(rng.choice(X.shape[0], size=size, replace=False))
    return X[idx], y[idx]


def get_python_memory_gb():
    try:
        out = subprocess.check_output(
            "ps -C python -o rss= | awk '{s+=$1} END {print s/1024/1024}'",
            shell=True,
            text=True,
        ).strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def get_n_python_processes():
    try:
        out = subprocess.check_output("ps -C python -o pid= | wc -l", shell=True, text=True).strip()
        return int(out)
    except Exception:
        return 0


def memory_logger(out_csv: Path, stop_event: threading.Event, interval_sec: int = 60):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "rss_gb_total", "n_python_processes"])
        while not stop_event.is_set():
            writer.writerow([
                round(time.time() - start, 1),
                round(get_python_memory_gb(), 4),
                get_n_python_processes(),
            ])
            f.flush()
            time.sleep(interval_sec)


def _parse_run_ids(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                start = int(a)
                end = int(b)
                if end < start:
                    raise ValueError(f"Bad run range: {part}")
                out.extend(range(start, end + 1))
            else:
                out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError("No run IDs were provided.")
    if min(out) < 0:
        raise ValueError("Run IDs must be non-negative.")
    return out


class SelectedRunsPatch:
    """Temporarily replace exp.make_fixed100_splits to return only requested runs."""

    def __init__(self, selected_run_ids: list[int]):
        self.selected_run_ids = sorted(set(int(x) for x in selected_run_ids))
        self.selected_set = set(self.selected_run_ids)
        self.original = exp.make_fixed100_splits

    def __enter__(self):
        selected_set = self.selected_set
        original = self.original
        required_n_runs = max(self.selected_run_ids) + 1

        def make_selected_splits(
            X,
            y,
            *,
            n_runs: int,
            n_fixed_test_points: int = 100,
            train_fraction: float = 0.75,
            fixed_test_seed: int = GLOBAL_FIXED_TEST_SEED,
            base_train_seed: int = GLOBAL_BASE_TRAIN_SEED,
        ):
            # Generate all splits up to max selected run so that run_id -> seed mapping
            # remains exactly the same as in the original pipeline.
            all_splits = original(
                X,
                y,
                n_runs=max(n_runs, required_n_runs),
                n_fixed_test_points=n_fixed_test_points,
                train_fraction=train_fraction,
                fixed_test_seed=fixed_test_seed,
                base_train_seed=base_train_seed,
            )
            selected = [s for s in all_splits if int(s["run_id"]) in selected_set]
            if len(selected) != len(selected_set):
                got = sorted(int(s["run_id"]) for s in selected)
                raise RuntimeError(f"Expected run IDs {sorted(selected_set)}, got {got}")
            return selected

        exp.make_fixed100_splits = make_selected_splits
        return self

    def __exit__(self, exc_type, exc, tb):
        exp.make_fixed100_splits = self.original
        return False



FriedmanGenerator = Callable[
    [int, int, int, float], tuple[np.ndarray, np.ndarray]
]


@dataclass(frozen=True)
class SparseVariantSpec:
    name: str
    dataset_tag: str
    family: str
    n_features: int
    n_informative: int
    description: str
    effective_rank: int | None = None


SPECS = {
    "friedman_p20_k5": SparseVariantSpec(
        name="friedman_p20_k5",
        dataset_tag="fixed100_FriedmanSparseDirP20K5",
        family="friedman1_independent",
        n_features=20,
        n_informative=5,
        description="Nonlinear Friedman #1; 5 active and 15 independent nuisance features.",
    ),
    "friedman_p200_k5": SparseVariantSpec(
        name="friedman_p200_k5",
        dataset_tag="fixed100_FriedmanSparseDirP200K5",
        family="friedman1_independent",
        n_features=200,
        n_informative=5,
        description="Nonlinear Friedman #1; 5 active and 195 independent nuisance features.",
    ),
}


def generate_sparse_variant(
    name: str,
    *,
    n_samples: int = 2000,
    seed: int = 42,
    noise_sd: float = 1.0,
    friedman_generator: FriedmanGenerator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Generate one named variant and return ``X, y, metadata``."""
    if name not in SPECS:
        raise KeyError(f"Unknown sparse variant {name!r}; choices={sorted(SPECS)}")
    if n_samples <= 100:
        raise ValueError("n_samples must exceed the fixed test size of 100")
    if noise_sd < 0:
        raise ValueError("noise_sd must be non-negative")

    spec = SPECS[name]
    signal: np.ndarray

    if spec.family == "friedman1_independent":
        # Formal experiments inject the repository's DataGenerator here.  The
        # sklearn fallback exists only so this module can be smoke-tested in
        # isolation; sklearn and the repo use different RNG implementations.
        if friedman_generator is None:
            def generate_friedman(n: int, p: int, s: int, noise: float):
                return make_friedman1(
                    n_samples=n,
                    n_features=p,
                    noise=noise,
                    random_state=s,
                )
            generator_backend = "sklearn_fallback"
        else:
            generate_friedman = friedman_generator
            generator_backend = "repo_DataGenerator"

        X, y = generate_friedman(n_samples, spec.n_features, seed, noise_sd)
        # Recover a noise-free signal only for recorded empirical SNR metadata.
        X_check, signal = generate_friedman(n_samples, spec.n_features, seed, 0.0)
        if not np.array_equal(X, X_check):
            raise RuntimeError("Unexpected Friedman feature mismatch across noise settings")

    else:
        raise RuntimeError(f"Unsupported family: {spec.family}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    signal_var = float(np.var(signal, ddof=0))
    noise_var = float(noise_sd**2)
    metadata = {
        **asdict(spec),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "noise_sd": float(noise_sd),
        "signal_variance_empirical": signal_var,
        "target_snr_variance_ratio": (signal_var / noise_var if noise_var > 0 else float("inf")),
        "informative_feature_indices_zero_based": list(range(spec.n_informative)),
        "dirichlet_prior": True,
        "s_alpha": 1.0,
    }
    if spec.family == "friedman1_independent":
        metadata["generator_backend"] = generator_backend
    return X, y, metadata


def smoke_check(friedman_generator: FriedmanGenerator | None = None) -> None:
    """Cheap deterministic shape/finiteness check for all primary variants."""
    for name, spec in SPECS.items():
        X, y, metadata = generate_sparse_variant(
            name,
            n_samples=200,
            seed=42,
            noise_sd=1.0,
            friedman_generator=friedman_generator,
        )
        assert X.shape == (200, spec.n_features)
        assert y.shape == (200,)
        assert np.isfinite(X).all() and np.isfinite(y).all()
        assert metadata["informative_feature_indices_zero_based"] == list(range(spec.n_informative))



def fail_fast_live_pipeline_check() -> None:
    """Stop before fitting if the live server pipeline lacks required fixes."""
    source = inspect.getsource(exp)
    required = {
        "prediction chains": '"preds"',
    }
    missing = [label for label, token in required.items() if token not in source]
    bad_seed_patterns = {
        "fixed test seed is overridden by a global":
            r"fixed_test_seed\s*=\s*GLOBAL_FIXED_TEST_SEED",
        "base train seed is overridden by a global":
            r"base_train_seed\s*=\s*GLOBAL_BASE_TRAIN_SEED",
    }
    bad = [label for label, pattern in bad_seed_patterns.items() if re.search(pattern, source)]
    if missing or bad:
        details = {"missing_required_outputs": missing, "bad_seed_patterns": bad}
        raise RuntimeError("LIVE PIPELINE PRE-FLIGHT FAILED:\n" + json.dumps(details, indent=2))


def repo_nested_friedman1_generator(
    n_samples: int,
    n_features: int,
    seed: int,
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nest p=20/200 around the exact live-repo p=100 Friedman dataset.

    This deliberately keeps x1,...,x5 and y identical across p.  Generating
    separate (n, p) random matrices with the same seed would change row-wise
    active-feature values when p changes, confounding sparsity with a new data
    realization.
    """
    if n_features < 5:
        raise ValueError("Friedman #1 requires at least five features")
    cfg = {
        "n_samples": int(n_samples),
        "n_features": 100,
        "seed": int(seed),
        "noise": float(noise_sd),
    }
    center_X, center_y = _load_generator_dataset(cfg, "friedman1")
    if n_features <= 100:
        return center_X[:, :n_features].copy(), center_y.copy()

    # Append only response-independent U(0,1) nuisance variables.  A separate
    # seed makes this block reproducible without perturbing the p=100 center.
    nuisance_rng = np.random.default_rng(int(seed) + 200_000)
    extra = nuisance_rng.uniform(
        0.0,
        1.0,
        size=(int(n_samples), int(n_features) - 100),
    )
    return np.column_stack([center_X, extra]), center_y.copy()


def verify_friedman_center_matches_repo() -> None:
    """Ensure the injected p=100 generator exactly reproduces the completed run."""

    repo_X, repo_y = load_dataset("friedman_sparse_dir")
    generated_X, generated_y = repo_nested_friedman1_generator(
        2000, 100, 42, 1.0
    )
    x_match = repo_X.shape == generated_X.shape and np.array_equal(
        repo_X, generated_X
    )
    y_match = repo_y.shape == generated_y.shape and np.array_equal(
        np.asarray(repo_y).reshape(-1), np.asarray(generated_y).reshape(-1)
    )
    if not (x_match and y_match):
        raise RuntimeError(
            "Injected repo DataGenerator does not exactly reproduce "
            "load_dataset('friedman_sparse_dir'). "
            f"x_match={x_match}, y_match={y_match}"
        )
