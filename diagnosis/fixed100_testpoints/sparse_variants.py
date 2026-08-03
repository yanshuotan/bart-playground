"""Deterministic sparse-regression variants for BART mixing experiments.

The four primary variants form two controlled comparisons:

1. Nonlinear sparsity severity (Friedman #1): 5/20 and 5/200 active
   features, with the already-completed 5/100 experiment as the midpoint.
2. Linear design geometry: 4/100 active features under independent and
   correlated designs.  The independent case is the Celeux et al. sparse
   uncorrelated benchmark exposed by scikit-learn.  The correlated case keeps
   the same feature count, signal scale, and noise scale but uses a low-rank
   Gaussian design to create correlated predictors.

All variants use n=2000, seed=42, and Gaussian noise SD=1 by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.datasets import make_friedman1, make_regression, make_sparse_uncorrelated


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
    "linear_independent_p100_k4": SparseVariantSpec(
        name="linear_independent_p100_k4",
        dataset_tag="fixed100_SparseLinearIndependentP100K4",
        family="sparse_linear_independent",
        n_features=100,
        n_informative=4,
        description="Celeux/scikit-learn sparse uncorrelated linear benchmark; 4 active features.",
    ),
    "linear_correlated_p100_k4": SparseVariantSpec(
        name="linear_correlated_p100_k4",
        dataset_tag="fixed100_SparseLinearCorrelatedP100K4",
        family="sparse_linear_correlated",
        n_features=100,
        n_informative=4,
        effective_rank=20,
        description="Sparse linear response with correlated low-rank Gaussian predictors.",
    ),
}


def _add_noise(signal: np.ndarray, *, noise_sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(signal, dtype=float) + rng.normal(0.0, noise_sd, size=len(signal))


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

    elif spec.family == "sparse_linear_independent":
        X, signal = make_sparse_uncorrelated(
            n_samples=n_samples,
            n_features=spec.n_features,
            random_state=seed,
        )
        # Keep feature generation independent of the additive-noise RNG.
        y = _add_noise(signal, noise_sd=noise_sd, seed=seed + 10_000)

    elif spec.family == "sparse_linear_correlated":
        X, raw_signal, coefficients = make_regression(
            n_samples=n_samples,
            n_features=spec.n_features,
            n_informative=spec.n_informative,
            effective_rank=spec.effective_rank,
            tail_strength=0.5,
            noise=0.0,
            shuffle=False,
            coef=True,
            random_state=seed,
        )
        # Match the theoretical signal SD of the independent benchmark:
        # Var(x1 + 2x2 - 2x3 - 1.5x4) = 11.25 for independent N(0,1) X.
        target_signal_sd = float(np.sqrt(11.25))
        raw_sd = float(np.std(raw_signal, ddof=0))
        if not np.isfinite(raw_sd) or raw_sd <= 0:
            raise RuntimeError("Degenerate correlated linear signal")
        signal = np.asarray(raw_signal, dtype=float) * (target_signal_sd / raw_sd)
        y = _add_noise(signal, noise_sd=noise_sd, seed=seed + 10_000)
        active = np.flatnonzero(np.asarray(coefficients) != 0).tolist()
        if active != list(range(spec.n_informative)):
            raise RuntimeError(f"Unexpected informative indices with shuffle=False: {active}")

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


if __name__ == "__main__":
    smoke_check()
    for variant in SPECS:
        X, y, metadata = generate_sparse_variant(variant)
        print(
            variant,
            "X=", X.shape,
            "y=", y.shape,
            "signal_var=", round(float(metadata["signal_variance_empirical"]), 4),
            "SNR=", round(float(metadata["target_snr_variance_ratio"]), 4),
        )
