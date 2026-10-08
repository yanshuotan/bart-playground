"""Loading and separation statistics for the stored long Default chains.

Shared by `budget_comparison.py`, which builds the two-budget table, and
`long_chain_figure.py`, which draws the paper figure.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def load_with_shape(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if "original_shape=" not in header:
        raise ValueError(f"Missing original_shape header: {path}")
    shape = ast.literal_eval(header.split("original_shape=")[-1])
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)


def dataset_tag(store_root: Path, dataset: str) -> str:
    """File-name tag of a store directory, read from the prediction files.

    Taken from the files themselves rather than dataset_metadata.csv, which every
    run rewrites and which can therefore be stale.
    """
    preds = sorted((store_root / dataset / "preds").glob("*__default_long__preds.csv"))
    if not preds:
        raise FileNotFoundError(f"No long-chain predictions in {store_root / dataset / 'preds'}")
    return preds[0].name.split("__run")[0]


def long_predictions(store_root: Path, dataset: str, run: int, burn: int) -> np.ndarray:
    """(chain, draw, test point) post-burn long-chain predictions."""
    tag = dataset_tag(store_root, dataset)
    path = store_root / dataset / "preds" / f"{tag}__run{run:03d}__default_long__preds.csv"
    return load_with_shape(path).transpose(0, 2, 1)[:, burn:, :]


def short_predictions(store_root: Path, dataset: str, run: int, burn: int,
                      method: str = "default") -> np.ndarray:
    tag = dataset_tag(store_root, dataset)
    path = store_root / dataset / "preds" / f"{tag}__run{run:03d}__{method}__preds.csv"
    return load_with_shape(path).transpose(0, 2, 1)[:, burn:, :]


def long_runs(store_root: Path, dataset: str) -> list[int]:
    """Run ids with a stored long chain; empty when the dataset has none."""
    try:
        tag = dataset_tag(store_root, dataset)
    except FileNotFoundError:
        return []
    return sorted(int(re.search(r"__run(\d+)__", p.name).group(1))
                  for p in (store_root / dataset / "preds").glob(f"{tag}__run*__default_long__preds.csv"))


def between_within_ratio(draws: np.ndarray) -> float:
    """Mean squared centroid displacement over mean within-chain squared radius."""
    centers = draws.mean(axis=1)
    between = float(np.mean(np.sum((centers - centers.mean(axis=0)) ** 2, axis=1)))
    within = np.mean([np.mean(np.sum((chain - chain.mean(axis=0)) ** 2, axis=1)) for chain in draws])
    return float(between / within) if within > 0 else float("nan")


def separation_index(draws: np.ndarray, n_blocks: int = 4) -> tuple[float, float]:
    """Chain separation measured against what one chain's own time blocks produce.

    `between_within_ratio` has no scale of its own, so divide it by the same
    statistic computed on `n_blocks` consecutive blocks of a single chain,
    averaged over chains. Because a block holds 1/n_blocks of a chain's draws,
    its centroid scatters n_blocks times as far, so the raw ratio is 1/n_blocks
    under perfect mixing whatever the autocorrelation; multiplying by `n_blocks`
    puts the null at one. Returns the index and the null denominator behind it.
    """
    across = between_within_ratio(draws)
    length = draws.shape[1] // n_blocks
    within = float(np.mean([
        between_within_ratio(chain[: n_blocks * length].reshape(n_blocks, length, draws.shape[2]))
        for chain in draws
    ]))
    return (float(n_blocks * across / within) if within > 0 else float("nan")), within


def worst_direction(draws: np.ndarray, ridge_fraction: float = 1e-8) -> np.ndarray:
    """Draws projected onto the leading between/within generalized eigenvector."""
    m, n, p = draws.shape
    means = draws.mean(axis=1)
    grand = means.mean(axis=0)
    within = np.zeros((p, p), dtype=float)
    for chain in draws:
        centered = chain - chain.mean(axis=0)
        within += centered.T @ centered / (n - 1)
    within /= m
    offsets = means - grand
    between = n * offsets.T @ offsets / (m - 1)
    ridge = max(ridge_fraction * np.trace(within) / p, ridge_fraction)
    eigenvalues, eigenvectors = eigh(between / n, within + ridge * np.eye(p))
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    direction /= np.linalg.norm(direction)
    return np.einsum("mnp,p->mn", draws, direction)
