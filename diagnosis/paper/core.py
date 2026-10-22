"""Loading and prediction-space diagnostics shared by every paper analysis.

One copy of the pieces the section scripts all need: reading a stored run,
pointwise R-hat, and the cross-fitted between/within directions behind every
projection figure. That last one was previously duplicated verbatim in the
section 3 and section 6 scripts, which meant the eigenproblem behind Figure 1
and the one behind Figures 4-5 could drift apart without anything failing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr
from scipy.linalg import eigh


def load_with_shape(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if "original_shape=" not in header:
        raise ValueError(f"Missing original_shape header: {path}")
    shape = ast.literal_eval(header.split("original_shape=")[-1])
    return np.loadtxt(path, delimiter=",", comments="#").reshape(shape)


def prediction_variance_terms(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise W and B of the finite equal-weight prediction mixture.

    Draws have shape (chain, draw, input). Population denominators give
    Var(all retained predictions at input j) = W_j + B_j exactly.
    """
    return draws.var(axis=1, ddof=0).mean(axis=0), draws.mean(axis=1).var(axis=0, ddof=0)


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


def pointwise_rhat_summary(draws: np.ndarray) -> dict[str, float]:
    """Median and upper-tail modern rank-split R-hat over prediction coordinates."""
    data = xr.DataArray(draws, dims=("chain", "draw", "test_point"), name="prediction")
    values = np.asarray(az.rhat(data, method="rank")["prediction"].values, dtype=float)
    return {
        "median": float(np.nanmedian(values)),
        "q90": float(np.nanquantile(values, 0.90)),
        "maximum": float(np.nanmax(values)),
    }


def worst_directions(
    draws: np.ndarray,
    n_directions: int = 2,
    ridge_fraction: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Leading between/within directions and the draws projected onto them.

    Directions are fitted on the first half of every chain; ``fit_draws``
    marks that boundary so plots can show the held second half on its own.
    No R-hat is reported on these axes: each method has its own fitted
    directions, so projected R-hat values are not comparable across methods.
    """
    m, n, p = draws.shape
    if n_directions < 1 or n_directions > p:
        raise ValueError(f"n_directions must be between 1 and {p}; got {n_directions}")
    half = n // 2
    if half < 2 or n - half < 2:
        raise ValueError(f"Need at least four draws per chain to cross-fit; got {n}")
    fit = draws[:, :half, :]
    means = fit.mean(axis=1)
    grand = means.mean(axis=0)
    within = np.zeros((p, p), dtype=float)
    for chain in fit:
        centered = chain - chain.mean(axis=0)
        within += centered.T @ centered / (half - 1)
    within /= m
    offsets = means - grand
    ridge = max(ridge_fraction * np.trace(within) / p, ridge_fraction)
    eigenvalues, eigenvectors = eigh(
        offsets.T @ offsets / (m - 1), within + ridge * np.eye(p)
    )
    order = np.argsort(eigenvalues)[::-1][:n_directions]
    values = np.maximum(eigenvalues[order], 0.0)
    directions = eigenvectors[:, order].copy()
    for column in range(directions.shape[1]):
        directions[:, column] /= np.linalg.norm(directions[:, column])
        anchor = int(np.argmax(np.abs(directions[:, column])))
        if directions[anchor, column] < 0:
            directions[:, column] *= -1
    projected = np.einsum("mnp,pk->mnk", draws - grand[None, None, :], directions)
    return {
        "eigenvalues": values,
        "directions": directions,
        "fit_draws": half,
        "projected": projected,
    }


def worst_direction(draws: np.ndarray, ridge_fraction: float = 1e-8) -> np.ndarray:
    """Compatibility wrapper returning the leading-direction projection."""
    return worst_directions(draws, n_directions=1, ridge_fraction=ridge_fraction)["projected"][:, :, 0]


def pointwise_rhat(draws: np.ndarray) -> np.ndarray:
    """Rank-normalized split R-hat at every test point."""
    data = xr.DataArray(draws, dims=("chain", "draw", "test_point"), name="prediction")
    return np.asarray(az.rhat(data, method="rank")["prediction"].values)


def summarize_rhat(values: np.ndarray) -> dict[str, float]:
    """Median, quartiles, upper tail and maximum across test points."""
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.nanmedian(values)),
        "q25": float(np.nanquantile(values, 0.25)),
        "q75": float(np.nanquantile(values, 0.75)),
        "q90": float(np.nanquantile(values, 0.90)),
        "maximum": float(np.nanmax(values)),
    }


def worst_direction_summary(draws: np.ndarray,
                            ridge_fraction: float) -> dict[str, np.ndarray | float]:
    """The two leading directions, packaged as the section scripts expect.

    A view of :func:`worst_directions`; the arithmetic lives there only.
    """
    found = worst_directions(draws, n_directions=2, ridge_fraction=ridge_fraction)
    values, projected = found["eigenvalues"], found["projected"]
    return {
        "lambda_max": float(values[0]),
        "lambda_second": float(values[1]),
        "eigen_gap": float(values[0] - values[1]),
        "directions": found["directions"],
        "fit_draws": found["fit_draws"],
        "projected": projected[:, :, 0],
        "projected_2d": projected,
    }


def markdown_table(frame) -> str:
    """A pipe table, for the summary markdown the scripts write beside the CSVs."""
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return chr(10).join(lines)


CACHE_ROOT = Path(__file__).resolve().parent / "cache"


def run_inputs(store_root: Path, dataset: str, run: int, methods) -> list[Path]:
    """Store files an analysis of one run reads: the split and each method's draws."""
    directory = store_root / dataset
    files = [*directory.glob(f"indices/*__run{run:03d}__*"),
             *directory.glob(f"subsample_y_test/*__run{run:03d}__*")]
    for method in methods:
        files += [*directory.glob(f"preds/*__run{run:03d}__{method}__preds.csv"),
                  *directory.glob(f"sigmas/*__run{run:03d}__{method}__sigmas.csv")]
    return files


def input_signature(settings: dict, paths) -> str:
    """Hash of the settings and of every input's name, size and mtime.

    Replacing a store file changes its mtime, so the entries that read it are
    recomputed; contents are not hashed because the draw files are large.
    """
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode())
    for path in sorted(Path(p) for p in paths):
        stat = path.stat()
        digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns};".encode())
    return digest.hexdigest()


class ResultCache:
    """Per-key results on disk, reused while the settings and inputs are unchanged.

    One pickle per key under cache/<name>/, so adding a dataset or replacing one
    method's draws recomputes only the entries that read them. Callers bump their
    own CACHE_VERSION, part of `settings`, when a calculation changes; `enabled`
    False recomputes everything and still rewrites the entries.
    """

    def __init__(self, name: str, settings: dict, enabled: bool = True):
        self.directory = CACHE_ROOT / name
        self.settings = settings
        self.enabled = enabled
        self.hits = self.misses = 0

    def _path(self, key) -> Path:
        return self.directory / ("__".join(str(part) for part in key) + ".pkl")

    def get(self, key, inputs):
        path = self._path(key)
        if self.enabled and path.is_file():
            with path.open("rb") as stream:
                entry = pickle.load(stream)
            if entry["signature"] == input_signature(self.settings, inputs):
                self.hits += 1
                return entry["payload"]
        self.misses += 1
        return None

    def put(self, key, inputs, payload) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump({"signature": input_signature(self.settings, inputs),
                         "payload": payload}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
