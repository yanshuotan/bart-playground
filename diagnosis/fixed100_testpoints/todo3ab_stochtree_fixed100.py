#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)



DATASET_CANON = {
    "abalone": "fixed100_Abalone",
    "concrete": "fixed100_Concrete",
    "friedman": "fixed100_Friedman",
}


def read_index_csv(path: Path) -> np.ndarray:
    """
    Robustly read fixed100 index CSV files.

    Handles common pandas outputs:
    - one column with header
    - one column without header
    - two columns: Unnamed row index + train_idx/fixed_test_idx
    - columns named idx/index/train_idx/fixed_test_idx
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required split index file: {path}")

    candidates = []

    # Try normal pandas header first.
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 0:
            candidates.append(df)
    except Exception:
        pass

    # Try no-header version.
    try:
        df_no_header = pd.read_csv(path, header=None)
        if df_no_header.shape[1] > 0:
            candidates.append(df_no_header)
    except Exception:
        pass

    best = None
    best_score = -1
    debug_cols = []

    for df in candidates:
        for col in df.columns:
            name = str(col)
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                debug_cols.append((name, 0, None, None))
                continue

            arr = vals.astype(int).to_numpy()

            # Score columns. Prefer explicit index columns and non-Unnamed columns.
            lname = name.lower()
            score = len(arr)

            if any(k in lname for k in ["idx", "index", "train", "test", "fixed"]):
                score += 100000
            if lname.startswith("unnamed"):
                score -= 10000

            # A real dataset index column should usually not just be 0..n-1.
            if len(arr) > 5 and np.array_equal(arr, np.arange(len(arr))):
                score -= 50000

            debug_cols.append((name, len(arr), int(arr.min()), int(arr.max())))

            if score > best_score:
                best_score = score
                best = arr

    if best is None or best.size == 0:
        raise ValueError(
            f"Could not read any integer indices from {path}. "
            f"Observed columns: {debug_cols}. First lines:\n{path.read_text()[:500]}"
        )

    # Drop accidental duplicated header-derived values if any.
    best = np.asarray(best, dtype=int)

    if best.min() < 0:
        raise ValueError(f"Negative index found in {path}: min={best.min()}")

    return best


def read_numeric_csv(path: Path) -> np.ndarray:
    """
    Read numeric snapshot CSVs saved by the fixed100 pipeline.

    These files may contain comment lines such as:
    # original_shape=(100, 7)

    Use comment="#" and header=None so pandas does not treat the
    comment line as a malformed CSV header.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    arr = df.to_numpy(dtype=float)

    if arr.size == 0:
        raise ValueError(
            f"Could not read numeric array from {path}. "
            f"First lines:\n{path.read_text()[:500]}"
        )

    return arr

def load_abalone():
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=1)
    Xdf = data.data.features.copy()
    ydf = data.data.targets.copy()

    # Current experiment used X=(4177, 7), so drop categorical Sex.
    if "Sex" in Xdf.columns:
        Xdf = Xdf.drop(columns=["Sex"])
    Xdf = Xdf.select_dtypes(include=[np.number])

    X = Xdf.to_numpy(dtype=float)
    y = ydf.iloc[:, 0].to_numpy(dtype=float)
    return X, y


def load_concrete():
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=165)
    Xdf = data.data.features.copy()
    ydf = data.data.targets.copy()

    Xdf = Xdf.select_dtypes(include=[np.number])
    X = Xdf.to_numpy(dtype=float)
    y = ydf.iloc[:, 0].to_numpy(dtype=float)
    return X, y


def load_friedman(index_store: Path, run_id: int):
    """
    Load Friedman from the original fixed100 pipeline instead of guessing
    make_friedman1 seed/noise.

    We also add repo root to sys.path because run_fixed100 imports
    bart_playground.DataGenerator.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]  # /root/bart-playground
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from run_fixed100 import load_dataset as fixed100_load_dataset

    X, y = fixed100_load_dataset("friedman")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    log(f"[Friedman loader] loaded from run_fixed100.load_dataset('friedman'): X={X.shape}, y={y.shape}")
    return X, y

def load_dataset(name: str, index_store: Path, run_id: int):
    name = name.lower()
    if name == "abalone":
        return load_abalone()
    if name == "concrete":
        return load_concrete()
    if name == "friedman":
        return load_friedman(index_store, run_id)
    raise ValueError(f"Unknown dataset: {name}")


def validate_split_and_data(dataset: str, run_id: int, index_store: Path, X: np.ndarray, y: np.ndarray):
    canon = DATASET_CANON[dataset]
    idx_dir = index_store / canon / "indices"

    train_idx = read_index_csv(idx_dir / f"{canon}__run{run_id:03d}__train_idx.csv")
    test_idx = read_index_csv(idx_dir / f"{canon}__run{run_id:03d}__fixed_test_idx.csv")

    if train_idx.max() >= len(y) or test_idx.max() >= len(y):
        raise ValueError(
            f"Index out of bounds for {dataset} run{run_id:03d}: "
            f"max train={train_idx.max()}, max test={test_idx.max()}, n={len(y)}"
        )

    x_ref = index_store / canon / "subsample_X_test" / f"{canon}__run{run_id:03d}__subsample_X_test.csv"
    y_ref = index_store / canon / "subsample_y_test" / f"{canon}__run{run_id:03d}__subsample_y_test.csv"

    if not x_ref.exists() or not y_ref.exists():
        raise FileNotFoundError(
            f"Hard verification requires stored subsample_X_test and subsample_y_test for {dataset} run{run_id:03d}. "
            f"Missing {x_ref} or {y_ref}."
        )

    X_ref = read_numeric_csv(x_ref)
    y_ref_arr = read_numeric_csv(y_ref).reshape(-1)

    X_test = X[test_idx]
    y_test = y[test_idx]

    if X_ref.shape[0] != X_test.shape[0]:
        raise ValueError(f"X_test row mismatch: loaded {X_test.shape}, stored {X_ref.shape}")

    if y_test.shape != y_ref_arr.shape:
        raise ValueError(f"y_test shape mismatch: loaded {y_test.shape}, stored {y_ref_arr.shape}")

    if not np.allclose(y_test, y_ref_arr, rtol=1e-8, atol=1e-8):
        raise ValueError(f"y_test values do not match stored fixed100 snapshot for {dataset} run{run_id:03d}")

    # Some fixed100 outputs only store the first few X columns in subsample_X_test.
    # If stored X has fewer columns than loaded X, verify the stored prefix only.
    if X_ref.shape[1] == X_test.shape[1]:
        X_to_check = X_test
        x_check_mode = "full_X"
    elif X_ref.shape[1] < X_test.shape[1]:
        X_to_check = X_test[:, :X_ref.shape[1]]
        x_check_mode = f"partial_X_first_{X_ref.shape[1]}_cols"
    else:
        raise ValueError(f"Stored X has more columns than loaded X: loaded {X_test.shape}, stored {X_ref.shape}")

    if not np.allclose(X_to_check, X_ref, rtol=1e-8, atol=1e-8):
        raise ValueError(
            f"X_test values do not match stored fixed100 snapshot for {dataset} run{run_id:03d}; "
            f"mode={x_check_mode}, loaded_check={X_to_check.shape}, stored={X_ref.shape}"
        )

    log(
        f"[split check] {canon} run{run_id:03d}: "
        f"train_n={len(train_idx)}, test_n={len(test_idx)}, "
        f"X_loaded={X_test.shape}, X_snapshot={X_ref.shape}, check={x_check_mode}, y_check=full"
    )

    return train_idx, test_idx


def extract_prediction_array(pred, n_test: int):
    """
    StochTree may return predictions either directly as an array
    or inside a dict. Extract the array whose shape contains n_test.
    """
    if isinstance(pred, dict):
        preferred_keys = [
            "y_hat_test",
            "yhat_test",
            "test_predictions",
            "predictions",
            "prediction",
            "y_hat",
            "yhat",
            "mean",
            "mu",
        ]

        for k in preferred_keys:
            if k in pred:
                try:
                    arr = np.asarray(pred[k], dtype=float)
                    if n_test in arr.shape:
                        log(f"[prediction extract] using dict key={k}, shape={arr.shape}")
                        return pred[k]
                except Exception:
                    pass

        # Fallback: scan all dict values.
        candidates = []
        for k, v in pred.items():
            try:
                arr = np.asarray(v, dtype=float)
                if n_test in arr.shape:
                    candidates.append((k, arr.shape, v))
            except Exception:
                continue

        if candidates:
            k, shape, v = candidates[0]
            log(f"[prediction extract] using scanned dict key={k}, shape={shape}")
            return v

        raise ValueError(
            "Could not find prediction array in StochTree dict. "
            f"Keys={list(pred.keys())}"
        )

    return pred


def normalize_prediction_array(pred, n_test: int):
    pred = extract_prediction_array(pred, n_test)
    arr = np.asarray(pred, dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] != n_test:
            raise ValueError(f"1D prediction length {arr.shape[0]} != n_test {n_test}")
        return arr.reshape(1, n_test)

    if arr.ndim == 2:
        if arr.shape[1] == n_test:
            return arr
        if arr.shape[0] == n_test:
            return arr.T
        raise ValueError(f"Cannot orient 2D predictions with shape {arr.shape}; n_test={n_test}")

    if arr.ndim == 3:
        # Move test dimension to last, then flatten non-test axes.
        axes_with_test = [i for i, s in enumerate(arr.shape) if s == n_test]
        if not axes_with_test:
            raise ValueError(f"Cannot find test dimension in 3D predictions shape {arr.shape}; n_test={n_test}")
        test_axis = axes_with_test[-1]
        arr = np.moveaxis(arr, test_axis, -1)
        return arr.reshape(-1, n_test)

    raise ValueError(f"Unsupported prediction array shape: {arr.shape}")


def normalize_chain_prediction_array(pred, n_test: int, num_chains: int, num_mcmc: int):
    flat = normalize_prediction_array(pred, n_test)
    total_needed = num_chains * num_mcmc

    if flat.shape[0] < total_needed:
        raise ValueError(f"Only got {flat.shape[0]} prediction draws, expected at least {total_needed}")

    flat = flat[-total_needed:]
    return flat.reshape(num_chains, num_mcmc, n_test)


def compute_metrics(draws_2d: np.ndarray, y_test: np.ndarray):
    pred_mean = draws_2d.mean(axis=0)
    err = pred_mean - y_test
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")

    lo = np.quantile(draws_2d, 0.025, axis=0)
    hi = np.quantile(draws_2d, 0.975, axis=0)
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    interval_width = float(np.mean(hi - lo))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coverage_95": coverage,
        "interval_width_95": interval_width,
        "n_draws": int(draws_2d.shape[0]),
        "n_test": int(draws_2d.shape[1]),
    }


def rhat_over_predictions(chains: np.ndarray):
    # chains: m chains × n draws × p test points
    m, n, p = chains.shape
    if m < 2 or n < 2:
        return {"rhat_median": float("nan"), "rhat_max": float("nan")}

    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    B = n * chain_means.var(axis=0, ddof=1)
    W = chain_vars.mean(axis=0)
    var_hat = ((n - 1) / n) * W + B / n

    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
    rhat = rhat[np.isfinite(rhat)]

    if rhat.size == 0:
        return {"rhat_median": float("nan"), "rhat_max": float("nan")}

    return {
        "rhat_median": float(np.median(rhat)),
        "rhat_max": float(np.max(rhat)),
    }


def fit_stochtree_bart(X_train, y_train, X_test, *, num_gfr, num_burnin, num_mcmc, num_chains, num_threads, seed):
    from stochtree import BARTModel

    model = BARTModel()

    # StochTree Python API expects chain/thread/seed controls under general_params.
    # This is safer than passing them as top-level kwargs.
    sample_kwargs = dict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        num_gfr=num_gfr,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            "num_chains": num_chains,
            "num_threads": num_threads,
            "random_seed": seed,
        },
    )

    try:
        result = model.sample(**sample_kwargs)
    except TypeError as e:
        # Fallback for older/newer API variants.
        print(f"[stochtree fallback] first sample call failed: {e}")
        sample_kwargs["general_params"].pop("random_seed", None)
        try:
            result = model.sample(**sample_kwargs)
        except TypeError as e2:
            print(f"[stochtree fallback] second sample call failed: {e2}")
            sample_kwargs.pop("general_params", None)
            result = model.sample(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
            )

    pred = None

    for call in [
        lambda: model.predict(X_test),
        lambda: model.predict(covariates=X_test),
        lambda: model.predict(X_test=X_test),
    ]:
        try:
            pred = call()
            break
        except Exception:
            pass

    if pred is None and result is not None:
        pred = result

    if pred is None:
        for attr in ["y_hat_test", "yhat_test", "predictions", "test_predictions"]:
            if hasattr(model, attr):
                pred = getattr(model, attr)
                break

    if pred is None:
        raise RuntimeError("Could not extract predictions from StochTree BARTModel. Check installed stochtree API.")

    return pred

def run_one(args, task: str, dataset: str, run_id: int):
    index_store = Path(args.index_store).resolve()
    out_root = Path(args.out_dir).resolve()
    canon = DATASET_CANON[dataset]

    X, y = load_dataset(dataset, index_store, run_id)
    train_idx, test_idx = validate_split_and_data(dataset, run_id, index_store, X, y)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    if task == "todo3a":
        params = dict(
            num_gfr=0,
            num_burnin=args.todo3a_burnin,
            num_mcmc=args.todo3a_mcmc,
            num_chains=args.todo3a_chains,
        )
        method_name = "todo3a_stochtree_bart_reference"
    elif task == "todo3b":
        params = dict(
            num_gfr=0,
            num_burnin=args.todo3b_burnin,
            num_mcmc=args.todo3b_mcmc,
            num_chains=args.todo3b_chains,
        )
        method_name = f"todo3b_multistart_{args.todo3b_chains}chains_{args.todo3b_mcmc}draws"
    else:
        raise ValueError(task)

    out_dir = out_root / canon / f"run{run_id:03d}" / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{task}] {canon} run{run_id:03d}")
    log(f"X_train={X_train.shape}, X_test={X_test.shape}, params={params}")

    t0 = time.time()
    pred = fit_stochtree_bart(
        X_train,
        y_train,
        X_test,
        num_gfr=params["num_gfr"],
        num_burnin=params["num_burnin"],
        num_mcmc=params["num_mcmc"],
        num_chains=params["num_chains"],
        num_threads=args.num_threads,
        seed=args.seed + run_id,
    )
    elapsed = time.time() - t0

    draws = normalize_prediction_array(pred, n_test=len(y_test))
    metrics = compute_metrics(draws, y_test)
    metrics.update(params)
    metrics.update({
        "task": task,
        "dataset": canon,
        "run_id": run_id,
        "method": method_name,
        "elapsed_sec": float(elapsed),
        "split_source": str(index_store),
        "hard_split_match": True,
    })

    np.save(out_dir / "pred_draws.npy", draws)
    np.savetxt(out_dir / "pred_mean.csv", draws.mean(axis=0), delimiter=",")
    np.savetxt(out_dir / "y_test.csv", y_test, delimiter=",")

    if task == "todo3b":
        chains = normalize_chain_prediction_array(
            pred,
            n_test=len(y_test),
            num_chains=params["num_chains"],
            num_mcmc=params["num_mcmc"],
        )
        np.save(out_dir / "pred_draws_by_chain.npy", chains)
        metrics.update(rhat_over_predictions(chains))

        chain_rows = []
        for c in range(chains.shape[0]):
            cm = compute_metrics(chains[c], y_test)
            cm["chain"] = c
            chain_rows.append(cm)
        pd.DataFrame(chain_rows).to_csv(out_dir / "chain_metrics.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"done: rmse={metrics['rmse']:.4f}, coverage={metrics['coverage_95']:.3f}, elapsed={elapsed/60:.1f} min")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=["todo3a", "todo3b"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=["abalone", "concrete", "friedman"], required=True)
    parser.add_argument("--run-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--index-store", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7000)

    parser.add_argument("--todo3a-burnin", type=int, default=1000)
    parser.add_argument("--todo3a-mcmc", type=int, default=2500)
    parser.add_argument("--todo3a-chains", type=int, default=4)

    # Defaults match the Notion TODO3b requirement:
    # 100 chains, burn-in 500, collect 100 posterior samples per chain.
    parser.add_argument("--todo3b-burnin", type=int, default=500)
    parser.add_argument("--todo3b-mcmc", type=int, default=100)
    parser.add_argument("--todo3b-chains", type=int, default=100)

    args = parser.parse_args()

    all_metrics = []
    for task in args.tasks:
        for dataset in args.datasets:
            for run_id in args.run_ids:
                all_metrics.append(run_one(args, task, dataset, run_id))

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Rebuild global summary from all metrics.json files in out_dir.
    # This avoids overwriting Abalone/Concrete summary when Friedman is run later.
    records = []
    for mp in sorted(out_root.rglob("metrics.json")):
        try:
            with open(mp, "r") as f:
                rec = json.load(f)
            rec["_metrics_path"] = str(mp)
            records.append(rec)
        except Exception as e:
            log(f"[summary warning] failed to read {mp}: {e}")

    if not records:
        records = all_metrics

    df = pd.DataFrame(records)
    sort_cols = [c for c in ["task", "dataset", "run_id", "method"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    summary_path = out_root / "todo3ab_summary.csv"
    df.to_csv(summary_path, index=False)
    log(f"Wrote global summary: {summary_path} with n_rows={len(df)}")


if __name__ == "__main__":
    main()
