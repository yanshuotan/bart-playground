#!/usr/bin/env python
"""Hypercube-D study of Lakshminarayanan, Roy and Teh (2015, section 4.1).

Ten points are drawn around each of the 2^D vertices of [-1, 1]^D
(x = v + N(0, 0.1^2 I)), each vertex gets its own value f ~ N(0, 3^2), and
y = f + N(0, 0.01^2). One tree (m = 1) with alpha = 0.95 and the paper's
beta for each D, so the prior expects about 2^D leaves. Every sampler runs
2,000 iterations from a stump; ESS of the training log-likelihood is computed
on the second half, as in the paper's Tables 1 and 2.

`--grid uniform` gives each variable 100 equally spaced thresholds over its
range, close to the paper's uniform split location; `--grid quantile` keeps
the package's 100-quantile grid, which puts about one threshold in the gap
between the two clusters of every coordinate.

    python diagnosis/pg/hypercube.py --output OUT --jobs 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BETAS = {2: 1.0, 3: 0.5, 4: 0.4, 5: 0.3, 7: 0.25}
SAMPLERS = ("cgm", "growprune", "pg")


def hypercube_data(D: int, seed: int, n_per_vertex: int = 10):
    rng = np.random.default_rng(seed)
    vertices = np.array(list(itertools.product([-1.0, 1.0], repeat=D)))
    f_vertex = rng.normal(0.0, 3.0, size=len(vertices))

    def draw():
        v = np.repeat(np.arange(len(vertices)), n_per_vertex)
        X = vertices[v] + rng.normal(0.0, 0.1, size=(len(v), D))
        y = f_vertex[v] + rng.normal(0.0, 0.01, size=len(v))
        return X, y

    return draw(), draw()


def uniform_grid(X, n_bins: int = 100):
    return {k: np.linspace(X[:, k].min(), X[:, k].max(), n_bins, endpoint=False) for k in range(X.shape[1])}


def make_model(sampler: str, D: int, n_iter: int, seed: int, n_particles: int, grid: str):
    from bart_playground import DefaultBART, PGBART

    common = dict(ndpost=n_iter, nskip=0, n_trees=1, tree_alpha=0.95, tree_beta=BETAS[D],
                  random_state=seed)
    if sampler == "pg":
        model = PGBART(n_particles=n_particles, **common)
    else:
        probs = ({"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1} if sampler == "cgm"
                 else {"grow": 0.5, "prune": 0.5})
        model = DefaultBART(proposal_probs=probs, tol=1, **common)
    if grid == "uniform":
        model.preprocessor.gen_thresholds = uniform_grid
    return model


def run_one(task):
    D, data_seed, sampler, n_iter, n_particles, grid = task
    import arviz as az

    (X, y), (X_test, y_test) = hypercube_data(D, data_seed)
    # Two warm-up iterations first, so the timing below excludes Numba compilation.
    make_model(sampler, D, 2, 1000 + data_seed, n_particles, grid).fit(X, y, quietly=True)
    model = make_model(sampler, D, n_iter, 1000 + data_seed, n_particles, grid)
    start = time.perf_counter()
    model.fit(X, y, quietly=True)
    seconds = time.perf_counter() - start

    data = model.data
    loglik, leaves, sigma2, test_mse = [], [], [], []
    for state in model.trace[1:]:
        s2 = float(state.global_params["eps_sigma2"][0])
        resid = data.y - state.evaluate(data.X)
        loglik.append(float(-0.5 * len(resid) * np.log(2 * np.pi * s2) - 0.5 * np.sum(resid ** 2) / s2))
        leaves.append(int(state.trees[0].n_leaves))
        sigma2.append(s2)
        pred = model.preprocessor.backtransform_y(state.evaluate(model.preprocessor.transform_X(X_test)))
        test_mse.append(float(np.mean((pred - y_test) ** 2)))
    half = len(loglik) // 2
    ess = float(az.ess(np.asarray(loglik[half:])[None, :]))
    return {
        "D": D, "data_seed": data_seed, "sampler": sampler, "n_iter": n_iter, "grid": grid,
        "n_particles": n_particles if sampler == "pg" else None,
        "seconds": seconds, "ess_loglik": ess, "ess_per_second": ess / seconds,
        "loglik": loglik, "leaves": leaves, "sigma2": sigma2, "test_mse": test_mse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=sorted(BETAS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLERS), choices=SAMPLERS)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--grid", choices=("uniform", "quantile"), default="uniform")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(D, s, m, args.n_iter, args.n_particles, args.grid)
             for D in args.dims for s in args.seeds for m in args.samplers]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(run_one, tasks):
            name = f"{res['grid']}_D{res['D']}_seed{res['data_seed']}_{res['sampler']}.json"
            (args.output / name).write_text(json.dumps(res))
            print(f"{res['grid']} D={res['D']} seed={res['data_seed']} {res['sampler']}: "
                  f"ESS={res['ess_loglik']:.1f} ESS/s={res['ess_per_second']:.3f} "
                  f"leaves={np.mean(res['leaves'][len(res['leaves']) // 2:]):.1f} "
                  f"time={res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
