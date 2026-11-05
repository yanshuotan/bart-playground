"""Compare SMC and MCMC on one fixed100 run-0 split at equal sample counts.

Every MCMC method contributes iterations 3000-10000 of its four stored chains
(7000 samples per chain). Every SMC setting contributes four populations of
7000 retained samples each, written by ``run_smc_population.py``. Conditional
SMC keeps 40 states per chain, so it is compared with 40 evenly spaced
post-burn draws of each chain instead.

``table`` prints, per method, the paper's measures: the between/within ratio
R_B/W of the four chains or populations (Section 3), relative RMSE and
relative CRPS against the training-mean and climatology references
(Section 6). Beside them are the mean noise variance (transformed scale) and
mean number of splits, to compare with the long run. It also prints CPU
seconds per population relative to one Default chain timed on the same node,
both read from the ``/usr/bin/time -v`` logs.

``figure`` projects all methods onto one PCA of the predictions and draws one
panel per method.

    python diagnosis/smc_equal_samples.py table --dataset abalone --runs RUNS --conditional COND
    python diagnosis/smc_equal_samples.py figure --dataset abalone --runs RUNS --conditional COND --figure OUT.png

``RUNS`` holds ``default/`` (smc_variants.py chain timings) and one
``run_smc_population.py`` output per SMC setting; ``COND`` is the
``run_conditional_smc.py`` output folder.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'diagnosis'))
sys.path.insert(0, str(ROOT / 'diagnosis/paper'))

BURN = 3000
MCMC = {'default': 'Default', 'default_pt': 'Default+PT', 'mtmh': 'MTMH', 'mtmh_pt': 'MTMH+PT'}
SMC = {'p50': 'SMC, $N=50$, rounds 1--140',
       'p50w860': 'SMC, $N=50$, rounds 861--1000',
       'p7000': 'SMC, $N=7000$, at $\\beta=1$'}
CONDITIONAL = 'Conditional SMC, $N=50$'
LONG = 'Long Default'


def stored(store, tag, method, kind):
    from core import load_with_shape
    return load_with_shape(Path(store) / tag / kind / f'{tag}__run000__{method}__{kind}.csv')


def mcmc_draws(store, tag, method):
    """Predictions (chain, draw, point), noise variances and split counts after burn-in."""
    preds = stored(store, tag, method, 'preds').transpose(0, 2, 1)[:, BURN:]
    sigmas = stored(store, tag, method, 'sigmas').reshape(4, -1)[:, BURN:]
    try:
        splits = stored(store, tag, method, 'vars_histogram')[:, BURN:].sum(-1)
    except FileNotFoundError:
        splits = None
    return preds, sigmas, splits


def smc_draws(folder):
    reps = [Path(folder) / 'replicates' / f'replicate{r:03d}' for r in range(4)]
    return (np.stack([np.load(r / 'preds.npy') for r in reps]),
            np.stack([np.load(r / 'sigmas.npy') for r in reps]),
            np.stack([np.load(r / 'vars_histogram.npy').sum(-1) for r in reps]))


def conditional_draws(folder, dataset):
    chains = [Path(folder) / dataset / f'chain{c:03d}' for c in range(4)]
    return (np.stack([np.load(c / 'preds.npy') for c in chains]),
            np.stack([np.load(c / 'sigmas.npy') for c in chains]),
            np.stack([np.load(c / 'vars_histogram.npy').sum(-1) for c in chains]))


def evenly(draws, count):
    # run_conditional_smc.py picks its matched baseline draws the same way.
    index = np.linspace(0, draws[0].shape[1] - 1, count, dtype=int)
    return tuple(None if a is None else a[:, index] for a in draws)


def measures(draws, scoring):
    """The paper's R_B/W, relative RMSE and relative CRPS, with mean sigma2 and splits."""
    from section6_methods import PREDICTIVE_NOISE_SEED, chain_geometry, predictive_metrics
    preds, sigmas, splits = draws
    truth, y_scale, baseline = scoring
    rmse, crps = predictive_metrics(preds, sigmas, truth, y_scale, PREDICTIVE_NOISE_SEED)
    return {'rbw': chain_geometry(preds)['between_within_ratio'],
            'rel_rmse': rmse / baseline['naive_rmse'],
            'rel_crps': crps / baseline['climatology_crps'],
            'sigma2': sigmas.mean() * 1e3,
            'splits': np.nan if splits is None else splits.mean()}


def show(label, m):
    print(f"  {label:34s} R_BW {m['rbw']:7.4f}  rel RMSE {m['rel_rmse']:.4f}  "
          f"rel CRPS {m['rel_crps']:.4f}  sigma2 {m['sigma2']:6.3f}  splits {m['splits']:6.1f}")


def cpu_seconds(log):
    text = Path(log).read_text()
    return sum(float(re.search(rf'{kind} time \(seconds\): ([\d.]+)', text).group(1))
               for kind in ('User', 'System'))


def collect(dataset, store, runs, conditional):
    from fixed100_support import DATASET_CONFIGS, load_dataset
    from experiment_fixed100 import make_fixed100_splits
    from section6_methods import predictive_baselines
    tag = DATASET_CONFIGS[dataset]['dataset_tag']
    from run_smc_population import validate_stored_split
    split = make_fixed100_splits(*load_dataset(dataset), n_runs=1)[0]
    validate_stored_split(split, Path(store) / tag, tag)
    truth, y_train = split['y_test_fixed'], split['y_train']
    reference = (truth, float(np.ptp(y_train)), predictive_baselines(y_train, truth))
    long = mcmc_draws(store, tag, 'default_long')
    methods = {label: mcmc_draws(store, tag, method) for method, label in MCMC.items()}
    methods.update({label: smc_draws(Path(runs) / name) for name, label in SMC.items()
                    if (Path(runs) / name / 'replicates').is_dir()})
    methods[LONG] = tuple(None if a is None else a[:, :7000] for a in long)
    cond = conditional_draws(conditional, dataset) if conditional else None
    return tag, reference, methods, cond


def table(dataset, store, runs, conditional):
    tag, reference, methods, cond = collect(dataset, store, runs, conditional)
    print(f'{dataset}: long-Default reference sigma2 {methods[LONG][1].mean() * 1e3:.3f}'
          f' (x1e-3, sd {methods[LONG][1].std() * 1e3:.3f}), splits {methods[LONG][2].mean():.1f}')
    print('\n7000 samples per chain or population')
    for label, draws in methods.items():
        show(label, measures(draws, reference))
    # Both sides count CPU seconds of the whole process tree, start-up included.
    log = Path(runs) / 'default.log'
    chain = cpu_seconds(log) / 4 if log.is_file() else np.nan
    print(f'\nDefault chain of 10^4 iterations on this node: {chain:.0f} CPU s')
    for name, label in SMC.items():
        log = Path(runs) / f'{name}.log'
        if log.is_file() and 'User time' in log.read_text():
            print(f'  {label:34s} {cpu_seconds(log) / 4:8.0f} CPU s per population '
                  f'= {cpu_seconds(log) / 4 / chain:5.1f} Default chains')
    if cond is not None:
        print('\n40 states per chain')
        rows = {CONDITIONAL: cond, **{label: evenly(methods[label], 40) for label in MCMC.values()}}
        for label, draws in rows.items():
            show(label, measures(draws, reference))


def figure(dataset, store, runs, conditional, path, shown=1000, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    tag, reference, methods, cond = collect(dataset, store, runs, conditional)
    panels = dict(methods)
    long = panels.pop(LONG)
    if cond is not None:
        panels[CONDITIONAL + ' (40 per chain)'] = cond
    panels[LONG] = long
    rng = np.random.default_rng(seed)
    # Up to `shown` draws per chain are drawn and used to fit the PCA, with
    # weights that give every method the same total weight.
    picks = {label: [rng.choice(p.shape[0], min(shown, p.shape[0]), replace=False) for p in d[0]]
             for label, d in panels.items()}
    fit_draws = [np.concatenate([p[i] for p, i in zip(d[0], picks[label])])
                 for label, d in panels.items()]
    weights = np.concatenate([np.full(len(x), 1 / len(x)) for x in fit_draws]) / len(fit_draws)
    fit = np.concatenate(fit_draws)
    centre = weights @ fit
    values, vectors = np.linalg.eigh((fit - centre).T @ ((fit - centre) * weights[:, None]))
    share = values[::-1] / values.sum()
    axes_ = vectors[:, ::-1][:, :2]
    project = lambda x: (x - centre) @ axes_
    ref = project(long[0].reshape(-1, long[0].shape[-1]))
    shown_xy = project(np.concatenate(fit_draws))
    low, high = np.quantile(shown_xy, 0.001, axis=0), np.quantile(shown_xy, 0.999, axis=0)
    low, high = low - 0.05 * (high - low), high + 0.05 * (high - low)
    grid = np.meshgrid(np.linspace(low[0], high[0], 120), np.linspace(low[1], high[1], 120))
    kde = gaussian_kde(ref[rng.choice(len(ref), 4000, replace=False)].T)
    density = kde(np.vstack([g.ravel() for g in grid])).reshape(grid[0].shape)
    # Highest-density regions holding 95%, 80% and 50% of the long-run samples.
    levels = np.quantile(kde(ref[rng.choice(len(ref), 4000, replace=False)].T), [0.05, 0.2, 0.5])
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.0), sharex=True, sharey=True)
    for ax, (label, draws) in zip(axes.ravel(), panels.items()):
        ax.contour(*grid, density, levels=levels, colors='#a8a7a2', linewidths=0.7)
        for chain, (preds, index) in enumerate(zip(draws[0], picks[label])):
            xy = project(preds[index])
            ax.scatter(*xy.T, s=3 if len(index) > 100 else 9, color=colors[chain], alpha=0.35, lw=0)
        for chain, preds in enumerate(draws[0]):
            ax.scatter(*project(preds.mean(0)[None]).T, marker='X', s=90, color=colors[chain],
                       edgecolor='black', lw=0.8, zorder=3)
        ax.set_title(label.replace('$\\beta=1$', r'$\beta=1$'), fontsize=9)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(f'PC1 ({share[0]:.1%})', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(f'PC2 ({share[1]:.1%})', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f'PC1 {share[0]:.3f}, PC2 {share[1]:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('table', 'figure'))
    parser.add_argument('--dataset', default='abalone')
    parser.add_argument('--store', default=str(ROOT / 'diagnosis/store'))
    parser.add_argument('--runs', required=True)
    parser.add_argument('--conditional')
    parser.add_argument('--figure')
    args = parser.parse_args()
    if args.command == 'table':
        table(args.dataset, args.store, args.runs, args.conditional)
    else:
        figure(args.dataset, args.store, args.runs, args.conditional, args.figure)


if __name__ == '__main__':
    main()
