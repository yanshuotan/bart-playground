"""Annealed-SMC variant study on one fixed100 run-0 split.

Each population runs the annealing loop of ``SMCBART.fit_with_data`` with one
of these changes: particle count N, sweeps per stage K, conditional-ESS target,
a Drovandi--Pettitt sweep count per stage, or MTMH instead of Default
mutations. A Default mutation is exactly one unchanged DefaultSampler sweep.
After beta=1 the population is resampled once and every particle runs further
beta=1 sweeps; snapshots are saved at chosen sweep counts. A config with
``"chain": "default"`` or ``"mtmh"`` instead reruns the paper's run-0 short
chain with seed 2024+rep from stumps, to time it on the same node and check
that its noise variances match the stored chain.

``summarize`` compares the snapshots with the stored long-Default chains:
distance of each population mean to the pooled long-Default mean over the 100
fixed test inputs, between/within ratio across populations, posterior sd ratio,
mean noise variance (transformed scale) and mean number of splits.

    python diagnosis/smc_variants.py run --configs variants.json --output OUT --jobs 4
    python diagnosis/smc_variants.py summarize --output OUT
    python diagnosis/smc_variants.py figure --output OUT --figure smc_variants.png

``variants.json`` is a list of
``{"name", "N", "K", "ess", "resample", "post", "snaps"}`` with
optional ``"adapt": [c, kmin, kmax]``, ``"kernel": "mtmh"`` and ``"reps"``;
a chain config is ``{"name", "chain", "iters"}``. Every task uses one core and
records its wall time and host.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diagnosis'))


def split_data(dataset, store, cache):
    import fixed100_support as support
    support.UCI_CACHE_DIR = Path(cache) / 'uci_cache'
    support.OPENML_CACHE_DIR = Path(cache) / 'openml_cache'
    from experiment_fixed100 import make_fixed100_splits
    from run_smc_population import validate_stored_split
    X, y = support.load_dataset(dataset)
    split = make_fixed100_splits(X, y, n_runs=1)[0]
    tag = support.DATASET_CONFIGS[dataset]['dataset_tag']
    validate_stored_split(split, Path(store) / tag, tag)
    return split, tag


def run_chain(cfg, rep, output, dataset, store, cache):
    from threadpoolctl import threadpool_limits
    from bart_playground import DefaultBART, MultiBART
    split, tag = split_data(dataset, store, cache)
    # experiment_fixed100: run-0 chain seeds are 2024 + chain id, tol=1.
    common = dict(ndpost=cfg['iters'], nskip=0, n_trees=100, tol=1, random_state=2024 + rep)
    model = (DefaultBART(**common) if cfg['chain'] == 'default'
             else MultiBART(tree_alpha=0.95, tree_beta=2.0, multi_tries=10, **common))
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(split['X_train'], split['y_train'], quietly=True)
    seconds = time.perf_counter() - started
    # The trace starts with the initial state; the stored chain does not.
    sigmas = np.array([state.global_params['eps_sigma2'][0] for state in model.trace[-cfg['iters']:]])
    stored = load_baseline(store, tag, cfg['chain'], 'sigmas')[rep, :len(sigmas), 0]
    matches = bool(np.allclose(sigmas[:len(stored)], stored, rtol=1e-8, atol=0))
    np.savez_compressed(Path(output) / f"{cfg['name']}_rep{rep}.npz", sigmas=sigmas,
                        seconds=seconds, matches_stored=matches, host=platform.node(),
                        config=json.dumps(cfg))
    return cfg['name'], rep, round(seconds), matches


def run_population(cfg, rep, output, dataset, store, cache):
    from threadpoolctl import threadpool_limits
    from bart_playground import SMCBART
    from bart_playground.samplers import MultiSampler, mtmh_proposal_probs
    from bart_playground.smc import (_copy_particle, _effective_sample_size,
                                     _normalize_log_weights, _systematic_resample)
    split, _ = split_data(dataset, store, cache)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model = SMCBART(n_particles=cfg['N'], n_trees=100, n_mcmc_steps=max(1, cfg['K']),
                        ess_target=cfg['ess'], resample_threshold=cfg['resample'],
                        random_state=2024 + rep)
        data = model.preprocessor.fit_transform(split['X_train'], split['y_train'])
        model.data = data
        sampler = model.sampler
        if cfg.get('kernel') == 'mtmh':
            sampler = MultiSampler(sampler.prior, dict(mtmh_proposal_probs), sampler.generator,
                                   tol=1, multi_tries=cfg.get('tries', 10))
            model.sampler = sampler
        sampler.add_data(data)
        sampler.add_thresholds(model.preprocessor.thresholds)
        sampler.global_prior.fit_hyperparameters(data)
        n = cfg['N']

        def sweep(state, temp):
            state = sampler.one_iter(state, temp=temp)
            sampler.accepted_moves_logmh.clear()
            return state

        def acceptance(before):
            selected = sum(sampler.move_selected_counts.values()) - before[0]
            return (sum(sampler.move_accepted_counts.values()) - before[1]) / max(1, selected)

        def counters():
            return (sum(sampler.move_selected_counts.values()),
                    sum(sampler.move_accepted_counts.values()))

        def snapshot(particles, weights):
            return {'pred': np.array([model.preprocessor.backtransform_y(p.evaluate(split['X_test_fixed']))
                                      for p in particles]),
                    'sig': np.array([p.global_params['eps_sigma2'][0] for p in particles]),
                    'splits': np.array([sum(t.n_leaves for t in p.trees) - p.n_trees for p in particles]),
                    'w': weights}

        particles = [model._sample_prior_state() for _ in range(n)]
        log_weights = np.full(n, -np.log(n))
        origins = np.arange(n)
        beta, stages = 0.0, []
        while beta < 1:
            log_likelihoods = model._log_likelihoods(particles)
            next_beta = model._next_beta(beta, log_weights, log_likelihoods)
            log_weights, _ = _normalize_log_weights(log_weights + (next_beta - beta) * log_likelihoods)
            ess = _effective_sample_size(log_weights)
            resampled = ess < cfg['resample'] * n
            if resampled:
                ancestors = _systematic_resample(log_weights, sampler.generator)
                particles = [_copy_particle(particles[a]) for a in ancestors]
                origins = origins[ancestors]
                log_weights = np.full(n, -np.log(n))
            stage_start = counters()
            if 'adapt' in cfg:
                # Drovandi and Pettitt (2011): sweeps until a tree has had an
                # accepted move with probability 1-c at the last sweep's rate.
                c, k_min, k_max = cfg['adapt']
                steps, needed = 0, k_min
                while steps < needed:
                    before = counters()
                    particles = [sweep(p, 1 / next_beta) for p in particles]
                    steps += 1
                    rate = acceptance(before)
                    needed = k_max if rate <= 0 else int(np.clip(
                        np.ceil(np.log(c) / np.log1p(-rate)), k_min, k_max))
            else:
                steps = cfg['K']
                for _ in range(steps):
                    particles = [sweep(p, 1 / next_beta) for p in particles]
            stages.append((next_beta, ess, resampled, acceptance(stage_start),
                           len(np.unique(origins)), steps))
            beta = next_beta
        anneal_seconds = time.perf_counter() - started
        snaps = {0: snapshot(particles, np.exp(log_weights))}
        ancestors = _systematic_resample(log_weights, sampler.generator)
        particles = [_copy_particle(particles[a]) for a in ancestors]
        for index in range(1, cfg['post'] + 1):
            particles = [sweep(p, 1.0) for p in particles]
            if index in cfg['snaps']:
                snaps[index] = snapshot(particles, np.full(n, 1 / n))
    path = Path(output) / f"{cfg['name']}_rep{rep}.npz"
    np.savez_compressed(path, **{f'{key}_{s}': value for s, snap in snaps.items()
                                 for key, value in snap.items()},
                        stages=np.array(stages, dtype=float), anneal_seconds=anneal_seconds,
                        seconds=time.perf_counter() - started, host=platform.node(),
                        config=json.dumps(cfg))
    return cfg['name'], rep, len(stages), round(anneal_seconds), round(time.perf_counter() - started)


def load_baseline(store, tag, method, kind):
    sys.path.insert(0, str(ROOT / 'diagnosis/paper'))
    from core import load_with_shape
    return load_with_shape(Path(store) / tag / kind / f'{tag}__run000__{method}__{kind}.csv')


def load_runs(output):
    runs = defaultdict(list)
    for path in sorted(Path(output).glob('*_rep*.npz')):
        runs[path.name.rsplit('_rep', 1)[0]].append(np.load(path))
    return runs


def is_chain(reps):
    return 'chain' in json.loads(str(reps[0]['config']))


# Labelled populations (config name, further beta=1 sweeps) -> label offset in points.
LABEL_OFFSETS = {('N50_K2', 0): (-12, 8), ('N400_K2', 0): (6, -12),
                 ('N16_K10_ess05_mtmh', 0): (-36, -14), ('N100_K10_ess05', 0): (-10, 8),
                 ('N50_K10', 0): (5, 6), ('N50_adaptive', 0): (-40, -14),
                 ('N16_K50_ess05', 0): (5, 6), ('N50_K2', 1000): (-30, -14),
                 ('N50_K10', 1000): (5, 6)}


def figure(output, dataset, store, path, burn=3000):
    """Noise variance at beta=1 against sweeps per lineage, beside chains from stumps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fixed100_support import DATASET_CONFIGS
    tag = DATASET_CONFIGS[dataset]['dataset_tag']
    long = load_baseline(store, tag, 'default_long', 'sigmas')[:, burn:].ravel() * 1e3
    colors = {'default': '#2a78d6', 'mtmh': '#1baf7a'}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(long.mean() - long.std(), long.mean() + long.std(), color='#d9d8d4', lw=0)
    ax.axhline(long.mean(), color='#8a8984', lw=1)
    ax.text(55, long.mean() + long.std() + 0.01, r'long Default ($10^7$ iterations) $\pm$1 sd',
            color='#52514e', fontsize=8, va='bottom')
    edges = np.unique(np.geomspace(1, 1e4, 30).astype(int))
    for method, label in (('default', 'Default chain'), ('mtmh', 'MTMH chain')):
        trace = load_baseline(store, tag, method, 'sigmas')[..., 0].mean(0) * 1e3
        centres = np.sqrt(edges[:-1] * edges[1:])
        ax.plot(centres, [trace[a:b].mean() for a, b in zip(edges[:-1], edges[1:])],
                color=colors[method], lw=2, label=f'{label} from stumps (mean of 4)')
    for name, reps in load_runs(output).items():
        if is_chain(reps):
            continue
        cfg = json.loads(str(reps[0]['config']))
        if 'leaf' in cfg and cfg['leaf']:
            continue  # the earlier, modified Default sweep is not an SMC of this code
        kernel = cfg.get('kernel', 'default')
        sweeps = np.mean([r['stages'][:, 5].sum() for r in reps])
        snaps = sorted(int(k[4:]) for k in reps[0].files if k.startswith('sig_'))
        for s in (0, snaps[-1]) if snaps[-1] >= 1000 else (0,):
            value = np.mean([r[f'w_{s}'] @ r[f'sig_{s}'] for r in reps]) * 1e3
            ax.plot(sweeps + s, value, 'o' if s == 0 else 's', ms=6, mfc=colors[kernel],
                    mec='white', mew=1.5, ls='none')
            text = (f"N={cfg['N']}, " + ('adaptive K' if 'adapt' in cfg else f"K={cfg['K']}")
                    + (', MTMH' if kernel == 'mtmh' else '') + (f', +{s}' if s else ''))
            if (name, s) in LABEL_OFFSETS:
                ax.annotate(text, (sweeps + s, value), xytext=LABEL_OFFSETS[name, s],
                            textcoords='offset points', fontsize=7, color='#52514e')
    ax.plot([], [], 'o', mfc='#52514e', mec='white', ls='none', label='SMC population at beta=1')
    ax.plot([], [], 's', mfc='#52514e', mec='white', ls='none', label='... after further beta=1 sweeps')
    ax.set_xscale('log')
    ax.set_xlim(50, 1.1e4)
    ax.set_ylim(4.9, 6.0)
    ax.set_xlabel('sweeps per chain or per particle lineage')
    ax.set_ylabel(r'mean $\sigma^2$ (transformed scale, $\times10^{-3}$)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color='#ecebe8', lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def summarize(output, dataset, store, burn=3000):
    from fixed100_support import DATASET_CONFIGS
    tag = DATASET_CONFIGS[dataset]['dataset_tag']
    long = load_baseline(store, tag, 'default_long', 'preds')[:, :, burn:]
    mean, sd = long.mean(axis=(0, 2)), long.std(axis=(0, 2))
    sig = load_baseline(store, tag, 'default_long', 'sigmas')[:, burn:]
    splits = load_baseline(store, tag, 'default_long', 'vars_histogram')[:, burn:].sum(-1)
    print(f'long Default reference: sigma2 {sig.mean():.6f} (sd {sig.std():.6f}), '
          f'splits {splits.mean():.1f} (sd {splits.std():.1f})')
    for method in ('default', 'default_pt', 'mtmh', 'mtmh_pt'):
        draws = load_baseline(store, tag, method, 'preds')[:, :, burn:]
        means = draws.mean(2)
        try:  # Some stores keep no split counts for the PT arms.
            counts = f'{load_baseline(store, tag, method, "vars_histogram")[:, burn:].sum(-1).mean():.1f}'
        except FileNotFoundError:
            counts = 'n/a'
        print(f'{method} chains: distance {np.linalg.norm(means - mean, axis=1).mean():.2f} '
              f'(pooled {np.linalg.norm(means.mean(0) - mean):.2f})  '
              f'B/W {means.var(0, ddof=1).mean() / draws.var(2).mean():.3f}  '
              f'sigma2 {load_baseline(store, tag, method, "sigmas")[:, burn:].mean():.6f}  '
              f'splits {counts}')
    runs = load_runs(output)
    for name, reps in runs.items():
        if is_chain(reps):
            print(f'\n{name}: {len(reps)} chains on {sorted({str(r["host"]) for r in reps})}, '
                  f'{np.mean([float(r["seconds"]) for r in reps]):.0f}s per chain, '
                  f'matches stored draws {[bool(r["matches_stored"]) for r in reps]}')
            continue
        snaps = sorted(int(k[5:]) for k in reps[0].files if k.startswith('pred_'))
        print(f'\n{name}: {len(reps)} populations, stages {[len(r["stages"]) for r in reps]}, '
              f'distinct origins at beta=1 {[int(r["stages"][-1, 4]) for r in reps]}, '
              f'anneal {np.mean([float(r["anneal_seconds"]) for r in reps]):.0f}s, '
              f'total {np.mean([float(r["seconds"]) for r in reps]):.0f}s per population, '
              f'hosts {sorted({str(r["host"]) for r in reps}) if "host" in reps[0].files else "?"}')
        for s in snaps:
            means, within, sigmas, counts = [], [], [], []
            for r in reps:
                P, w = r[f'pred_{s}'], r[f'w_{s}']
                m = w @ P
                means.append(m)
                within.append(w @ (P - m) ** 2)
                sigmas.append(w @ r[f'sig_{s}'])
                counts.append(w @ r[f'splits_{s}'])
            means, within = np.array(means), np.array(within)
            print(f'  sweep {s:5d}: distance {np.linalg.norm(means - mean, axis=1).mean():.2f} '
                  f'(pooled {np.linalg.norm(means.mean(0) - mean):.2f})  '
                  f'B/W {means.var(0, ddof=1).mean() / within.mean():.3f}  '
                  f'sd ratio {np.sqrt(within.mean(0)).mean() / sd.mean():.3f}  '
                  f'sigma2 {np.mean(sigmas):.6f}  splits {np.mean(counts):.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('run', 'summarize', 'figure'))
    parser.add_argument('--figure', help='output image for the figure command')
    parser.add_argument('--output', required=True)
    parser.add_argument('--configs')
    parser.add_argument('--dataset', default='abalone')
    parser.add_argument('--store', default=str(ROOT / 'diagnosis/store'),
                        help='folder holding the fixed100_* baseline records')
    parser.add_argument('--cache', default=str(ROOT / 'diagnosis/store'),
                        help='folder holding uci_cache/ and openml_cache/')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'summarize':
        return summarize(args.output, args.dataset, args.store)
    if args.command == 'figure':
        return figure(args.output, args.dataset, args.store, args.figure)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    configs = json.loads(Path(args.configs).read_text())
    tasks = [(cfg, rep) for cfg in configs for rep in range(cfg.get('reps', 4))]
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(run_chain if 'chain' in cfg else run_population,
                                   cfg, rep, args.output, args.dataset, args.store, args.cache)
                   for cfg, rep in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
