"""Annealed-SMC variant study on one fixed100 run-0 split.

Each population runs the annealing loop of ``SMCBART.fit_with_data`` with one
of these changes: particle count N, sweeps per stage K, conditional-ESS target,
an optional extra leaf Gibbs sweep after each Default sweep (the earlier SMC),
a Drovandi--Pettitt sweep count per stage, or MTMH instead of Default
mutations. After beta=1 the population is resampled once and every particle
runs further beta=1 sweeps; snapshots are saved at chosen sweep counts.

``summarize`` compares the snapshots with the stored long-Default chains:
distance of each population mean to the pooled long-Default mean over the 100
fixed test inputs, between/within ratio across populations, posterior sd ratio,
mean noise variance (transformed scale) and mean number of splits.

    python diagnosis/smc_variants.py run --configs variants.json --output OUT --jobs 4
    python diagnosis/smc_variants.py summarize --output OUT

``variants.json`` is a list of
``{"name", "N", "K", "ess", "resample", "leaf", "post", "snaps"}`` with
optional ``"adapt": [c, kmin, kmax]``, ``"kernel": "mtmh"`` and ``"reps"``.
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
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diagnosis'))


def split_data(dataset, store):
    import fixed100_support as support
    from experiment_fixed100 import make_fixed100_splits
    from run_smc_population import validate_stored_split
    X, y = support.load_dataset(dataset)
    split = make_fixed100_splits(X, y, n_runs=1)[0]
    tag = support.DATASET_CONFIGS[dataset]['dataset_tag']
    validate_stored_split(split, Path(store) / tag, tag)
    return split, tag


def run_population(cfg, rep, output, dataset, store):
    from threadpoolctl import threadpool_limits
    from bart_playground import SMCBART
    from bart_playground.samplers import MultiSampler, mtmh_proposal_probs
    from bart_playground.smc import (_copy_particle, _effective_sample_size,
                                     _normalize_log_weights, _systematic_resample)
    split, _ = split_data(dataset, store)
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
            if cfg['leaf']:
                for tree_id in range(state.n_trees):
                    state.update_leaf_vals([tree_id], sampler.tree_prior.resample_leaf_vals(
                        state, data.y, [tree_id], temp=temp))
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
                        seconds=time.perf_counter() - started, config=json.dumps(cfg))
    return cfg['name'], rep, len(stages), round(anneal_seconds), round(time.perf_counter() - started)


def load_baseline(store, tag, method, kind):
    sys.path.insert(0, str(ROOT / 'diagnosis/paper'))
    from core import load_with_shape
    return load_with_shape(Path(store) / tag / kind / f'{tag}__run000__{method}__{kind}.csv')


def summarize(output, dataset, store, burn=3000):
    from fixed100_support import DATASET_CONFIGS
    tag = DATASET_CONFIGS[dataset]['dataset_tag']
    long = load_baseline(store, tag, 'default_long', 'preds')[:, :, burn:]
    mean, sd = long.mean(axis=(0, 2)), long.std(axis=(0, 2))
    sig = load_baseline(store, tag, 'default_long', 'sigmas')[:, burn:]
    splits = load_baseline(store, tag, 'default_long', 'vars_histogram')[:, burn:].sum(-1)
    print(f'long Default reference: sigma2 {sig.mean():.6f} (sd {sig.std():.6f}), '
          f'splits {splits.mean():.1f} (sd {splits.std():.1f})')
    runs = defaultdict(list)
    for path in sorted(Path(output).glob('*_rep*.npz')):
        runs[path.name.rsplit('_rep', 1)[0]].append(np.load(path))
    for name, reps in runs.items():
        snaps = sorted(int(k[5:]) for k in reps[0].files if k.startswith('pred_'))
        print(f'\n{name}: {len(reps)} populations, stages {[len(r["stages"]) for r in reps]}, '
              f'distinct origins at beta=1 {[int(r["stages"][-1, 4]) for r in reps]}, '
              f'anneal {np.mean([float(r["anneal_seconds"]) for r in reps]):.0f}s, '
              f'total {np.mean([float(r["seconds"]) for r in reps]):.0f}s')
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
                  f'R_BW {means.var(0, ddof=1).mean() / within.mean():.3f}  '
                  f'sd ratio {np.sqrt(within.mean(0)).mean() / sd.mean():.3f}  '
                  f'sigma2 {np.mean(sigmas):.6f}  splits {np.mean(counts):.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('run', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--configs')
    parser.add_argument('--dataset', default='abalone')
    parser.add_argument('--store', default=str(ROOT / 'diagnosis/store'),
                        help='folder holding the fixed100_* baseline records')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'summarize':
        return summarize(args.output, args.dataset, args.store)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    configs = json.loads(Path(args.configs).read_text())
    tasks = [(cfg, rep) for cfg in configs for rep in range(cfg.get('reps', 4))]
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(run_population, cfg, rep, args.output, args.dataset, args.store)
                   for cfg, rep in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
