"""Adaptive resample-move SMC using PT's collapsed targets and Default kernels.

This implementation is independent of the earlier smc branch. Each temperature
uses ordinary importance-weight ESS/N=0.5, systematic resampling, an exact
PT-style conditional leaf refresh, and K=100 unmodified DefaultSampler sweeps.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import invgamma
from threadpoolctl import threadpool_limits

from .bart import (BART, DefaultBART, _compress_state_for_trace,
                   _refresh_tempered_state_for_sampler, resolve_pt_workers)
from .params import Parameters, Tree
from .samplers import DefaultSampler
from .util import Dataset


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def resolve_workers(n_jobs, n_particles):
    if n_jobs is not None and (isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0):
        raise ValueError("n_jobs must be a nonzero integer or None.")
    return resolve_pt_workers(n_jobs, n_particles)


def _pack(state):
    state = _compress_state_for_trace(state)
    state.global_params = {key: value.copy() for key, value in state.global_params.items()}
    for tree in state.trees:
        tree.dataX = None
    return state


def _restore(state, X):
    trees = []
    for source in state.trees:
        tree = source.copy(copy_cache=False)
        tree.dataX = X
        tree.n = np.zeros(len(tree.vars), dtype=np.int32)
        tree.n[0] = len(X)
        tree.leaf_ids = np.zeros(len(X), dtype=np.int32)
        if not tree.update_n():
            raise ValueError("Particle contains an invalid training partition.")
        tree.update_outputs()
        trees.append(tree)
    return Parameters(trees, {key: value.copy() for key, value in state.global_params.items()})


def _resample(log_weights, rng):
    cdf = np.cumsum(np.exp(log_weights))
    cdf[-1] = 1.
    return np.searchsorted(cdf, (rng.random()+np.arange(len(cdf)))/len(cdf), side='right')


@dataclass
class PTWeight:
    """Temperature-independent sufficient quantities of the existing PT SVD.

    Evaluates exactly the expression in _collapsed_loglik_for_sampler_state,
    including the Gaussian normalization. U and the leaf basis are discarded:
    storing those dense matrices for 1,000 particles would exhaust RAM.
    """
    singular_squared: np.ndarray
    projected_squared: np.ndarray
    residual_ss: float
    variance: float
    leaf_variance: float
    n: int

    def log_at(self, beta):
        if beta == 0:
            return 0.
        if not 0 < beta <= 1:
            raise ValueError("beta must be in [0, 1].")
        tempered_variance = (1./beta)*self.variance
        ratio = self.singular_squared.astype(float)/(tempered_variance/self.leaf_variance)
        logdet = np.log1p(ratio).sum()
        ridge = np.sum(self.projected_squared.astype(float)/(1.+ratio))
        return float(-.5*(logdet+(self.residual_ss+ridge)/tempered_variance)
                     -.5*self.n*np.log(2*np.pi*self.variance)*beta)


def _pt_weight(sampler, state):
    # Reuse the very same leaf-basis/SVD implementation as PT. Only its
    # temperature-independent projections are moved to the coordinator.
    basis = state.leaf_basis(np.arange(state.n_trees))
    U, S = sampler.likelihood.leaf_basis_svd(basis)
    y = sampler.data.y-state.evaluate(all_except=np.arange(state.n_trees))
    coefficients = U.T @ y
    residual = y-U @ coefficients
    return PTWeight(S**2, coefficients**2, float(np.sum(residual**2)),
                    float(state.global_params['eps_sigma2'][0]),
                    sampler.tree_prior.f_sigma2, len(y))


def _importance(weights, old_beta, new_beta):
    increments = np.array([w.log_at(new_beta)-w.log_at(old_beta) for w in weights])
    normalizer = float(logsumexp(increments))
    if not np.isfinite(normalizer):
        raise FloatingPointError("No finite importance-weight normalizer.")
    log_weights = increments-normalizer
    ess = float(np.exp(-logsumexp(2*log_weights)))
    return log_weights, ess, normalizer-np.log(len(weights))


def _choose_beta(weights, beta, ess_ratio):
    """Bracket/bisect the ordinary ESS of NEW weights, starting equally weighted.

    For collapsed targets ESS need not be globally monotone in beta. Bisection
    maintains a feasible/infeasible bracket and finds a crossing; no claim is
    made that it is the first crossing. A feasible endpoint goes straight to 1.
    """
    target = ess_ratio*len(weights)
    if _importance(weights, beta, 1.)[1] >= target:
        return 1.
    low, high = beta, 1.
    for _ in range(60):
        middle = low+(high-low)/2
        if _importance(weights, beta, middle)[1] >= target:
            low = middle
        else:
            high = middle
    if low <= beta:
        raise RuntimeError("Adaptive beta cannot advance at floating-point precision.")
    return float(low)


def _draw_prior(sampler, max_attempts):
    """Calibrated BART prior conditioned on nonempty training leaves."""
    trees, prior, rng = [], sampler.tree_prior, sampler.generator
    for _ in range(prior.n_trees):
        for _ in range(max_attempts):
            tree = Tree.new(sampler.data.X)
            pending, valid = [(0, 0)], True
            while pending:
                node, depth = pending.pop()
                if rng.random() >= prior.alpha/(1+depth)**prior.beta:
                    continue
                variable = int(rng.integers(sampler.data.p))
                thresholds = sampler.possible_thresholds[variable]
                if not len(thresholds) or not tree.split_leaf(
                        node, variable, thresholds[rng.integers(len(thresholds))]):
                    valid = False
                    break
                pending.extend([(2*node+1, depth+1), (2*node+2, depth+1)])
            if valid:
                leaves = tree.leaves
                tree.leaf_vals[leaves] = rng.normal(0, np.sqrt(prior.f_sigma2), len(leaves))
                tree.update_outputs()
                trees.append(tree)
                break
        else:
            raise RuntimeError("Failed to draw an admissible prior tree.")
    global_prior = sampler.global_prior
    variance = global_prior.fixed_eps_sigma2
    if variance is None:
        variance = invgamma.rvs(global_prior.eps_nu/2,
            scale=global_prior.eps_nu*global_prior.eps_lambda/2, random_state=rng)
    return Parameters(trees, {'eps_sigma2': np.array([variance], dtype=float)})


_CONTEXT = None


def _init_worker(folder, config):
    global _CONTEXT
    arrays = {name: np.load(Path(folder)/f'{name}.npy', mmap_mode='r') for name in ('X', 'y', 'test')}
    _CONTEXT = {**config, 'data': Dataset(arrays['X'], arrays['y']), 'test': arrays['test']}


def _particle_task(task, context):
    particle_id, packed, seed, beta, keep = task
    started = time.process_time()
    rng = np.random.default_rng(seed)
    prior = deepcopy(context['prior'])
    prior.tree_prior.generator = prior.global_prior.generator = rng
    # This is the original class and original variance sampler, not a subclass.
    sampler = DefaultSampler(prior, context['proposals'], rng, tol=context['tol'])
    sampler.add_data(context['data'])
    sampler.add_thresholds(context['preprocessor'].thresholds)
    saved = None
    with threadpool_limits(limits=1):
        if packed is None:
            state = _draw_prior(sampler, context['max_prior_attempts'])
        else:
            state = _restore(packed, context['data'].X)
            # Reweighting targets (T,sigma2) with leaves integrated out. Lift to
            # the NEW joint target before calling DefaultSampler; old leaves
            # have the wrong conditional distribution after a beta change.
            _refresh_tempered_state_for_sampler(sampler, state, 1./beta)
            if keep:
                saved = {'preds': np.empty((keep, len(context['test']))),
                         'sigmas': np.empty(keep),
                         'vars_histogram': np.empty((keep, context['data'].p), dtype=np.int32)}
            for step in range(context['n_mcmc_steps']):
                state = sampler.one_iter(state, temp=1./beta)
                sampler.accepted_moves_logmh.clear()
                if keep and step >= context['n_mcmc_steps']-keep:
                    index = step-context['n_mcmc_steps']+keep
                    saved['preds'][index] = context['preprocessor'].backtransform_y(state.evaluate(context['test']))
                    saved['sigmas'][index] = state.global_params['eps_sigma2'][0]
                    saved['vars_histogram'][index] = state.vars_histogram
        weight = _pt_weight(sampler, state)
    counts = {name: getattr(sampler, name).copy() for name in
              ('move_selected_counts','move_success_counts','move_accepted_counts')}
    return particle_id, _pack(state), weight, saved, counts, time.process_time()-started


def _worker(task):
    return _particle_task(task, _CONTEXT)


class AdaptiveSMCBART(BART):
    """Supervisor configuration: N=1000, K=100, ESS/N=0.5 and PT weights.

    Systematic resampling occurs at EVERY temperature, including beta=1, so
    each importance step starts equally weighted. K counts full, unchanged
    Default sweeps, in addition to the PT-style conditional leaf refresh.
    n_jobs follows PT's convention and defaults to serial. The final `trace`
    contains N equally weighted endpoints; optional final-round numeric output
    saves the last `retained_sweeps` of the existing K updates, never extra ones.
    """
    def __init__(self, n_particles=1000, n_mcmc_steps=100, ess_ratio=.5,
                 n_trees=100, n_jobs=1, random_state=42, max_stages=1000,
                 max_prior_attempts=10000, tol=1, **bart_parameters):
        self.n_particles = _integer('n_particles', n_particles)
        self.n_mcmc_steps = _integer('n_mcmc_steps', n_mcmc_steps)
        self.max_stages = _integer('max_stages', max_stages)
        self.max_prior_attempts = _integer('max_prior_attempts', max_prior_attempts)
        self.tol = _integer('tol', tol)
        _integer('n_trees', n_trees)
        if not np.isfinite(ess_ratio) or not 0 < ess_ratio < 1:
            raise ValueError('ess_ratio must be in (0, 1).')
        self.ess_ratio = float(ess_ratio)
        resolve_workers(n_jobs, self.n_particles)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.seed_sequence = np.random.SeedSequence(random_state)
        if bart_parameters.get('dirichlet_prior', False) or bart_parameters.get('quick_decay', False):
            raise ValueError('SMC2 supports uniform variables and the usual depth prior only.')
        if 'ndpost' in bart_parameters or 'nskip' in bart_parameters:
            raise ValueError('SMC2 stores terminal particles; use retained_sweeps for numeric output.')
        if bart_parameters.get('init_trees') is not None or bart_parameters.get('init_sigma2') is not None:
            raise ValueError('SMC2 initializes from the prior, not a seeded forest or variance.')
        default = DefaultBART(ndpost=self.n_particles, nskip=0, n_trees=n_trees,
                              tol=self.tol, random_state=random_state, **bart_parameters)
        super().__init__(default.preprocessor, default.sampler, self.n_particles, 0)
        self.beta_history_ = [0.]
        self.stage_diagnostics_ = []
        self.particles_, self.weights_ = [], np.empty(0)

    def get_params(self):
        prior = self.sampler.prior
        return {'model_type':'AdaptiveSMCBART', 'n_particles':self.n_particles,
                'n_mcmc_steps':self.n_mcmc_steps, 'ess_ratio':self.ess_ratio,
                'n_trees':prior.tree_prior.n_trees, 'tree_alpha':prior.tree_prior.alpha,
                'tree_beta':prior.tree_prior.beta, 'f_k':prior.tree_prior.f_k,
                'eps_nu':prior.global_prior.eps_nu,'eps_q':prior.global_prior.eps_q,
                'specification':prior.global_prior.specification,
                'fixed_eps_sigma2':prior.global_prior.fixed_eps_sigma2,
                'proposal_probs':self.sampler.proposals.copy(), 'tol':self.tol,
                'max_bins':self.preprocessor.max_bins,'max_stages':self.max_stages,
                'max_prior_attempts':self.max_prior_attempts,'n_jobs':self.n_jobs,
                'random_state':self.random_state}

    def fit(self, X, y, quietly=False, **options):
        X, y = np.asarray(X), np.asarray(y)
        if (X.ndim != 2 or y.ndim != 1 or not len(y) or X.shape[0] != len(y)
                or not X.shape[1] or not np.isfinite(X).all() or not np.isfinite(y).all()):
            raise ValueError('Expected finite, nonempty X(n,p) and y(n).')
        data = self.preprocessor.fit_transform(X, y)
        return self.fit_with_data(data, quietly=quietly, **options)

    def fit_with_data(self, data, quietly=False, X_test=None, retained_sweeps=0,
                      output_dir=None, checkpoint_path=None, resume=False,
                      progress_callback=None, temp_dir=None):
        retained_sweeps = _integer('retained_sweeps', retained_sweeps, 0)
        if retained_sweeps > self.n_mcmc_steps:
            raise ValueError('retained_sweeps cannot exceed the K rejuvenation steps.')
        test = data.X if X_test is None else np.asarray(X_test)
        if test.ndim != 2 or test.shape[1] != data.p or not np.isfinite(test).all():
            raise ValueError('Prediction inputs must match training features and be finite.')
        if retained_sweeps and not len(test):
            raise ValueError('Prediction inputs must be nonempty when saving rounds.')
        self.is_fitted = False
        self.trace, self.particles_ = [], []
        self.data = data
        self.sampler.add_data(data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.sampler.global_prior.fit_hyperparameters(data)
        self.effective_workers_ = resolve_workers(self.n_jobs, self.n_particles)
        config = {'prior':self.sampler.prior,'proposals':dict(self.sampler.proposals),
                  'tol':self.tol,'preprocessor':self.preprocessor,
                  'n_mcmc_steps':self.n_mcmc_steps,'max_prior_attempts':self.max_prior_attempts}
        self.posterior_draws_ = None
        self.mutation_counts_ = {name:dict.fromkeys(self.sampler.proposals, 0) for name in
                                ('move_selected_counts','move_success_counts','move_accepted_counts')}
        self.task_cpu_seconds_ = 0.
        self.beta_history_, self.stage_diagnostics_ = [0.], []
        self.log_evidence_ = 0.
        self.ancestry_ = np.arange(self.n_particles)
        rng = np.random.default_rng(np.random.SeedSequence(self.seed_sequence.entropy, spawn_key=(9,)))
        params = {k:v for k,v in self.get_params().items() if k != 'n_jobs'}
        digest = hashlib.sha256(pickle.dumps(params))
        digest.update(str(retained_sweeps).encode())
        digest.update(pickle.dumps(self.preprocessor))
        for array in (data.X, data.y, test):
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        fingerprint = digest.hexdigest()
        checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)

        def report(phase, **values):
            if progress_callback is not None:
                progress_callback({'phase':phase, 'beta':self.beta_history_[-1],
                    'stage':len(self.stage_diagnostics_), **values})

        def save_checkpoint(particles, weights):
            if checkpoint_path is None:
                return
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {'fingerprint':fingerprint,'particles':particles,'summaries':weights,
                'beta_history':self.beta_history_,'history':self.stage_diagnostics_,
                'log_evidence':self.log_evidence_,'ancestry':self.ancestry_,
                'rng':rng.bit_generator.state,'seed_sequence':self.seed_sequence,
                'counts':self.mutation_counts_,'task_cpu_seconds':self.task_cpu_seconds_}
            temporary = checkpoint_path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, checkpoint_path)

        def seed(phase, stage, particle):
            return np.random.SeedSequence(self.seed_sequence.entropy, spawn_key=(phase, stage, particle))

        def allocate_outputs():
            specs = {'preds':((retained_sweeps*self.n_particles, len(test)), float),
                     'sigmas':((retained_sweeps*self.n_particles,),float),
                     'vars_histogram':((retained_sweeps*self.n_particles,data.p), np.int32)}
            if output_dir is None:
                return {k:np.empty(shape,dtype=dtype) for k,(shape,dtype) in specs.items()}
            path = Path(output_dir)
            path.mkdir(parents=True,exist_ok=True)
            return {k:np.lib.format.open_memmap(path/f'{k}.npy', mode='w+',shape=shape,dtype=dtype)
                    for k,(shape,dtype) in specs.items()}

        if resume and checkpoint_path is not None and checkpoint_path.exists():
            with checkpoint_path.open('rb') as stream:
                previous = pickle.load(stream)
            if previous['fingerprint'] != fingerprint:
                raise ValueError('Checkpoint parameters or data do not match this experiment.')
            particles, summaries = previous['particles'], previous['summaries']
            self.beta_history_, self.stage_diagnostics_ = previous['beta_history'], previous['history']
            self.log_evidence_, self.ancestry_ = previous['log_evidence'], previous['ancestry']
            self.seed_sequence = previous['seed_sequence']
            self.mutation_counts_, self.task_cpu_seconds_ = previous['counts'], previous['task_cpu_seconds']
            rng.bit_generator.state = previous['rng']
            if self.beta_history_[-1] == 1 and retained_sweeps:
                if output_dir is None:
                    raise ValueError('Completed-checkpoint output requires the original output_dir.')
                self.posterior_draws_ = {name:np.load(Path(output_dir)/f'{name}.npy',mmap_mode='r+')
                                        for name in ('preds','sigmas','vars_histogram')}
        else:
            particles, summaries = None, None

        def execute(tasks, phase, executor, context, keep=0):
            next_particles, next_summaries = [None]*self.n_particles, [None]*self.n_particles
            if keep:
                self.posterior_draws_ = allocate_outputs()
            def collect(result, completed):
                index, state, weight, saved, counts, seconds = result
                next_particles[index], next_summaries[index] = state, weight
                self.task_cpu_seconds_ += seconds
                for name, moves in counts.items():
                    for move, count in moves.items():
                        self.mutation_counts_[name][move] += count
                if saved is not None:
                    for name, values in saved.items():
                        if not np.isfinite(values).all():
                            raise FloatingPointError(f'Non-finite {name}.')
                        self.posterior_draws_[name][index::self.n_particles] = values
                if completed == 1 or completed % 20 == 0 or completed == self.n_particles:
                    report(phase,completed_particles=completed,total_particles=self.n_particles)
            if executor is None:
                for completed, task in enumerate(tasks,1):
                    collect(_particle_task(task, context),completed)
            else:
                futures = {executor.submit(_worker, task) for task in tasks}
                try:
                    for completed, future in enumerate(as_completed(futures),1):
                        futures.remove(future)
                        collect(future.result(),completed)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            if keep:
                for array in self.posterior_draws_.values():
                    if hasattr(array,'flush'):
                        array.flush()
            return next_particles, next_summaries

        def run(executor, context):
            nonlocal particles, summaries
            if particles is None:
                tasks = ((i,None,seed(0,0,i),0.,0) for i in range(self.n_particles))
                particles, summaries = execute(tasks, 'prior', executor, context)
                save_checkpoint(particles, summaries)
            while self.beta_history_[-1] < 1:
                stage = len(self.stage_diagnostics_)+1
                if stage > self.max_stages:
                    raise RuntimeError('SMC2 exceeded max_stages before beta=1.')
                old_beta = self.beta_history_[-1]
                beta = _choose_beta(summaries, old_beta, self.ess_ratio)
                log_weights, ess, increment = _importance(summaries, old_beta, beta)
                ancestors = _resample(log_weights, rng)
                report('temperature', next_beta=beta, importance_ess=ess)
                keep = retained_sweeps if beta == 1 else 0
                tasks = ((i,particles[ancestors[i]],seed(1,stage,i),beta,keep)
                         for i in range(self.n_particles))
                particles, summaries = execute(tasks, 'rejuvenation', executor, context, keep)
                self.ancestry_ = self.ancestry_[ancestors]
                self.log_evidence_ += increment
                self.beta_history_.append(beta)
                self.stage_diagnostics_.append({'stage':stage,'beta':beta,'importance_ess':ess,
                    'ess_ratio':ess/self.n_particles,'resampled':True,'ancestors':ancestors,
                    'initial_ancestors':len(np.unique(self.ancestry_)),
                    'log_evidence_increment':increment,'mcmc_steps':self.n_mcmc_steps})
                save_checkpoint(particles, summaries)
                report('stage_complete')

        if self.effective_workers_ == 1:
            arrays = [np.asarray(a).view() for a in (data.X,data.y,test)]
            for array in arrays:
                array.flags.writeable = False
            run(None,{**config,'data':Dataset(*arrays[:2]),'test':arrays[2]})
        else:
            with TemporaryDirectory(prefix='bart-smc2-',dir=temp_dir) as folder:
                for name,array in (('X',data.X),('y',data.y),('test',test)):
                    np.save(Path(folder)/f'{name}.npy',array,allow_pickle=False)
                with ProcessPoolExecutor(max_workers=self.effective_workers_,mp_context=mp.get_context('spawn'),
                        initializer=_init_worker,initargs=(folder,config)) as executor:
                    run(executor,None)
        self.particles_ = particles
        for state in particles:
            for tree in state.trees:
                tree.dataX = data.X
        self.trace = particles
        self.sampler.trace = self.trace
        self.weights_ = np.full(self.n_particles,1/self.n_particles)
        self.is_fitted = True
        report('complete')
        return self

    def update_fit(self, *args, **kwargs):
        raise NotImplementedError('Use a fresh SMC2 fit for each fixed dataset.')

    def update_fit_with_data(self, *args, **kwargs):
        raise NotImplementedError('Use a fresh SMC2 fit for each fixed dataset.')
