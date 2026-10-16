"""Likelihood-tempered, resample-move SMC for Gaussian regression BART.

The targets are p(theta) p(y | theta)**beta, from beta=0 to beta=1.
See Del Moral, Doucet and Jasra (2006), Sequential Monte Carlo Samplers,
https://doi.org/10.1111/j.1467-9868.2006.00553.x.
"""

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import invgamma
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from .bart import BART, DefaultBART, _compress_state_for_trace, resolve_pt_workers
from .moves import contrary_moves
from .params import Parameters, Tree
from .priors import ComprehensivePrior, GlobalParamPrior
from .samplers import DefaultSampler, default_proposal_probs
from .util import Dataset, DefaultPreprocessor


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def resolve_smc_workers(n_jobs, n_particles):
    """PT conventions: 1 serial, None one per particle, -1 available physical CPUs.

    Explicit positive values allow server allocations larger than the local CPU
    count; the number of particles is the only upper bound in that case.
    """
    if n_jobs is not None and (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0
    ):
        raise ValueError("n_jobs must be a nonzero integer or None.")
    return resolve_pt_workers(n_jobs, _positive_int("n_particles", n_particles))


def _normalize_log_weights(log_weights):
    log_weights = np.asarray(log_weights, dtype=float)
    normalizer = float(logsumexp(log_weights))
    if not np.isfinite(normalizer):
        raise FloatingPointError("SMC particle weights have no finite normalizer.")
    return log_weights - normalizer, normalizer


def _effective_sample_size(log_weights):
    return float(np.exp(-logsumexp(2.0 * log_weights)))


def _systematic_resample(log_weights, generator, size=None):
    """Return ancestry indices, including for an output size different from N."""
    size = len(log_weights) if size is None else size
    cdf = np.cumsum(np.exp(log_weights))
    cdf[-1] = 1.0
    positions = (generator.random() + np.arange(size)) / size
    return np.searchsorted(cdf, positions, side="right")


def _copy_particle(state, compress=False):
    copied = _compress_state_for_trace(state) if compress else state.copy()
    # Parameters.copy() only copies the dictionary; particles must own the arrays.
    copied.global_params = {key: value.copy() for key, value in state.global_params.items()}
    return copied


def _restore_particle(state, X):
    """Independent training-ready copy of a cache-free population member."""
    trees = []
    for source in state.trees:
        tree = source.copy(copy_cache=False)
        tree.dataX = X
        tree.n = np.zeros(len(tree.vars), dtype=np.int32)
        tree.n[0] = len(X)
        tree.leaf_ids = np.zeros(len(X), dtype=np.int32)
        if not tree.update_n():
            raise RuntimeError("SMC particle has an invalid training partition.")
        tree.update_outputs()
        trees.append(tree)
    return Parameters(trees, {key: value.copy() for key, value in state.global_params.items()})


def _transport_particle(state):
    """Strip training data as well as caches before crossing process boundaries."""
    result = _copy_particle(state, compress=True)
    for tree in result.trees:
        tree.dataX = None
    return result


# One immutable configuration per worker, rather than serializing X/y for every
# particle or sending forests across a process boundary every posterior sweep.
_POSTERIOR_CONTEXT = None


def _init_posterior_worker(folder, config):
    global _POSTERIOR_CONTEXT
    arrays = {name: np.load(Path(folder) / f"{name}.npy", mmap_mode="r")
              for name in ("X", "y", "X_pred")}
    _POSTERIOR_CONTEXT = {**config, "data": Dataset(arrays["X"], arrays["y"]),
                          "X_pred": arrays["X_pred"]}


def _posterior_particle_path(task, context):
    """Run one whole particle path; ordering of tasks cannot affect its RNG."""
    particle_id, initial, seed = task
    started = time.process_time()
    rng = np.random.default_rng(seed)
    prior = deepcopy(context["prior"])
    prior.tree_prior.generator = rng
    prior.global_prior.generator = rng
    sampler = _SMCMutationSampler(prior, context["proposals"], rng, tol=1)
    sampler.add_data(context["data"])
    sampler.add_thresholds(context["preprocessor"].thresholds)
    state = _restore_particle(initial, context["data"].X)
    sweeps, burn = context["n_sweeps"], context["burn_sweeps"]
    preds = np.empty((sweeps, len(context["X_pred"])), dtype=float)
    sigmas = np.empty(sweeps, dtype=float)
    histogram = np.empty((sweeps, context["data"].p), dtype=np.int32)
    # Also limit BLAS in spawned processes; the caller may run several chains.
    with threadpool_limits(limits=1):
        for sweep in range(burn + sweeps):
            state = sampler.one_iter(state, temp=1.0)
            sampler.accepted_moves_logmh.clear()
            if sweep >= burn:
                index = sweep - burn
                preds[index] = context["preprocessor"].backtransform_y(
                    state.evaluate(context["X_pred"]))
                sigmas[index] = state.global_params["eps_sigma2"][0]
                histogram[index] = state.vars_histogram
    counts = {name: getattr(sampler, name).copy() for name in (
        "move_selected_counts", "move_success_counts", "move_accepted_counts")}
    return (particle_id, {"preds": preds, "sigmas": sigmas,
                          "vars_histogram": histogram},
            _transport_particle(state), counts, time.process_time() - started)


def _run_posterior_worker(task):
    return _posterior_particle_path(task, _POSTERIOR_CONTEXT)


class _SMCGlobalParamPrior(GlobalParamPrior):
    def _sample_eps_sigma2(self, residuals, temp=1.0):
        # SMC importance weights require the same variance law in initialization
        # and mutation. Clipping tiny inverse gamma draws would introduce a point
        # mass absent from the prior and from the weighting likelihood.
        residuals = np.asarray(residuals, dtype=float)
        return invgamma.rvs(
            a=self.eps_nu / 2 + len(residuals) / (2 * temp),
            scale=self.eps_nu * self.eps_lambda / 2 + residuals @ residuals / (2 * temp),
            size=1, random_state=self.generator,
        )


class _SMCMutationSampler(DefaultSampler):
    """Existing tempered structure kernel followed by a full leaf Gibbs sweep.

    tol=1 leaves invalid proposals as self transitions. Retrying until a valid
    split is found would require additional proposal normalization factors.
    """

    def one_iter(self, current, temp, return_trace=False):
        if return_trace:
            raise NotImplementedError("SMC mutation stores particle populations, not substeps.")
        state = super().one_iter(current, temp=temp)
        for tree_id in range(state.n_trees):
            values = self.tree_prior.resample_leaf_vals(
                state, self.data.y, [tree_id], temp=temp,
            )
            state.update_leaf_vals([tree_id], values)
        return state


class SMCBART(BART):
    """Annealed SMC over complete BART ensembles for Gaussian regression.

    ``n_particles`` controls the population, ``ndpost`` the number of equally
    weighted draws exposed through the usual BART prediction API (defaults to
    n_particles). Draws can share ancestors and are not independent MCMC chains.
    ``ess_target`` selects beta increments using conditional ESS / N;
    ``resample_threshold`` triggers systematic resampling using ordinary ESS / N.
    ``n_mcmc_steps`` complete backfitting sweeps run at every new beta.

    Supply ``betas`` (strictly increasing, starting at 0 and ending at 1) for a
    fixed schedule. Otherwise beta is selected adaptively. ``max_stages`` guards
    against a stalled adaptive schedule; failure never returns a tempered fit.

    Tree priors use uniform variables and the preprocessor's global threshold
    grid, conditioned on nonempty leaves, matching the existing admissible tree
    space. No depth truncation or forced stumps are used for initialization.
    The noise prior is calibrated to the preprocessed data just as in DefaultBART.
    ``fixed_eps_sigma2`` is expressed on that preprocessed response scale.
    Classification, sparse Dirichlet split priors and seeded forests are not
    supported by this class. ``update_fit`` refits the combined data with SMC.
    ``n_jobs`` defaults to 1 and only controls ``sample_posterior``: annealing
    and the legacy ``iter_posterior_populations`` iterator remain serial.
    """

    preprocessor_class = DefaultPreprocessor

    def __init__(
        self, n_particles=64, ndpost=None, n_trees=200,
        tree_alpha=0.95, tree_beta=2.0, f_k=2.0, eps_q=0.9,
        eps_nu=3.0, specification="linear", proposal_probs=None,
        max_bins=100, random_state=42, fixed_eps_sigma2=None,
        quick_decay=False, ess_target=0.8, resample_threshold=0.5,
        n_mcmc_steps=2, betas=None, max_stages=1000,
        max_prior_attempts=10000, n_jobs=1,
    ):
        self.n_particles = _positive_int("n_particles", n_particles)
        resolve_smc_workers(n_jobs, self.n_particles)
        self.n_jobs = None if n_jobs is None else int(n_jobs)
        ndpost = self.n_particles if ndpost is None else _positive_int("ndpost", ndpost)
        _positive_int("n_trees", n_trees)
        _positive_int("max_bins", max_bins)
        self.n_mcmc_steps = _positive_int("n_mcmc_steps", n_mcmc_steps)
        self.max_stages = _positive_int("max_stages", max_stages)
        self.max_prior_attempts = _positive_int("max_prior_attempts", max_prior_attempts)
        if not 0 < tree_alpha < (0.5 if quick_decay else 1.0):
            raise ValueError("tree_alpha must lie in (0, 1), or (0, 0.5) for quick_decay.")
        if not np.isfinite(tree_beta) or tree_beta <= 0:
            raise ValueError("tree_beta must be finite and strictly positive.")
        for name, value in (("f_k", f_k), ("eps_nu", eps_nu)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive.")
        if not 0 < eps_q < 1:
            raise ValueError("eps_q must lie strictly between 0 and 1.")
        if fixed_eps_sigma2 is not None and (
            not np.isfinite(fixed_eps_sigma2) or fixed_eps_sigma2 <= 0
        ):
            raise ValueError("fixed_eps_sigma2 must be finite and strictly positive.")
        if not 0 < ess_target < 1:
            raise ValueError("ess_target must lie strictly between 0 and 1.")
        if not 0 < resample_threshold <= 1:
            raise ValueError("resample_threshold must lie in (0, 1].")
        self.ess_target = float(ess_target)
        self.resample_threshold = float(resample_threshold)
        self.betas = None if betas is None else np.asarray(betas, dtype=float).copy()
        if self.betas is not None and (
            self.betas.ndim != 1 or len(self.betas) < 2
            or not np.all(np.isfinite(self.betas))
            or self.betas[0] != 0 or self.betas[-1] != 1
            or np.any(np.diff(self.betas) <= 0)
        ):
            raise ValueError("betas must increase strictly from 0 to 1.")
        if self.betas is not None and len(self.betas) - 1 > self.max_stages:
            raise ValueError("The beta schedule exceeds max_stages.")

        proposals = dict(default_proposal_probs if proposal_probs is None else proposal_probs)
        if not proposals or any(key not in default_proposal_probs for key in proposals):
            raise ValueError("SMC supports grow, prune, change and swap proposals only.")
        probs = np.asarray(list(proposals.values()), dtype=float)
        if np.any(~np.isfinite(probs)) or np.any(probs <= 0) or not np.isclose(probs.sum(), 1):
            raise ValueError("Proposal probabilities must be positive and sum to 1.")
        if any(contrary_moves[key] not in proposals for key in proposals):
            raise ValueError("Each move must include its reverse proposal.")

        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(
            n_trees, tree_alpha, tree_beta, f_k, eps_q, eps_nu,
            specification, rng, False, quick_decay=quick_decay,
            s_alpha=1.0, fixed_eps_sigma2=fixed_eps_sigma2,
        )
        prior.global_prior = _SMCGlobalParamPrior(
            eps_q, eps_nu, specification, rng, False, s_alpha=1.0,
            fixed_eps_sigma2=fixed_eps_sigma2,
        )
        sampler = _SMCMutationSampler(prior, proposals, rng, tol=1)
        super().__init__(self.preprocessor_class(max_bins=max_bins), sampler, ndpost, nskip=0)
        self.particles_ = []
        self.weights_ = np.empty(0)
        self.beta_history_ = [0.0]
        self.stage_diagnostics_ = []
        self.log_evidence_ = None

    def fit(self, X, y, quietly=False):
        X, y = np.asarray(X), np.asarray(y)
        self._validate_data(X, y)
        return super().fit(X, y, quietly=quietly)

    @staticmethod
    def _validate_data(X, y):
        if (X.ndim != 2 or y.ndim != 1 or X.shape[0] == 0 or X.shape[1] == 0
                or len(y) != X.shape[0] or not np.all(np.isfinite(X))
                or not np.all(np.isfinite(y))):
            raise ValueError("SMCBART requires finite, nonempty X (n, p) and y (n,).")

    def _sample_prior_tree(self):
        """Reject the entire tree on invalid splits, preserving the tree prior.

        Resampling just an invalid rule, or forcing a leaf, would instead change
        its prior probability. Conditioning the whole tree is essential here.
        """
        prior = self.sampler.tree_prior
        rng = self.sampler.generator
        for _ in range(self.max_prior_attempts):
            tree = Tree.new(self.data.X)
            queue = [(0, 0)]
            valid = True
            while queue:
                node, depth = queue.pop()
                split_prob = prior.alpha ** depth if prior.quick_decay else prior.alpha / (1 + depth) ** prior.beta
                if rng.random() >= split_prob:
                    continue
                var = int(rng.integers(self.data.p))
                thresholds = self.preprocessor.thresholds[var]
                if len(thresholds) == 0 or not tree.split_leaf(
                    node, var, thresholds[rng.integers(len(thresholds))],
                ):
                    valid = False
                    break
                queue.extend([(2 * node + 1, depth + 1), (2 * node + 2, depth + 1)])
            if valid:
                leaves = tree.leaves
                tree.leaf_vals[leaves] = rng.normal(0, np.sqrt(prior.f_sigma2), size=len(leaves))
                tree.update_outputs()
                return tree
        raise RuntimeError(
            "Could not sample an admissible prior tree; increase max_prior_attempts "
            "or check the data and tree prior (quick_decay forces the root to split)."
        )

    def _sample_prior_state(self):
        prior = self.sampler.global_prior
        # DefaultSampler.get_init_state() draws sigma conditional on y and sets
        # all trees to stumps, so it is not an SMC prior draw.
        sigma2 = prior.fixed_eps_sigma2
        if sigma2 is None:
            sigma2 = invgamma.rvs(
                a=prior.eps_nu / 2, scale=prior.eps_nu * prior.eps_lambda / 2,
                random_state=self.sampler.generator,
            )
        trees = [self._sample_prior_tree() for _ in range(self.sampler.tree_prior.n_trees)]
        return Parameters(trees, {"eps_sigma2": np.array([sigma2], dtype=float)})

    def _log_likelihoods(self, particles):
        # Include the Gaussian constant for the evidence estimate. Accumulate
        # squared residuals in float64, including when the tree cache is float32.
        result = []
        for state in particles:
            variance = float(state.global_params["eps_sigma2"][0])
            residuals = self.data.y.astype(float) - state.evaluate().astype(float)
            result.append(-0.5 * (self.data.n * np.log(2 * np.pi * variance) + residuals @ residuals / variance))
        result = np.asarray(result)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("SMC encountered a non-finite particle log likelihood.")
        return result

    def _next_beta(self, beta, log_weights, log_likelihoods):
        """Bisection on conditional ESS, which starts at N even with unequal W."""
        centered = log_likelihoods - np.max(log_likelihoods)

        def log_cess_fraction(delta):
            incremental = delta * centered
            return 2 * logsumexp(log_weights + incremental) - logsumexp(log_weights + 2 * incremental)

        target = np.log(self.ess_target)
        if log_cess_fraction(1 - beta) >= target:
            return 1.0
        low, high = beta, 1.0
        for _ in range(60):
            midpoint = low + (high - low) / 2
            if log_cess_fraction(midpoint - beta) < target:
                high = midpoint
            else:
                low = midpoint
        if low <= beta:
            raise RuntimeError("Adaptive SMC beta cannot advance at floating-point precision.")
        return float(low)

    def fit_with_data(self, data, quietly=False):
        """Run a fresh population through the full beta=0 to beta=1 sequence."""
        self.is_fitted = False
        self.trace = []
        self.sampler.trace = []
        self.particles_ = []
        self.weights_ = np.empty(0)
        self.log_evidence_ = None
        self.beta_history_ = [0.0]
        self.stage_diagnostics_ = []
        self._validate_data(data.X, data.y)
        self.data = data
        self.sampler.add_data(data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.sampler.global_prior.fit_hyperparameters(data)
        self.sampler.moves_str_cache = None
        self.sampler.moves_cache_iterator = None
        self.sampler.accepted_moves_logmh = []
        for counts in (self.sampler.move_selected_counts, self.sampler.move_success_counts,
                       self.sampler.move_accepted_counts):
            for key in counts:
                counts[key] = 0
        particles = [self._sample_prior_state() for _ in range(self.n_particles)]
        log_weights = np.full(self.n_particles, -np.log(self.n_particles))
        beta = 0.0
        log_evidence = 0.0
        with tqdm(total=1.0, desc="SMC beta", disable=quietly) as progress:
            for stage in range(self.max_stages):
                log_likelihoods = self._log_likelihoods(particles)
                next_beta = (float(self.betas[stage + 1]) if self.betas is not None
                             else self._next_beta(beta, log_weights, log_likelihoods))
                log_weights, increment = _normalize_log_weights(
                    log_weights + (next_beta - beta) * log_likelihoods,
                )
                log_evidence += increment
                ess = _effective_sample_size(log_weights)
                resampled = ess < self.resample_threshold * self.n_particles
                ancestors = np.arange(self.n_particles)
                if resampled:
                    ancestors = _systematic_resample(log_weights, self.sampler.generator)
                    particles = [_copy_particle(particles[index]) for index in ancestors]
                    log_weights = np.full(self.n_particles, -np.log(self.n_particles))
                for index, particle in enumerate(particles):
                    for _ in range(self.n_mcmc_steps):
                        particle = self.sampler.one_iter(particle, temp=1.0 / next_beta)
                    particles[index] = particle
                self.stage_diagnostics_.append({
                    "beta": next_beta, "ess": ess, "resampled": resampled,
                    "ancestors": ancestors.copy(), "log_evidence_increment": increment,
                })
                self.beta_history_.append(next_beta)
                progress.update(next_beta - beta)
                beta = next_beta
                if beta == 1.0:
                    break
            else:
                raise RuntimeError("SMC did not reach beta=1; increase max_stages.")

        self.weights_ = np.exp(log_weights)
        self.particles_ = [_copy_particle(state, compress=True) for state in particles]
        indices = _systematic_resample(log_weights, self.sampler.generator, self.ndpost)
        self.trace = [_copy_particle(particles[index], compress=True) for index in indices]
        self.sampler.trace = self.trace
        # Evidence is for transformed y and the data-calibrated prior.
        self.log_evidence_ = float(log_evidence)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Weighted particle posterior mean, without final resampling noise."""
        if not self.is_fitted:
            raise ValueError("Fit SMCBART before predicting.")
        X = np.asarray(X)
        mean = sum(weight * state.evaluate(X) for weight, state in zip(self.weights_, self.particles_))
        return self.preprocessor.backtransform_y(mean)

    def update_fit(self, X, y, add_ndpost=None, quietly=False):
        """Refit all observed data; this is not an online particle filter."""
        if add_ndpost is not None:
            self.ndpost = _positive_int("add_ndpost", add_ndpost)
        if self.data is None:
            return self.fit(X, y, quietly=quietly)
        data = self.preprocessor.update_transform(X, y, self.data)
        return self.update_fit_with_data(data, add_ndpost=add_ndpost, quietly=quietly)

    def update_fit_with_data(self, data, add_ndpost=None, quietly=False):
        if add_ndpost is not None:
            self.ndpost = _positive_int("add_ndpost", add_ndpost)
        return self.fit_with_data(data, quietly=quietly)

    def init_from_xgboost(self, *args, **kwargs):
        raise NotImplementedError("SMCBART initializes particles from the prior; seeded forests are unsupported.")

    def initialize_posterior_chain(self, random_state=None):
        """Select a weighted particle to initialize an ordinary DefaultBART chain.

        The returned model contains one initial state, not a production draw.
        Its sampler can then run consecutive beta=1 updates. This deliberately
        uses the existing DefaultSampler (tol=1), so comparisons with Default
        isolate the initialization strategy. Each independent chain should have
        its own SMC fit and a separate production random stream.
        """
        if not self.is_fitted:
            raise ValueError("Fit SMCBART before initializing a posterior chain.")
        rng = np.random.default_rng(random_state)
        index = int(rng.choice(self.n_particles, p=self.weights_))
        params = self.get_params()
        keys = ("n_trees", "tree_alpha", "tree_beta", "f_k", "eps_q", "eps_nu",
                "specification", "proposal_probs", "max_bins", "quick_decay",
                "fixed_eps_sigma2")
        chain = DefaultBART(
            ndpost=1, nskip=0, tol=1, random_state=rng,
            **{key: params[key] for key in keys},
        )
        chain.preprocessor = self.preprocessor
        chain.data = self.data
        chain.sampler.add_data(self.data)
        chain.sampler.add_thresholds(self.preprocessor.thresholds)
        chain.sampler.global_prior.fit_hyperparameters(self.data)
        # Saved particles are cache-free. Reconstruct training caches without
        # mutating the population or assuming the default tree-array size.
        state = _restore_particle(self.particles_[index], self.data.X)
        chain.trace = [state]
        chain.sampler.trace = chain.trace
        chain.is_fitted = True
        chain.smc_particle_index_ = index
        return chain

    def iter_posterior_populations(self, n_sweeps=100, burn_sweeps=50,
                                  random_state=None, progress_callback=None):
        """Stream complete beta=1 populations after terminal resampling/mutation.

        Keeps all N particles, never selects one particle for a long chain.
        Terminal systematic resampling makes weights equal. Each retained sweep
        applies one full SMC mutation sweep to EVERY particle, after optional
        beta=1 warm-up. N*n_sweeps states may then be pooled as an unordered
        posterior approximation; they are dependent, not iid observations or
        one chronological MCMC trajectory. Earlier beta<1 states are excluded.

        Yields (sweep_index, particles). The list and its states are borrowed:
        summarize immediately, or copy states before advancing the iterator.
        Only O(N) working states are retained. A separate RNG/prior/sampler
        leaves the annealed population and its diagnostics unchanged.
        This legacy iterator is serial and preserves its shared random stream.
        Use ``sample_posterior`` for worker-count-invariant particle parallelism.
        """
        if not self.is_fitted:
            raise ValueError("Fit SMCBART before sampling posterior populations.")
        n_sweeps = _positive_int("n_sweeps", n_sweeps)
        if (isinstance(burn_sweeps, (bool, np.bool_))
                or not isinstance(burn_sweeps, (int, np.integer)) or burn_sweeps < 0):
            raise ValueError("burn_sweeps must be a non-negative integer.")
        rng = np.random.default_rng(random_state)
        prior = deepcopy(self.sampler.prior)
        prior.tree_prior.generator = rng
        prior.global_prior.generator = rng
        sampler = _SMCMutationSampler(prior, dict(self.sampler.proposals), rng, tol=1)
        sampler.add_data(self.data)
        sampler.add_thresholds(self.preprocessor.thresholds)
        log_weights = np.full(self.n_particles, -np.inf)
        np.log(self.weights_, out=log_weights, where=self.weights_ > 0)
        indices = _systematic_resample(log_weights, rng)
        particles = [_restore_particle(self.particles_[index], self.data.X) for index in indices]
        self.posterior_resampling_indices_ = indices.copy()
        self.population_sampler_ = sampler
        self.posterior_population_ = []
        for sweep in range(int(burn_sweeps) + n_sweeps):
            for index, state in enumerate(particles):
                particles[index] = sampler.one_iter(state, temp=1.0)
            sampler.accepted_moves_logmh.clear()
            if progress_callback is not None:
                progress_callback(sweep + 1, int(burn_sweeps) + n_sweeps)
            if sweep >= burn_sweeps:
                yield sweep - int(burn_sweeps), particles
        self.posterior_population_ = [_copy_particle(state, compress=True) for state in particles]

    def sample_posterior(self, X, n_sweeps=100, burn_sweeps=50,
                         random_state=None, progress_callback=None,
                         outputs=None, temp_dir=None):
        """Collect beta=1 particle paths, serial by default or using ``n_jobs``.

        Returns a dictionary with ``preds`` (N*n_sweeps, len(X)) in response
        units, ``sigmas`` (N*n_sweeps,) containing eps_sigma2 on the transformed
        response scale, and ``vars_histogram`` (N*n_sweeps, p). Row s*N+i is
        particle i after retained sweep s. Optional ``outputs`` supplies writable
        arrays/memmaps of these shapes (float64, float64, int32 respectively).
        Forests are saved only at the endpoint in ``posterior_population_``;
        annealed particles, prediction trace and annealing RNG are unchanged.

        Terminal resampling has its own RNG; every offspring has a separate
        stream indexed by particle ID. For a fixed seed and fitted population,
        results are identical across worker counts in the same environment.
        Streams differ from the legacy population iterator. The full mutation
        sweep, prior and posterior target are the same in both interfaces.

        Each task runs a complete path including burn-in, computes predictions
        locally and returns only numeric summaries plus its endpoint forest.
        Read-only training/prediction arrays are shared through temporary file
        mappings; ``temp_dir`` can select server-local scratch. No per-sweep
        process synchronization or forest transfer is needed. Each worker uses
        one BLAS thread. ``progress_callback(completed_paths, N)`` runs in the
        caller after storing each completed path, not after population sweeps.
        Call multiprocessing programs under ``if __name__ == '__main__':``.
        """
        if not self.is_fitted:
            raise ValueError("Fit SMCBART before sampling posterior populations.")
        n_sweeps = _positive_int("n_sweeps", n_sweeps)
        if (isinstance(burn_sweeps, (bool, np.bool_))
                or not isinstance(burn_sweeps, (int, np.integer)) or burn_sweeps < 0):
            raise ValueError("burn_sweeps must be a non-negative integer.")
        X = np.asarray(X)
        if (X.ndim != 2 or not len(X) or X.shape[1] != self.data.p
                or not np.all(np.isfinite(X))):
            raise ValueError("X must be finite, nonempty and have the training feature count.")
        workers = resolve_smc_workers(self.n_jobs, self.n_particles)
        count = n_sweeps * self.n_particles
        specs = {"preds": ((count, len(X)), np.dtype(float)),
                 "sigmas": ((count,), np.dtype(float)),
                 "vars_histogram": ((count, self.data.p), np.dtype(np.int32))}
        if outputs is None:
            outputs = {name: np.empty(shape, dtype=dtype)
                       for name, (shape, dtype) in specs.items()}
        for name, (shape, dtype) in specs.items():
            array = outputs.get(name)
            if (not isinstance(array, np.ndarray) or array.shape != shape
                    or array.dtype != dtype or not array.flags.writeable):
                raise ValueError(f"outputs['{name}'] must be writable {dtype} with shape {shape}.")
        names = list(specs)
        if any(np.shares_memory(outputs[left], outputs[right])
               for i, left in enumerate(names) for right in names[i+1:]):
            raise ValueError("Posterior output arrays must not overlap.")

        # A fixed stream per offspring, independent of resampled ancestor,
        # process count, task scheduling, and number of draws on other paths.
        entropy = np.random.default_rng(random_state).integers(0, 2**32, size=4, dtype=np.uint32)
        streams = np.random.SeedSequence(entropy).spawn(self.n_particles + 1)
        log_weights = np.full(self.n_particles, -np.inf)
        np.log(self.weights_, out=log_weights, where=self.weights_ > 0)
        indices = _systematic_resample(log_weights, np.random.default_rng(streams[0]))
        self.posterior_resampling_indices_ = indices.copy()
        self.posterior_workers_ = workers
        self.posterior_sampling_complete_ = False
        self.posterior_population_ = []
        self.posterior_path_cpu_seconds_ = 0.0
        self.posterior_move_counts_ = {name: dict.fromkeys(self.sampler.proposals, 0) for name in (
            "move_selected_counts", "move_success_counts", "move_accepted_counts")}
        endpoints = [None] * self.n_particles
        config = {"prior": self.sampler.prior, "proposals": dict(self.sampler.proposals),
                  "preprocessor": self.preprocessor,
                  "n_sweeps": n_sweeps, "burn_sweeps": int(burn_sweeps)}

        def task(index):
            return index, _transport_particle(self.particles_[indices[index]]), streams[index+1]

        def collect(result, completed):
            index, summaries, endpoint, counts, cpu_seconds = result
            for name in specs:
                if not np.all(np.isfinite(summaries[name])):
                    raise FloatingPointError(f"Non-finite posterior {name}.")
                outputs[name][index::self.n_particles] = summaries[name]
            if not np.all(summaries["sigmas"] > 0):
                raise FloatingPointError("Nonpositive posterior variance.")
            # Reattach one shared caller-side reference for forest properties
            # such as vars_histogram, without restoring O(n) training caches.
            for tree in endpoint.trees:
                tree.dataX = self.data.X
            endpoints[index] = endpoint
            for name, moves in counts.items():
                for move, value in moves.items():
                    self.posterior_move_counts_[name][move] += value
            self.posterior_path_cpu_seconds_ += cpu_seconds
            if progress_callback is not None:
                progress_callback(completed, self.n_particles)

        if workers == 1:
            # Match the workers' read-only array signatures in Numba as well as
            # their math. Views do not change the caller's writeability flags.
            arrays = [value.view() for value in (self.data.X, self.data.y, X)]
            for array in arrays:
                array.flags.writeable = False
            context = {**config, "data": Dataset(*arrays[:2]), "X_pred": arrays[2]}
            for index in range(self.n_particles):
                collect(_posterior_particle_path(task(index), context), index+1)
        else:
            # Wait for workers to exit/close mappings before deleting files on
            # Windows too. Exceptions propagate; partial runs never claim success.
            with TemporaryDirectory(prefix="bart-smc-", dir=temp_dir) as folder:
                for name, array in (("X", self.data.X), ("y", self.data.y), ("X_pred", X)):
                    np.save(Path(folder) / f"{name}.npy", array, allow_pickle=False)
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                        initializer=_init_posterior_worker, initargs=(folder, config)) as executor:
                    pending = {executor.submit(_run_posterior_worker, task(i)): i
                               for i in range(self.n_particles)}
                    try:
                        for completed, future in enumerate(as_completed(pending), 1):
                            pending.pop(future)
                            collect(future.result(), completed)
                    except BaseException:
                        # Cancel paths that have not started; the executor joins
                        # running workers before the shared files are removed.
                        for future in pending:
                            future.cancel()
                        raise
        self.posterior_population_ = endpoints
        self.posterior_sampling_complete_ = True
        return outputs

    def get_params(self):
        prior = self.sampler.prior
        return {
            "model_type": "SMCBART", "n_particles": self.n_particles,
            "ndpost": self.ndpost, "n_trees": prior.tree_prior.n_trees,
            "tree_alpha": prior.tree_prior.alpha, "tree_beta": prior.tree_prior.beta,
            "f_k": prior.tree_prior.f_k, "eps_q": prior.global_prior.eps_q,
            "eps_nu": prior.global_prior.eps_nu,
            "specification": prior.global_prior.specification,
            "fixed_eps_sigma2": prior.global_prior.fixed_eps_sigma2,
            "quick_decay": prior.tree_prior.quick_decay,
            "max_bins": self.preprocessor.max_bins,
            "proposal_probs": self.sampler.proposals.copy(),
            "ess_target": self.ess_target, "resample_threshold": self.resample_threshold,
            "n_mcmc_steps": self.n_mcmc_steps,
            "betas": None if self.betas is None else self.betas.tolist(),
            "max_stages": self.max_stages, "max_prior_attempts": self.max_prior_attempts,
            "n_jobs": self.n_jobs,
        }
