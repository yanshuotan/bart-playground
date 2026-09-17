from warnings import warn
from contextlib import ExitStack
import multiprocessing as mp
import numpy as np
from typing import Optional, Callable, Dict, Any, Sequence
from scipy.stats import norm
from tqdm import tqdm

from joblib import effective_n_jobs

from .samplers import Sampler, DefaultSampler, MultiSampler, ProbitSampler, LogisticSampler, TemperatureSchedule, default_proposal_probs, mtmh_proposal_probs
from .priors import ComprehensivePrior, ProbitPrior, LogisticPrior
from .util import Preprocessor, DefaultPreprocessor, ClassificationPreprocessor, Dataset
from .params import Parameters


def _physical_core_count() -> int:
    """Physical cores this process may use.

    PT workers are compute-bound, and on an SMT machine one worker per logical
    core is measurably slower than one per physical core, so negative `n_jobs`
    resolves against this rather than against the logical count.

    Never exceeds what joblib reports as available, so a cgroup quota, a CPU
    affinity mask or a scheduler allocation still caps the worker count on a
    shared machine. Falls back to that count when psutil is not installed.
    """
    available = int(effective_n_jobs(-1))
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    if not physical:
        return max(1, available)
    return max(1, min(int(physical), available))


def resolve_pt_workers(n_jobs: Optional[int], n_temperatures: int) -> int:
    """How many worker processes `n_jobs` asks for with this many temperatures.

    1 means serial, None one process per temperature, a negative value follows
    joblib's convention counted in physical cores (-1 all of them, -2 all but
    one), and the result never exceeds the number of temperatures. Exposed so
    that callers can report or log the worker count without constructing a
    model; see `ParallelTemperingBART`.
    """
    if n_temperatures <= 1:
        return 1
    if n_jobs is None:
        return int(n_temperatures)
    if n_jobs < 0:
        requested = _physical_core_count() + 1 + int(n_jobs)
    else:
        requested = int(n_jobs)
    return int(max(1, min(n_temperatures, requested)))


class _ConstantTemperature:
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, _t):
        return self.temperature


class _Missing:
    pass


_MISSING = _Missing()


def _compress_state_for_trace(state):
    """Cache-free copy of a state for storing in a trace.

    Idempotent: an already-compressed state (cache is None) is copied without
    re-evaluating the trees.
    """
    if state.cache is None:
        state_out = Parameters(
            trees=[tree.copy(copy_cache=False) for tree in state.trees],
            global_params=state.global_params.copy(),
            cache=np.empty(0),
        )
    else:
        state_out = state.copy(copy_cache=False)
    state_out.clear_cache()
    return state_out


def _leaf_basis_and_svd_for_sampler_state(sampler, state):
    """Leaf basis and its SVD for all trees (both None for a single tree).

    Both depend only on the tree structures, so they stay valid for a whole PT
    swap step: a swap only permutes states and the post-swap refresh only
    changes leaf values. The basis has to be built anyway to compute the SVD,
    so keeping it costs nothing and saves the refresh from rebuilding it.
    """
    if state.n_trees <= 1:
        return None, None
    leaf_basis = state.leaf_basis(np.arange(state.n_trees))
    return leaf_basis, sampler.likelihood.leaf_basis_svd(leaf_basis)


def _leaf_basis_svd_for_sampler_state(sampler, state):
    """Just the SVD; see `_leaf_basis_and_svd_for_sampler_state`."""
    return _leaf_basis_and_svd_for_sampler_state(sampler, state)[1]


class _PTSwapLoglikCache:
    """Per-swap-step cache for PT swap likelihoods, indexed by chain position.

    Swap sweeps never change tree structures (a swap permutes states and the
    post-swap refresh only resamples leaf values), so each state's leaf-basis
    SVD is computed once and moves with the state. Collapsed log-likelihood
    values are memoized per temperature and dropped whenever the state at a
    position changes. The same floating-point operations are performed as
    without the cache, so results are bitwise identical.
    """

    def __init__(self, n_chains: int):
        # Each entry is _MISSING or a (leaf_basis, svd) pair for the state at
        # that position; the basis is reused by the post-swap leaf refresh.
        self.entries = [_MISSING] * n_chains
        self.values = [dict() for _ in range(n_chains)]

    def swap(self, i: int, j: int):
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]
        self.values[i] = {}
        self.values[j] = {}

    def leaf_basis(self, chain_id: int):
        """Cached basis for the state now at `chain_id`, or None if not built yet."""
        entry = self.entries[chain_id]
        return None if entry is _MISSING else entry[0]


def _run_sampler_block(
    sampler,
    current_state,
    temp,
    n_steps: int,
    keep_states_from: Optional[int] = None,
):
    states = [] if keep_states_from is not None else None
    state = current_state
    for step in range(int(n_steps)):
        state = sampler.one_iter(state, temp=temp, return_trace=False)
        if keep_states_from is not None and step >= keep_states_from:
            # DefaultSampler.one_iter updates `state` in place when no tree move is
            # accepted, so a later step could overwrite an earlier kept state's
            # global params. Keep a cache-free snapshot: intermediate states are
            # only used for traces, and this keeps inter-process transfers small.
            states.append(_compress_state_for_trace(state))
    return sampler, state, states, keep_states_from


def _refresh_tempered_state_for_sampler(sampler, state, temp, leaf_basis=None):
    tree_ids = np.arange(state.n_trees, dtype=int)

    new_leaf_vals = sampler.tree_prior.resample_leaf_vals(
        state,
        data_y=sampler.data.y,
        tree_ids=tree_ids,
        temp=temp,
        leaf_basis=leaf_basis,
    )
    state.update_leaf_vals(tree_ids.tolist(), new_leaf_vals)
    return state


def _loglik_as_float(value) -> float:
    """Scalar log-likelihood.

    `trees_log_marginal_lkhd` returns a plain float for a single tree but a
    one-element array for an ensemble, because it is handed the one-element
    `eps_sigma2` array. Picking the element out keeps the value bitwise
    identical and avoids numpy's deprecated array-to-scalar conversion.
    """
    return float(np.asarray(value).reshape(-1)[0])


def _collapsed_loglik_for_sampler_state(sampler, state, temp: float, leaf_basis_svd=None) -> float:
    collapsed = _loglik_as_float(
        sampler.likelihood.trees_log_marginal_lkhd(
            state,
            sampler.data.y,
            np.arange(state.n_trees),
            temp=float(temp),
            leaf_basis_svd=leaf_basis_svd,
        )
    )
    eps_sigma2 = float(state.global_params["eps_sigma2"][0])
    n = int(sampler.data.y.shape[0])
    return collapsed - 0.5 * n * np.log(2.0 * np.pi * eps_sigma2) / float(temp)


def _pt_chain_worker(conn):
    """Persistent PT worker process, hosting one or more chains.

    Every message is ``(slot, command, *args)``; ``slot`` selects one of the
    chains this process was given. Replies are sent in the order the requests
    arrived, so the main process can pipeline requests across all slots and
    then collect the answers in the same order.

    A slot owns one chain's sampler (its RNG, move statistics and temperature)
    and holds exactly one state. During a swap step states are not moved: the
    main process tracks which slot holds the state of each temperature
    position, and the chain RNGs travel instead (their exact bit-generator
    states), so every random draw happens in the same order and from the same
    stream as in the serial implementation. Displaced states are moved to
    their positions once, at the end of the swap step.

    The chain payloads arrive over the pipe rather than as process arguments;
    see `_PTWorkerProcess` for why.
    """
    payloads = conn.recv()
    samplers = [payload[0] for payload in payloads]
    states = [payload[1] for payload in payloads]
    temps = [payload[2] for payload in payloads]
    # _MISSING or a (leaf_basis, svd) pair for the state this slot holds; both
    # depend only on the tree structures, so they survive a post-swap refresh.
    leaf_caches = [_MISSING] * len(payloads)
    loglik_memos = [{} for _ in payloads]
    try:
        while True:
            msg = conn.recv()
            slot, cmd = msg[0], msg[1]
            if cmd == "close":
                conn.send(None)
                break
            elif cmd == "advance":
                n_steps, keep_states_from = msg[2], msg[3]
                samplers[slot], states[slot], kept_states, _ = _run_sampler_block(
                    samplers[slot],
                    states[slot],
                    temps[slot],
                    n_steps,
                    keep_states_from=keep_states_from,
                )
                leaf_caches[slot] = _MISSING
                loglik_memos[slot] = {}
                conn.send(kept_states)
            elif cmd == "collapsed_logliks":
                # Tree structures are fixed during a swap step, so the leaf-basis
                # SVD is computed once; values are memoized until leaf values change.
                if leaf_caches[slot] is _MISSING:
                    leaf_caches[slot] = _leaf_basis_and_svd_for_sampler_state(
                        samplers[slot], states[slot]
                    )
                memo = loglik_memos[slot]
                values = []
                for t in msg[2]:
                    if t not in memo:
                        memo[t] = _collapsed_loglik_for_sampler_state(
                            samplers[slot],
                            states[slot],
                            t,
                            leaf_basis_svd=leaf_caches[slot][1],
                        )
                    values.append(memo[t])
                conn.send(tuple(values))
            elif cmd == "refresh_with_rng":
                # Refresh the held state at another position's temperature, using
                # that position's chain RNG; return the advanced RNG state.
                chain_temp, rng_state = msg[2], msg[3]
                bit_generator = samplers[slot].generator.bit_generator
                own_rng_state = bit_generator.state
                bit_generator.state = rng_state
                try:
                    cached = leaf_caches[slot]
                    states[slot] = _refresh_tempered_state_for_sampler(
                        samplers[slot],
                        states[slot],
                        chain_temp,
                        leaf_basis=None if cached is _MISSING else cached[0],
                    )
                    new_rng_state = bit_generator.state
                finally:
                    bit_generator.state = own_rng_state
                loglik_memos[slot] = {}
                conn.send(new_rng_state)
            elif cmd == "export_state":
                conn.send(states[slot])
            elif cmd == "import_state":
                states[slot] = conn.recv()
                leaf_caches[slot] = _MISSING
                loglik_memos[slot] = {}
                conn.send(None)
            elif cmd == "get_rng_state":
                conn.send(samplers[slot].generator.bit_generator.state)
            elif cmd == "set_rng_state":
                samplers[slot].generator.bit_generator.state = msg[2]
                conn.send(None)
            elif cmd == "compressed_state":
                conn.send(_compress_state_for_trace(states[slot]))
            elif cmd == "get_sampler":
                conn.send(samplers[slot])
            else:
                raise ValueError(f"Unknown PT worker command: {cmd}")
    finally:
        conn.close()


class _PTWorkerProcess:
    """One OS process hosting the chains of a single worker.

    The chain payloads are deliberately not passed as process arguments. Under
    the spawn start method the parent blocks in `start()` until the child
    drains the pickled arguments from the pipe, and the child only gets there
    after importing this package (seconds). Starting workers one at a time
    would therefore serialize one package import per worker. Instead each
    process is started with nothing but its pipe, and payloads are sent once
    every process is running, so the child imports overlap.
    """

    def __init__(self, payloads):
        ctx = mp.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(
            target=_pt_chain_worker,
            args=(child_conn,),
            daemon=True,
        )
        self.process.start()
        child_conn.close()
        self._payloads = list(payloads)

    def send_payloads(self):
        """Hand the chains over to the process; blocks until the child is up."""
        if self._payloads is not None:
            payloads, self._payloads = self._payloads, None
            self.conn.send(payloads)

    def close(self):
        if self.process.is_alive():
            if self._payloads is not None:
                # Still waiting for its payloads, so it cannot answer commands.
                self.process.terminate()
            else:
                self.conn.send((0, "close"))
                self.conn.recv()
        self.conn.close()
        self.process.join()


class _PTChainWorker:
    """Handle for one chain inside a `_PTWorkerProcess`.

    Several handles can share a process. The process answers in request order,
    so callers must receive replies in the order they issued the requests,
    which is what the fit loop already does.
    """

    def __init__(self, worker_process: "_PTWorkerProcess", slot: int):
        self._worker_process = worker_process
        self.slot = int(slot)

    @property
    def conn(self):
        return self._worker_process.conn

    def request(self, *msg):
        self.conn.send((self.slot,) + msg)

    def recv(self):
        return self.conn.recv()

    def request_export_state(self):
        self.request("export_state")

    def recv_state_bytes(self) -> bytes:
        # Relay the pickled state without unpickling it in the main process.
        return self.conn.recv_bytes()

    def request_import_state_bytes(self, payload: bytes):
        self.request("import_state")
        self.conn.send_bytes(payload)

    def get_sampler(self):
        self.request("get_sampler")
        return self.conn.recv()


def _split_chains_across_workers(n_chains: int, n_workers: int) -> list[list[int]]:
    """Contiguous, balanced chain groups; the first groups take the remainder."""
    if not 1 <= n_workers <= n_chains:
        raise ValueError("n_workers must be between 1 and n_chains.")
    base, extra = divmod(n_chains, n_workers)
    groups = []
    first = 0
    for worker_id in range(n_workers):
        size = base + (1 if worker_id < extra else 0)
        groups.append(list(range(first, first + size)))
        first += size
    return groups


def _start_pt_chain_workers(samplers, states, temps, n_workers: int):
    """Start every worker process, then send the payloads.

    Returns one chain handle per chain plus the processes hosting them; the
    deferred payloads and the chain-to-process packing are explained in
    `_PTWorkerProcess` and `_PTChainWorker`.
    """
    groups = _split_chains_across_workers(len(samplers), n_workers)
    processes = []
    workers = [None] * len(samplers)
    try:
        for group in groups:
            worker_process = _PTWorkerProcess(
                [(samplers[chain_id], states[chain_id], temps[chain_id]) for chain_id in group]
            )
            processes.append(worker_process)
            for slot, chain_id in enumerate(group):
                workers[chain_id] = _PTChainWorker(worker_process, slot)
        for worker_process in processes:
            worker_process.send_payloads()
    except BaseException:
        for worker_process in processes:
            worker_process.close()
        raise
    return workers, processes


class BART:
    """
    API for the BART model.
    """
    preprocessor_class = None  # Must be overridden by subclasses
    
    def __init__(self, preprocessor : Preprocessor, sampler : Sampler, 
                 ndpost=1000, nskip=100):
        """
        Initialize the BART model.
        """
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.ndpost = int(ndpost)
        self.nskip = int(nskip)
        self.trace = []
        self.is_fitted = False
        self.data = None

    def get_params(self) -> Dict[str, Any]:
        """Get effective parameters for this model instance."""
        return {"ndpost": self.ndpost, "nskip": self.nskip}

    def fit(self, X, y, quietly = False):
        """
        Fit the BART model.
        """
        data = self.preprocessor.fit_transform(X, y)
        return self.fit_with_data(data, quietly=quietly)
    
    def fit_with_data(self, data: Dataset, quietly=False):
        """
        Fit the BART model using a preprocessed dataset.
        """
        self.data = data
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.trace = self.sampler.run(self.ndpost + self.nskip, quietly=quietly, n_skip=self.nskip)
        self.is_fitted = True
        return self
    
    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        """
        Update an existing fitted model with new data points.
        
        Parameters:
            X: New feature data to add
            y: New target data to add
            add_ndpost: Number of more posterior samples to draw
            quietly: Whether to suppress output
            
        Returns:
            self
        """
        if self.data is None:
            self.fit(X, y, quietly=quietly)
            return self
        if not self.is_fitted: # or self.data.n <= 10:
            # If not fitted yet, or data is empty, or not enough data, just do a regular fit
            X_combined = np.vstack((self.data.X, X))
            y_combined = np.hstack((self.data.y, y))
            self.fit(X_combined, y_combined, quietly=quietly)
            return self

        updated_data = self.preprocessor.update_transform(X, y, self.data)
        return self.update_fit_with_data(updated_data, add_ndpost=add_ndpost, quietly=quietly)
    
    def update_fit_with_data(self, data: Dataset, add_ndpost=20, quietly=False):
        """
        Update an existing fitted model with a new preprocessed dataset.
        """
        if self.data is None or not self.is_fitted:
            return self.fit_with_data(data, quietly=quietly)
        additional_iters = add_ndpost
        # Set all previous iterations as burn-in
        self.nskip += self.ndpost
        # Set new add_ndpost iterations as post-burn-in
        self.ndpost = add_ndpost

        self.data = data
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        
        # Run the sampler for additional iterations
        new_trace = self.sampler.continue_run(additional_iters, new_data=self.data, quietly=quietly)
        # Previous samples are treated as burn-in (via nskip adjustment above), so only the latest posterior samples are kept.
        self.trace = new_trace
        
        return self
    
    @property
    def _trace_length(self):
        return len(self.trace)
    
    @property
    def range_post(self):
        """
        Get the range of posterior samples.
        """
        total_iterations = self._trace_length
        if total_iterations < self.ndpost:
            raise ValueError(f"Not enough posterior samples: {total_iterations} < {self.ndpost} (provided ndpost).")
        return range(total_iterations - self.ndpost, total_iterations)
    
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for i, k in enumerate(self.range_post):
            preds[:, i] = self.predict_trace(k, X, backtransform=backtransform)
        return preds
    
    # WeightSchedule: Callable that takes a trace index k and returns a normalized probability (sum over all k must equal 1.0)
    WeightSchedule = Callable[[int], float]
    def posterior_sample(self, X, schedule: WeightSchedule, backtransform=True):
        """
        Get a posterior sample of f(x) for each row in X.
        """
        pred = np.zeros((X.shape[0]))
        # sample a k using the schedule
        k = self.sampler.generator.choice(
            range(self._trace_length), 
            p=[schedule(k) for k in range(self._trace_length)]
        )
        y_eval = self.trace[k].evaluate(X)
        if backtransform:
            pred = self.preprocessor.backtransform_y(y_eval)
        else:
            pred = y_eval
        return pred
    
    def predict(self, X):
        """
        Predict using the BART model.
        """
        return np.mean(self.posterior_f(X), axis=1)
    
    def predict_trace(self, k: int, X, backtransform=True):
        """
        Predict using a single trace state.
        """
        y_eval = self.trace[k].evaluate(X)
        if backtransform:
            return self.preprocessor.backtransform_y(y_eval)
        else:
            return y_eval
    
    def posterior_predict(self, X):
        """
        Get the full posterior distribution of predictions.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with posterior samples
        """
        preds = self.posterior_f(X, backtransform=False)
        for k in range(self.ndpost):
            eps_sigma2 = self.trace[k].global_params['eps_sigma2']
            preds[:, k] += self.sampler.generator.normal(0, np.sqrt(eps_sigma2), size=preds[:, k].shape)
            preds[:, k] = self.preprocessor.backtransform_y(preds[:, k])
        return preds

    def init_from_xgboost(
            self,
            xgb_model,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            xgb_kwargs: dict | None = None,
            debug: bool = False
    ) -> "BART":
        # Ensure self.data is correctly populated. 
        # If X, y are different from self.data, an update or re-fit might be needed.
        # We assume that X and y are train_data.X and train_data.y,
        # and self.data is already train_data.
        if self.data is None: 
            self.data = self.preprocessor.fit_transform(X,y)
        elif X is not self.data.X or y is not self.data.y: # Check if X,y are different objects
            # This path is taken if X, y are new/different from what self.data currently holds.
            # If they are actually different datasets, a full re-fit or careful update is needed.
            print("[WARN BART.init_from_xgboost] X or y are different objects than self.data.X/y. Calling update_transform.")
            self.data = self.preprocessor.update_transform(X, y, self.data)

        dataX = self.data.X # Use self.data which should be correctly set

        from .xgb_init import fit_and_init_trees
        xgb_kwargs = xgb_kwargs or {}

        n_trees = self.sampler.tree_prior.n_trees

        model, init_trees = fit_and_init_trees(
            X, y,
            model=xgb_model,
            dataX=dataX,
            n_estimators=n_trees,
            debug=debug,
            **xgb_kwargs
        )

        self.sampler = DefaultSampler(
            prior=self.sampler.prior,
            proposal_probs=self.sampler.proposals,
            generator=self.sampler.generator,
            temp_schedule=self.sampler.temp_schedule,
            tol=self.sampler.tol,
            init_trees=init_trees
        )

        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)

        # ——— warm-start a BART draw by resampling leaf-values & global params ———
        init_state = self.sampler.get_init_state()
        if debug: # Check if debug flag is True
            print(f"[DEBUG XGB_INIT] Initial state from get_init_state():")
            print(f"[DEBUG XGB_INIT]   Tree 0 Leaf Vals (from XGB): {init_state.trees[0].leaf_vals[init_state.trees[0].leaves]}")
            print(f"[DEBUG XGB_INIT]   Global eps_sigma2: {init_state.global_params['eps_sigma2']}")

        # 1) for each tree, draw new leaf-values under BART's posterior
        for k in range(self.sampler.tree_prior.n_trees):
            new_leaf_vals = self.sampler.tree_prior.resample_leaf_vals(
                init_state,
                data_y=self.data.y,
                tree_ids=[k],
            )
            if debug:
                print(f"[DEBUG XGB_INIT] Resampled Leaf Vals for tree {k}: {new_leaf_vals}")
            init_state.update_leaf_vals([k], new_leaf_vals)
        # 2) draw the global μ/σ
        init_state.global_params = self.sampler.global_prior.resample_global_params(
            init_state,
            data_y=self.data.y
        )
        if debug:
            print(f"[DEBUG XGB_INIT] Resampled Global eps_sigma2: {init_state.global_params['eps_sigma2']}")
            print(f"[DEBUG XGB_INIT] Final state for trace - Tree 0 Leaf Vals: {init_state.trees[0].leaf_vals[init_state.trees[0].leaves]}")

        # 3) overwrite the sampler's "trace" so .run() will start from a BART-sampled state
        self.sampler.trace = [init_state]

        return self

    def _check_temperature(self, temperature):
        """
        Check if the temperature is a valid type.
        """
        is_temperature_number = type(temperature) in [float, int]
        if is_temperature_number:
            temp_func = _ConstantTemperature(temperature)
            return TemperatureSchedule(temp_func)
        elif type(temperature) == TemperatureSchedule:
            return temperature
        else:
            raise ValueError("Invalid temperature type ", type(temperature))
        
    def clean_trace(self, k, keep_indices=True):
        """
        Clean the trace by removing the k-th element.
        If keep_indices is True, it will set the k-th element to None and keep the originial indices.
        If keep_indices is False, it will remove the k-th element from the trace.
        """
        if not keep_indices:
            self.trace = [t for i, t in enumerate(self.trace) if i != k]
        else:
            self.trace[k] = None

class DefaultBART(BART):
    preprocessor_class = DefaultPreprocessor

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, dirichlet_prior=False, quick_decay: bool = False,
                 s_alpha: float = 1.0, fixed_eps_sigma2: Optional[float] = None,
                 init_trees=None, init_sigma2=None):
        if max_bins is None:
            max_bins = 100
        preprocessor = self.preprocessor_class(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay, s_alpha=s_alpha, fixed_eps_sigma2=fixed_eps_sigma2)
        temp_schedule = self._check_temperature(temperature)
        sampler = DefaultSampler(prior=prior, proposal_probs=proposal_probs, generator=rng, 
                                 tol=tol, temp_schedule=temp_schedule, init_trees=init_trees)
        super().__init__(preprocessor, sampler, ndpost, nskip)
        
    def get_params(self) -> Dict[str, Any]:
        """Get all effective parameters for this model instance."""
        return {
            "model_type": "DefaultBART",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "n_trees": self.sampler.tree_prior.n_trees,
            "tree_alpha": self.sampler.tree_prior.alpha,
            "tree_beta": self.sampler.tree_prior.beta,
            "f_k": self.sampler.tree_prior.f_k,
            "eps_nu": self.sampler.prior.global_prior.eps_nu,
            "eps_q": self.sampler.prior.global_prior.eps_q,
            "specification": self.sampler.prior.global_prior.specification,
            "dirichlet_prior": self.sampler.prior.global_prior.dirichlet_prior,
            "quick_decay": self.sampler.tree_prior.quick_decay,
            "proposal_probs": self.sampler.proposals,
            "fixed_eps_sigma2": self.sampler.prior.global_prior.fixed_eps_sigma2
        }
        
    def predict_proba(self, X):
        """
        DefaultBART doesn't support classification probabilities.
        Use naive prediction instead.
        Returns:
            Array of shape (n_samples, 1) with predicted values
        """
        warn("predict_proba not recommended for regression BART. Use LogisticBART for classification.")
        prob_1 = np.clip(self.predict(X).reshape(-1, 1), 0.0, 1.0)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

    def feature_inclusion_probability(self):
        """
        Compute posterior inclusion probability for each feature.

        For each posterior draw k in range_post, mark 1 if feature i is used
        at least once as a split variable in any tree (hist_k[i] > 0), else 0.
        Returns the average over posterior draws.

        Returns
        -------
        np.ndarray
            Array of shape (p,) where p is the number of features.
        """
        if not self.is_fitted or self.data is None:
            raise ValueError("Model must be fitted before computing inclusion probability.")

        p = self.data.X.shape[1]
        probs = np.zeros(p, dtype=float)

        for k in self.range_post:
            # trace[k] is Parameters for regression BART
            # vars_histogram is now a numpy array of shape (p,)
            hist = self.trace[k].vars_histogram
            if hist.size == 0:
                continue
            # Mark features that were used at least once
            probs += (hist > 0).astype(float)

        probs /= float(self.ndpost)
        return probs

    def feature_inclusion_frequency(self, normalize: str = 'split'):
        """
        Compute feature inclusion frequency (VIP-style) across posterior draws.

        Parameters
        ----------
        normalize : str, default 'split'
            - 'split': aggregate counts across draws then divide by total split count.
            - 'per_draw': normalize each draw's histogram to sum 1, then average over draws.

        Returns
        -------
        np.ndarray
            Array of shape (p,) with frequencies summing to 1 when normalize='split'.
        """
        if not self.is_fitted or self.data is None:
            raise ValueError("Model must be fitted before computing inclusion frequency.")

        if normalize not in ('split', 'per_draw'):
            raise ValueError("normalize must be one of {'split', 'per_draw'}.")

        p = self.data.X.shape[1]
        freq = np.zeros(p, dtype=float)

        if normalize == 'split':
            total_splits = 0.0
            for k in self.range_post:
                # vars_histogram is now a numpy array of shape (p,)
                hist = self.trace[k].vars_histogram
                if hist.size == 0:
                    continue
                freq += hist.astype(float)
                total_splits += float(hist.sum())
            if total_splits > 0.0:
                freq /= total_splits
            else:
                # no splits observed; return zeros
                freq[:] = 0.0
            return freq

        # per_draw: average normalized-per-draw histograms
        draws_count = 0
        for k in self.range_post:
            hist = self.trace[k].vars_histogram
            if hist.size == 0:
                continue
            draw_total = float(hist.sum())
            if draw_total <= 0.0:
                continue
            freq += hist.astype(float) / draw_total
            draws_count += 1

        if draws_count > 0:
            freq /= float(draws_count)
        else:
            freq[:] = 0.0
        return freq


class ParallelTemperingBART(BART):
    """
    Regression BART with parallel tempering (PT).

    Temperature affects likelihood only; tree/global priors are untouched.

    `n_jobs` sets how many worker processes run the chains: 1 keeps everything
    in this process, -1 uses one process per physical core (-2 all but one),
    None gives every temperature its own process, and any other value is
    capped at the number of temperatures. The chains are split into that many
    contiguous groups. It is a performance knob only — results are bitwise
    identical for every value.

    -1 is usually the fastest choice. One process per temperature oversubscribes
    as soon as there are more temperatures than cores, and one process per
    *logical* core is slower than one per physical core on an SMT machine.
    """
    preprocessor_class = DefaultPreprocessor

    def __init__(
        self,
        ndpost=1000,
        nskip=100,
        n_trees=200,
        tree_alpha: float = 0.95,
        tree_beta: float = 2.0,
        f_k=2.0,
        eps_q: float = 0.9,
        eps_nu: float = 3,
        specification="linear",
        proposal_probs=default_proposal_probs,
        tol=100,
        max_bins=100,
        random_state=42,
        temperatures: Optional[Sequence[float]] = None,
        n_temperatures: int = 4,
        max_temperature: float = 5.0,
        swap_interval: int = 50,
        swap_sweeps: Optional[int] = None,
        post_swap_repair_steps: int = 0,
        dirichlet_prior=False,
        quick_decay: bool = False,
        s_alpha: float = 1.0,
        fixed_eps_sigma2: Optional[float] = None,
        init_trees=None,
        init_sigma2=None,
        store_chain_traces: bool = False,
        store_swap_diagnostics: bool = False,
        print_swap_diagnostics: bool = False,
        n_jobs: Optional[int] = 1,
        local_move_backend: str = "multiprocessing-pipe",
        sampler_kind: str = "default",
        multi_tries: Optional[int] = None,
    ):
        if max_bins is None:
            max_bins = 100
        if swap_interval <= 0:
            raise ValueError("swap_interval must be a positive integer.")
        if swap_sweeps is not None and int(swap_sweeps) <= 0:
            raise ValueError("swap_sweeps must be a positive integer or None.")
        if post_swap_repair_steps < 0:
            raise ValueError("post_swap_repair_steps must be a non-negative integer.")

        preprocessor = self.preprocessor_class(max_bins=max_bins)

        seed_seq = random_state if isinstance(random_state, np.random.SeedSequence) else np.random.SeedSequence(int(random_state))
        temps = self._build_temperature_ladder(
            temperatures=temperatures,
            n_temperatures=n_temperatures,
            max_temperature=max_temperature,
        )
        child_seeds = seed_seq.spawn(len(temps))

        chain_samplers = []
        for chain_idx, chain_seed in enumerate(child_seeds):
            rng = np.random.default_rng(chain_seed)
            prior = ComprehensivePrior(
                n_trees,
                tree_alpha,
                tree_beta,
                f_k,
                eps_q,
                eps_nu,
                specification,
                rng,
                dirichlet_prior,
                quick_decay=quick_decay,
                s_alpha=s_alpha,
                fixed_eps_sigma2=fixed_eps_sigma2,
                init_sigma2=init_sigma2,
            )
            # In PT, each chain has a fixed temperature.
            chain_temp = temps[chain_idx]
            temp_schedule = TemperatureSchedule(_ConstantTemperature(chain_temp))
            # Allow using MultiSampler inside PT when requested.
            if str(sampler_kind).lower() == "multi" or str(sampler_kind).lower() == "mtmh":
                sampler = MultiSampler(
                    prior=prior,
                    proposal_probs=mtmh_proposal_probs,
                    generator=rng,
                    tol=tol,
                    temp_schedule=temp_schedule,
                    multi_tries=multi_tries if multi_tries is not None else 10,
                    init_trees=init_trees,
                )
            else:
                sampler = DefaultSampler(
                    prior=prior,
                    proposal_probs=proposal_probs,
                    generator=rng,
                    tol=tol,
                    temp_schedule=temp_schedule,
                    init_trees=init_trees,
                )
            chain_samplers.append(sampler)

        # Keep BART base compatibility via the cold-chain sampler.
        super().__init__(preprocessor, chain_samplers[0], ndpost, nskip)

        self.temperatures = temps
        self.n_temperatures = len(temps)
        self.swap_interval = int(swap_interval)
        self.swap_sweeps = None if swap_sweeps is None else int(swap_sweeps)
        self.post_swap_repair_steps = int(post_swap_repair_steps)
        self.chain_samplers = chain_samplers
        self.store_chain_traces = bool(store_chain_traces)
        self.store_swap_diagnostics = bool(store_swap_diagnostics)
        self.print_swap_diagnostics = bool(print_swap_diagnostics)
        if n_jobs is not None and int(n_jobs) == 0:
            raise ValueError("n_jobs cannot be 0.")
        self.n_jobs = None if n_jobs is None else int(n_jobs)
        backend = str(local_move_backend).strip().lower()
        valid_backends = {"multiprocessing-pipe"}
        if backend not in valid_backends:
            raise ValueError(f"local_move_backend must be one of {sorted(valid_backends)}.")
        self.local_move_backend = backend

        self.swap_attempt_counts = np.zeros(max(0, self.n_temperatures - 1), dtype=np.int64)
        self.swap_accept_counts = np.zeros(max(0, self.n_temperatures - 1), dtype=np.int64)
        self.chain_traces = [[] for _ in range(self.n_temperatures)] if self.store_chain_traces else None
        self.swap_diagnostics = []
        self.sampler_kind = str(sampler_kind).lower()
        self.multi_tries = None if multi_tries is None else int(multi_tries)

    def _effective_parallel_workers(self) -> int:
        """Number of worker processes; each hosts one or more temperature chains."""
        return resolve_pt_workers(self.n_jobs, self.n_temperatures)

    def _swap_sweeps_per_interval(self) -> int:
        if self.n_temperatures <= 1:
            return 0
        if self.swap_sweeps is None:
            return self.n_temperatures - 1
        return int(self.swap_sweeps)

    def _advance_all_chains(self, current_states):
        for chain_id, sampler in enumerate(self.chain_samplers):
            current_states[chain_id] = sampler.one_iter(
                current_states[chain_id],
                temp=self.temperatures[chain_id],
                return_trace=False,
            )
        return current_states

    def _advance_all_chains_block(
        self,
        current_states,
        n_steps: int,
        keep_states_from: Optional[int] = None,
    ):
        per_chain_results = []
        for chain_id, sampler in enumerate(self.chain_samplers):
            chain_keep_states_from = keep_states_from if (self.store_chain_traces or chain_id == 0) else None
            result = _run_sampler_block(
                sampler,
                current_states[chain_id],
                self.temperatures[chain_id],
                n_steps,
                keep_states_from=chain_keep_states_from,
            )
            self.chain_samplers[chain_id] = result[0]
            current_states[chain_id] = result[1]
            per_chain_results.append(result)
        return per_chain_results

    def _advance_all_chains_workers(self, workers, n_steps: int, keep_states_from: Optional[int] = None):
        """Advance every chain; returns each chain's kept (compressed) states.

        All requests are issued before any reply is read, so chains hosted by
        different processes run concurrently and chains sharing a process run
        back to back.
        """
        for chain_id, worker in enumerate(workers):
            chain_keep_states_from = keep_states_from if (self.store_chain_traces or chain_id == 0) else None
            worker.request("advance", int(n_steps), chain_keep_states_from)
        return [worker.recv() for worker in workers]

    def _record_swap_diagnostic(
        self, i, j, temp_a, temp_b, ll_aa, ll_bb, ll_ab, ll_ba, delta, accepted,
        iteration=None, sweep=None, swap_step=None,
    ) -> None:
        if not (self.store_swap_diagnostics or self.print_swap_diagnostics):
            return
        diag = {
            "pair_index": int(i),
            "iteration": None if iteration is None else int(iteration),
            "swap_step": None if swap_step is None else int(swap_step),
            "sweep": None if sweep is None else int(sweep),
            "temp_a": temp_a,
            "temp_b": temp_b,
            "ll_aa": ll_aa,
            "ll_bb": ll_bb,
            "ll_ab": ll_ab,
            "ll_ba": ll_ba,
            "delta": delta,
            "accepted": accepted,
        }
        if self.store_swap_diagnostics:
            self.swap_diagnostics.append(diag)
        if self.print_swap_diagnostics:
            print(
                "[PT swap collapsed] "
                f"iter={diag['iteration']} "
                f"step={diag['swap_step']} "
                f"sweep={diag['sweep']} "
                f"pair={i}-{j} "
                f"temp_a={temp_a:.6g} temp_b={temp_b:.6g} "
                f"ll_aa={ll_aa:.6g} ll_bb={ll_bb:.6g} "
                f"ll_ab={ll_ab:.6g} ll_ba={ll_ba:.6g} "
                f"delta={delta:.6g} "
                f"accepted={accepted}"
            )

    def _swap_step_with_workers(self, workers, iter_idx: int) -> bool:
        """All swap sweeps of one swap step, bitwise identical to the serial path.

        `holder[p]` is the chain handle currently holding the state of
        temperature position p. Instead of moving states back and forth, the
        chain RNGs (exact bit-generator states) are kept here: uniforms are
        drawn here and each refresh runs in the holding worker with the
        position's RNG state. Per-chain draw order is the same as serial
        (pairs in a sweep are disjoint). States are moved to their positions
        once at the end.

        Several handles may share one worker process, which answers in request
        order, so every request loop below is followed by a receive loop over
        the same handles in the same order. `holder` is only ever reassigned
        for the pair being resolved, and pairs within a sweep are disjoint, so
        that order is stable while a batch is in flight.
        """
        n_chains = self.n_temperatures
        for worker in workers:
            worker.request("get_rng_state")
        rngs = []
        for worker in workers:
            rng_state = worker.recv()
            bit_generator = getattr(np.random, rng_state["bit_generator"])()
            bit_generator.state = rng_state
            rngs.append(np.random.Generator(bit_generator))

        holder = list(range(n_chains))
        swap_step = (iter_idx + 1) // self.swap_interval
        base_offset = ((iter_idx + 1) // self.swap_interval) % 2
        in_posterior = iter_idx >= self.nskip
        accepted_any = False

        for sweep in range(self._swap_sweeps_per_interval()):
            offset = (base_offset + sweep) % 2
            pairs = [(left, left + 1) for left in range(offset, n_chains - 1, 2)]
            if not pairs:
                continue

            for i, j in pairs:
                temp_a = float(self.temperatures[i])
                temp_b = float(self.temperatures[j])
                workers[holder[i]].request("collapsed_logliks", (temp_a, temp_b))
                workers[holder[j]].request("collapsed_logliks", (temp_b, temp_a))

            refresh_positions = []
            for i, j in pairs:
                temp_a = float(self.temperatures[i])
                temp_b = float(self.temperatures[j])
                ll_aa, ll_ba = workers[holder[i]].recv()
                ll_bb, ll_ab = workers[holder[j]].recv()
                delta = float(ll_ab + ll_ba - ll_aa - ll_bb)
                if in_posterior:
                    self.swap_attempt_counts[i] += 1
                u = rngs[i].uniform(0.0, 1.0)
                accepted = bool(np.log(u) < delta)
                self._record_swap_diagnostic(
                    i, j, temp_a, temp_b, ll_aa, ll_bb, ll_ab, ll_ba, delta, accepted,
                    iteration=iter_idx + 1, sweep=sweep + 1, swap_step=swap_step,
                )
                if accepted:
                    accepted_any = True
                    holder[i], holder[j] = holder[j], holder[i]
                    if in_posterior:
                        self.swap_accept_counts[i] += 1
                    refresh_positions.extend((i, j))

            for position in refresh_positions:
                workers[holder[position]].request(
                    "refresh_with_rng",
                    float(self.temperatures[position]),
                    rngs[position].bit_generator.state,
                )
            for position in refresh_positions:
                rngs[position].bit_generator.state = workers[holder[position]].recv()

        # Move displaced states to the worker of their temperature position.
        moves = [(position, holder[position]) for position in range(n_chains) if holder[position] != position]
        for _position, source in moves:
            workers[source].request_export_state()
        payloads = [workers[source].recv_state_bytes() for _position, source in moves]
        for (position, _source), payload in zip(moves, payloads):
            workers[position].request_import_state_bytes(payload)
        for position, _source in moves:
            workers[position].recv()

        for position, worker in enumerate(workers):
            worker.request("set_rng_state", rngs[position].bit_generator.state)
        for worker in workers:
            worker.recv()
        return accepted_any

    @staticmethod
    def _build_temperature_ladder(
        temperatures: Optional[Sequence[float]],
        n_temperatures: int,
        max_temperature: float,
    ) -> list[float]:
        if temperatures is not None:
            if len(temperatures) == 0:
                raise ValueError("temperatures cannot be empty.")
            temps = sorted(float(t) for t in temperatures)
            if any(t <= 0 for t in temps):
                raise ValueError("All temperatures must be strictly positive.")
            if temps[0] != 1.0:
                temps = [1.0] + [t for t in temps if t != 1.0]
            return temps

        if n_temperatures < 1:
            raise ValueError("n_temperatures must be >= 1.")
        if max_temperature < 1.0:
            raise ValueError("max_temperature must be >= 1.0.")
        if n_temperatures == 1:
            return [1.0]
        return list(np.geomspace(1.0, float(max_temperature), int(n_temperatures)).astype(float))

    def _compress_state_for_trace(self, state):
        return _compress_state_for_trace(state)

    def _state_loglik_rss_eps(self, state):
        if state.cache is not None:
            fitted = state.cache
        else:
            fitted = state.evaluate()
        residuals = self.data.y - fitted
        rss = float(np.sum(residuals ** 2))
        eps_sigma2 = float(state.global_params["eps_sigma2"][0])
        if eps_sigma2 <= 0.0:
            raise ValueError("eps_sigma2 must be strictly positive.")
        n = residuals.shape[0]
        loglik = float(-0.5 * (n * np.log(eps_sigma2) + rss / eps_sigma2))
        return loglik, rss, eps_sigma2

    def _state_collapsed_loglik(self, state, temp: float, leaf_basis_svd=None) -> float:
        collapsed = _loglik_as_float(
            self.sampler.likelihood.trees_log_marginal_lkhd(
                state,
                self.data.y,
                np.arange(state.n_trees),
                temp=temp,
                leaf_basis_svd=leaf_basis_svd,
            )
        )
        eps_sigma2 = float(state.global_params["eps_sigma2"][0])
        n = int(self.data.y.shape[0])
        return collapsed - 0.5 * n * np.log(2.0 * np.pi * eps_sigma2) / float(temp)

    def _swap_collapsed_logliks(self, states, i: int, j: int, temp_a: float, temp_b: float, swap_cache=None):
        if swap_cache is None:
            ll_aa = self._state_collapsed_loglik(states[i], temp_a)
            ll_bb = self._state_collapsed_loglik(states[j], temp_b)
            ll_ab = self._state_collapsed_loglik(states[j], temp_a)
            ll_ba = self._state_collapsed_loglik(states[i], temp_b)
            return ll_aa, ll_bb, ll_ab, ll_ba
        ll_aa = self._cached_collapsed_loglik(states, swap_cache, i, temp_a)
        ll_bb = self._cached_collapsed_loglik(states, swap_cache, j, temp_b)
        ll_ab = self._cached_collapsed_loglik(states, swap_cache, j, temp_a)
        ll_ba = self._cached_collapsed_loglik(states, swap_cache, i, temp_b)
        return ll_aa, ll_bb, ll_ab, ll_ba

    def _cached_collapsed_loglik(self, states, swap_cache, chain_id: int, temp: float) -> float:
        values = swap_cache.values[chain_id]
        if temp in values:
            return values[temp]
        entry = swap_cache.entries[chain_id]
        if entry is _MISSING:
            entry = _leaf_basis_and_svd_for_sampler_state(self.sampler, states[chain_id])
            swap_cache.entries[chain_id] = entry
        value = self._state_collapsed_loglik(states[chain_id], temp, leaf_basis_svd=entry[1])
        values[temp] = value
        return value

    def _refresh_state_tempered_params(self, state, chain_id: int, leaf_basis=None) -> None:
        sampler = self.chain_samplers[chain_id]
        temp = float(self.temperatures[chain_id])
        tree_ids = np.arange(state.n_trees, dtype=int)

        new_leaf_vals = sampler.tree_prior.resample_leaf_vals(
            state,
            data_y=self.data.y,
            tree_ids=tree_ids,
            temp=temp,
            leaf_basis=leaf_basis,
        )
        state.update_leaf_vals(tree_ids.tolist(), new_leaf_vals)

    def _attempt_adjacent_swap(
        self,
        states,
        i: int,
        j: int,
        iteration: int | None = None,
        sweep: int | None = None,
        swap_step: int | None = None,
        count_for_stats: bool = True,
        swap_cache=None,
    ) -> bool:
        sampler = self.chain_samplers[i]
        temp_a = float(self.temperatures[i])
        temp_b = float(self.temperatures[j])

        ll_aa, ll_bb, ll_ab, ll_ba = self._swap_collapsed_logliks(
            states, i, j, temp_a, temp_b, swap_cache=swap_cache
        )
        delta = float(ll_ab + ll_ba - ll_aa - ll_bb)

        if count_for_stats:
            self.swap_attempt_counts[i] += 1
        u = sampler.generator.uniform(0.0, 1.0)
        accepted = bool(np.log(u) < delta)

        self._record_swap_diagnostic(
            i, j, temp_a, temp_b, ll_aa, ll_bb, ll_ab, ll_ba, delta, accepted,
            iteration=iteration, sweep=sweep, swap_step=swap_step,
        )

        if accepted:
            states[i], states[j] = states[j], states[i]
            if swap_cache is not None:
                # After `swap`, entry k belongs to the state now at position k.
                swap_cache.swap(i, j)
                basis_i, basis_j = swap_cache.leaf_basis(i), swap_cache.leaf_basis(j)
            else:
                basis_i = basis_j = None
            self._refresh_state_tempered_params(states[i], i, leaf_basis=basis_i)
            self._refresh_state_tempered_params(states[j], j, leaf_basis=basis_j)
            if count_for_stats:
                self.swap_accept_counts[i] += 1
            return True
        return False

    def fit(self, X, y, quietly=False):
        data = self.preprocessor.fit_transform(X, y)
        return self.fit_with_data(data, quietly=quietly)

    def fit_with_data(self, data: Dataset, quietly=False):
        self.data = data
        self.trace = []
        if self.chain_traces is not None:
            self.chain_traces = [[] for _ in range(self.n_temperatures)]
        self.swap_attempt_counts[:] = 0
        self.swap_accept_counts[:] = 0
        if self.store_swap_diagnostics:
            self.swap_diagnostics = []

        current_states = []
        for sampler in self.chain_samplers:
            sampler.add_data(self.data)
            sampler.add_thresholds(self.preprocessor.thresholds)
            current_states.append(sampler.get_init_state())

        total_iters = self.ndpost + self.nskip
        progress = None if quietly else tqdm(total=total_iters, desc="Iterations")
        if self._effective_parallel_workers() > 1:
            with ExitStack() as stack:
                workers, worker_processes = _start_pt_chain_workers(
                    self.chain_samplers,
                    current_states,
                    self.temperatures,
                    self._effective_parallel_workers(),
                )
                for worker_process in worker_processes:
                    stack.callback(worker_process.close)
                del current_states  # the workers own the states from here on
                self._fit_loop_workers(workers, total_iters, progress)
                for chain_id, worker in enumerate(workers):
                    self.chain_samplers[chain_id] = worker.get_sampler()
        else:
            self._fit_loop_serial(current_states, total_iters, progress)
        if progress is not None:
            progress.close()

        self.is_fitted = True
        self.sampler = self.chain_samplers[0]
        return self

    def _block_bounds(self, it: int, total_iters: int):
        steps_to_boundary = self.swap_interval - (it % self.swap_interval)
        block_steps = min(steps_to_boundary, total_iters - it)
        keep_states_from = None
        if it + block_steps > self.nskip:
            keep_states_from = max(0, self.nskip - it)
        return block_steps, keep_states_from

    def _fit_loop_serial(self, current_states, total_iters: int, progress) -> None:
        it = 0
        while it < total_iters:
            block_steps, keep_states_from = self._block_bounds(it, total_iters)
            block_chain_states = self._advance_all_chains_block(
                current_states,
                n_steps=block_steps,
                keep_states_from=keep_states_from,
            )

            for local_step in range(block_steps):
                iter_idx = it + local_step
                if local_step < block_steps - 1 and keep_states_from is not None and local_step >= keep_states_from:
                    current_states[0] = block_chain_states[0][2][local_step - keep_states_from]

                accepted_swap_this_iter = False
                # The final step of every block must use each chain's final
                # state, even if this block does not end on a swap boundary.
                if local_step == block_steps - 1:
                    for chain_id in range(self.n_temperatures):
                        current_states[chain_id] = block_chain_states[chain_id][1]

                # Swap only at interval boundaries.
                if self.n_temperatures > 1 and ((iter_idx + 1) % self.swap_interval == 0):
                    swap_step = (iter_idx + 1) // self.swap_interval
                    swap_cache = _PTSwapLoglikCache(self.n_temperatures)
                    base_offset = ((iter_idx + 1) // self.swap_interval) % 2
                    for sweep in range(self._swap_sweeps_per_interval()):
                        offset = (base_offset + sweep) % 2
                        in_posterior = iter_idx >= self.nskip
                        for left in range(offset, self.n_temperatures - 1, 2):
                            accepted_swap_this_iter = self._attempt_adjacent_swap(
                                current_states,
                                left,
                                left + 1,
                                iteration=iter_idx + 1,
                                sweep=sweep + 1,
                                swap_step=swap_step,
                                count_for_stats=in_posterior,
                                swap_cache=swap_cache,
                            ) or accepted_swap_this_iter

                if accepted_swap_this_iter and self.post_swap_repair_steps > 0:
                    for _ in range(self.post_swap_repair_steps):
                        current_states = self._advance_all_chains(current_states)

                if iter_idx >= self.nskip:
                    self.trace.append(self._compress_state_for_trace(current_states[0]))
                    if self.chain_traces is not None:
                        for chain_id in range(self.n_temperatures):
                            if local_step < block_steps - 1 and keep_states_from is not None:
                                current_states[chain_id] = block_chain_states[chain_id][2][local_step - keep_states_from]
                            self.chain_traces[chain_id].append(
                                self._compress_state_for_trace(current_states[chain_id])
                            )

                if progress is not None:
                    progress.update(1)

            it += block_steps

    def _fit_loop_workers(self, workers, total_iters: int, progress) -> None:
        """Parallel counterpart of `_fit_loop_serial` with identical results.

        Full states stay in the workers. Only compressed trace states, swap
        likelihood values, RNG states and displaced states cross the pipes.
        """
        traced_chains = list(range(self.n_temperatures)) if self.chain_traces is not None else [0]
        it = 0
        while it < total_iters:
            block_steps, keep_states_from = self._block_bounds(it, total_iters)
            kept_states = self._advance_all_chains_workers(
                workers,
                n_steps=block_steps,
                keep_states_from=keep_states_from,
            )

            for local_step in range(block_steps):
                iter_idx = it + local_step
                accepted_swap_this_iter = False
                if self.n_temperatures > 1 and ((iter_idx + 1) % self.swap_interval == 0):
                    accepted_swap_this_iter = self._swap_step_with_workers(workers, iter_idx)

                if accepted_swap_this_iter and self.post_swap_repair_steps > 0:
                    self._advance_all_chains_workers(
                        workers,
                        n_steps=self.post_swap_repair_steps,
                        keep_states_from=None,
                    )

                if iter_idx >= self.nskip:
                    if accepted_swap_this_iter:
                        # States changed after the block: fetch them (cache-free).
                        for chain_id in traced_chains:
                            workers[chain_id].request("compressed_state")
                        step_states = {chain_id: workers[chain_id].recv() for chain_id in traced_chains}
                    else:
                        step_states = {
                            chain_id: kept_states[chain_id][local_step - keep_states_from]
                            for chain_id in traced_chains
                        }
                    # Step states are already cache-free and not shared elsewhere.
                    self.trace.append(step_states[0])
                    if self.chain_traces is not None:
                        for chain_id in traced_chains:
                            state = step_states[chain_id]
                            if chain_id == 0:
                                state = self._compress_state_for_trace(state)
                            self.chain_traces[chain_id].append(state)

                if progress is not None:
                    progress.update(1)

            it += block_steps

    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        # For PT, a full re-fit is the safest behavior to keep chain coupling coherent.
        warn("ParallelTemperingBART.update_fit currently refits from scratch with updated data.")
        X_combined = X if self.data is None else np.vstack((self.data.X, X))
        y_combined = y if self.data is None else np.hstack((self.data.y, y))
        self.ndpost = int(add_ndpost)
        self.nskip = 0
        return self.fit(X_combined, y_combined, quietly=quietly)

    def get_params(self) -> Dict[str, Any]:
        base = {
            "model_type": "ParallelTemperingBART",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "n_temperatures": self.n_temperatures,
            "temperatures": list(self.temperatures),
            "swap_interval": self.swap_interval,
            "swap_sweeps": self.swap_sweeps,
            "effective_swap_sweeps": self._swap_sweeps_per_interval(),
            "post_swap_repair_steps": self.post_swap_repair_steps,
            "n_jobs": self.n_jobs,
            "effective_parallel_workers": self._effective_parallel_workers(),
            "chains_per_worker": [
                len(group)
                for group in _split_chains_across_workers(
                    self.n_temperatures, self._effective_parallel_workers()
                )
            ],
            "local_move_backend": self.local_move_backend,
            "store_chain_traces": self.store_chain_traces,
            "store_swap_diagnostics": self.store_swap_diagnostics,
            "print_swap_diagnostics": self.print_swap_diagnostics,
        }
        base["sampler_kind"] = self.sampler_kind
        base["multi_tries"] = self.multi_tries
        if self.swap_attempt_counts.size > 0:
            rates = np.divide(
                self.swap_accept_counts,
                np.maximum(1, self.swap_attempt_counts),
            )
            base["swap_attempts"] = self.swap_attempt_counts.tolist()
            base["swap_accepts"] = self.swap_accept_counts.tolist()
            base["swap_accept_rates"] = rates.tolist()
        if self.store_swap_diagnostics:
            base["swap_diagnostics"] = self.swap_diagnostics
        return base

    def predict_proba(self, X):
        warn("predict_proba not recommended for regression BART. Use LogisticBART for classification.")
        prob_1 = np.clip(self.predict(X).reshape(-1, 1), 0.0, 1.0)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

class ProbitBART(BART):
    """
    Binary BART implementation using Albert-Chib data augmentation and probit link.
    """
    preprocessor_class = ClassificationPreprocessor

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95,
                 tree_beta: float=2.0,
                 f_k=2.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, quick_decay: bool = False):
        preprocessor = self.preprocessor_class(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ProbitPrior(n_trees, tree_alpha, tree_beta, f_k, rng, quick_decay=quick_decay)
        temp_schedule = self._check_temperature(temperature)
        sampler = ProbitSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        super().__init__(preprocessor, sampler, ndpost, nskip)
    
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        For binary BART, this returns the latent function values.
        Sort of categories: lexicographical, the same as np.unique
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for i, k in enumerate(self.range_post):
            y_eval = self.trace[k].evaluate(X)
            preds[:, i] = y_eval
        return preds
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the probit link.
        
        Returns:
            Array of shape (n_samples, 2) with probabilities for classes 0 and 1
        """
        # Get posterior samples of probabilities
        prob_1 = self.posterior_predict_proba(X)
        
        # Average over posterior samples
        mean_prob_1 = np.mean(prob_1, axis=1)
        mean_prob_0 = 1 - mean_prob_1
        
        return np.column_stack([mean_prob_0, mean_prob_1])
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary classes.
        
        Parameters:
            X: Input features
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def posterior_predict_proba(self, X):
        """
        Get full posterior distribution of predicted probabilities.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with probability samples
        """
        f_samples = self.posterior_f(X)
        return norm.cdf(f_samples)
    
    def posterior_predict(self, X):
        """
        Get full posterior distribution of predicted classes.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with class samples
        """
        prob_samples = self.posterior_predict_proba(X)
        draws = self.sampler.generator.binomial(1, prob_samples, size=prob_samples.shape).astype(int)
        y_labels = np.zeros((draws.shape[0], draws.shape[1]), dtype=int)
        for k in range(draws.shape[1]):
            y_labels[:, k] = self.preprocessor.backtransform_y(draws[:, k])
        return y_labels
    
class LogisticBART(BART):
    """
    Logistic BART implementation using logistic link function.
    """
    preprocessor_class = ClassificationPreprocessor

    def __init__(self, ndpost=1000, nskip=100, n_trees=25, tree_alpha: float=0.95,
                 tree_beta: float=2.0, 
                 c: float = 0.0, d: float = 0.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, quick_decay: bool = False):
        if max_bins is None:
            max_bins = 100
        preprocessor = self.preprocessor_class(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = LogisticPrior(n_trees, tree_alpha, tree_beta, c, d, rng, quick_decay=quick_decay)
        temp_schedule = self._check_temperature(temperature)
        sampler = LogisticSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        self.sampler : LogisticSampler
        super().__init__(preprocessor, sampler, ndpost, nskip)

    def get_params(self) -> Dict[str, Any]:
        """Get all effective parameters for this model instance."""
        return {
            "model_type": "LogisticBART",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "n_trees": self.sampler.tree_prior.n_trees,
            "tree_alpha": self.sampler.tree_prior.alpha,
            "tree_beta": self.sampler.tree_prior.beta,
            "c": self.sampler.tree_prior.c,
            "d": self.sampler.tree_prior.d,
            "quick_decay": self.sampler.tree_prior.quick_decay,
            "proposal_probs": self.sampler.proposals
        }
        
    @property
    def n_categories(self):
        return self.sampler.n_categories
    @n_categories.setter
    def n_categories(self, value):
        self.sampler.n_categories = value
        
    def fit(self, X, y, quietly=False):
        y = y.flatten()
        self.sampler.n_categories = np.unique(y).size
        return super().fit(X, y, quietly=quietly)

    def fit_with_data(self, data: Dataset, quietly=False):
        # data.y is already encoded to 0..K-1 by ClassificationPreprocessor
        self.sampler.n_categories = int(np.max(data.y)) + 1
        return super().fit_with_data(data, quietly=quietly)
        
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        For logistic BART, this returns the latent function values.
        """
        preds = np.zeros((X.shape[0], self.ndpost, self.n_categories))
        for i, k in enumerate(self.range_post):
            for category in range(self.n_categories):
                y_eval = self.trace[k][category].evaluate(X)
                preds[:, i, category] = y_eval
        return preds
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the logistic link.
        """
        prob = self.posterior_predict_proba(X)
        
        # Average over posterior samples
        mean_prob = np.mean(prob, axis=1)
        return mean_prob
    
    def predict(self, X):
        """
        Predict classes.
        
        Parameters:
            X: Input features

        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def posterior_predict_proba(self, X):
        """
        Get full posterior distribution of predicted probabilities.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples, n_categories) with probability samples
        """
        f_samples = self.posterior_f(X)
        prob = np.zeros_like(f_samples)
        for category in range(self.n_categories):
            prob[:, :, category] = np.exp(f_samples[:, :, category])
        # Normalize to get probabilities
        prob_sum = np.sum(prob, axis=2, keepdims=True)
        prob /= prob_sum
        return prob
    
    def posterior_sample(self, X, schedule: Callable[[int], float], backtransform=False):
        """
        Get a posterior sample of predicted probabilities (posterior mean) for each row in X.
        
        Parameters:
            X: Input features
            schedule: Callable that returns a temperature for sampling
            
        Returns:
            Sampled predictions
        """
        pred = np.zeros((X.shape[0], self.n_categories))
        # sample a k using the schedule
        k = self.sampler.generator.choice(
            range(len(self.trace)), 
            p=[schedule(k) for k in range(len(self.trace))]
        )
        f_sample = np.zeros((X.shape[0], self.n_categories))
        for category in range(self.n_categories):
            f_sample[:, category] = self.trace[k][category].evaluate(X)
        prob = np.exp(f_sample)
        # Normalize to get probabilities
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob /= prob_sum
        if backtransform:
            raise NotImplementedError("Backtransform not implemented for LogisticBART")
        else:
            pred = prob
        return pred
    
    def posterior_predict(self, X):
        """
        Get full posterior distribution of predicted classes.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with class samples
        """
        prob_samples = self.posterior_predict_proba(X)
        draws = self.sampler.generator.multinomial(
            n=1, pvals=prob_samples,
            size=(prob_samples.shape[0], prob_samples.shape[1])
        )
        labels = np.argmax(draws, axis=2)
        y_labels = np.zeros((labels.shape[0], labels.shape[1]), dtype=int)
        for k in range(labels.shape[1]):
            y_labels[:, k] = self.preprocessor.backtransform_y(labels[:, k])
        return y_labels

    def predict_trace(self, k: int, X, backtransform=True):
        """
        Predict class probabilities using a single trace state for LogisticBART.
        Returns an array shaped (n_samples, n_categories).
        """
        n_categories = self.n_categories
        f_sample = np.zeros((X.shape[0], n_categories))
        for category in range(n_categories):
            f_sample[:, category] = self.trace[k][category].evaluate(X)
        prob = np.exp(f_sample)
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob /= prob_sum
        if backtransform:
            # Nothing to backtransform for probabilities
            return prob
        return prob
    
class MultiBART(BART):

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=mtmh_proposal_probs, tol=1, max_bins=100,
                 random_state=42, temperature=1.0, multi_tries=10, dirichlet_prior=False, 
                 s_alpha: float = 1.0, fixed_eps_sigma2: Optional[float] = None,
                 quick_decay: bool = False, init_trees=None, init_sigma2=None):
        preprocessor = DefaultPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay, s_alpha=s_alpha, fixed_eps_sigma2=fixed_eps_sigma2, init_sigma2=init_sigma2)
        temp_schedule = self._check_temperature(temperature)
        sampler = MultiSampler(
            prior=prior, proposal_probs=proposal_probs, generator=rng, tol=tol, 
            temp_schedule=temp_schedule, multi_tries=multi_tries, init_trees=init_trees)
        super().__init__(preprocessor, sampler, ndpost, nskip)

    def predict_proba(self, X):
        """
        MultiBART doesn't support classification probabilities.
        Use naive prediction instead.
        Returns:
            Array of shape (n_samples, 1) with predicted values
        """
        warn("predict_proba not recommended for regression BART. Use LogisticBART for classification.")
        prob_1 = np.clip(self.predict(X).reshape(-1, 1), 0.0, 1.0)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

class PipelineBART(BART):
    """
    A BART model that first uses MultiSampler and then DefaultSampler.
    """
    def __init__(self, ndpost=1000, nskip=0, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, eps_nu: float=3, 
                 specification="linear", multi_proposal_probs=mtmh_proposal_probs, 
                 proposal_probs=default_proposal_probs, tol=100, 
                 max_bins=100, random_state=42, temperature=1.0, multi_tries=10, dirichlet_prior=False, 
                 quick_decay: bool = False, init_trees=None):
        """
        Initialize the PipelineBART model.

        Parameters:
            ndpost (int): Number of posterior samples to draw.
            nskip (int): Number of burn-in iterations to skip.
            n_trees (int): Number of trees in the model.
            tree_alpha, tree_beta, f_k, eps_q, eps_nu: Prior parameters.
            specification (str): Model specification.
            proposal_probs (dict): Proposal probabilities for moves.
            tol (int): Tolerance for samplers.
            max_bins (int): Maximum number of bins for preprocessing.
            random_state (int): Random seed.
            temperature (float): Temperature for the sampler.
            multi_tries (list[int]): Multi-try MCMC parameters for MultiSampler.
        """
        # Initialize preprocessor
        preprocessor = DefaultPreprocessor(max_bins=max_bins)

        # Initialize random generator
        rng = np.random.default_rng(random_state)

        # Initialize prior
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay)

        # Initialize temperature schedule
        temp_schedule = self._check_temperature(temperature)

        # Initialize MultiSampler
        self.multi_sampler = MultiSampler(
            prior=prior,
            proposal_probs=multi_proposal_probs,
            generator=rng,
            temp_schedule=temp_schedule,
            tol=1,
            multi_tries=multi_tries, 
            init_trees=init_trees
        )

        # Initialize DefaultSampler
        self.default_sampler = DefaultSampler(
            prior=prior,
            proposal_probs=proposal_probs,
            generator=rng,
            temp_schedule=temp_schedule,
            tol=tol
        )

        # Call the parent constructor with DefaultSampler
        super().__init__(preprocessor, self.default_sampler, ndpost, nskip)

    def fit(self, X, y, multi_iter=1000, quietly=False):
        """
        Fit the PipelineBART model.

        Parameters:
            X: Feature matrix.
            y: Target vector.
            multi_iter (int): Number of iterations for MultiSampler.
            quietly (bool): Whether to suppress output.
        """
        # Step 1: Preprocess the data
        self.data = self.preprocessor.fit_transform(X, y)
        self.multi_sampler.add_data(self.data)
        self.multi_sampler.add_thresholds(self.preprocessor.thresholds)
        self.multi_iter = multi_iter

        # Step 2: Run MultiSampler
        if not quietly:
            print(f"Running MultiSampler for {self.multi_iter + self.nskip} iterations...")
        self.multi_sampler.run(self.multi_iter + self.nskip, quietly=quietly, n_skip=self.nskip)

        # Step 3: Get the final state from MultiSampler
        final_state = self.multi_sampler.trace[-1]

        # Step 4: Initialize DefaultSampler with the final state
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.sampler.trace = [final_state]

        # Step 5: Run DefaultSampler
        if not quietly:
            print(f"Running DefaultSampler for {self.ndpost} iterations...")
        self.trace = self.multi_sampler.trace[:-1] + self.sampler.run(self.ndpost, quietly=quietly)
        self.is_fitted = True

    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.multi_iter + self.ndpost))
        for k in range(self.multi_iter + self.ndpost):
            y_eval = self.trace[k].evaluate(X)
            if backtransform:
                preds[:, k] = self.preprocessor.backtransform_y(y_eval)
            else:
                preds[:, k] = y_eval
        return preds
