import numpy as np
import ray
from typing import Callable, Any, List, Dict
from .bart import DefaultBART 
from .serializer import bart_to_json

@ray.remote
class BARTActor:
    """A Ray Actor to hold and manage a single stateful BART instance."""
    def __init__(self, bart_class, random_state, **kwargs):
        # Each actor gets its own BART instance, initialized with a unique random state.
        self.model = bart_class(random_state=random_state, **kwargs)

    def fit(self, X, y, quietly=False):
        self.model.fit(X, y, quietly=quietly)
        # We don't need to return the model itself, as its state is maintained within the actor.
        return True # Return a success signal

    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        self.model.update_fit(X, y, add_ndpost=add_ndpost, quietly=quietly)
        return True # Return a success signal

    def predict(self, X):
        return self.model.predict(X)

    def posterior_predict(self, X):
        return self.model.posterior_predict(X)
        
    def posterior_f(self, X, backtransform=True):
        return self.model.posterior_f(X, backtransform=backtransform)
    
    def posterior_sample(self, X, schedule: Callable[[int], float]):
        # Use default backtransform behavior
        return self.model.posterior_sample(X, schedule)

    def predict_trace(self, k: int, X, backtransform: bool = True):
        """Evaluate prediction at a specific trace index k using the in-actor model."""
        return self.model.predict_trace(k, X, backtransform=backtransform)

    def get_attributes(self):
        """Helper to retrieve attributes from the model inside the actor."""
        if self.model.is_fitted:
            return {
                'data': self.model.data,
                'is_fitted': self.model.is_fitted,
                'ndpost': self.model.ndpost,
                'nskip': self.model.nskip,
                '_trace_length': self.model._trace_length,
            }
        return {'is_fitted': False}

    def get_model(self):
        """Return the in-actor BART model."""
        return self.model

    def get_model_json(self):
        """Return the serialized in-actor BART model."""
        return bart_to_json(self.model, include_dataX=False, include_cache=True)

    def apply(self, func, *args, **kwargs):
        return func(self.model, *args, **kwargs)

    def feature_inclusion_frequency(self, normalize: str = "split"):
        """
        Return per-feature inclusion frequency for this actor's model.
        """
        return self.model.feature_inclusion_frequency(normalize=normalize)

    def get_params(self):
        """Proxy to get params from the underlying model."""
        return self.model.get_params()

class MultiChainBART:
    """
    Multi-chain BART model that runs multiple BART chains in parallel using Ray.
    This allows for efficient, low-overhead updates.
    """
    def __init__(self, n_ensembles, bart_class=DefaultBART, random_state=42, ndpost=1000, **kwargs):
        self.n_ensembles = n_ensembles
        self.bart_class = bart_class
        self.ndpost = ndpost
        
        # Initialize Ray. ignore_reinit_error is useful in interactive environments.
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Initialize parent random state for chain picking
        self.rng = np.random.default_rng(random_state)

        # Generate children random states for reproducibility
        def _make_child_states(master_seed: int, chain_id: int) -> np.random.SeedSequence:
            return np.random.SeedSequence(master_seed, spawn_key=(chain_id,))

        child_states = [_make_child_states(int(random_state), i) for i in range(n_ensembles)]

        # Create stateful actors. Each actor will build and hold one BART instance.
        self.bart_actors: List[Any] = [
            BARTActor.remote(bart_class, random_state=seed_sequence, ndpost=ndpost, **kwargs)  # type: ignore
            for seed_sequence in child_states
        ]
        print(f"Created {n_ensembles} BARTActor(s) using BART class: {bart_class.__name__}")
        
    @property
    def is_fitted(self):
        """Check if the MultiChainBART model is fitted by checking all actors' models."""
        if not self.bart_actors:
            return False

        # Check if all actors are fitted
        all_attrs = ray.get([actor.get_attributes.remote() for actor in self.bart_actors])
        return all(attrs.get('is_fitted', False) for attrs in all_attrs)

    @property
    def _trace_length(self):
        """Get the trace length from the first actor's model."""
        if self.bart_actors:
            attrs: Dict[str, Any] = ray.get(self.bart_actors[0].get_attributes.remote())
            return attrs.get('_trace_length', 0)
        return 0

    @property
    def range_post(self):
        """
        Get the range of posterior samples.
        """
        total_iterations = self._trace_length
        if total_iterations < self.ndpost:
            raise ValueError(f"Not enough posterior samples: {total_iterations} < {self.ndpost} (provided ndpost).")
        return range(total_iterations - self.ndpost, total_iterations)

    def fit(self, X, y, quietly=False):
        """Fit all BART instances in parallel using Ray actors."""
        # This is a non-blocking call. It returns futures immediately.
        fit_futures = [actor.fit.remote(X, y, quietly) for actor in self.bart_actors]
        # This is a blocking call that waits for all actors to finish their fit.
        ray.get(fit_futures)
    
    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        """
        Update all BART instances in parallel with very low overhead.
        This sends a command to the existing actors without process creation.
        """
        update_futures = [actor.update_fit.remote(X, y, add_ndpost, quietly) for actor in self.bart_actors]
        ray.get(update_futures)

    def predict(self, X):
        """Predict using all BART instances and average the results."""
        preds_futures = [actor.predict.remote(X) for actor in self.bart_actors]
        all_preds = np.array(ray.get(preds_futures))
        return np.mean(all_preds, axis=0)
    
    def posterior_predict(self, X):
        """Get full posterior distribution from all instances."""
        preds_futures = [actor.posterior_predict.remote(X) for actor in self.bart_actors]
        preds_list = ray.get(preds_futures)
        return np.concatenate(preds_list, axis=1)

    def posterior_f(self, X, backtransform=True):
        """Get posterior distribution of f(x) from all instances.

        Returns an array of shape (n_rows, ndpost_per_chain * n_ensembles).
        """
        preds_futures = [actor.posterior_f.remote(X, backtransform=backtransform) for actor in self.bart_actors]
        preds_list = ray.get(preds_futures)
        return np.concatenate(preds_list, axis=1)
    
    def posterior_sample(self, X, schedule: Callable[[int], float]):
        """
        Get a posterior sample from a randomly selected instance (chain).
        """
        actor_idx = self.rng.integers(0, self.n_ensembles)
        chosen_actor = self.bart_actors[actor_idx]
        sample_future = chosen_actor.posterior_sample.remote(X, schedule)
        return ray.get(sample_future)

    def predict_trace(self, k: int, X, backtransform: bool = True):
        """Predict using a specific trace index by averaging across all actors (chains)."""
        preds_futures = [actor.predict_trace.remote(k, X, backtransform) for actor in self.bart_actors]
        all_preds = np.array(ray.get(preds_futures))
        return np.mean(all_preds, axis=0)

    def collect_model_states(self):
        """Return the in-actor BART models."""
        return ray.get([actor.get_model.remote() for actor in self.bart_actors])

    def collect_model_json(self):
        """Return the serialized in-actor BART models."""
        return ray.get([actor.get_model_json.remote() for actor in self.bart_actors])

    def collect(self, func: Callable[..., Any], *args, **kwargs):
        """
        Map a callable across all actors and return their results.

        Parameters
        ----------
        func : Callable[[Any], Any]
            A top-level, Ray-serializable function with signature
            func(model, *args, **kwargs) -> Any. It will be executed inside each
            actor process, receiving the in-actor model as the first argument.
        *args, **kwargs :
            Additional arguments passed to the callable.

        Returns
        -------
        List[Any]
            One result per actor, in actor order.
        """
        futures = [actor.apply.remote(func, *args, **kwargs) for actor in self.bart_actors]
        return ray.get(futures)

    def feature_inclusion_frequency(self, normalize: str = "split") -> np.ndarray:
        """
        Aggregate per-feature inclusion frequency across chains.

        Each actor computes its own normalized frequency vector; we average them.
        """
        per_chain = ray.get(
            [actor.feature_inclusion_frequency.remote(normalize) for actor in self.bart_actors]
        )
        if not per_chain:
            return np.zeros(0, dtype=float)
        return np.mean(np.stack(per_chain, axis=0), axis=0)

    def clean_up(self):
        for actor in self.bart_actors:
            ray.kill(actor)
        print("Ray Actors have been cleaned up.")
        
    def get_params(self) -> Dict[str, Any]:
        """Get parameters from the first actor and add ensemble info."""
        if not self.bart_actors:
            return {}
        # Get params from the first actor
        base_params : Dict[str, Any] = ray.get(self.bart_actors[0].get_params.remote())
        # Add ensemble-level params
        base_params["n_ensembles"] = self.n_ensembles
        base_params["model_type"] = f"MultiChainBART({base_params.get('model_type', 'Unknown')})"
        return base_params
        