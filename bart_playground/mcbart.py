import copy
import numpy as np
import ray
from typing import Callable, Any, List, Dict, Optional
from .bart import DefaultBART 
from .serializer import bart_to_json
from .util import Dataset

@ray.remote
class BARTActor:
    """A Ray Actor that can hold multiple stateful BART instances (one per arm)."""
    def __init__(self, bart_class, random_state, n_models: int = 1, **kwargs):
        self._bart_class = bart_class
        self._kwargs = dict(kwargs)
        self._n_models = int(n_models)
        seed_seq = random_state if isinstance(random_state, np.random.SeedSequence) else np.random.SeedSequence(int(random_state))
        self._seeds = seed_seq.spawn(self._n_models)
        self.models = [
            bart_class(random_state=s, **self._kwargs)
            for s in self._seeds
        ]
        self._active = 0

    def set_active_model(self, model_id: int):
        self._active = int(model_id)
        return True

    @property
    def model(self):
        return self.models[self._active]

    def _reinit_active_model(self):
        """Re-build the active BART instance from scratch (fresh sampler state)."""
        self.models[self._active] = self._bart_class(
            random_state=self._seeds[self._active],
            **self._kwargs,
        )

    def fit(self, dataset, preprocessor, quietly=False):
        self._reinit_active_model()
        self.model.preprocessor = preprocessor
        self.model.fit_with_data(dataset, quietly=quietly)
        return True

    def update_fit(self, dataset, preprocessor, add_ndpost=20, quietly=False):
        self.model.preprocessor = preprocessor
        self.model.update_fit_with_data(dataset, add_ndpost=add_ndpost, quietly=quietly)
        return True

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

    def predict_trace_batch(self, k: int, X, backtransform: bool = True):
        """Evaluate prediction at trace index k for ALL models in this actor."""
        return [m.predict_trace(k, X, backtransform) for m in self.models]

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
        base = self.model.get_params()
        base["n_models"] = self._n_models
        return base

    def set_ndpost(self, ndpost: int):
        nd = int(ndpost)
        self._kwargs["ndpost"] = nd
        for m in self.models:
            m.ndpost = nd
        return True

class MultiChainBART:
    """
    Multi-chain BART model that runs multiple BART chains in parallel using Ray.
    This allows for efficient, low-overhead updates.
    """
    def __init__(self, n_ensembles, bart_class=DefaultBART, random_state=42, ndpost=1000, n_models: int = 1, **kwargs):
        self.n_ensembles = int(n_ensembles)
        self.bart_class = bart_class
        self.ndpost = int(ndpost)
        self.n_models = int(n_models)
        self._driver_preprocessor = bart_class.preprocessor_class(max_bins=kwargs.get('max_bins', 100))
        self._dataset: Optional[Dataset] = None
        self._active = 0
        
        # Initialize Ray. ignore_reinit_error is useful in interactive environments.
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Initialize parent random state for chain picking
        parent_seed = random_state if isinstance(random_state, np.random.SeedSequence) else np.random.SeedSequence(int(random_state))
        self.rng = np.random.default_rng(parent_seed)

        # Generate children random states for reproducibility
        child_states = parent_seed.spawn(self.n_ensembles)

        # Create stateful actors. Each actor will build and hold one BART instance.
        self.bart_actors: List[Any] = [
            BARTActor.remote(
                bart_class,
                random_state=seed_sequence,
                n_models=self.n_models,
                ndpost=self.ndpost,
                **kwargs,
            )  # type: ignore
            for seed_sequence in child_states
        ]
        print(f"Created {self.n_ensembles} BARTActor(s) using {bart_class.__name__} with {self.n_models} model(s) each.")

    def set_active_model(self, model_id: int):
        idx = int(model_id)
        self._active = idx
        ray.get([actor.set_active_model.remote(idx) for actor in self.bart_actors])
        return self

    def set_ndpost(self, ndpost: int):
        self.ndpost = int(ndpost)
        ray.get([actor.set_ndpost.remote(self.ndpost) for actor in self.bart_actors])
        return self

    def __len__(self):
        return max(1, self.n_models)

    def __getitem__(self, i: int):
        if not (0 <= i < len(self)):
            raise IndexError("model index out of range")
        return self.set_active_model(i)
        
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
        dataset = self._driver_preprocessor.fit_transform(X, y)
        data_ref, prep_ref = self._share_dataset(dataset)
        fit_futures = [actor.fit.remote(data_ref, prep_ref, quietly) for actor in self.bart_actors]
        # This is a blocking call that waits for all actors to finish their fit.
        ray.get(fit_futures)
        return self
    
    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        """
        Update all BART instances in parallel with very low overhead.
        This sends a command to the existing actors without process creation.
        """
        if self._dataset is None:
            return self.fit(X, y, quietly=quietly)
        updated_dataset = self._driver_preprocessor.update_transform(X, y, self._dataset)
        data_ref, prep_ref = self._share_dataset(updated_dataset)
        update_futures = [
            actor.update_fit.remote(data_ref, prep_ref, add_ndpost, quietly)
            for actor in self.bart_actors
        ]
        ray.get(update_futures)
        return self

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

    def predict_trace_batch(self, k: int, X, backtransform: bool = True):
        """
        Predict using a specific trace index for ALL models (arms).
        Returns: np.ndarray of shape (n_models, ...) averaged across chains.
        """
        futures = [a.predict_trace_batch.remote(k, X, backtransform) for a in self.bart_actors]
        # shape: (n_chains, n_models, ...) -> mean over chains -> (n_models, ...)
        return np.mean(ray.get(futures), axis=0)

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

    def _share_dataset(self, dataset: Dataset):
        self._dataset = dataset
        data_ref = ray.put(dataset)
        preproc_ref = ray.put(self._driver_preprocessor)
        return data_ref, preproc_ref
        