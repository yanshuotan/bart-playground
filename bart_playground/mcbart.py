import numpy as np
import ray
from typing import Callable
from .bart import DefaultBART 

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
        
    def posterior_f(self, X):
        return self.model.posterior_f(X)
    
    def posterior_sample(self, X, schedule):
        # Use default backtransform behavior
        return self.model.posterior_sample(X, schedule)

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

class MultiChainBART:
    """
    Multi-chain BART model that runs multiple BART chains in parallel using Ray.
    This allows for efficient, low-overhead updates.
    """
    def __init__(self, n_ensembles, bart_class=DefaultBART, random_state=42, **kwargs):
        self.n_ensembles = n_ensembles
        self.bart_class = bart_class

        # Initialize Ray. ignore_reinit_error is useful in interactive environments.
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Generate children random states for reproducibility
        self.rng = np.random.default_rng(random_state)
        random_states = [self.rng.integers(0, 2**32 - 1) for _ in range(n_ensembles)]

        # Create stateful actors. Each actor will build and hold one BART instance.
        self.bart_actors = [BARTActor.remote(bart_class, random_state=rs, **kwargs)
                            for rs in random_states]
        print(f"Created {n_ensembles} BARTActor(s) using BART class: {bart_class.__name__}")
        
    @property
    def _trace_length(self):
        """Get the trace length from the first actor's model."""
        if self.bart_actors:
            return ray.get(self.bart_actors[0].get_attributes.remote()).get('_trace_length', 0)
        return 0

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

    def posterior_f(self, X):
        """Get posterior distribution of f(x) from all instances."""
        preds_futures = [actor.posterior_f.remote(X) for actor in self.bart_actors]
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

    def clean_up(self):
        for actor in self.bart_actors:
            ray.kill(actor)
        print("Ray Actors have been cleaned up.")
        