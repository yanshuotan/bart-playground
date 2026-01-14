import numpy as np
from typing import Optional, Dict, Any, Callable
from .bart_ts_agents import BARTTSAgent, _prepare_bart_kwargs
from .external_wrappers import BartzWrapper, StochTreeWrapper
from bart_playground.mcbart import MultiChainBART
from bart_playground.bart import DefaultBART

class HybridBARTTSAgent(BARTTSAgent):
    """
    A hybrid agent that switches from StochTree to MultiChainBART.
    Enforces encoding='separate'.
    """
    def __init__(self, n_arms: int, n_features: int,
                 switch_t: int = 100,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'separate',
                 n_chains: int = 1,
                 refresh_schedule: str = 'log',
                 use_gfr: bool = True,
                 bart_kwargs: Optional[Dict[str, Any]] = None,
                 feel_good_lambda: float = 0.0) -> None:
        
        if encoding != 'separate':
            raise NotImplementedError("HybridBARTTSAgent currently only supports 'separate' encoding.")

        # Define default parameters aligned with DefaultBARTTSAgent
        default_bart_kwargs: Dict[str, Any] = {
            "ndpost": 500,
            "nskip": 500,
            "n_trees": 100,
            "specification": "naive",
            "eps_nu": 1.0
        }
        
        # Merge defaults with user-provided bart_kwargs
        base_ndpost, user_max_bins, merged_bart_kwargs = _prepare_bart_kwargs(default_bart_kwargs, bart_kwargs)

        self.switch_t = int(switch_t)
        self.n_chains = int(n_chains)
        self._hybrid_random_state = int(random_state)
        self._hybrid_use_gfr = bool(use_gfr)
        self._hybrid_bart_kwargs = merged_bart_kwargs
        self._mc: Optional[MultiChainBART] = None

        def early_factory(new_ndpost: int = base_ndpost, max_bins: Optional[int] = None):
            return StochTreeWrapper(
                ndpost=new_ndpost,
                use_gfr=self._hybrid_use_gfr,
                random_state=self._hybrid_random_state,
                **self._hybrid_bart_kwargs,
            )
        
        super().__init__(n_arms, n_features, early_factory,
                         initial_random_selections, random_state, encoding, refresh_schedule,
                         feel_good_lambda)

        # Carry over the max_bins if provided
        if user_max_bins is not None:
            self._fixed_max_bins_value = user_max_bins
        
        # Initialize max_ndpost from processed base_ndpost
        self.max_ndpost = int(base_ndpost)

    def _ensure_multichain_initialized(self) -> None:
        if self._mc is None:
            self._mc = MultiChainBART(
                n_ensembles=self.n_chains,
                bart_class=DefaultBART,
                random_state=self._hybrid_random_state,
                ndpost=int(self.max_ndpost),
                max_bins=int(self._current_max_bins()),
                n_models=self.n_arms,
                **self._hybrid_bart_kwargs,
            )
        if self.models is not self._mc:
            self.models = self._mc

    def _model_factory_for_refresh(self) -> Callable:
        """Switch decision lives here; factories are pure (no agent mutation)."""
        if self.t < self.switch_t:
            return self.model_factory
        self._ensure_multichain_initialized()

        def _mc_factory(new_ndpost: int = 500, max_bins: Optional[int] = None):
            return self._mc

        return _mc_factory


class BartzTSAgent(BARTTSAgent):
    """
    BART agent using the bartz library backend.
    """
    def __init__(self, n_arms: int, n_features: int,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'multi',
                 refresh_schedule: str = 'log',
                 bart_kwargs: Optional[Dict[str, Any]] = None,
                 feel_good_lambda: float = 0.0) -> None:
        
        default_bart_kwargs: Dict[str, Any] = {
            "ndpost": 500,
            "nskip": 500,
            "ntree": 100,
        }
        
        base_ndpost, user_max_bins, merged_bart_kwargs = _prepare_bart_kwargs(default_bart_kwargs, bart_kwargs)
        
        def model_factory(new_ndpost=base_ndpost, max_bins=None):
            # bartz wrapper handles ndpost and nskip
            return BartzWrapper(
                ndpost=new_ndpost,
                random_state=random_state,
                **merged_bart_kwargs
            )
            
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding, refresh_schedule,
                         feel_good_lambda)
        
        if user_max_bins is not None:
            self._fixed_max_bins_value = user_max_bins


class StochTreeTSAgent(BARTTSAgent):
    """
    BART agent using the stochtree library backend.
    """
    def __init__(self, n_arms: int, n_features: int,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'multi',
                 refresh_schedule: str = 'log',
                 use_gfr: bool = False,
                 bart_kwargs: Optional[Dict[str, Any]] = None,
                 feel_good_lambda: float = 0.0) -> None:
        
        default_bart_kwargs: Dict[str, Any] = {
            "ndpost": 500,
            "nskip": 500,
        }
        
        # Note: stochtree uses its own parameter names inside bart_kwargs if passed
        base_ndpost, user_max_bins, merged_bart_kwargs = _prepare_bart_kwargs(default_bart_kwargs, bart_kwargs)
        
        def model_factory(new_ndpost=base_ndpost, max_bins=None):
            return StochTreeWrapper(
                ndpost=new_ndpost,
                use_gfr=use_gfr,
                random_state=random_state,
                **merged_bart_kwargs
            )
            
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding, refresh_schedule,
                         feel_good_lambda)
        
        if user_max_bins is not None:
            self._fixed_max_bins_value = user_max_bins
