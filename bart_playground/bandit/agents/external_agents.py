import numpy as np
from typing import Optional, Dict, Any, Callable
from .bart_ts_agents import BARTTSAgent, _prepare_bart_kwargs
from .external_wrappers import BartzWrapper, StochTreeWrapper

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
                **merged_bart_kwargs
            )
            
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding, refresh_schedule,
                         feel_good_lambda)
        
        if user_max_bins is not None:
            self._fixed_max_bins_value = user_max_bins
