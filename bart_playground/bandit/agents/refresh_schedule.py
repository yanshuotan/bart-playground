"""
Refresh schedule logic for BART Thompson Sampling agents.

This module contains the RefreshScheduleMixin class which implements various
refresh scheduling strategies (log, sqrt, hybrid, rev_hybrid) for determining
when to re-fit BART models.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports while providing type hints
    from bart_playground.bandit.agents.bart_ts_agents import BARTTSAgent


class RefreshScheduleMixin:
    """
    Mixin to handle model refresh scheduling logic (Log, Sqrt, Hybrid, etc.).
    
    This mixin provides methods to determine when a BART model should be refreshed
    based on different scheduling strategies. It requires the host class to have:
      - self.refresh_schedule (str)
      - self.t (int)
      - self.initial_random_selections (int)
      - self._has_sufficient_data() (method)
    """

    def _refresh_idx(self: "BARTTSAgent", t: int) -> int:
        """
        Determine the refresh index at time t based on the schedule.
        
        The refresh index increases with time according to the chosen schedule.
        A refresh is triggered when the index jumps to a new integer value.
        
        Parameters:
            t (int): Current time step
            
        Returns:
            int: Refresh index value
            
        Raises:
            ValueError: If refresh_schedule is not recognized
        """
        # Total number of refreshes at time t is normalized to ~56 at t=10k steps.
        if self.refresh_schedule == 'log':
            # Original: Aggressive early (t<100), very sparse late.
            return int(np.ceil(8.0 * np.log(t)))
            
        elif self.refresh_schedule == 'sqrt':
            # Balanced: Smoother early phase, consistent updates later.
            return int(np.ceil(0.57 * np.sqrt(t)))
            
        elif self.refresh_schedule == 'hybrid':
            # Hybrid (Switch @ 100): Sqrt -> Log
            # Mitigates early overhead (Sqrt), then switches to efficient Log decay.
            if t <= 100:
                return int(np.ceil(1.848 * np.sqrt(t)))
            else:
                return int(np.ceil(9.240 * np.log(t) - 24.072))
                
        elif self.refresh_schedule == 'rev_hybrid':
            # Rev-Hybrid (Switch @ 200): Log -> Sqrt
            # Maintains aggressive early learning (Log), but keeps "alertness" later (Sqrt).
            # Best for non-stationary or hard-to-converge environments.
            if t <= 200:
                return int(np.ceil(3.620 * np.log(t)))
            else:
                return int(np.ceil(0.512 * np.sqrt(t) + 11.940))
                
        else:
            raise ValueError(f"Unknown refresh_schedule: {self.refresh_schedule}")

    def _should_refresh(self: "BARTTSAgent") -> bool:
        """
        Check if model should be refreshed at the current time step.
        
        A refresh is triggered when:
        1. We're past the initial warm-start phase (n_arms * initial_random_selections)
        2. We have sufficient data
        3. The refresh index increases from the previous time step
        
        Returns:
            bool: True if a refresh should occur, False otherwise
        """
        warmstart_steps = getattr(self, "n_arms", 1) * self.initial_random_selections
        if self.t <= warmstart_steps or not self._has_sufficient_data():
            return False
        return self._refresh_idx(self.t) > self._refresh_idx(self.t - 1)
    
    def _steps_until_next_refresh(self: "BARTTSAgent", t_now: int) -> int:
        """
        Compute the number of selections until the next refresh trigger.
        
        This is used to determine how many posterior samples we need to generate
        to cover the period until the next refresh.
        
        Parameters:
            t_now (int): Current time step
            
        Returns:
            int: Number of steps until next refresh
            
        Note:
            We assume we are in steady-state (enough data and past initial random phase)
        """
        steps = 1
        while self._refresh_idx(t_now + steps) <= self._refresh_idx(t_now + steps - 1):
            steps += 1
        return steps

