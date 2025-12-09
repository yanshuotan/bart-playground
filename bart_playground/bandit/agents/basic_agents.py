import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple, Sequence
from bart_playground.bandit.agents.agent import BanditAgent
from enum import Enum
class SillyAgent(BanditAgent):
    """
    A simple agent that selects arms randomly.
    """
    def __init__(self, n_arms: int, n_features: int, random_state: Optional[int] = None, probabilities: Optional[Sequence[float]] = None) -> None:
        """
        Initialize the agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        # Optional per-arm probabilities
        self._probabilities: Optional[np.ndarray] = None
        if probabilities is not None:
            p = np.asarray(list(probabilities), dtype=float)
            if p.shape[0] != n_arms:
                raise ValueError("SillyAgent: probabilities length must equal n_arms")
            s = float(np.sum(p))
            self._probabilities = (p / s).astype(float)
        super().__init__(n_arms, n_features)
    
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm randomly.
        
        Parameters:
            x (array-like): Feature vector (not used by this agent).
            
        Returns:
            int: The randomly selected arm index.
        """
        if self._probabilities is not None:
            return int(self.rng.choice(self.n_arms, p=self._probabilities))
        return int(self.rng.integers(0, self.n_arms))
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "SillyAgent":
        """
        Update agent state (no-op for this agent).
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Unchanged instance.
        """
        return self

class AgentType(Enum):
    TS = "ts"
    UCB = "ucb"

class LinearAgentStable(BanditAgent):
    """
    Linear Thompson Sampling / UCB agent.
    This agent supports both TS and UCB.
    """
    def __init__(
        self, agent_type: AgentType | str,
        n_arms: int, n_features: int, v: float = 1.0,
        alpha: float = 1.0, random_state: Optional[int] = None
        ) -> None:
        """
        Initialize the LinearAgentStable agent.
        
        Parameters:
            agent_type: AgentType
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            v (float): Exploration parameter for TS.
            alpha (float): Alpha parameter for UCB.
        """
        self.rng = np.random.default_rng(random_state)
        super().__init__(n_arms, n_features + 1)  # +1 for the intercept term
        
        # Set exploration parameter v
        self.v = v
        self.agent_type = AgentType(agent_type)
        self.alpha = alpha
        
        # Initialize matrices for each arm
        self.B = [np.eye(self.n_features) for _ in range(n_arms)]
        self.m2_r = [np.zeros((self.n_features, 1)) for _ in range(n_arms)]
        self.L = [np.eye(self.n_features) for _ in range(n_arms)]  # cache Cholesky

    def _normalize_x(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return np.vstack([x, [[1.0]]])

    def _get_post_mean(self, arm_idx: int) -> np.ndarray:
        """
        Get the posterior mean for each arm.
        """
        return np.linalg.solve(self.B[arm_idx], self.m2_r[arm_idx])

    def _sample_w(self, arm_idx: int) -> np.ndarray:
        """
        Sample parameters from the posterior for each arm.
        """
        return self._get_post_mean(arm_idx) + self.v * np.linalg.solve(
            self.L[arm_idx].T,
            self.rng.standard_normal((self.n_features, 1))
            )

    def _get_action_estimates(self, x: np.ndarray) -> List[float]:
        """
        Get action estimates for all arms based on input features x.
        """
        x = self._normalize_x(x)
        
        if (self.agent_type == AgentType.TS):
            # Sample parameters from the posterior for each arm
            weights = [self._sample_w(a) for a in range(self.n_arms)]
            # Compute the expected reward for each arm
            u = [(w.T @ x).item() for w in weights]
        else:
            u: List[float] = []
            for a in range(self.n_arms):
                post_mean = self._get_post_mean(a)
                # variance term: || L^{-1} x ||^2
                y = np.linalg.solve(self.L[a], x)
                var = (y.T @ y).item()
                u.append((post_mean.T @ x).item() + self.alpha * np.sqrt(max(var, 0.0)))

        return u

    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Thompson Sampling.
        
        Parameters:
            x (array-like): Feature vector.
            
        Returns:
            int: The index of the selected arm.
        """
        u = self._get_action_estimates(x)
        
        # Return the arm with the highest expected reward (with sampled parameters)
        return int(np.argmax(u))
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], y: float) -> "LinearAgentStable":
        """
        Update the agent's state with new observation data.
        
        Parameters:
            arm (int): The index of the arm chosen (0-indexed).
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        arm_idx = arm
        x = self._normalize_x(x)
        
        # Update the precision matrix
        self.B[arm_idx] = self.B[arm_idx] + x @ x.T
        
        # Update the weighted sum of rewards
        self.m2_r[arm_idx] = self.m2_r[arm_idx] + x * y

        # Update the Cholesky decomposition
        self.L[arm_idx] = np.linalg.cholesky(self.B[arm_idx])

        return self

    def posterior_draws_on_probes(self, X_probes: np.ndarray, n_post: int = 200) -> Tuple[np.ndarray, int]:
        """
        Generate posterior draws for LinearTS across all probes and arms.

        Args:
            X_probes: shape (n_probes, n_features) without intercept term.
            n_post: number of posterior samples to draw for diagnostics.

        Returns:
            draws: np.ndarray shaped (n_post, n_probes, n_arms)
            n_post: int, number of draws actually used

        Notes:
            - Uses the agent's posterior w ~ N(mu, v^2 * (B^{-1})) via cached Cholesky L.
            - For UCB mode, this diagnostic is not meaningful; method still returns TS-style draws using v.
        """
        X_probes = np.asarray(X_probes, dtype=float)
        if X_probes.ndim == 1:
            X_probes = X_probes.reshape(1, -1)
        n_probes = X_probes.shape[0]
        n_arms = self.n_arms

        draws = np.zeros((n_post, n_probes, n_arms), dtype=float)

        # Normalize probes by adding intercept once
        def _norm(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.hstack([X, ones])

        Xn = _norm(X_probes)  # (n_probes, n_features+1) matches self.n_features

        for s in range(n_post):
            # Sample per-arm weights and evaluate u = x^T w
            for a in range(n_arms):
                w = self._sample_w(a)  # shape (n_features, 1)
                draws[s, :, a] = (Xn @ w).reshape(-1)

        return draws, int(n_post)
