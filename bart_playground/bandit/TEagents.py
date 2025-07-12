from .baselines.Hannes.TETS import BernoulliXGBoostTSAgent, BernoulliRandomForestTSAgent
from .agent import BanditAgent
import numpy as np
from typing import Dict, Any, Union, List

class MockEnvironment:
    """Mock environment to satisfy TETS agent requirements."""
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.arm_set = list(range(n_arms))
        self.arm_weights = {arm: 0.0 for arm in self.arm_set}
        
    def pick_random_arm(self):
        """Pick a random arm."""
        arm = np.random.choice(self.arm_set)
        return MockAction(arm)
        
    def overwrite_arm_weight(self, samples: Dict[int, float]):
        """Update arm weights with samples."""
        for arm, weight in samples.items():
            self.arm_weights[arm] = weight
            
    def get_optimal_action(self):
        """Get the arm with highest weight."""
        best_arm = max(self.arm_weights, key=self.arm_weights.get)
        return MockAction(best_arm)

class MockAction:
    """Mock action to satisfy TETS agent requirements."""
    def __init__(self, arm: int):
        self.label = arm

class MockObservation:
    """Mock observation to satisfy TETS agent requirements."""
    def __init__(self, context: np.ndarray, n_arms: int):
        self.context = context
        self._arms = {arm: MockContextualArm(context, arm, n_arms) for arm in range(n_arms)}
        
    def __getitem__(self, arm):
        return self._arms[arm]

class MockContextualArm:
    """Mock contextual arm."""
    def __init__(self, context: np.ndarray, arm: int, n_arms: int):
        # Ensure context is properly shaped for TETS agents
        if context.ndim == 1:
            self.context = context.reshape(-1, 1)  # Column vector for TETS
        else:
            self.context = context
        # use one-hot encoding for arm now
        arm_arr = np.zeros((n_arms, 1))
        arm_arr[arm, 0] = 1  # One-hot encoding for the arm
        self.context = np.vstack((arm_arr, self.context))  # Append arm identifier to context
        
class TEAgent(BanditAgent):
    def __init__(self, n_arms: int, n_features: int, agent_type: str = 'xgboost',
                 random_state: int = 0, **kwargs):
        """
        Initialize the TEAgent with the specified type.
        
        :param n_arms: Number of arms in the bandit problem.
        :param n_features: Number of features in the context.
        :param agent_type: Type of agent to use ('xgboost' or 'random_forest').
        :param kwargs: Additional parameters for the underlying TETS agent.
        """
        super().__init__(n_arms, n_features)
        
        # Create environment constructor for TETS agents
        def env_constructor():
            return MockEnvironment(n_arms)
        
        # Initialize the appropriate TETS agent
        if agent_type == 'xgboost':
            self.model = BernoulliXGBoostTSAgent(
                env_constructor=env_constructor,
                # use one-hot encoding for arm now
                context_size=n_arms+n_features,
                **kwargs
            )
        elif agent_type == 'random_forest':
            self.model = BernoulliRandomForestTSAgent(
                env_constructor=env_constructor,
                context_size=n_arms+n_features,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def choose_arm(self, x: Union[np.ndarray, List[float]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm based on input features x.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm.
            
        Returns:
            int: The index of the selected arm.
        """
        # Convert to numpy array and ensure proper shape for TETS
        x = np.array(x).reshape(-1)
        
        # Create mock observation
        observation = MockObservation(x, self.n_arms)
        
        # Use TETS agent to pick action
        arm = self.model.pick_action(observation)
        
        return int(arm)
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], y: float) -> "TEAgent":
        """
        Update the agent's state after observing a reward.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        # Convert to numpy array and ensure proper shape for TETS
        x = np.array(x).reshape(-1)
        
        # Create mock observation and action
        observation = MockObservation(x, self.n_arms)
        action = MockAction(arm)
        reward = np.float64(y)
        
        # Update the TETS agent
        self.model.update_observation(observation, action, reward)

        return self
