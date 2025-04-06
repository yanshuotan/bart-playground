import numpy as np
from ..sim_util import Scenario
from .data_generator import DataGenerator


class DataGeneratorScenario(Scenario):
    """Base class for scenarios that use the same data generation mechanisms as DataGenerator"""
    
    def __init__(self, P, K, sigma2, assumptions="nonlinear", 
                 nonlinear_scale=1.0, theta_base = None, linear_baseline_coef=None, random_generator=None):
        """
        Parameters:
            P (int): Number of covariates (features).
            K (int): Number of arms (including control).
            sigma2 (float): Noise variance.
            assumptions (str): "nonlinear", "homogeneous", or "heterogeneous"
            random_seed (int): Random seed for reproducibility
            nonlinear_scale (float): Scaling factor for nonlinear baseline
            linear_baseline_coef (array): Coefficients for linear baseline
            random_generator: Random number generator instance. If None, np.random is used.
        """
        self.assumptions = assumptions
        self.linear = assumptions != "nonlinear"
        self.nonlinear_scale = nonlinear_scale
        self.linear_baseline_coef = linear_baseline_coef
        # Theta parameters for advantage function
        if (theta_base is None) or (len(theta_base) != P + 1):
            raise ValueError("theta_base must have length P + 1")
        self.theta_base = theta_base

        self.homogeneous = True
        self.time_stable = True
        self.n_max = 1
        self.t_max = 1
        self.user_idx = 0
        self.time_idx = 0

        super().__init__(P, K, sigma2, random_generator)
        
    def init_params(self):
        """Initialize parameters for the scenario"""
        self.p = self.theta_base.size
        
        if not self.linear:
            self.normal_vars = self.rng.normal(size=(20, 100))
            self.uniform_vars = np.random.uniform(size=(1000,))

        # Create theta parameters for users (in case of multiple users)
        if not self.homogeneous:
            raise NotImplementedError("Heterogeneous users not implemented")
            # self.theta_user = np.random.normal(scale=1.0, size=(n_max, self.p))
        else:
            self.theta_user = np.zeros((self.n_max, self.p))
            
        self.theta_base_plus_user = self.theta_user + self.theta_base
        
        # For nonlinear case, create random variables used in baseline function
        if not self.time_stable:
            raise NotImplementedError("Time-varying parameters not implemented")
            # # Time-specific parameters
            # theta_time_init = np.random.uniform(-0.5, 0.5, size=self.p)
            # self.theta_time = []
            # for t in range(self.t_max):  # Using K as our max time steps
            #     theta_time_t = theta_time_init / (1 + 6*t/self.t_max) + np.random.normal(scale=0.2, size=self.p)
            #     self.theta_time.append(theta_time_t)
            # self.theta_time = np.asarray(self.theta_time)
        else:
            self.theta_time = np.zeros((self.t_max, self.p))

    def generate_covariates(self):
        """Generate a vector of P covariates (features)"""
        return self.rng.uniform(-1.0, 1.0, size= self.P).astype(np.float32)
    
    @staticmethod
    def _featurize(context):
        """Create feature vector from context, matching DataGenerator's method"""
        if len(context.shape) == 1:
            context_dim = context.shape[0]
        else:
            context_dim = context.shape[1]
        context = context.reshape((-1, context_dim))
        n_obs = context.shape[0]
        features = np.ones((n_obs, 1 + context_dim))
        features[:, 1:] = context
        return features
    
    def _nonlinear_baseline_function(self, context, scale=1.0):
        """Nonlinear baseline function similar to DataGenerator's implementation"""
        context = context.reshape(1, -1)
        if context.shape[1] > 1:
            n_obs = context.shape[0]
            context1 = context[:, 0].copy()
            context2 = context[:, 1].copy()
        else:
            raise ValueError("Context must have at least 2 dimensions")
        
        # Create some hard change points via recursive partitioning
        baseline = self.__recursive_split(
            np.array([True]*n_obs), context1, context2, -1., 1.,
            -1., 1., self.uniform_vars, 0, min_size=0.4, scale=scale)
        
        # Add smooth component on top 
        for i in range(self.normal_vars.shape[0]):
            normal_iter = iter(self.normal_vars[i])
            x1 = next(normal_iter) * context1 + next(normal_iter) * context2
            x2 = next(normal_iter) * context1 + next(normal_iter) * context2
            baseline += scale * 2.0 * next(normal_iter) * np.exp(
                -(x1 - next(normal_iter))**2
                -(x2 - next(normal_iter))**2
            )
            
        return baseline
    
    def _standard_baseline_function(self, context):
        """Linear baseline function, matching DataGenerator's implementation"""
        features = self._featurize(context)
        baseline = features @ self.linear_baseline_coef
        return baseline
    
    def reward_function(self, x):
        """
        Given a feature vector x, compute:
          - outcome_mean: Expected rewards for each arm.
          - reward: Outcome_mean plus noise.
        """
        # Compute baseline (common to all arms)
        if self.assumptions == "nonlinear":
            baseline = self._nonlinear_baseline_function(x, scale=self.nonlinear_scale)
        else:
            baseline = self._standard_baseline_function(x)
        
        # Compute arm-specific advantages
        features = self._featurize(x)
        outcome_mean = np.zeros(self.K)
        
        for arm in range(self.K):
            theta = self.theta_base_plus_user[self.user_idx]
            theta += self.theta_time[self.time_idx]
            
            # Calculate advantage for this arm
            potential_advantage = features @ theta
            advantage = arm * potential_advantage  # Simple scaling by arm index
            
            # Final outcome mean for this arm
            outcome_mean[arm] = baseline + advantage
            
        # Add noise to get rewards
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        reward = outcome_mean + epsilon_t
        
        return {"outcome_mean": outcome_mean, "reward": reward}
    
    def __recursive_split(
        self, in_region: np.ndarray, x: np.ndarray, y: np.ndarray, xl: float, xr: float,
        yl: float, yr: float, u: np.ndarray, i: int, min_size: float = 0.1, scale: float = 1.):

        if min(abs(xl-xr), abs(yl-yr)) < min_size:
            return np.zeros_like(in_region)
        
        u1, u2, u3 = u[i], u[i+1], u[i+2]

        if u1 < 0.5:
            split = u2 * xl + (1.-u2) * xr
            x_below_split = x < split
            in_region_l = in_region & x_below_split
            in_region_r = in_region & (~x_below_split)
            values_increment_l = self.__recursive_split(in_region_l, x, y, xl, split, yl, yr, u, i+3, min_size=min_size, scale=scale)
            values_increment_r = self.__recursive_split(in_region_l, x, y, split, xr, yl, yr, u, i+4, min_size=min_size, scale=scale)
        else:
            split = u2 * yl + (1.-u2) * yr
            y_below_split = y < split
            in_region_l = in_region & y_below_split
            in_region_r = in_region & (~y_below_split)
            values_increment_l = self.__recursive_split(in_region_l * y_below_split, x, y, xl, xr, yl, split, u, i+3, min_size=min_size, scale=scale)
            values_increment_r = self.__recursive_split(in_region_r * (~y_below_split), x, y, xl, xr, split, yr, u, i+4, min_size=min_size, scale=scale)

        values = scale * 12. * (u3 - 0.5) * in_region.astype(float)
        values += (values_increment_l.astype(float) + values_increment_r.astype(float))
        return values


class NonlinearScenario(DataGeneratorScenario):
    """Scenario with nonlinear baseline and homogeneous! effects"""
    
    def __init__(self, random_generator=None):
        P = 2 
        K = 2 
        sigma2 = 1
        nonlinear_scale=1.0
        super().__init__(
            P, K, sigma2, 
            assumptions="nonlinear", 
            theta_base=np.array([1., 0.5, -4.]),
            nonlinear_scale=nonlinear_scale,
            random_generator=random_generator
        )


class HomogeneousScenario(DataGeneratorScenario):
    """Scenario with linear baseline and homogeneous users"""
    
    def __init__(self, random_generator=None):
        P = 2 
        K = 2 
        sigma2 = 1
        linear_baseline_coef = np.array([2., -2., 3.])
        super().__init__(
            P, K, sigma2, 
            assumptions="homogeneous", 
            theta_base=np.array([1., 0.5, -4.]),
            linear_baseline_coef=linear_baseline_coef,
            random_generator=random_generator
        )


class HeterogeneousScenario(DataGeneratorScenario):
    """Scenario with linear baseline and heterogeneous users"""
    
    def __init__(self, P, K, sigma2, random_seed=1, linear_baseline_coef=None, random_generator=None):
        raise NotImplementedError("Heterogeneous scenario not implemented")
        super().__init__(
            P, K, sigma2, 
            assumptions="heterogeneous", 
            random_generator=random_generator
        )
        