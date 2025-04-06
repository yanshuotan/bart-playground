# Standard library imports
from collections import UserList
from typing import Callable, Iterator

# Third-party imports
import numpy as np
import river
from river import base, linear_model, utils
from river.base.estimator import Estimator
from river.base.wrapper import Wrapper
from river.stream import iter_array
from river.tree import SGTRegressor
from river.tree.splitter import DynamicQuantizer
import scipy
import scipy.sparse
from scipy.stats import norm
import sklearn
from sklearn.base import BaseEstimator
from sksparse.cholmod import cholesky

class BaseTS:
    """Base class for Thompson sampling algorithms"""
    
    def calculate_optimal_action(self, context: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the optimal action for a given context and theta"""
        features = self.featurize(context)
        n_obs = features.shape[0]
        return (features @ theta.T > 0).astype(float).flatten()

    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        """Sample an action based on the current state"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, 
               time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'BaseTS':
        """Update the model based on observed rewards"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def reset(self) -> 'BaseTS':
        """Reset the model to its initial state"""
        raise NotImplementedError("Subclasses must implement this method")
    

class Ensemble(UserList):
    def __init__(self, models: Iterator[Estimator]):
        super().__init__(models)

        if len(self) < self._min_number_of_models:
            raise ValueError(
                f"At least {self._min_number_of_models} models are expected, "
                + f"only {len(self)} were passed"
            )

    @property
    def _min_number_of_models(self):
        return 2

    @property
    def models(self):
        return self.data
    
class WrapperEnsemble(Ensemble, Wrapper):
    def __init__(self, models, n_models, seed):
        super().__init__(model for model in models)
        self.model = models[0]
        self.n_models = n_models
        self.seed = seed
        from random import Random
        self._rng = Random(seed)

    @property
    def _wrapped_model(self):
        return self.model
    
class RiverBatchEstimator(sklearn.base.BaseEstimator):
    def __init__(self, model: river.base.Estimator) -> None:
        super().__init__()
        self.model = model

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'RiverBatchEstimator':
        for xi, yi in iter_array(X, y):
            self.model = self.model.learn_one(xi, yi)
        return self

    def predict(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        predictions = [self.model.predict_one(xi, i) for xi, i in iter_array(X, idx)]
        return np.asarray(predictions)

class BaseBagging(WrapperEnsemble):
    def learn_one(self, x, y, subsample: float = 0.5, **kwargs):
        # subsample <= 0. performs the bootstrap
        # Otherwise, observations are included in a tree with probability = subsample

        for model in self:
            if subsample <= 0:
                poisson_val = utils.random.poisson(1, self._rng)
            else:
                poisson_val = int(self._rng.random() < subsample)
            if not hasattr(model, "poisson_vals"):
                model.poisson_vals = []
            model.poisson_vals.append(poisson_val)
            for _ in range(poisson_val):
                model.learn_one(x, y, **kwargs)

        return self


class BaggingRegressor(BaseBagging, base.Regressor):
    def __init__(self, model: base.Regressor, n_models=10, seed: int = None, subsample: float = 0.5):
        self.subsample = subsample

        # Randomly change radius
        models = [model.clone() for _ in range(n_models)]
        if hasattr(model, "feature_quantizer"):
            for model_i in models:
                model_i.feature_quantizer.radius = 0.1 + np.random.exponential(0.1)
                model_i.feature_quantizer.std_prop = 0.1 + np.random.exponential(0.1)

        super().__init__(models, n_models, seed)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LinearRegression()}

    def learn_one(self, x, y, subsample: float = 0.5, **kwargs):
        subsample = self.subsample if subsample is None else subsample
        return super().learn_one(x, y, subsample, **kwargs)

    def predict_one(self, x, i=None, **kwargs):
        """Averages the predictions of each regressor."""
        prediction_sum = 0.
        count = 0
        for regressor in self:
            if (i is None) or (i < 0) or (i >= len(regressor.poisson_vals)) or (regressor.poisson_vals[i] == 0):
                count += 1
                prediction_sum += regressor.predict_one(x, **kwargs)
        prediction = prediction_sum / count if count else 0.
        return prediction


class RoME(BaseTS):
    """Class for running the RoME (Robust Mixed-Effects) Thompson sampler"""
    def __init__(self, n_max: int, t_max: int, p: int, featurize: Callable,
        L_user: scipy.sparse._csr.csr_matrix, L_time: scipy.sparse._csr.csr_matrix,
        user_cov: np.ndarray, time_cov: np.ndarray, lambda_penalty: float = 1., ml_interactions: bool = False,
        ml_model: BaseEstimator = RiverBatchEstimator(BaggingRegressor(SGTRegressor(delta=0.05, grace_period=50, feature_quantizer=DynamicQuantizer(), lambda_value=0., gamma=0.), n_models=100, subsample=0.8)),
        v: float = 1., delta: float = 0.01, zeta: float = 10.,
        n_neighbors: int= 1, pool_users: bool = True) -> None:
        self.pool_users = pool_users
        if not pool_users:
            n_max = 1
            L_user = scipy.sparse.csr_matrix([[0.]])
        self.n_max = n_max
        self.t_max = t_max
        self.K = max(n_max, t_max)
        self.p = p
        self.num_thetas = 1 + n_max + t_max
        self.theta_dim = p * self.num_thetas
        self.featurize = featurize
        self.L_user = L_user
        self.L_time = L_time
        self.ml_interactions = ml_interactions
        self.ml_model = ml_model
        self.user_cov = user_cov
        self.user_precision = np.linalg.inv(user_cov)
        # self.gamma_user = np.linalg.eigvals(self.user_precision).max()
        self.time_cov = time_cov
        self.time_precision = np.linalg.inv(time_cov)
        # self.gamma_time = np.linalg.eigvals(self.time_precision).max()
        self.lambda_penalty = lambda_penalty
        self.v = v
        self.delta = delta
        self.n_neighbors = n_neighbors
        self.zeta = zeta
        self.beta_const = self.zeta * max(1., np.log(self.K)**(0.75))

        self._initialize_priors()
    
    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        if not self.pool_users:
            user_idx = np.zeros_like(user_idx, dtype=int)
        n_obs = len(user_idx)

        # Update theta_hat
        theta_hat = self.V_cholesky(self.b)

        # Calculate probability that advantage is negative
        phi_ones = self._make_phi(context, np.ones(n_obs), user_idx, time_idx).tocsr()
        advantage_mean = phi_ones @ theta_hat
        advantage_vars = []
        for j in range(n_obs):
            phi_j = phi_ones[[j]]
            i = user_idx[j]
            t = time_idx[j]
            I_p = scipy.sparse.identity(self.p, format="csr")
            C = scipy.sparse.hstack([
                I_p,
                scipy.sparse.csr_matrix((self.p, self.p * i)),
                I_p,
                scipy.sparse.csr_matrix((self.p, self.p * (self.n_max-i+t-1))),
                I_p,
                scipy.sparse.csr_matrix((self.p, self.p * (self.t_max-t-1))),
            ])
            Ct = C.T.tocsc()
            Vi_Ct = self.V_cholesky(Ct)
            V_bar = (C @ Vi_Ct).toarray()
            Lambda = (Vi_Ct.T @ self.V0 @ Vi_Ct).toarray()
            k = i + t + 1
            beta = self.v * np.sqrt(
                2 * np.log(2*self.K*(self.K+1)/self.delta) +
                + np.linalg.slogdet(V_bar)[1]
                - np.linalg.slogdet(Lambda)[1]
            ) + self.beta_const
            advantage_var_j = phi_j @ self.V_cholesky(phi_j.T.tocsc() * (beta / self.first_beta)**2)
            advantage_vars.append(advantage_var_j[0, 0])
        advantage_vars = np.asarray(advantage_vars)
        self.pi = norm.cdf(np.zeros(n_obs), loc=advantage_mean, scale=np.sqrt(advantage_vars))

        # Sample actions: select A=1 with probability 1 - pi(0 | s)
        action = (np.random.uniform(size=n_obs) > self.pi).astype(float)

        return action

    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'RoME':
        # Assumption: users appear no more than once
        n_obs = context.shape[0]
        negative_list = -np.ones(n_obs)
        if not self.pool_users:
            user_idx = np.zeros_like(user_idx, dtype=int)
        phi = self._make_phi(context, action, user_idx, time_idx).tocsr()
        adjusted_reward_den = action - self.pi
        sigma_tilde_2 = self.pi * (1. - self.pi)
        weights = np.sqrt(sigma_tilde_2)
        action_2d = action.reshape((-1, 1))
        action0_2d = np.zeros_like(action_2d)
        action1_2d = np.ones_like(action_2d)
        if self.ml_interactions:
            features_extra = np.hstack([context, context_extra, action_2d, action_2d*context, action_2d*context_extra])
            features_extra_0 = np.hstack([context, context_extra, action0_2d, action0_2d*context, action0_2d*context_extra])
            features_extra_1 = np.hstack([context, context_extra, action1_2d, action1_2d*context, action1_2d*context_extra])
        else:
            features_extra = np.hstack([context, context_extra, action_2d])
            features_extra_0 = np.hstack([context, context_extra, action0_2d])
            features_extra_1 = np.hstack([context, context_extra, action1_2d])

        # Update parameter values
        phi_weighted = phi.copy()
        phi_weighted.data *= weights.repeat(np.diff(phi_weighted.indptr))
        self.V += phi_weighted.T @ phi_weighted
        self.V_cholesky.update_inplace(phi_weighted.T.tocsc())
        if self.stage == 1:
            f_A = np.zeros(n_obs)
            f_1 = np.zeros(n_obs)
            f_0 = np.zeros(n_obs)
        else:
            f_A = self.ml_model.predict(features_extra, negative_list)
            f_0 = self.ml_model.predict(features_extra_0, negative_list)
            f_1 = self.ml_model.predict(features_extra_1, negative_list)
        adjusted_reward_num = reward - f_A
        adjusted_reward = adjusted_reward_num / adjusted_reward_den + f_1 - f_0
        phi_csc = phi.tocsc()
        phi_csc.data *= (sigma_tilde_2 * adjusted_reward)[phi_csc.indices]
        phi_csc_summed = np.asarray(phi_csc.sum(axis=0)).flatten()
        self.b += phi_csc_summed

        # Update ML model
        self.ml_model.partial_fit(features_extra, reward)

        # Increment stage
        self.stage += 1

    def _initialize_priors(self) -> None:
        I_p = scipy.sparse.identity(self.p)
        I_n = scipy.sparse.identity(self.n_max)
        I_t = scipy.sparse.identity(self.t_max)

        # Shared
        V_shared = 25. * I_p

        # User
        V_user = (
            scipy.sparse.kron(I_n, self.user_precision) + 
            scipy.sparse.kron(self.L_user, self.lambda_penalty * I_p)
        )
        if not self.pool_users:
            V_user *= 1e18  # Effectively force these to be 0

        # Time
        V_time = (
            scipy.sparse.kron(I_t, self.time_precision) + 
            scipy.sparse.kron(self.L_time, self.lambda_penalty * I_p)
        )

        # Final matrices        
        self.V = scipy.sparse.block_diag([V_shared, V_user, V_time], format="csc")
        self.V0 = self.V.copy()
        self.V_cholesky = cholesky(self.V.tocsc())  # Could use Taylor series approximation
        self.log_det_V0 = self.V_cholesky.logdet()
        self.b = np.zeros((1+self.n_max+self.t_max) * self.p)
        self.stage = 1
        self.first_beta = self.v * np.sqrt(2 * np.log(2*self.K*(self.K+1)/self.delta)) + self.beta_const
        # NOTE: The determinant term is one at the first stage

    def reset(self) -> 'RoME':
        self._initialize_priors()

    def _make_phi(self, context: np.ndarray, action: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        if not self.pool_users:
            user_idx = np.zeros_like(user_idx, dtype=int)
        n_obs = context.shape[0]
        features = self.featurize(context)
        ones = np.ones(3*n_obs)
        row_idx = np.tile(np.arange(n_obs), 3)
        col_idx = np.concatenate([np.zeros(n_obs), user_idx+1, (self.n_max+1) + time_idx])
        indicators = scipy.sparse.csr_matrix((ones, (row_idx, col_idx)), shape=(n_obs, self.num_thetas))
        phi = scipy.sparse.vstack([
            scipy.sparse.kron(indicators[i], features[i])
            for i in range(n_obs)])
        phi_csr = phi.tocsr()
        return phi_csr
