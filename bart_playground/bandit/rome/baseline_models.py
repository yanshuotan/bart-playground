from .standalone_rome import BaseTS, RiverBatchEstimator, BaggingRegressor
from river.tree import SGTRegressor
from river.tree.splitter import DynamicQuantizer  # The default quantizer was throwing errors
import numpy as np
import scipy
import scipy.sparse
from scipy.stats import norm, gamma
from sksparse.cholmod import cholesky
from sklearn.base import BaseEstimator
from typing import Callable

class StandardTS(BaseTS):
    """Class for running a standard Thompson sampler that considers each individual in isolation"""
    def __init__(self, n_max: int, t_max: int, p: int, featurize: Callable, nn: Callable = None, nn_dim: int = None) -> None:
        self.n_max = n_max
        self.t_max = t_max
        self.p = p  # Dimension of covariates including intercept
        self.featurize = featurize
        self.nn = nn
        self.nn_dim = nn_dim
        if self.nn is None:
            self.bayes_model_dim = 2 * self.p
        else:
            self.bayes_model_dim = self.nn_dim + self.p
        self._initialize_priors()
    
    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        n_obs = context.shape[0]

        # Sample theta
        theta_precisions = self.theta_precisions[user_idx]
        theta_chol_precisions = np.linalg.cholesky(theta_precisions)
        theta_sqrt_transpose_covariances = np.linalg.solve(
            theta_chol_precisions,
            self.identity_stack[:n_obs])
        theta_sqrt_covariances = np.swapaxes(theta_sqrt_transpose_covariances, 1, 2)
        z = np.random.normal(size=(n_obs, self.bayes_model_dim, 1))
        precisions = gamma.rvs(
            a=self.precision_shapes[user_idx],
            scale=1./self.precision_rates[user_idx],
            size=n_obs)
        sigma = np.sqrt(1/precisions)
        theta_centered_raw = (theta_sqrt_covariances @ z)[:, :, 0]
        theta_centered = (theta_centered_raw.T * sigma).T
        theta = self.theta_means[user_idx] + theta_centered
        theta_action = theta[:, :self.p]

        # Compute optimal actions
        optimal_actions = self.calculate_optimal_action(context, theta_action)

        return optimal_actions

    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'StandardTS':
        # Assumption: users appear no more than once
        n_obs = context.shape[0]
        prior_theta_means = self.theta_means[user_idx].copy()
        prior_theta_precisions = self.theta_precisions[user_idx].copy()

        features = self.featurize(context)
        if self.nn is None:
            features_expanded = np.hstack([(action*features.T).T, features])
        else:
            nn_features = self.nn(np.hstack([context, context_extra]))
            features_expanded = np.hstack([(action*features.T).T, nn_features])
        features_stack_of_columns = np.expand_dims(features_expanded, axis=2)
        features_stack_of_rows = np.expand_dims(features_expanded, axis=1)
        features_crossprod = features_stack_of_columns @ features_stack_of_rows
        theta_precisions = prior_theta_precisions + features_crossprod
        self.theta_precisions[user_idx] = theta_precisions
        
        theta_mean_left = features_stack_of_columns * np.repeat(reward, self.bayes_model_dim).reshape((n_obs, self.bayes_model_dim, 1))
        theta_mean_right = prior_theta_precisions @ np.expand_dims(prior_theta_means, axis=2)
        theta_means = np.linalg.solve(
            theta_precisions,
            theta_mean_left + theta_mean_right)[:, :, 0]
        self.theta_means[user_idx] = theta_means

        self.precision_shapes[user_idx] += 0.5
        self.precision_rates[user_idx] += 0.5 * (
            reward**2
                + self._quadratic_form_3d(prior_theta_means, prior_theta_precisions)
                - self._quadratic_form_3d(theta_means, theta_precisions))       

    def reset(self) -> 'StandardTS':
        self._initialize_priors()

    @staticmethod
    def _quadratic_form_3d(rectangle, squares):
        rectangle_stack_of_rows = np.expand_dims(rectangle, axis=1)
        rectangle_stack_of_columns = np.expand_dims(rectangle, axis=2)
        multiplication_result = rectangle_stack_of_rows @ squares @ rectangle_stack_of_columns
        return multiplication_result.flatten()

    def _initialize_priors(self) -> None:
        # NOTE: Need to update dimensions below
        self.identity_stack = np.broadcast_to(np.identity(self.bayes_model_dim)[None,...], (self.n_max, self.bayes_model_dim, self.bayes_model_dim)).copy()
        if self.nn is not None:
            self.identity_stack[:, self.p:, self.p:] *= self.p / self.bayes_model_dim
        self.theta_precisions = self.identity_stack.copy()
        self.theta_means = np.zeros((self.n_max, self.bayes_model_dim))
        self.precision_shapes = np.ones(self.n_max) 
        self.precision_rates = np.ones(self.n_max)

class ActionCenteredTS(StandardTS):
    """Class for running an action-centered Thompson sampler as described here: https://doi.org/10.48550/arXiv.1711.03596"""
    def __init__(self, n_max: int, t_max: int, p: int, featurize: Callable, pi_min: float = 0., pi_max: float = 1.) -> None:
        self.n_max = n_max
        self.t_max = t_max
        self.p = p
        self.featurize = featurize
        self.pi_min = pi_min
        self.pi_max = pi_max
        self._initialize_priors()
    
    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        n_obs = context.shape[0]

        # Sample theta
        theta_mean = self.theta_means[user_idx]
        theta_precisions = self.theta_precisions[user_idx]
        theta_chol_precisions = np.linalg.cholesky(theta_precisions)
        theta_sqrt_transpose_covariances = np.linalg.solve(
            theta_chol_precisions,
            self.identity_stack[:n_obs]
        )
        theta_sqrt_covariances = np.swapaxes(theta_sqrt_transpose_covariances, 1, 2)

        # Select action
        a_bar = np.ones(n_obs, dtype=int)
        features = self.featurize(context)
        reward_a_bar_mean = (features * theta_mean).sum(axis=1)
        theta_covariances = theta_sqrt_transpose_covariances @ theta_sqrt_covariances
        reward_a_bar_var = self._quadratic_form_3d(features, theta_covariances)
        reward_a_bar_sd = np.sqrt(reward_a_bar_var)
        p_a_bar_positive = 1. - norm.cdf(
            np.zeros_like(reward_a_bar_mean),
            loc=reward_a_bar_mean,
            scale=reward_a_bar_sd)
        minned_pi_max = np.minimum(self.pi_max, p_a_bar_positive)
        self.pi_t = np.maximum(self.pi_min, minned_pi_max)
        action = a_bar * (np.random.uniform(size=a_bar.shape) < self.pi_t)

        return action

    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'StandardTS':
        # Assumption: users appear no more than once
        n_obs = context.shape[0]
        prior_theta_means = self.theta_means[user_idx].copy()
        prior_theta_precisions = self.theta_precisions[user_idx].copy()

        features = (action * self.featurize(context).T).T
        features_stack_of_columns = np.expand_dims(features, axis=2)
        features_scaled = (features.T * (self.pi_t * (1. - self.pi_t))).T
        features_scaled_stack_of_rows = np.expand_dims(features_scaled, axis=1)
        features_crossprod = features_stack_of_columns @ features_scaled_stack_of_rows
        theta_precisions = prior_theta_precisions + features_crossprod
        self.theta_precisions[user_idx] = theta_precisions
        
        action_centered_reward = (action - self.pi_t) * reward
        theta_mean_left = features_stack_of_columns * np.repeat(action_centered_reward, self.p).reshape((n_obs, self.p, 1))
        theta_mean_right = prior_theta_precisions @ np.expand_dims(prior_theta_means, axis=2)
        theta_means = np.linalg.solve(
            theta_precisions,
            theta_mean_left + theta_mean_right)[:, :, 0]
        self.theta_means[user_idx] = theta_means

    def _initialize_priors(self) -> None:
        self.identity_stack = np.broadcast_to(np.identity(self.p)[None,...], (self.n_max, self.p, self.p)).copy()
        self.theta_precisions = self.identity_stack.copy()
        self.theta_means = np.zeros((self.n_max, self.p))

class RoME(BaseTS):
    """Class for running the DML Thompson sampler"""
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
        self.gamma_user = np.linalg.eigvals(self.user_precision).max()
        self.time_cov = time_cov
        self.time_precision = np.linalg.inv(time_cov)
        self.gamma_time = np.linalg.eigvals(self.time_precision).max()
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


class IntelligentPooling(BaseTS):
    """Class for running a user- and time-pooled Thompson sampler (no DML)
    Notice that self.pi is Pr(A=1 | s), not Pr(A=0 | s) as it is in the DML versions
    """
    def __init__(self, n_max: int, t_max: int, p: int, featurize: Callable,
        user_cov: np.ndarray, time_cov: np.ndarray, sigma: float = 1.) -> None:
        self.n_max = n_max
        self.t_max = t_max
        self.p = p
        self.num_thetas = 2 + n_max + t_max
        self.theta_dim = p * self.num_thetas
        self.featurize = featurize
        self.user_cov = user_cov
        self.time_cov = time_cov
        self.sigma = sigma
        self._initialize_priors()
    
    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        n_obs = len(user_idx)
        theta_hat = self.V_cholesky(self.b)

        # Calculate probability that advantage is positive
        phi_zeros = self._make_phi(context, np.zeros(n_obs), user_idx, time_idx, pi=0.).tocsr()
        phi_ones = self._make_phi(context, np.ones(n_obs), user_idx, time_idx, pi=0.).tocsr()
        advantage_mean = (phi_ones - phi_zeros) @ theta_hat
        advantage_vars = []
        for i in range(n_obs):
            phi_i = phi_ones[[i]]
            advantage_var_i = phi_i @ self.V_cholesky(phi_i.T)
            advantage_vars.append(advantage_var_i[0, 0])
        advantage_vars = np.asarray(advantage_vars)
        self.pi = 1. - norm.cdf(np.zeros(n_obs), loc=advantage_mean, scale=np.sqrt(advantage_vars))

        # Sample actions: select A=1 with probability pi(A=1 | s)
        action = (np.random.uniform(size=n_obs) < self.pi).astype(float)

        return action

    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'IntelligentPooling':
        phi = self._make_phi(context, action, user_idx, time_idx, self.pi).tocsr()
        self.V += phi.T @ phi
        self.V_cholesky.update_inplace(phi.T.tocsc())
        phi_csc = phi.copy().tocsc()
        phi_csc.data *= reward[phi_csc.indices]
        phi_csc_summed = np.asarray(phi_csc.sum(axis=0)).flatten()
        self.b += phi_csc_summed
        self.stage += 1

    def _initialize_priors(self) -> None:
        # Shared
        V_shared = 25. * scipy.sparse.identity(2 * self.p)

        # User
        I_n = scipy.sparse.identity(self.n_max)
        V_user = scipy.sparse.kron(I_n, np.linalg.inv(self.user_cov))

        # Time
        I_t = scipy.sparse.identity(self.t_max)
        V_time = scipy.sparse.kron(I_t, np.linalg.inv(self.time_cov))
        
        # V
        self.V = scipy.sparse.block_diag([V_shared, V_user, V_time], format="csc")
        self.V_cholesky = cholesky(self.V.tocsc())

        # Other quantities
        self.b = np.zeros(self.theta_dim)
        self.stage = 1
   
    def _make_phi(self, context: np.ndarray, action: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, pi: np.ndarray) -> np.ndarray:
        # pi can also be specified as a float
        n_obs = context.shape[0]
        a_centered = action - pi
        features = self.featurize(context)
        a_centered_features = (a_centered * features.T).T

        # Make the causal portion of phi
        ones = np.ones(3*n_obs)
        row_idx = np.tile(np.arange(n_obs), 3)
        col_idx = np.concatenate([np.zeros(n_obs), user_idx+1, (self.n_max+1) + time_idx])
        indicators = scipy.sparse.csr_matrix((ones, (row_idx, col_idx)), shape=(n_obs, self.num_thetas - 1))
        phi_theta = scipy.sparse.vstack([
            scipy.sparse.kron(indicators[i], a_centered_features[i])
            for i in range(n_obs)])
    
        # Prepend the baseline
        phi = scipy.sparse.hstack([features, phi_theta])

        # Convert to csr and return
        phi_csr = phi.tocsr()
        return phi_csr
    
    def reset(self) -> 'IntelligentPooling':
        self._initialize_priors()


class UserLaplacian(BaseTS):
    """Class for running a user-pooled Thompson sampler with NN Laplacian regularization.
    Notice that self.pi is Pr(A=1 | s), not Pr(A=0 | s) as it is in the DML versions.
    """
    def __init__(self, n_max: int, t_max: int, p: int, featurize: Callable,
        L_user: scipy.sparse._csr.csr_matrix, user_cov: np.ndarray, lambda_penalty: float = 1.,
        sigma: float = 1.) -> None:
        self.n_max = n_max
        self.t_max = t_max
        self.p = p
        self.num_thetas = 2 + n_max  # The 2 comes from the fact that we have a baseline AND advantage function model
        self.theta_dim = p * self.num_thetas
        self.featurize = featurize
        self.L_user = L_user
        self.user_precision = np.linalg.inv(user_cov)
        self.lambda_penalty = lambda_penalty
        self.sigma = sigma
        self._initialize_priors()
    
    def sample_action(self, context: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
        n_obs = len(user_idx)
        theta_hat = self.V_cholesky(self.b)

        # Calculate probability that advantage is positive
        phi_zeros = self._make_phi(context, np.zeros(n_obs), user_idx, pi=0.).tocsr()
        phi_ones = self._make_phi(context, np.ones(n_obs), user_idx, pi=0.).tocsr()
        advantage_mean = (phi_ones - phi_zeros) @ theta_hat
        advantage_vars = []
        for i in range(n_obs):
            phi_i = phi_ones[[i]]
            advantage_var_i = phi_i @ self.V_cholesky(phi_i.T)
            advantage_vars.append(advantage_var_i[0, 0])
        advantage_vars = np.asarray(advantage_vars)
        self.pi = 1. - norm.cdf(np.zeros(n_obs), loc=advantage_mean, scale=np.sqrt(advantage_vars))

        # Sample actions: select A=1 with probability pi(A=1 | s)
        action = (np.random.uniform(size=n_obs) < self.pi).astype(float)

        return action

    def update(self, context: np.ndarray, context_extra: np.ndarray, user_idx: np.ndarray, time_idx: np.ndarray, action: np.ndarray, reward: np.ndarray) -> 'IntelligentPooling':
        phi = self._make_phi(context, action, user_idx, self.pi).tocsr()
        self.V += phi.T @ phi
        self.V_cholesky.update_inplace(phi.T.tocsc())
        phi_csc = phi.copy().tocsc()
        phi_csc.data *= reward[phi_csc.indices]
        phi_csc_summed = np.asarray(phi_csc.sum(axis=0)).flatten()
        self.b += phi_csc_summed
        self.stage += 1

    def _initialize_priors(self) -> None:
        # Shared
        V_shared = 25. * scipy.sparse.identity(2 * self.p)

        # User
        # Need Laplacian regularization here
        I_n = scipy.sparse.identity(self.n_max)
        I_p = scipy.sparse.identity(self.p)
        V_user = (
            scipy.sparse.kron(I_n, self.user_precision) + 
            scipy.sparse.kron(self.L_user, self.lambda_penalty * I_p)
        )
        
        # V
        self.V = scipy.sparse.block_diag([V_shared, V_user], format="csc")
        self.V_cholesky = cholesky(self.V.tocsc())

        # Other quantities
        self.b = np.zeros(self.theta_dim)
        self.stage = 1
   
    def _make_phi(self, context: np.ndarray, action: np.ndarray, user_idx: np.ndarray, pi: np.ndarray) -> np.ndarray:
        # pi can also be specified as a float
        n_obs = context.shape[0]
        features = self.featurize(context)
        # action_features = (action * features.T).T
        a_centered = action - pi
        action_features = (a_centered * features.T).T

        # Make the causal portion of phi
        ones = np.ones(2*n_obs)
        row_idx = np.tile(np.arange(n_obs), 2)
        col_idx = np.concatenate([np.zeros(n_obs), user_idx+1])
        indicators = scipy.sparse.csr_matrix((ones, (row_idx, col_idx)), shape=(n_obs, self.num_thetas - 1))
        phi_theta = scipy.sparse.vstack([
            scipy.sparse.kron(indicators[i], action_features[i])
            for i in range(n_obs)])
    
        # Prepend the baseline
        phi = scipy.sparse.hstack([features, phi_theta])

        # Convert to csr and return
        phi_csr = phi.tocsr()
        return phi_csr
    
    def reset(self) -> 'IntelligentPooling':
        self._initialize_priors()
