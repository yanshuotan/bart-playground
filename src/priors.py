import numpy as np
from abc import ABC, abstractmethod

from scipy.stats import invgamma
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

from params import Tree, Parameters
from moves import Move

class Prior:
    """
    Represents the priors for the BART model.
    """
    @abstractmethod
    def init_global_params(self, data):
        pass

    @abstractmethod
    def resample_global_params(self, bart_params):
        pass
    
    @abstractmethod
    def resample_leaf_vals(self, bart_params, tree_ids):
        pass

    @abstractmethod
    def trees_log_prior(self, bart_params, tree_ids):
        pass

    @abstractmethod
    def trees_log_marginal_lkhd(self, bart_params, tree_ids):
        pass

    def trees_log_prior_ratio(self, move : Move):
        log_prior_current = self.trees_log_prior(move.current, move.trees_changed)
        log_prior_proposed = self.trees_log_prior(move.proposed, move.trees_changed)
        return log_prior_proposed - log_prior_current

    def trees_log_marginal_lkhd_ratio(self, move : Move, marginalize: bool=False):
        """
        Compute the ratio of marginal likelihoods for a given move.

        Parameters:
        - move: Move
            The move to compute the marginal likelihood ratio for.
        - marginalize: bool
            Whether to marginalize over the ensemble.

        Returns:
        - float
            Marginal likelihood ratio.
        """
        if not marginalize:
            log_lkhd_current = self.trees_log_marginal_lkhd(move.current, move.trees_changed)
            log_lkhd_proposed = self.trees_log_marginal_lkhd(move.proposed, move.trees_changed)
        else:
            log_lkhd_current = self.trees_log_marginal_lkhd(move.current, 
                                                            np.arange(self.current.n_trees))
            log_lkhd_proposed = self.trees_log_marginal_lkhd(move.proposed, 
                                                             np.arange(self.current.n_trees))
        return log_lkhd_proposed - log_lkhd_current
    
    def trees_log_mh_ratio(self, move : Move, marginalize : bool=False):
         return self.trees_log_prior_ratio(move) + \
            self.trees_log_marginal_lkhd_ratio(move, marginalize)

class DefaultPrior(Prior):
    """
    Default implementation of the BART priors.
    """
    def __init__(self, n_trees=200, tree_alpha: float=0.95, tree_beta: float=2.0, f_k=2.0, 
                 eps_q: float=0.9, eps_nu: float=3, specification="linear"):
        """
        Initialize the default BART priors.

        Parameters:
        - alpha: float
            Parameter for the tree prior.
        - beta: float
            Parameter for the tree prior.
        - mu_func: callable
            Function for the mean of the leaf parameters.
        - sigma2_func: callable
            Function for the variance of the leaf parameters.
        - lambda_noise: float
            Scale parameter for the noise prior.
        - nu_noise: float
            Shape parameter for the noise prior.
        """
        self.n_trees = n_trees
        self.tree_alpha = tree_alpha
        self.tree_beta = tree_beta
        self.f_k = f_k
        self.f_sigma2 = 0.25 / (self.k ** 2 * n_trees)
        self.eps_q = eps_q
        self.eps_nu = eps_nu
        self.eps_lambda = None
        self.specification = specification

    def fit(self, data):
        self.eps_lambda = self._fit_eps_lambda(data, self.specification)

    def init_global_params(self, data):
        eps_sigma2 = self._sample_eps_sigma2(data.X.shape[1], data.y)
        return {"eps_sigma2" : eps_sigma2}

    def resample_global_params(self, bart_params):
        eps_sigma2 = self._sample_eps_sigma2(bart_params.data.n, 
                                             bart_params.data.y - bart_params.evaluate())
        return {"eps_sigma2" : eps_sigma2}

    def resample_leaf_vals(self, bart_params, tree_ids):
        residuals = bart_params.y - bart_params.evaluate(all_except=tree_ids)
        leaf_basis = bart_params.leaf_basis(tree_ids)
        p = leaf_basis.shape[1]
        post_cov = np.linalg.inv(leaf_basis.T @ leaf_basis / 
                                 bart_params.global_params["eps_sigma2"] + 
                                 np.eye(p) / self.f_sigma2)
        post_mean = post_cov @ leaf_basis.T @ residuals / \
            bart_params.global_params["eps_sigma2"]
        leaf_params_new = sqrtm(post_cov) @ np.np.random.randn(p) + post_mean
        return leaf_params_new

    def trees_log_prior(self, bart_params, tree_ids):
        log_prior = 0
        for tree_id in tree_ids:
            tree = bart_params.trees[tree_id]
            d = np.ceil(np.log2(np.arange(len(tree.vars)) + 2)) - 1
            log_p_split = np.log(self.alpha) - self.tree_beta * np.log(1 + d)
            log_probs = np.where(self.var == -1, np.log(1 - np.exp(log_p_split)), 
                                 log_p_split)
            log_prior += np.sum(log_probs[~np.isnan(tree.var)])
        return log_prior

    def trees_log_marginal_lkhd(self, bart_params, tree_ids):
        resids = bart_params.data.y - bart_params.evaluate(all_except=tree_ids)
        leaf_basis = bart_params.leaf_basis(tree_ids)
        U, S, _ = np.linalg.svd(leaf_basis)
        noise_ratio = bart_params.global_params["eps_sigma2"] / self.f_sigma2
        logdet = np.sum(np.log(S ** 2 / noise_ratio + 1))
        resid_u_coefs = U.T @ resids
        resids_u = U @ resid_u_coefs
        ls_resids = np.sum((resids - resids_u) ** 2)
        ridge_bias = np.sum(resid_u_coefs ** 2 / (S ** 2 / noise_ratio + 1))
        return - (logdet + (ls_resids + ridge_bias) / 
                  bart_params.global_params["eps_sigma2"]) / 2

    def _fit_eps_lambda(self, data, specification="linear"):
        """
        Compute the lambda parameter for the noise variance prior.
        Find lambda such that x ~ Gamma(nu/2, nu/(2*lambda) and P(x < q) = sigma_hat.
        """
        if specification == "naive":
            sigma_hat = np.std(data.y)
        elif specification == "linear":
            # Fit a linear model to the data
            model = LinearRegression().fit(data.X, data.y)
            y_hat = model.predict(data.X)
            resids = data.y - y_hat
            sigma_hat = np.std(resids)
        else:
            raise ValueError("Invalid specification for the noise variance prior.")
        def objective(l):
            return np.abs(invgamma.cdf(self.eps_q, a=self.eps_nu/2, 
                                       scale=self.eps_nu * l / 2) - sigma_hat)
        result = minimize_scalar(objective)
        return result.x
    
    def _sample_eps_sigma2(self, n, residuals):
        # Convert to inverse gamma params
        prior_alpha = self.eps_nu / 2
        prior_beta = self.eps_nu * self.eps_lambda / 2
        post_alpha = prior_alpha + n / 2
        post_beta = prior_beta + np.sum(residuals ** 2) / 2
        eps_sigma2 = invgamma.rvs(a=post_alpha, scale=post_beta, size=1)[0]
        return eps_sigma2

all_priors = {"default" : DefaultPrior}