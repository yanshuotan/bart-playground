import numpy as np

from scipy.stats import invgamma
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

from params import Tree, BARTParams

class BARTPrior:
    """
    Represents the priors for the BART model.
    """
    def tree_log_prior(self):
        """
        Compute the prior for the tree structure.
        """
        pass

    def n_trees_prior(self):
        """
        Compute the prior for the number of trees.
        """
        pass

    def features_prior(self):
        """
        Compute the prior for the feature selection.
        """
        pass

    def leaf_params_prior(self):
        """
        Compute the prior for the leaf parameters.
        """
        pass

    def resample_global_params(self, bart_params):
        """
        Resample the global parameters of the BART model conditioned on the tree parameters
        """
        pass


class DefaultBARTPrior(BARTPrior):
    """
    Default implementation of the BART priors.
    """
    def __init__(self, X, y, n_trees, tree_alpha: float, tree_beta: float, f_k=2.0, 
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
        self.tree_alpha = tree_alpha
        self.tree_beta = tree_beta
        self.f_k = f_k
        self.f_sigma2 = 0.25 / (self.k ** 2 * n_trees)
        self.eps_q = eps_q
        self.eps_nu = eps_nu
        self.eps_lambda = self._fit_eps_lambda(X, y, specification)

    def _fit_eps_lambda(self, X, y, specification="linear"):
        """
        Compute the lambda parameter for the noise variance prior.
        Find lambda such that x ~ Gamma(nu/2, nu/(2*lambda) and P(x < q) = sigma_hat.
        """
        if specification == "naive":
            sigma_hat = np.std(y)
        elif specification == "linear":
            # Fit a linear model to the data
            model = LinearRegression().fit(X, y)
            y_hat = model.predict(X)
            resids = y - y_hat
            sigma_hat = np.std(resids)
        else:
            raise ValueError("Invalid specification for the noise variance prior.")
        def objective(l):
            return np.abs(invgamma.cdf(self.eps_q, a=self.eps_nu/2, scale=self.eps_nu * l / 2) - sigma_hat)
        result = minimize_scalar(objective)
        return result.x

    def tree_log_prior(self, bart_params, tree_ids):
        log_prior = 0
        for tree_id in tree_ids:
            tree = bart_params.trees[tree_id]
            d = np.ceil(np.log2(np.arange(len(tree.vars)) + 2)) - 1
            log_p_split = np.log(self.alpha) - self.tree_beta * np.log(1 + d)
            log_probs = np.where(self.var == -1, np.log(1 - np.exp(log_p_split)), log_p_split)
            log_prior += np.sum(log_probs[~np.isnan(tree.var)])
        return log_prior

    def tree_log_marginal_lkhd(self, bart_params, tree_ids):
        resids = self.y - self.evaluate(self.X, all_except=tree_ids)
        leaf_basis = bart_params.leaf_basis(tree_ids)
        U, S, _ = np.linalg.svd(leaf_basis)
        noise_ratio = bart_params.global_params["eps_sigma2"] / self.f_sigma2
        logdet = np.sum(np.log(S ** 2 / noise_ratio + 1))
        resid_u_coefs = U.T @ resids
        resids_u = U @ resid_u_coefs
        ls_resids = np.sum((resids - resids_u) ** 2)
        ridge_bias = np.sum(resid_u_coefs ** 2 / (S ** 2 / noise_ratio + 1))
        return - (logdet + (ls_resids + ridge_bias) / bart_params.global_params["eps_sigma2"]) / 2
    
    def resample_global_params(self, bart_params):
        eps_sigma2 = self._sample_eps_sigma2(bart_params.n, bart_params.y - bart_params.evaluate())
        bart_params.global_params = {"eps_sigma2" : eps_sigma2}
        return bart_params.global_params
    
    def init_global_params(self, X, y):
        eps_sigma2 = self._sample_eps_sigma2(X.shape[1], y)
        return {"eps_sigma2" : eps_sigma2}
    
    def _sample_eps_sigma2(self, n, residuals):
        # Convert to inverse gamma params
        prior_alpha = self.eps_nu / 2
        prior_beta = self.eps_nu * self.eps_lambda / 2
        post_alpha = prior_alpha + n / 2
        post_beta = prior_beta + np.sum(residuals ** 2) / 2
        eps_sigma2 = invgamma.rvs(a=post_alpha, scale=post_beta, size=1)[0]
        return eps_sigma2

    def resample_leaf_params(self, bart_params, tree_ids):
        residuals = bart_params.y - bart_params.evaluate(all_except=tree_ids)
        leaf_basis = bart_params.leaf_basis(tree_ids)
        p = leaf_basis.shape[1]
        post_cov = np.linalg.inv(leaf_basis.T @ leaf_basis / bart_params.global_params["eps_sigma2"] + np.eye(p) / self.f_sigma2)
        post_mean = post_cov @ leaf_basis.T @ residuals / bart_params.global_params["eps_sigma2"]
        leaf_params_new = sqrtm(post_cov) @ np.np.random.randn(p) + post_mean
        bart_params.update_leaf_params(tree_ids, leaf_params_new)

all_priors = {"default" : DefaultBARTPrior}