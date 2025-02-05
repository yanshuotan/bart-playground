import numpy as np
from abc import ABC, abstractmethod

from scipy.stats import invgamma
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

from .params import Tree, Parameters
from .moves import Move
from .util import Dataset

class Prior(ABC):
    """
    Represents the prior for the BART model.
    """
    @abstractmethod
    def fit(self, data : Dataset):
        """
        Fits the prior's hyperparameters to the provided dataset.
        """
        pass

    @abstractmethod
    def init_global_params(self, data : Dataset):
        """
        Initialize global parameters for the model.

        Args:
            data (object): An object containing the data with attributes:
                - X (numpy.ndarray): The input features matrix.
                - y (numpy.ndarray): The target values array.

        Returns:
            dict: A dictionary containing the initialized global parameters
        """
        pass

    @abstractmethod
    def resample_global_params(self, bart_params : Parameters):
        """
        Resamples the global parameters for the BART model.

        This function resamples the global parameters based on the provided BART parameters.

        Args:
            bart_params (Parameters): An instance of the Parameters class containing 
                                      the data and model parameters for BART.

        Returns:
            dict: A dictionary containing the resampled global parameters.
        """
        pass
    
    @abstractmethod
    def resample_leaf_vals(self, bart_params : Parameters, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees.

        This function updates the leaf parameters by resampling from the posterior
        distribution given the current residuals and leaf basis.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters.
        tree_ids : list or array-like
            A list or array of tree indices for which the leaf values are to be resampled.

        Returns:
        --------
        leaf_params_new : numpy.ndarray
            The resampled leaf parameters.
        """
        pass

    @abstractmethod
    def trees_log_prior(self, bart_params : Parameters, tree_ids):
        """
        Calculate the log prior probability of a set of trees.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters, 
            including the list of trees.
        tree_ids : list of int
            A list of tree indices for which the log prior is to be calculated.

        Returns:
        --------
        float
            The log prior probability of the specified trees.
        """
        pass

    @abstractmethod
    def trees_log_marginal_lkhd(self, bart_params : Parameters, tree_ids):
        """
        Calculate the log marginal likelihood of the trees in a BART model.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters,
            including data, global parameters, and methods for evaluation and leaf basis computation.
        tree_ids : list or array-like
            A list or array of tree identifiers for which the log marginal likelihood is to be computed.

        Returns:
        --------
        float
            The log marginal likelihood of the specified trees.
        """
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
        Initializes the parameters for the prior distribution.

        Args:
            n_trees (int, optional): Number of trees. Defaults to 200.
            tree_alpha (float, optional): Alpha parameter for the tree prior. Defaults to 0.95.
            tree_beta (float, optional): Beta parameter for the tree prior. Defaults to 2.0.
            f_k (float, optional): Scaling factor for the variance of the leaf parameters. Defaults to 2.0.
            eps_q (float, optional): Quantile used for setting the hyperprior for noise sigma2. Defaults to 0.9.
            eps_nu (float, optional): Inverse chi-squared nu hyperparameter for noise sigma2. Defaults to 3.
            specification (str, optional): Specification for a data-driven initial estimate for noise sigma2. Defaults to "linear".

        Attributes:
            n_trees (int): Number of trees.
            tree_alpha (float): Alpha parameter for the tree prior.
            tree_beta (float): Beta parameter for the tree prior.
            f_k (float): Scaling factor for the variance of the leaf parameters.
            f_sigma2 (float): Variance of the leaf parameters.
            eps_q (float): Parameter for the epsilon prior.
            eps_nu (float): Parameter for the epsilon prior.
            eps_lambda (None): Placeholder for epsilon lambda parameter.
            specification (str): Model specification.
        """

        self.n_trees = n_trees
        self.alpha = tree_alpha
        self.beta = tree_beta
        self.f_k = f_k
        self.f_sigma2 = 0.25 / (self.f_k ** 2 * n_trees)
        self.eps_q = eps_q
        self.eps_nu = eps_nu
        self.eps_lambda = None
        self.specification = specification

    def fit(self, data : Dataset):
        """
        Fits the prior's hyperparameters to the provided dataset.
        """
        self.eps_lambda = self._fit_eps_lambda(data, self.specification)

    def init_global_params(self, data):
        """
        Initialize global parameters for the model.

        This method samples the epsilon sigma squared (eps_sigma2) parameter 
        based on the provided data and returns it in a dictionary.

        Args:
            data (object): An object containing the data with attributes:
                - X (numpy.ndarray): The input features matrix.
                - y (numpy.ndarray): The target values array.

        Returns:
            dict: A dictionary containing the initialized global parameter:
                - eps_sigma2 (float): The sampled epsilon sigma squared value.
        """
        eps_sigma2 = self._sample_eps_sigma2(data.X.shape[1], data.y)
        return {"eps_sigma2" : eps_sigma2}

    def resample_global_params(self, bart_params : Parameters):
        """
        Resamples the global parameters for the BART model.

        This function resamples the global parameters, specifically `eps_sigma2`, 
        based on the provided BART parameters.

        Args:
            bart_params (Parameters): An instance of the Parameters class containing 
                                      the data and model parameters for BART.

        Returns:
            dict: A dictionary containing the resampled global parameters.
        """
        eps_sigma2 = self._sample_eps_sigma2(bart_params.data.n, 
                                             bart_params.data.y - bart_params.evaluate())
        return {"eps_sigma2" : eps_sigma2}

    def resample_leaf_vals(self, bart_params : Parameters, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees.

        This function updates the leaf parameters by resampling from the posterior
        distribution given the current residuals and leaf basis.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters.
        tree_ids : list or array-like
            A list or array of tree indices for which the leaf values are to be resampled.

        Returns:
        --------
        leaf_params_new : numpy.ndarray
            The resampled leaf parameters.
        """
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

    def trees_log_prior(self, bart_params : Parameters, tree_ids):
        """
        Calculate the log prior probability of a set of trees.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters, 
            including the list of trees.
        tree_ids : list of int
            A list of tree indices for which the log prior is to be calculated.

        Returns:
        --------
        float
            The log prior probability of the specified trees.
        """
        log_prior = 0
        for tree_id in tree_ids:
            tree = bart_params.trees[tree_id]
            d = np.ceil(np.log2(np.arange(len(tree.vars)) + 2)) - 1
            log_p_split = np.log(self.alpha) - self.beta * np.log(1 + d)
            log_probs = np.where(tree.vars == -1, np.log(1 - np.exp(log_p_split)), 
                                 log_p_split)
            is_non_empty = np.where(tree.vars != -2)[0]
            log_prior += np.sum(log_probs[is_non_empty])
        return log_prior

    def trees_log_marginal_lkhd(self, bart_params : Parameters, tree_ids):
        """
        Calculate the log marginal likelihood of the trees in a BART model.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters,
            including data, global parameters, and methods for evaluation and leaf basis computation.
        tree_ids : list or array-like
            A list or array of tree identifiers for which the log marginal likelihood is to be computed.

        Returns:
        --------
        float
            The log marginal likelihood of the specified trees.

        Notes:
        ------
        This method computes the log marginal likelihood by performing the following steps:
        1. Compute the residuals by subtracting the evaluated values of all trees except the specified ones from the observed data.
        2. Compute the leaf basis for the specified trees.
        3. Perform Singular Value Decomposition (SVD) on the leaf basis.
        4. Calculate the noise ratio using the global parameters.
        5. Compute the log determinant of the modified singular values.
        6. Project the residuals onto the left singular vectors.
        7. Compute the least squares residuals and ridge bias.
        8. Combine the log determinant, least squares residuals, and ridge bias to obtain the log marginal likelihood.
        """
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
            scale_value = max(self.eps_nu * l / 2, 1e-6)  # Ensure scale is never zero or negative
            return np.abs(invgamma.cdf(self.eps_q, a=self.eps_nu/2, 
                                       scale=scale_value) - sigma_hat)
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
