import math
import numpy as np
from scipy.stats import invgamma, chi2
from sklearn.linear_model import LinearRegression
from numba import njit

from .params import Parameters
from .moves import Move
from .util import Dataset, GIG

import logging

logger = logging.getLogger(__name__)

# Standalone Numba-optimized functions

@njit(cache=True) 
def _get_resid_all(leaf_ids, node_counts, residuals):
    resid_all = np.zeros(node_counts, dtype=np.float32)
    leaves = np.zeros(node_counts, dtype=np.bool_)
    for i in range(len(leaf_ids)):
        leaf_sample = leaf_ids[i]
        resid_all[leaf_sample] += residuals[i]
        # Here, we suppose a leaf contains at least one sample
        # We need to record leaves because resid_all[node] may add to 0.0
        leaves[leaf_sample] = True
    return resid_all, leaves

@njit(cache=True)
def _single_tree_resample_leaf_vals(leaf_ids, sample_n_in_node, residuals, eps_sigma2, f_sigma2, random_normal_p):
    """
    Numba-optimized function to resample leaf values using leaf_ids and sample_n_in_node.
    """
    node_counts = len(sample_n_in_node)
    resid_all, leaves = _get_resid_all(leaf_ids, node_counts, residuals)
        
    n_leaves = 0
    for node in range(node_counts):
        if leaves[node]:
            n_leaves += 1
            
    noise_ratio = eps_sigma2 / f_sigma2
    
    # Compute posterior parameters only for leaf nodes
    post_cov_diag = np.zeros(n_leaves, dtype=np.float32)
    post_mean = np.zeros(n_leaves, dtype=np.float32)
    
    leaf_idx = 0
    for node in range(node_counts):
        if leaves[node]:
            post_cov_diag[leaf_idx] = eps_sigma2 / (sample_n_in_node[node] + noise_ratio)
            post_mean[leaf_idx] = resid_all[node] / (sample_n_in_node[node] + noise_ratio)
            leaf_idx += 1
    
    leaf_params_new = np.sqrt(post_cov_diag) * random_normal_p + post_mean
    return leaf_params_new

@njit(cache=True)
def _resample_leaf_vals_numba(leaf_basis, residuals, eps_sigma2, f_sigma2, random_normal_p):
    """
    Numba-optimized function to resample leaf values.
    """
    p = leaf_basis.shape[1]
    # Explicitly convert boolean array to float32
    num_lbs = leaf_basis
    post_cov = np.linalg.inv(
        num_lbs.T @ num_lbs / eps_sigma2 + np.eye(p) / f_sigma2
        ).astype(np.float32)
    post_mean = post_cov @ num_lbs.T @ residuals / eps_sigma2
    
    leaf_params_new = np.sqrt(np.diag(post_cov)) * random_normal_p + post_mean
    return leaf_params_new

@njit(cache=True)
def _single_tree_log_marginal_lkhd_numba(leaf_ids, sample_n_in_node, resids, eps_sigma2, f_sigma2):
    """
    Numba-optimized function to calculate log marginal likelihood when there is only one tree.
    sample_n_in_node is the number of samples in each node, which is not limited to the samples counts in the leaves.
    """
    # Calculate counts for each leaf
    node_counts = len(sample_n_in_node)
    resid_all, leaves = _get_resid_all(leaf_ids, node_counts, resids)
    
    ls_resids = np.sum(resids ** 2)
    ridge_bias = 0.0
    logdet = 0.0
    noise_ratio = eps_sigma2 / f_sigma2
    
    for node in range(node_counts):
        if leaves[node]:
            logdet += math.log(sample_n_in_node[node] / noise_ratio + 1.0)
            ls_resids -= (resid_all[node] ** 2) / sample_n_in_node[node]
            ridge_bias += (resid_all[node] ** 2) / (sample_n_in_node[node] * (sample_n_in_node[node] / noise_ratio + 1.0))

    return - (logdet + (ls_resids + ridge_bias) / eps_sigma2) / 2

@njit(cache=True)
def _trees_log_marginal_lkhd_numba(leaf_basis, resids, eps_sigma2, f_sigma2):
    """
    Numba-optimized function to calculate log marginal likelihood when there are multiple trees.
    This function uses SVD for the leaf basis matrix.
    """
    # Explicitly convert boolean array to float32
    leaf_basis_float = leaf_basis.astype(np.float32)
    
    # Now use the float32 array with SVD
    U, S, _ = np.linalg.svd(leaf_basis_float, full_matrices=False)
    noise_ratio = eps_sigma2 / f_sigma2
    logdet = np.sum(np.log(S ** 2 / noise_ratio + 1))
    resid_u_coefs = U.T @ resids
    resids_u = U @ resid_u_coefs
    ls_resids = np.sum((resids - resids_u) ** 2)
    ridge_bias = np.sum(resid_u_coefs ** 2 / (S ** 2 / noise_ratio + 1))
    return - (logdet + (ls_resids + ridge_bias) / eps_sigma2) / 2

@njit(cache=True)
def _trees_log_prior_numba(tree_vars, alpha, beta, quick_decay=False):
    # Calculate depth for each node
    d = np.ceil(np.log2(np.arange(len(tree_vars)) + 2)) - 1
    # Calculate log probability of split
    log_p_split = np.log(alpha) - beta * np.log(1 + d) if not quick_decay else np.log(alpha) * d
    
    # Use loops instead of vectorized operations for better performance with Numba
    #   and better readability
    log_prior = 0.0
    for i in range(len(tree_vars)):
        if tree_vars[i] == -1:  # Leaf node
            log_prior += np.log(1 - np.exp(log_p_split[i]))
        elif tree_vars[i] != -2:  # Split node (not leaf and not empty)
            log_prior += log_p_split[i]
    
    return log_prior

class TreesPrior:
    """
    Prior for tree structure and leaf values.

    Attributes:
        n_trees (int): Number of trees.
        alpha (float): Alpha parameter for the tree prior.
        beta (float): Beta parameter for the tree prior.
        f_k (float): Scaling factor for the variance of the leaf parameters.
        f_sigma2 (float): Variance of the leaf parameters.
        generator: Random number generator.
    """
    def __init__(self, n_trees, tree_alpha, tree_beta, f_k, generator, quick_decay: bool = False):
        self.n_trees = n_trees
        self.alpha = tree_alpha
        self.beta = tree_beta
        self.f_k = f_k
        self.f_sigma2 = 0.25 / (self.f_k ** 2 * n_trees)
        self.generator = generator
        # If True, use quick decay parameterization for split probabilities
        self.quick_decay = quick_decay
        if quick_decay:
            if tree_alpha <= 0.0 or tree_alpha >= 0.5:
                raise ValueError("tree_alpha must be between 0.0 and 0.5 for quick decay parameterization. See Rockova and Saha (2018) for more details.")
                
    def resample_leaf_vals(self, bart_params : Parameters, data_y, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees.
        
        This function updates the leaf parameters by resampling from the posterior
        distribution given the current residuals and leaf basis.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters.
        data_y (numpy.ndarray): The target values array.
        tree_ids : list or array-like
            A list or array of tree indices for which the leaf values are to be resampled.

        Returns:
        --------
        leaf_params_new : numpy.ndarray
            The resampled leaf parameters.
        """
        residuals = data_y - bart_params.evaluate(all_except=tree_ids)
        if len(tree_ids) == 1:
            # For single tree, we can use the optimized function with leaf_ids
            tree = bart_params.trees[tree_ids[0]]
            leaf_ids = tree.leaf_ids
            return _single_tree_resample_leaf_vals(
                leaf_ids,
                tree.n,
                residuals,
                eps_sigma2=bart_params.global_params["eps_sigma2"][0],
                f_sigma2=self.f_sigma2,
                random_normal_p=self.generator.standard_normal(size=len(tree.leaves))
            )
        else:
            leaf_basis = bart_params.leaf_basis(tree_ids)

            leaf_params_new = _resample_leaf_vals_numba(
                leaf_basis,
                residuals,
                eps_sigma2=bart_params.global_params["eps_sigma2"],
                f_sigma2 = self.f_sigma2,
                random_normal_p = self.generator.standard_normal(size=leaf_basis.shape[1])
            )
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
            log_prior += _trees_log_prior_numba(tree.vars, self.alpha, self.beta, self.quick_decay)
        return log_prior
    
    def trees_log_prior_ratio(self, move : Move):
        """Calculate log prior ratio for proposed move"""
        log_prior_current = self.trees_log_prior(move.current, move.trees_changed)
        log_prior_proposed = self.trees_log_prior(move.proposed, move.trees_changed)
        return log_prior_proposed - log_prior_current

class GlobalParamPrior:
    """
    Prior for global parameters (noise variance).
    
        Args:
        eps_q (float, optional): Quantile used for setting the hyperprior for noise sigma2. Defaults to 0.9.
        eps_nu (float, optional): Inverse chi-squared nu hyperparameter for noise sigma2. Defaults to 3.
        specification (str, optional): Specification for a data-driven initial estimate for noise sigma2. Defaults to "linear".
            
    Attributes:
        eps_q (float): Quantile for noise variance prior
        eps_nu (float): Degrees of freedom for noise variance prior
        eps_lambda (float): Scale parameter for noise variance
        specification (str): Method for initial variance estimate
        generator: Random number generator
    """
    def __init__(self, eps_q, eps_nu, specification, generator, dirichlet_prior, s_alpha, fixed_eps_sigma2=None):
        self.eps_q = eps_q
        self.eps_nu = eps_nu
        self.eps_lambda : float
        self.specification = specification
        self.generator = generator
        self.dirichlet_prior = dirichlet_prior
        self.s_alpha = s_alpha
        self.fixed_eps_sigma2 = fixed_eps_sigma2

    def fit_hyperparameters(self, data):
        """Fit the prior hyperparameters to the data"""
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
        if self.fixed_eps_sigma2 is not None:
            eps_sigma2 = np.array([self.fixed_eps_sigma2], dtype=float)
        else:
            self.fit_hyperparameters(data)
            eps_sigma2 = self._sample_eps_sigma2(data.y)
        global_params = {"eps_sigma2": eps_sigma2}
        if self.dirichlet_prior:
            s = np.ones(data.X.shape[1]) / data.X.shape[1]
            global_params["s"] = s
            global_params["s_cumsum"] = np.cumsum(s)
        return global_params
    
    def resample_global_params(self, bart_params : Parameters, data_y):
        """
        Resamples the global parameters for the BART model.

        This function resamples the global parameters, specifically `eps_sigma2`, 
        based on the provided BART parameters.

        Args:
            bart_params (Parameters): An instance of the Parameters class containing 
                                      the data and model parameters for BART.
            data_y (numpy.ndarray): The target values array.

        Returns:
            dict: A dictionary containing the resampled global parameters.
        """
        global_params = dict({})
        if self.fixed_eps_sigma2 is not None:
            global_params["eps_sigma2"] = np.array([self.fixed_eps_sigma2], dtype=float)
        else:
            global_params["eps_sigma2"] = self._sample_eps_sigma2(data_y - bart_params.evaluate())
        if self.dirichlet_prior:
            s, s_cumsum = self._resample_s(bart_params)
            global_params["s"] = s
            global_params["s_cumsum"] = s_cumsum
        return global_params
    
    def _resample_s(self, bart_params : Parameters):
        """
        Resample the split probabilities s.

        Args:
            bart_params (Parameters): An instance of the Parameters class containing the data and model parameters for BART.

        Returns:
            tuple: (s, s_cumsum) where s is the resampled probabilities and s_cumsum is the precomputed cumulative sum.
        """
        if not self.dirichlet_prior:
            raise ValueError("Dirichlet prior is not enabled.")
        # vars_histogram now returns numpy array directly
        vars_histogram_array = bart_params.vars_histogram
        p = len(vars_histogram_array)
        s = self.generator.dirichlet(self.s_alpha / p + vars_histogram_array)
        s_cumsum = np.cumsum(s)
        return s, s_cumsum
    
    def _fit_eps_lambda(self, data : Dataset, specification: str) -> float:
        """
        Compute the lambda parameter for the noise variance prior.
        Find lambda such that x ~ Gamma(nu/2, nu/(2*lambda) and P(x < q) = sigma_hat.
        """
        sigma_hat : float
        if specification == "naive":
            sigma_hat = float(np.std(data.y))
        elif specification == "linear":
            # Fit a linear model to the data
            model = LinearRegression().fit(data.X, data.y)
            y_hat = model.predict(data.X)
            resids = data.y - y_hat
            sigma_hat = float(np.std(resids))
        else:
            raise ValueError("Invalid specification for the noise variance prior.")
        
        if sigma_hat <= 1e-5:
            logger.warning(f"Estimated sigma_hat {sigma_hat} is too small, returning a small positive value (1e-5) to avoid numerical issues. Specification: {specification}.")
            sigma_hat = float(1e-5)
        
        # chi2.ppf suffices
        c = chi2.ppf(1 - self.eps_q, df=self.eps_nu).item()
        eps_lambda_val = (sigma_hat**2 * c) / self.eps_nu
        return eps_lambda_val

    def _sample_eps_sigma2(self, residuals):
        """
        Sample noise variance parameter.

        Parameters:
            residuals: Model residuals

        Returns:
            float: Sampled noise variance
        """
        n = len(residuals)
        # Convert to inverse gamma params
        prior_alpha = self.eps_nu / 2
        prior_beta = self.eps_nu * self.eps_lambda / 2
        post_alpha = prior_alpha + n / 2
        post_beta = prior_beta + np.sum(residuals ** 2) / 2
        # size=1 returns a 1-element numpy array, so we use [0] for indexing
        eps_sigma2 = invgamma.rvs(a=post_alpha, scale=post_beta, size=1, random_state = self.generator)
        if eps_sigma2[0] <= 1e-8:
            # logger.info(f"Sampled eps_sigma2 is non-positive, returning 1e-8. Sampled eps_sigma2: {eps_sigma2}, prior_alpha: {prior_alpha}, prior_beta: {prior_beta}, post_alpha: {post_alpha}, post_beta: {post_beta}")
            eps_sigma2[0] = 1e-8
        return eps_sigma2

class BARTLikelihood:
    """
    BART likelihood calculations and MCMC utilities.

    Combines tree and global parameter priors for full model inference.
    """
    def __init__(self, f_sigma2 : float): 
        """
        f_sigma2 (float): Variance of the leaf parameters.
        """
        self.f_sigma2 = f_sigma2

    def trees_log_marginal_lkhd(self, bart_params : Parameters, data_y, tree_ids):
        """
        Calculate the log marginal likelihood of the trees in a BART model.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters,
            including data, global parameters, and methods for evaluation and leaf basis computation.
        data_y (numpy.ndarray): The target values array.
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
        resids = (data_y - bart_params.evaluate(all_except=tree_ids))
        if len(tree_ids) == 1:
            # For single tree, we can use the optimized function with leaf_ids
            tree = bart_params.trees[tree_ids[0]]
            try:
                return _single_tree_log_marginal_lkhd_numba(
                    tree.leaf_ids, 
                    tree.n,
                    resids, 
                    bart_params.global_params["eps_sigma2"][0], 
                    self.f_sigma2
                )     
            except Exception as e:
                logger.error(f"Error calculating likelihood for tree {tree_ids[0]}: {e}")
                logger.error(f"Data: {data_y}")
                logger.error(f"Related information: leaf_ids: {tree.leaf_ids}; sample_n_in_node: {tree.n}; residuals: {resids}; eps_sigma2: {bart_params.global_params['eps_sigma2'][0]}; f_sigma2: {self.f_sigma2}", exc_info=True)
                raise ValueError("Error calculating likelihood for single tree. Check the tree structure and residuals.")
        else:
            leaf_basis = bart_params.leaf_basis(tree_ids)
            return _trees_log_marginal_lkhd_numba(
                leaf_basis, 
                resids, 
                bart_params.global_params["eps_sigma2"], 
                self.f_sigma2
            )
            

    def trees_log_marginal_lkhd_ratio(self, move : Move, data_y, marginalize: bool=False):
        """
        Compute the ratio of marginal likelihoods for a given move.

        Parameters:
        - move: Move
            The move to compute the marginal likelihood ratio for.
        - data_y (numpy.ndarray): The target values array.
        - marginalize: bool
            Whether to marginalize over the ensemble.

        Returns:
        - float
            Marginal likelihood ratio.
        """
        if not marginalize:
            log_lkhd_current = self.trees_log_marginal_lkhd(move.current, data_y, move.trees_changed)
            log_lkhd_proposed = self.trees_log_marginal_lkhd(move.proposed, data_y, move.trees_changed)
        else:
            log_lkhd_current = self.trees_log_marginal_lkhd(move.current, data_y, np.arange(move.current.n_trees))
            log_lkhd_proposed = self.trees_log_marginal_lkhd(move.proposed, data_y, np.arange(move.current.n_trees))
        return log_lkhd_proposed - log_lkhd_current

class ComprehensivePrior:
    def __init__(self, n_trees, tree_alpha, tree_beta, f_k, eps_q, eps_nu, 
                 specification, generator, dirichlet_prior, quick_decay, s_alpha, fixed_eps_sigma2=None):
        self.tree_prior = TreesPrior(int(n_trees), tree_alpha, tree_beta, f_k, generator, quick_decay=quick_decay)
        self.global_prior = GlobalParamPrior(eps_q, eps_nu, specification, generator, dirichlet_prior, s_alpha, fixed_eps_sigma2=fixed_eps_sigma2)
        self.likelihood = BARTLikelihood(self.tree_prior.f_sigma2)

class ProbitPrior:
    """
    BART Prior for binary classification tasks.
    """
    def __init__(self, n_trees, tree_alpha, tree_beta, f_k, generator, quick_decay: bool = False):
        self.tree_prior = TreesPrior(n_trees, tree_alpha, tree_beta, f_k, generator, quick_decay=quick_decay)
        self.likelihood = BARTLikelihood(self.tree_prior.f_sigma2)

class LogisticTreesPrior(TreesPrior):
    """
    Prior for logistic regression trees.
    
    Inherits from TreesPrior and overrides the resample_leaf_vals method to handle logistic regression.
    """
    def __init__(self, n_trees, tree_alpha, tree_beta, c, d, generator, parent, quick_decay: bool = False):
        """
        Initialize the logistic trees prior with parameters.
        
        Parameters:
        -----------
        n_trees : int
            Number of trees in the BART model.
        tree_alpha : float
            Alpha parameter for the tree prior.
        tree_beta : float
            Beta parameter for the tree prior.
        c : float
            Parameter for the GIG distribution.
        d : float
            Parameter for the GIG distribution.
        generator : numpy.random.Generator, optional
            Random number generator for reproducibility (default is np.random.default_rng()).
        """
        if c == 0.0 or d == 0.0:
            raise ValueError("Parameters c and d must be provided for logistic regression prior.")
        self.c = c
        self.d = d
        self.parent = parent  # LogisticPrior
        super().__init__(n_trees, tree_alpha, tree_beta, f_k=2.0, generator=generator, quick_decay=quick_decay)

    def set_latents(self, latents):
        self.latents = latents
        
    def resample_leaf_vals(self, bart_params : Parameters, data_y, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees.
        
        This function updates the leaf parameters by resampling from the posterior
        distribution given the current residuals and leaf basis.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters.
        data_y (numpy.ndarray): The target values array.
        tree_ids : list or array-like
            A list or array of tree indices for which the leaf values are to be resampled.

        Returns:
        --------
        leaf_params_new : numpy.ndarray
            The resampled leaf parameters.
        """
        # Cached values of rh, sh and even pi_h from parent may be used. 
        # The speedup is small though (2-4%)
        if self.parent is None:
            raise ValueError("Parent LogisticPrior not set")
        if self.parent.param is not bart_params:
            print("Error: BART Parameter used by calculated values is not the same as that provided to LogisticTreesPrior.resample_leaf_vals, this may lead to incorrect results.")
            print("Please contact the developer to see if we need fall back to re-calculating rh, sh and pi_h.")
            raise ValueError("BART Parameter mismatch")
        if self.parent.rh is None or self.parent.sh is None or self.parent.pi_h is None:
            raise ValueError("Parent LogisticPrior attributes (rh, sh, pi_h) not initialized")
        rh = self.parent.rh
        sh = self.parent.sh
        pi_h = self.parent.pi_h
        
        leaf_params_new = np.zeros(rh.shape[0])
        for i in range(leaf_params_new.shape[0]):
            leaf_params_new[i] = pi_h[i] * GIG.rvs_gig_scalar(
                    -self.c + rh[i], 2 * self.d, 2 * sh[i],
                    generator=self.generator
                ) + (1 - pi_h[i]) * GIG.rvs_gig_scalar(
                    self.c + rh[i], 0, 2 * (self.d + sh[i]),
                    generator=self.generator
                )
        
        # vectorized calculation of normalizing constants
        # and generation of RVs are actually slower than the loops above
        
        return np.log(leaf_params_new)
    
_log_gig_const_global = GIG.log_gig_normalizing_constant_numba

class LogisticLikelihood(BARTLikelihood):
    """
    Likelihood for logistic regression tasks.
    
    Inherits from BARTLikelihood and overrides the trees_log_marginal_lkhd method.
    """
    def __init__(self, c, d, parent=None):
        """
        Initialize the logistic likelihood with parameters c and d.

        Parameters:
        -----------
        c : float
            Parameter for the GIG distribution.
        d : float
            Parameter for the GIG distribution.
        """
        self.c = c
        self.d = d
        self.parent: LogisticPrior | None = parent  # LogisticPrior
        # f_sigma2 useless
        super().__init__(f_sigma2=0.0)

    def set_latents(self, latents):
        self.latents = latents
        
    @staticmethod
    @njit(cache=True)
    def _get_rh_sh(leaf_ids, latent_tree_product, data_y, node_counts, leaves):
        rh_all = np.zeros(node_counts, dtype=np.float64)
        sh_all = np.zeros(node_counts, dtype=np.float64)

        for i in range(len(leaf_ids)):
            leaf_sample = leaf_ids[i]
            rh_all[leaf_sample] += data_y[i]
            sh_all[leaf_sample] += latent_tree_product[i]

        return rh_all[leaves], sh_all[leaves]

    @staticmethod
    @njit(cache=True)
    def _trees_log_marginal_lkhd_numba_backend(c, d, rh, sh):
        log_likelihood = np.zeros(len(rh))
        pi_h = np.zeros(len(rh))
        for i in range(len(rh)):
            log_z1 = _log_gig_const_global(-c + rh[i], 2 * d, 2 * sh[i])
            log_z2 = _log_gig_const_global(c + rh[i], 0, 2 * (d + sh[i]))
            max_log_z = max(log_z1, log_z2)
            min_log_z = min(log_z1, log_z2)
            log_z1_p_z2 = max_log_z + math.log1p(math.exp(min_log_z - max_log_z))
            log_likelihood[i] = log_z1_p_z2 - (math.log(2) + _log_gig_const_global(c, 0, 2 * d))
            pi_h[i] = math.exp(log_z1 - log_z1_p_z2)
        return log_likelihood.sum(), pi_h

    def trees_log_marginal_lkhd(self, bart_params : Parameters, data_y, tree_ids):
        """
        Calculate the log marginal likelihood of the trees in a logistic regression BART model.

        Parameters:
        -----------
        bart_params : Parameters
            An instance of the Parameters class containing the BART model parameters.
        data_y (bool, numpy.ndarray): The target values array that contain boolean values representing for matching the category or not.
        tree_ids : list or array-like
            A list or array of tree identifiers for which the log marginal likelihood is to be computed.

        Returns:
        --------
        float
            The log marginal likelihood of the specified trees.
        """
        # here, we assume tree_ids contain only one tree
        if len(tree_ids) != 1:
            raise ValueError("Logistic likelihood only supports single tree evaluation.")
        
        tree_eval = bart_params.evaluate(all_except=tree_ids)
        # dim of tree_eval is (n_samples)
        latent_tree_product = self.latents * np.exp(tree_eval)
        # dim of latent_tree_product is (n_samples)
        
        tree = bart_params.trees[tree_ids[0]]
        
        rh, sh = self._get_rh_sh(
            tree.leaf_ids, latent_tree_product, data_y, len(tree.vars), tree.leaves
        )
        if self.parent is not None:
            self.parent.rh = rh
            self.parent.sh = sh
            self.parent.param = bart_params
        
        # assert np.allclose(rh, rhn) and np.allclose(sh, shn), "Mismatch in rh and sh calculation."
        log_likelihood_sum, pi_h = self._trees_log_marginal_lkhd_numba_backend(self.c, self.d, rh, sh)
        if self.parent is not None:
            self.parent.pi_h = pi_h

        return log_likelihood_sum

class LogisticPrior:
    """
    BART Prior for logistic classification tasks.
    """
    def __init__(self, n_trees=25, tree_alpha=0.95, tree_beta=2.0, c=0.0, d=0.0, generator=np.random.default_rng(), quick_decay: bool = False):
        if c == 0.0 or d == 0.0:
            a0 = 3.5 / math.sqrt(2)
            c = n_trees/(a0 ** 2) + 0.5
            d = n_trees/(a0 ** 2)
        
        self.tree_prior = LogisticTreesPrior(n_trees, tree_alpha, tree_beta, c, d, generator, parent=self, quick_decay=quick_decay)
        self.likelihood = LogisticLikelihood(c, d, parent=self)
        
        # Placeholders for values reused in resampling - will be set during likelihood calculation
        self.rh: np.ndarray | None = None  
        self.sh: np.ndarray | None = None
        self.pi_h: np.ndarray | None = None
        self.param: Parameters | None = None  # Will hold Parameters object
    
    def set_latents(self, latents):
        self.tree_prior.set_latents(latents)
        self.likelihood.set_latents(latents)
