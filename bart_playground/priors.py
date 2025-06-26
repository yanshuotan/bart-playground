
import math
from matplotlib.pylab import f
import numpy as np
from scipy.stats import invgamma, chi2, gamma
from sklearn.linear_model import LinearRegression
from numba import njit

from .params import Parameters
from .moves import Move
from .moves import Break, Combine, Birth, Death
from .util import Dataset, GIG

# Standalone Numba-optimized functions
@njit
def _resample_leaf_vals_numba(leaf_basis, residuals, eps_sigma2, f_sigma2, random_normal_p):
    """
    Numba-optimized function to resample leaf values.
    """
    p = leaf_basis.shape[1]
    # Explicitly convert boolean array to float64
    num_lbs = leaf_basis.astype(np.float64)
    post_cov = np.linalg.inv(num_lbs.T @ num_lbs / eps_sigma2 + np.eye(p) / f_sigma2)
    post_mean = post_cov @ num_lbs.T @ residuals / eps_sigma2
    
    leaf_params_new = np.sqrt(np.diag(post_cov)) * random_normal_p + post_mean
    return leaf_params_new

@njit
def _trees_log_marginal_lkhd_numba(leaf_basis, resids, eps_sigma2, f_sigma2):
    """
    Numba-optimized function to calculate log marginal likelihood.
    """
    # Explicitly convert boolean array to float64
    leaf_basis_float = leaf_basis.astype(np.float64)
    
    # Now use the float64 array with SVD
    U, S, _ = np.linalg.svd(leaf_basis_float, full_matrices=False)
    noise_ratio = eps_sigma2 / f_sigma2
    logdet = np.sum(np.log(S ** 2 / noise_ratio + 1))
    resid_u_coefs = U.T @ resids
    resids_u = U @ resid_u_coefs
    ls_resids = np.sum((resids - resids_u) ** 2)
    ridge_bias = np.sum(resid_u_coefs ** 2 / (S ** 2 / noise_ratio + 1))
    return - (logdet + (ls_resids + ridge_bias) / eps_sigma2) / 2

@njit
def _trees_log_prior_numba(tree_vars, alpha, beta):
    """
    Numba-optimized function to calculate log prior probability of a tree.
    
    Parameters:
    -----------
    tree_vars : numpy.ndarray
        An array of variables used for splitting at each node.
    alpha : float
        Alpha parameter for the tree prior.
    beta : float
        Beta parameter for the tree prior.
        
    Returns:
    --------
    float
        The log prior probability of the tree.
    """
    # Calculate depth for each node
    d = np.ceil(np.log2(np.arange(len(tree_vars)) + 2)) - 1
    # Calculate log probability of split
    log_p_split = np.log(alpha) - beta * np.log(1 + d)
    
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
    def __init__(self, n_trees=200, tree_alpha=0.95, tree_beta=2.0, f_k=2.0, generator=np.random.default_rng()):
        self.n_trees = n_trees
        self.alpha = tree_alpha
        self.beta = tree_beta
        self.f_k = f_k
        self.f_sigma2 = 0.25 / (self.f_k ** 2 * n_trees)
        self.generator = generator

    def update_f_sigma2(self, n_trees):
        """
        Update f_sigma2 based on the current number of trees.
        """
        self.f_sigma2 = 0.25 / (self.f_k ** 2 * n_trees)

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
        leaf_basis = bart_params.leaf_basis(tree_ids)
        
        leaf_params_new = _resample_leaf_vals_numba(
            leaf_basis, 
            residuals, 
            eps_sigma2 = bart_params.global_params["eps_sigma2"], 
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
            log_prior += _trees_log_prior_numba(tree.vars, self.alpha, self.beta)
        return log_prior
    
    def trees_log_prior_ratio(self, move : Move):
        """Calculate log prior ratio for proposed move"""
        log_prior_current = self.trees_log_prior(move.current, move.trees_changed)
        if isinstance(move, Break) or isinstance(move, Birth):
            trees_proposed_ids = move.trees_changed + [-1]
        elif isinstance(move, Combine) or isinstance(move, Death):
            trees_proposed_ids = [move.trees_changed[0] if move.trees_changed[0] < move.trees_changed[1] else move.trees_changed[0] - 1]
        else:
            trees_proposed_ids = move.trees_changed
        log_prior_proposed = self.trees_log_prior(move.proposed, trees_proposed_ids)
        return log_prior_proposed - log_prior_current

class GlobalParamPrior:
    """
    Prior for global parameters (noise variance).
    
        Args:
        eps_q (float, optional): Quantile used for setting the hyperprior for noise sigma2. Defaults to 0.9.
        eps_nu (float, optional): Inverse chi-squared nu hyperparameter for noise sigma2. Defaults to 3.
        specification (str, optional): Specification for a data-driven initial estimate for noise sigma2. Defaults to "linear".
        theta_0(int, optional): Control the theta which is the poisson parameter for the number of trees. Defaults to 200.
        theta_df(int, optional): Control the theta degree of freedom. Defaults to 100.
        
    Attributes:
        eps_q (float): Quantile for noise variance prior
        eps_nu (float): Degrees of freedom for noise variance prior
        eps_lambda (float): Scale parameter for noise variance
        specification (str): Method for initial variance estimate
        generator: Random number generator
    """
    def __init__(self, eps_q=0.9, eps_nu=3.0, theta_0 : int=200, theta_df :int = 100,
                 specification="linear", generator=np.random.default_rng()):
        self.eps_q = eps_q
        self.eps_nu = eps_nu
        self.eps_lambda : float
        self.theta_0 = theta_0
        self.theta_df = theta_df
        self.specification = specification
        self.generator = generator

    def fit_hyperparameters(self, data):
        """Fit the prior hyperparameters to the data"""
        self.eps_lambda = self._fit_eps_lambda(data)
        
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
                - ntree_theta (int): The initial theta value for the number of trees.
        """
        self.fit_hyperparameters(data)
        eps_sigma2 = self._sample_eps_sigma2(data.y)
        ntree_theta = self.theta_0
        return {"eps_sigma2" : eps_sigma2,
                "ntree_theta" : ntree_theta}
    
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
        eps_sigma2 = self._sample_eps_sigma2(data_y - bart_params.evaluate())
        if self.theta_df == np.inf:
            # If theta_df is infinite, we don't need to sample ntree_theta
            ntree_theta = self.theta_0
        else:
            ntree_theta = self._sample_ntree_theta(bart_params.n_trees)
        return {"eps_sigma2" : eps_sigma2,
                "ntree_theta" : ntree_theta}
    
    def _fit_eps_lambda(self, data : Dataset, specification="linear") -> float:
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
        eps_sigma2 = invgamma.rvs(a=post_alpha, scale=post_beta, size=1, random_state = self.generator)# [0]
        return eps_sigma2
    
    def _sample_ntree_theta(self, n_trees):
        alpha_posterior = self.theta_df / 2 + n_trees
        beta_posterior = self.theta_df / (2 * self.theta_0) + 1

        theta_new = gamma.rvs(alpha_posterior, scale=1 / beta_posterior)
        return theta_new
    
class TreeNumPrior:
    def __init__(self, prior_type="poisson",  gp_eta=0.0, com_nu=1.0):
        """
        Initialize the TreeNumPrior.

        Parameters:
        - prior_type: str, default="poisson"
            The type of prior to use for the number of trees. Options are:
            - "poisson": Poisson distribution (default).
            - "bernoulli": Bernoulli distribution with m = 1 or 2, each with probability 0.5.
            - "generalized_poisson": Generalized Poisson distribution with parameters theta and gp_eta.
        - gp_eta: float, parameter for Generalized Poisson (default 0.0, |gp_eta| < 1)
        - com_nu: float, parameter ν for COM-Poisson (default 1.0)
        """
        if prior_type not in ["poisson", "bernoulli", "generalized_poisson", "com_poisson"]:
            raise ValueError("Invalid prior_type. Must be 'poisson', 'bernoulli', 'generalized_poisson', or 'com_poisson'.")
        if prior_type == "generalized_poisson" and abs(gp_eta) >= 1:
            raise ValueError("For generalized_poisson, |gp_eta| must be less than 1.")
        self.prior_type = prior_type
        self.gp_eta = gp_eta
        self.com_nu = com_nu

    def tree_num_log_prior(self, bart_params: Parameters, ntree_theta):
        m = bart_params.n_trees

        if self.prior_type == "poisson":
            theta = ntree_theta
            log_prior = m * np.log(theta) - math.lgamma(m + 1) # Omit "-theta"
        elif self.prior_type == "bernoulli":
            log_prior = np.log(0.5)
        elif self.prior_type == "generalized_poisson":
            theta = ntree_theta
            gp_eta = self.gp_eta
            if theta + gp_eta * m <= 0:
                # Avoid invalid log or negative probability
                return -np.inf # TODO: Handle this case more gracefully if needed
            # log P(X=m) = log(theta) + (m-1)*log(theta+gp_eta*m) - theta - gp_eta*m - lgamma(m+1)
            log_prior = (
                np.log(theta)
                + (m - 1) * np.log(theta + gp_eta * m)
                - theta
                - gp_eta * m
                - math.lgamma(m + 1)
            )
        elif self.prior_type == "com_poisson":
            theta = ntree_theta
            nu = self.com_nu
            # Omit normalizing constant because it does not depend on m
            log_prior = m * np.log(theta) - nu * math.lgamma(m + 1) 
        return log_prior

    def tree_num_log_prior_ratio(self, move: Move):
        log_prior_current = self.tree_num_log_prior(move.current, move.current.global_params["ntree_theta"])
        log_prior_proposed = self.tree_num_log_prior(move.proposed, move.proposed.global_params["ntree_theta"])
        return log_prior_proposed - log_prior_current
    
class LeafValPrior:
    def __init__(self, tau_k = 2.0):
        self.tau_k = tau_k

    def tau_square_inv(self, m):
        """
        Computes the inverse of tau squared (1 / tau^2)
        """
        return 4*(self.tau_k**2)*m
    
    def leaf_vals_squared_sum(self, bart_params, trees_changed):
        """
        Computes the sum of squared leaf values for all trees in bart_params.trees,
        excluding the trees specified in trees_changed.
        """    
        return np.nansum([
            val ** 2 for idx, tree in enumerate(bart_params.trees) 
            if idx not in set(trees_changed) for val in tree.leaf_vals
        ])
    
    def total_leaf_count(self, bart_params, trees_changed):
        """
        Computes the total number of leaves across all trees in bart_params.trees,
        excluding the trees specified in trees_changed.
        """     
        return sum(
            tree.n_leaves 
            for idx, tree in enumerate(bart_params.trees) 
            if idx not in set(trees_changed)
        )


    def leaf_vals_log_prior_ratio(self, move: Move):
        m_current = move.current.n_trees
        m_proposed = move.proposed.n_trees
        
        return -0.5*self.total_leaf_count(move.current, move.trees_changed) * np.log(m_current/m_proposed) - \
            0.5*(self.tau_square_inv(m_proposed) - self.tau_square_inv(m_current)) * self.leaf_vals_squared_sum(move.current, move.trees_changed)

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
        resids = data_y - bart_params.evaluate(all_except=tree_ids)
        leaf_basis = bart_params.leaf_basis(tree_ids)
        
        # Use the standalone numba function instead
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
            if isinstance(move, Break) or isinstance(move, Birth):
                trees_proposed_ids = move.trees_changed + [-1]
            elif isinstance(move, Combine) or isinstance(move, Death):
                trees_proposed_ids = [move.trees_changed[0] if move.trees_changed[0] < move.trees_changed[1] else move.trees_changed[0] - 1]
            else:
                trees_proposed_ids = move.trees_changed
            log_lkhd_current = self.trees_log_marginal_lkhd(move.current, data_y, move.trees_changed)
            log_lkhd_proposed = self.trees_log_marginal_lkhd(move.proposed, data_y, trees_proposed_ids)
        else:
            log_lkhd_current = self.trees_log_marginal_lkhd(move.current, data_y, np.arange(move.current.n_trees))
            log_lkhd_proposed = self.trees_log_marginal_lkhd(move.proposed, data_y, np.arange(move.proposed.n_trees))
        return log_lkhd_proposed - log_lkhd_current

class ComprehensivePrior:
    def __init__(self, n_trees=200, tree_alpha=0.95, tree_beta=2.0, f_k=2.0, 
                 eps_q=0.9, eps_nu=3.0, specification="linear", generator=np.random.default_rng(),
                 theta_0=200, theta_df=100, tau_k = 2.0, tree_num_prior_type="poisson", 
                 tree_num_eta=0.0, com_nu=1.0):
        self.tree_prior = TreesPrior(n_trees, tree_alpha, tree_beta, f_k, generator)
        self.global_prior = GlobalParamPrior(eps_q, eps_nu, theta_0, theta_df, specification, generator)
        self.likelihood = BARTLikelihood(self.tree_prior.f_sigma2)
        self.tree_num_prior = TreeNumPrior(prior_type=tree_num_prior_type, gp_eta=tree_num_eta, com_nu=com_nu)
        self.leaf_val_prior = LeafValPrior(tau_k)

class ProbitPrior:
    """
    BART Prior for binary classification tasks.
    """
    def __init__(self, n_trees=200, tree_alpha=0.95, tree_beta=2.0, f_k=2.0, generator=np.random.default_rng()):
        self.tree_prior = TreesPrior(n_trees, tree_alpha, tree_beta, f_k, generator)
        self.likelihood = BARTLikelihood(self.tree_prior.f_sigma2)

class LogisticTreesPrior(TreesPrior):
    """
    Prior for logistic regression trees.
    
    Inherits from TreesPrior and overrides the resample_leaf_vals method to handle logistic regression.
    """
    def __init__(self, n_trees=200, tree_alpha=0.95, tree_beta=2.0, c = 0.0, d = 0.0, generator=np.random.default_rng(), parent = None):
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
        super().__init__(n_trees, tree_alpha, tree_beta, f_k=2.0, generator=generator)

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
        if(self.parent.param is not bart_params):
            print("Error: BART Parameter used by calculated values is not the same as that provided to LogisticTreesPrior.resample_leaf_vals, this may lead to incorrect results.")
            print("Please contact the developer to see if we need fall back to re-calculating rh, sh and pi_h.")
            raise ValueError("BART Parameter mismatch")
        rh = self.parent.rh
        sh = self.parent.sh
        pi_h = self.parent.pi_h
        
        # lb_bool = bart_params.leaf_basis(tree_ids)
        # leaf_basis = lb_bool.astype(np.float64)
        # tree_eval = bart_params.evaluate(all_except = tree_ids)
        # latent_tree_product = self.latents * np.exp(tree_eval)
        # 
        # rh = leaf_basis.T @ data_y  # Shape: (n_leaves,)
        # sh = leaf_basis.T @ latent_tree_product  # Shape: (n_leaves,)
        # 
        # pi_h = np.zeros(len(rh))
        # for i in range(len(rh)):
        #     Z1 = GIG.gig_normalizing_constant_numba(-self.c + rh[i], 2 * self.d, 2 * sh[i])
        #     Z2 = GIG.gig_normalizing_constant_numba(self.c + rh[i], 0, 2 * (self.d + sh[i]))
        #     pi_h[i] = Z1 / (Z1 + Z2)
        
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
        self.parent : LogisticPrior = parent  # LogisticPrior
        # f_sigma2 useless
        super().__init__(f_sigma2=0.0)

    def set_latents(self, latents):
        self.latents = latents
    
    @staticmethod
    @njit
    def trees_log_marginal_lkhd_numba_backend(c, d, rh, sh):
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
        
        lb_bool = bart_params.leaf_basis(tree_ids)
        leaf_basis = lb_bool.astype(np.float64)
        # leaf_basis is an array of shape (n_samples, n_leaves)
        tree_eval = bart_params.evaluate(all_except=tree_ids)
        # dim of tree_eval is (n_samples)
        latent_tree_product = self.latents * np.exp(tree_eval)
        # dim of latent_tree_product is (n_samples)

        # Vectorized computation of rh and sh for all leaves
        # rh[t] = sum of data_y values where leaf_basis[i, t] == 1
        rh = leaf_basis.T @ data_y  # Shape: (n_leaves,)

        # sh[t] = sum of latents[i] * np.exp(tree_eval)[i] where leaf_basis[i, t] == 1
        sh = leaf_basis.T @ latent_tree_product  # Shape: (n_leaves,)
        
        self.parent.rh = rh
        self.parent.sh = sh
        self.parent.param = bart_params
        log_likelihood_sum, self.parent.pi_h = self.trees_log_marginal_lkhd_numba_backend(self.c, self.d, rh, sh)

        return log_likelihood_sum

class LogisticPrior:
    """
    BART Prior for logistic classification tasks.
    """
    def __init__(self, n_trees=25, tree_alpha=0.95, tree_beta=2.0, c=0.0, d=0.0, generator=np.random.default_rng()):
        if c == 0.0 or d == 0.0:
            a0 = 3.5 / math.sqrt(2)
            c = n_trees/(a0 ** 2) + 0.5
            d = n_trees/(a0 ** 2)
        
        self.tree_prior = LogisticTreesPrior(n_trees, tree_alpha, tree_beta, c, d, generator, parent=self)
        self.likelihood = LogisticLikelihood(c, d, parent=self)
        
        # Placeholders for values reused in resampling
        self.rh = None  
        self.sh = None
        self.pi_h = None
        self.param = None
    
    def set_latents(self, latents):
        self.tree_prior.set_latents(latents)
        self.likelihood.set_latents(latents)
        