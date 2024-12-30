import numpy as np

from scipy.stats import gamma
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

from params import TreeParams

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

    def sigma2_prior_icdf(self, **kwargs):
        """
        Compute the prior for the noise variance.
        """
        pass

def _get_lambda(nu, sigma_hat, q):
    """
    Compute the lambda parameter for the noise variance prior.
    Find lambda such that x ~ Gamma(nu/2, nu/(2*lambda) and P(x < q) = sigma_hat.

    Parameters:
    - nu: float
        Shape parameter for the noise variance prior.
    - sigma_hat: float
        Variance of the noise.
    - q: float
        Quantile to compute.

    """

    def objective(l):
        return np.abs(gamma.cdf(q, nu/2, scale=nu/(2*l)) - sigma_hat)

    res = minimize_scalar(objective)
    return res.x


class DefaultBARTPrior(BARTPrior):
    """
    Default implementation of the BART priors.
    """
    def __init__(self, alpha: float, beta: float, mu_func, sigma2_func, lambda_noise: float, nu_noise: float):
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
        self.alpha = alpha
        self.beta = beta
        self.mu_func = mu_func
        self.sigma2_func = sigma2_func
        self.lambda_noise = lambda_noise
        self.nu_noise = nu_noise

    def tree_log_prior(self, tree_params: TreeParams):
        d = np.ceil(np.log2(np.arange(len(tree_params.vars)) + 2)) - 1
        log_p_split = np.log(self.alpha) - self.beta * np.log(1 + d)
        log_probs = np.where(self.var == -1, np.log(1 - np.exp(log_p_split)), log_p_split)
        return np.sum(log_probs[~np.isnan(tree_params.var)])
    
    def sigma2_prior_icdf(self, X, y,specification, q=0.9, nu=3): # Change to add hyperparameters
        """
        Compute the prior for the noise variance.

        According to the BART paper, the noise variance is drawn from an inverse chi-squared distribution.
        We select the parameters of the gamma distribution: alpha and beta in a data dependent way.


        """
        # sigma hat is the variance of y under the naive specification and the variance of the residuals under the "linear" specification
        if specification == "naive":
            sigma_hat = np.std(y)
        elif specification == "linear":
            # Fit a linear model to the data
            model = LinearRegression().fit(X, y)
            y_hat = model.predict(X)
            residuals = y - y_hat
            sigma_hat = np.std(residuals)
        else:
            raise ValueError("Invalid specification for the noise variance prior.")
        
        lambda_prior = _get_lambda(nu, sigma_hat, q)

        alpha = nu / 2
        beta = nu / (2 * lambda_prior)
        return alpha, beta

        


        

all_priors = {"default" : DefaultBARTPrior}