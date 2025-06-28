import numpy as np
from typing import Dict, Any, Optional, Union, List
from .agent import BanditAgent


class LinearTSAgent2(BanditAgent):
    """
    Linear Thompson Sampling agent for contextual bandits by Aouali, et al. 2023 (Mixed-Effects Thompson Sampling).
    """
    def __init__(self, n_arms: int, n_features: int, sigma: float = 1.0, 
                 prior_mean: Optional[np.ndarray] = None,
                 prior_cov: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None) -> None:
        """
        Initialize the LinearTS agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            sigma (float): Noise standard deviation.
            prior_mean (array-like, optional): Prior means of arm parameters.
            prior_cov (array-like, optional): Prior covariance of arm parameters.
            random_state (int, optional): Random state for reproducibility.
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        super().__init__(n_arms, n_features)
        self.sigma = sigma
        
        # Set prior parameters
        self.Theta0 = prior_mean if prior_mean is not None else np.zeros((n_arms, n_features))
        
        if prior_cov is not None:
            self.Sigma0 = prior_cov
            if prior_cov.ndim == 2:
                self.Sigma0 = np.array([prior_cov.copy() for _ in range(n_arms)])
        else:
            self.Sigma0 = np.array([np.eye(n_features) for _ in range(n_arms)])
        
        # Calculate precision matrices from covariance
        self.Lambda0 = np.zeros((n_arms, n_features, n_features))
        for i in range(n_arms):
            self.Lambda0[i] = np.linalg.inv(self.Sigma0[i])
        
        # Initialize sufficient statistics
        self.G = np.zeros((n_arms, n_features, n_features))  # Precision matrix
        self.B = np.zeros((n_arms, n_features))  # Weighted sum of rewards
    
    def choose_arm(self, x: Union[np.ndarray, List[List[float]]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Thompson Sampling.
        
        Parameters:
            x (array-like): Feature matrix. Can be either:
                - A matrix of shape (n_arms, n_features) where each row corresponds to an arm's features.
                - A single feature vector of shape (n_features,) or (1, n_features) to be used for all arms.
            
        Returns:
            int: The index of the selected arm.
        """
        x = np.atleast_2d(x)  # Convert to 2D array if necessary
        
        # Check if we have features per arm or a single feature vector
        if x.shape[0] == 1 and self.n_arms > 1:
            # Single feature vector for all arms - replicate it
            x = np.tile(x, (self.n_arms, 1))
        elif x.shape[0] != self.n_arms:
            raise ValueError(f"Feature matrix shape mismatch. Expected {self.n_arms} rows, got {x.shape[0]}")
        
        # Sample arm parameters and compute expected rewards
        mu = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            # Linear model posterior
            Gt = self.Lambda0[i] + self.G[i]
            Sigma_hat = np.linalg.inv(Gt)
            theta_hat = np.linalg.solve(Gt, self.Lambda0[i].dot(self.Theta0[i]) + self.B[i])
            
            # Posterior sampling
            theta_tilde = self.rng.multivariate_normal(theta_hat, Sigma_hat)
            mu[i] = x[i].dot(theta_tilde)
        
        return int(np.argmax(mu))
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "LinearTSAgent2":
        """
        Update the agent's state after observing a reward.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector for the chosen arm.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        x = np.array(x).flatten()
        
        # Update sufficient statistics
        self.G[arm] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm] += x * y / np.square(self.sigma)
        
        return self


class LinearUCBAgent(BanditAgent):
    """
    Linear Upper Confidence Bound agent for contextual bandits.
    """
    def __init__(self, n_arms: int, n_features: int, sigma: float = 1.0, 
                 prior_mean: Optional[np.ndarray] = None,
                 prior_cov: Optional[np.ndarray] = None,
                 delta: float = 0.01) -> None:
        """
        Initialize the LinearUCB agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            sigma (float): Noise standard deviation.
            prior_mean (array-like, optional): Prior means of arm parameters.
            prior_cov (array-like, optional): Prior covariance of arm parameters.
            delta (float): Confidence parameter.
        """
        super().__init__(n_arms, n_features)
        self.sigma = sigma
        self.delta = delta
        
        # Set prior parameters
        self.Theta0 = prior_mean if prior_mean is not None else np.zeros((n_arms, n_features))
        
        if prior_cov is not None:
            self.Sigma0 = prior_cov
            if prior_cov.ndim == 2:
                self.Sigma0 = np.array([prior_cov.copy() for _ in range(n_arms)])
        else:
            self.Sigma0 = np.array([np.eye(n_features) for _ in range(n_arms)])
        
        # Calculate precision matrices from covariance
        self.Lambda0 = np.zeros((n_arms, n_features, n_features))
        for i in range(n_arms):
            self.Lambda0[i] = np.linalg.inv(self.Sigma0[i])
        
        # Initialize sufficient statistics
        self.G = np.zeros((n_arms, n_features, n_features))  # Precision matrix
        self.B = np.zeros((n_arms, n_features))  # Weighted sum of rewards
        
        # Counter for calculating confidence width
        self.t = 0
    
    def confidence_ellipsoid_width(self) -> float:
        """
        Calculate the width of the confidence ellipsoid.
        Based on Theorem 2 in Abassi-Yadkori (2011).
        
        Returns:
            float: The width of the confidence ellipsoid.
        """
        L = 1.0  # Assume bound on feature norm
        Lambda = np.trace(self.Lambda0, axis1=-2, axis2=-1).max() / self.n_features
        R = self.sigma
        S = np.sqrt(self.n_features)
        width = np.sqrt(Lambda) * S + \
            R * np.sqrt(self.n_features * np.log((1 + self.t * np.square(L) / Lambda) / self.delta))
        return width
    
    def choose_arm(self, x: Union[np.ndarray, List[List[float]]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Upper Confidence Bound.
        
        Parameters:
            x (array-like): Feature matrix. Can be either:
                - A matrix of shape (n_arms, n_features) where each row corresponds to an arm's features.
                - A single feature vector of shape (n_features,) or (1, n_features) to be used for all arms.
            
        Returns:
            int: The index of the selected arm.
        """
        x = np.atleast_2d(x)  # Convert to 2D array if necessary
        
        # Check if we have features per arm or a single feature vector
        if x.shape[0] == 1 and self.n_arms > 1:
            # Single feature vector for all arms - replicate it
            x = np.tile(x, (self.n_arms, 1))
        elif x.shape[0] != self.n_arms:
            raise ValueError(f"Feature matrix shape mismatch. Expected {self.n_arms} rows, got {x.shape[0]}")
        
        self.t += 1  # Increment time counter
        
        # Calculate confidence width
        cew = self.confidence_ellipsoid_width()
        
        # Compute UCB for each arm
        mu = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            # Linear model posterior
            Gt = self.Lambda0[i] + self.G[i]
            Sigma_hat = np.linalg.inv(Gt)
            theta_hat = np.linalg.solve(Gt, self.Lambda0[i].dot(self.Theta0[i]) + self.B[i])
            
            # UCB calculation
            Sigma_hat_scaled = Sigma_hat / np.square(self.sigma)
            mu[i] = x[i].dot(theta_hat) + cew * \
                np.sqrt(x[i].dot(Sigma_hat_scaled).dot(x[i]))
        
        return int(np.argmax(mu))
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "LinearUCBAgent":
        """
        Update the agent's state after observing a reward.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector for the chosen arm.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        x = np.array(x).flatten()
        
        # Update sufficient statistics
        self.G[arm] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm] += x * y / np.square(self.sigma)
        
        return self


class METSAgent(BanditAgent):
    """
    Matrix-variate Emission Thompson Sampling agent.
    """
    def __init__(self, n_arms: int, n_features: int, A: Optional[np.ndarray] = None, sigma: float = 1.0,
                 prior_mean_psi: Optional[np.ndarray] = None, 
                 prior_cov_psi: Optional[np.ndarray] = None,
                 prior_cov: Optional[np.ndarray] = None) -> None:
        """
        Initialize the METS agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            A (array-like, optional): Mixing coefficients matrix (n_arms × L). Default is n_arms x 1 matrix of ones.
            sigma (float): Noise standard deviation.
            prior_mean_psi (array-like, optional): Prior mean of effect parameters.
            prior_cov_psi (array-like, optional): Prior covariance of effect parameters.
            prior_cov (array-like, optional): Prior covariance of arm parameters.
        """
        super().__init__(n_arms, n_features)
        self.sigma = sigma
        
        # Set default A if not provided
        if A is None:
            A = np.ones((n_arms, 1))  # Default: n_arms x 1 matrix of ones
            
        self.A = np.array(A)
        self.L = self.A.shape[1]
        
        # Set prior parameters for effects
        self.mu_psi = prior_mean_psi if prior_mean_psi is not None else np.zeros(self.L * n_features)
        self.Sigma_psi = prior_cov_psi if prior_cov_psi is not None else np.eye(self.L * n_features)
        self.Lambda_psi = np.linalg.inv(self.Sigma_psi)
        
        # Set prior parameters for arms
        if prior_cov is not None:
            self.Sigma0 = prior_cov
            if prior_cov.ndim == 2:
                self.Sigma0 = np.array([prior_cov.copy() for _ in range(n_arms)])
        else:
            self.Sigma0 = np.array([np.eye(n_features) for _ in range(n_arms)])
        
        # Calculate precision matrices from covariance
        self.Lambda0 = np.zeros((n_arms, n_features, n_features))
        for i in range(n_arms):
            self.Lambda0[i] = np.linalg.inv(self.Sigma0[i])
        
        # Initialize sufficient statistics
        self.G = np.zeros((n_arms, n_features, n_features))  # Precision matrix
        self.B = np.zeros((n_arms, n_features))  # Weighted sum of rewards
    
    def choose_arm(self, x: Union[np.ndarray, List[List[float]]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Matrix-variate Emission Thompson Sampling.
        
        Parameters:
            x (array-like): Feature matrix. Can be either:
                - A matrix of shape (n_arms, n_features) where each row corresponds to an arm's features.
                - A single feature vector of shape (n_features,) or (1, n_features) to be used for all arms.
            
        Returns:
            int: The index of the selected arm.
        """
        x = np.atleast_2d(x)  # Convert to 2D array if necessary
        
        # Check if we have features per arm or a single feature vector
        if x.shape[0] == 1 and self.n_arms > 1:
            # Single feature vector for all arms - replicate it
            x = np.tile(x, (self.n_arms, 1))
        elif x.shape[0] != self.n_arms:
            raise ValueError(f"Feature matrix shape mismatch. Expected {self.n_arms} rows, got {x.shape[0]}")
        
        small_eye = 1e-3 * np.eye(self.n_features)
        
        # Effect posterior
        Lambda_t = np.copy(self.Lambda_psi)
        mu_t = self.Lambda_psi.dot(self.mu_psi)
        
        for i in range(self.n_arms):
            aiai = np.outer(self.A[i, :], self.A[i, :])
            inv_Gi = np.linalg.inv(self.G[i] + small_eye)
            prior_adjusted_Gi = np.linalg.inv(self.Sigma0[i] + inv_Gi)
            Lambda_t += np.kron(aiai, prior_adjusted_Gi)
            prior_adjusted_Bi = prior_adjusted_Gi.dot(inv_Gi.dot(self.B[i]))
            mu_t += np.outer(self.A[i, :], prior_adjusted_Bi).flatten()
        
        # Posterior sampling
        Sigma_t = np.linalg.inv(Lambda_t)
        mu_t = Sigma_t.dot(mu_t)
        Psi_tilde = np.random.multivariate_normal(mu_t, Sigma_t)
        matPsi_tilde = np.reshape(Psi_tilde, (self.L, self.n_features))  # matrix version
        
        # Compute expected rewards
        mu = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            # Arm parameter posterior
            Sigma_ti = np.linalg.inv(self.Lambda0[i] + self.G[i])
            mu_ti = self.Lambda0[i].dot(self.A[i, :].dot(matPsi_tilde)) + self.B[i]
            mu_ti = Sigma_ti.dot(mu_ti)
            
            # Posterior sampling
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            mu[i] = x[i].dot(theta_tilde)
        
        return int(np.argmax(mu))
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "METSAgent":
        """
        Update the agent's state after observing a reward.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector for the chosen arm.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        x = np.array(x).flatten()
        
        # Update sufficient statistics
        self.G[arm] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm] += x * y / np.square(self.sigma)
        
        return self


class HierTSAgent(BanditAgent):
    """
    Hierarchical Thompson Sampling agent.
    """
    def __init__(self, n_arms: int, n_features: int, A: Optional[np.ndarray] = None, sigma: float = 1.0,
                 prior_mean_psi: Optional[np.ndarray] = None, 
                 prior_cov_psi: Optional[np.ndarray] = None,
                 prior_cov: Optional[np.ndarray] = None) -> None:
        """
        Initialize the HierTS agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            A (array-like): Mixing coefficients matrix (n_arms × L).
            sigma (float): Noise standard deviation.
            prior_mean_psi (array-like, optional): Prior mean of effect parameters.
            prior_cov_psi (array-like, optional): Prior covariance of effect parameters.
            prior_cov (array-like, optional): Prior covariance of arm parameters.
        """
        super().__init__(n_arms, n_features)
        self.sigma = sigma
        
        # Set default A if not provided
        if A is None:
            A = np.ones((n_arms, 1))  # Default: n_arms x 1 matrix of ones

        self.A = np.array(A)
        self.L = self.A.shape[1]
        
        # Calculate average effect parameters and covariance
        if prior_mean_psi is not None:
            self.mu_psi = np.mean(prior_mean_psi.reshape(self.L, n_features), axis=0)
        else:
            self.mu_psi = np.zeros(n_features)
            
        if prior_cov_psi is not None:
            self.Sigma_psi = (1/(self.L**2)) * np.sum(
                [prior_cov_psi[l*n_features:(l+1)*n_features, l*n_features:(l+1)*n_features] 
                 for l in range(self.L)], axis=0
            )
        else:
            self.Sigma_psi = np.eye(n_features)
            
        self.Lambda_psi = np.linalg.inv(self.Sigma_psi)
        
        # Set prior parameters for arms
        if prior_cov is not None:
            self.Sigma0 = prior_cov
            if prior_cov.ndim == 2:
                self.Sigma0 = np.array([prior_cov.copy() for _ in range(n_arms)])
        else:
            self.Sigma0 = np.array([np.eye(n_features) for _ in range(n_arms)])
        
        # Calculate precision matrices from covariance
        self.Lambda0 = np.zeros((n_arms, n_features, n_features))
        for i in range(n_arms):
            self.Lambda0[i] = np.linalg.inv(self.Sigma0[i])
        
        # Initialize sufficient statistics
        self.G = np.zeros((n_arms, n_features, n_features))  # Precision matrix
        self.B = np.zeros((n_arms, n_features))  # Weighted sum of rewards
    
    def choose_arm(self, x: Union[np.ndarray, List[List[float]]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Hierarchical Thompson Sampling.
        
        Parameters:
            x (array-like): Feature matrix. Can be either:
                - A matrix of shape (n_arms, n_features) where each row corresponds to an arm's features.
                - A single feature vector of shape (n_features,) or (1, n_features) to be used for all arms.
            
        Returns:
            int: The index of the selected arm.
        """
        x = np.atleast_2d(x)  # Convert to 2D array if necessary
        
        # Check if we have features per arm or a single feature vector
        if x.shape[0] == 1 and self.n_arms > 1:
            # Single feature vector for all arms - replicate it
            x = np.tile(x, (self.n_arms, 1))
        elif x.shape[0] != self.n_arms:
            raise ValueError(f"Feature matrix shape mismatch. Expected {self.n_arms} rows, got {x.shape[0]}")
        
        small_eye = 1e-3 * np.eye(self.n_features)
        
        # Effect parameter posterior
        Lambda_t = np.copy(self.Lambda_psi)
        mu_t = self.Lambda_psi.dot(self.mu_psi)
        
        for i in range(self.n_arms):
            inv_Gi = np.linalg.inv(self.G[i] + small_eye)
            prior_adjusted_Gi = np.linalg.inv(self.Sigma0[i] + inv_Gi)
            Lambda_t += prior_adjusted_Gi
            prior_adjusted_Bi = prior_adjusted_Gi.dot(inv_Gi.dot(self.B[i]))
            mu_t += prior_adjusted_Bi
        
        # Posterior sampling
        Sigma_t = np.linalg.inv(Lambda_t)
        mu_t = Sigma_t.dot(mu_t)
        Psi_tilde = np.random.multivariate_normal(mu_t, Sigma_t)
        
        # Compute expected rewards
        mu = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            # Arm parameter posterior
            Sigma_ti = np.linalg.inv(self.Lambda0[i] + self.G[i])
            mu_ti = self.Lambda0[i].dot(Psi_tilde) + self.B[i]
            mu_ti = Sigma_ti.dot(mu_ti)
            
            # Posterior sampling
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            mu[i] = x[i].dot(theta_tilde)
        
        return int(np.argmax(mu))
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "HierTSAgent":
        """
        Update the agent's state after observing a reward.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector for the chosen arm.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        x = np.array(x).flatten()
        
        # Update sufficient statistics
        self.G[arm] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm] += x * y / np.square(self.sigma)
        
        return self
