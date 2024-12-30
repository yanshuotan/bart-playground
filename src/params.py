import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder


class TreeParams:
    """
    Represents the parameters of a single tree in the BART model, combining both
    the tree structure and leaf values into a single object.
    """
    def __init__(self):
        """
        Initialize the tree parameters.

        Parameters:
        - var: np.ndarray
            Array of variables used for splitting at each node.
        - thresholds: np.ndarray
            Array of split values at each node.
        - leaf_vals: np.ndarray
            Values at the leaf nodes.
        """
        self.vars = np.full(8, -2, dtype=int)
        self.vars[0] = -1
        self.thresholds = np.full(8, np.nan, dtype=float)
        self.leaf_vals = np.full(8, np.nan, dtype=float)
        self.n_vals = np.full(8, -2, dtype=int)

    def traverse_tree(self, X: np.ndarray) -> int:
        """
        Traverse the tree to find the leaf nodes for a given input data matrix.

        Parameters:
        - X: np.ndarray
            Input data (2D array).

        Returns:
        - int
            Index of the leaf node.
        """
        n = X.shape[0]
        node_ids = np.zeros(n, dtype=int)
        for i in range(n):
            node_ids[i] = self._traverse_tree(X[i,:])
        return node_ids

    def _traverse_tree(self, x: np.ndarray) -> int:
        """
        Traverse the tree to find the leaf node for a given input.

        Parameters:
        - x: np.ndarray
            Input data point (1D array).

        Returns:
        - int
            Index of the leaf node.
        """
        node_id = 0  # Start at the root node

        while True:
            var = self.vars[node_id]
            threshold = self.thresholds[node_id]

            if var == -1:  # Check if the node is a leaf
                return node_id  # Return the leaf node index

            # Determine the next node based on the split condition
            if x[var] <= threshold:
                node_id = node_id * 2 + 1  # Move to the left child
            else:
                node_id = node_id * 2 + 2  # Move to the right child

    def evaluate(self, X: np.ndarray) -> float:
        """
        Evaluate the tree for a given input data matrix.


        Parameters:
        - x: np.ndarray
            Input data (2D array).

        Returns:
        - float
            Output values of the tree.
        """
        leaf_ids = self.traverse_tree(X)  # Find the leaf node for the input
        return self.leaf_vals[leaf_ids]  # Return the value at the leaf node

    # def _evaluate(self, x: np.ndarray) -> float:
    #     """
    #     Evaluate the tree for a given input.

    #     Parameters:
    #     - x: np.ndarray
    #         Input data point (1D array).

    #     Returns:
    #     - float
    #         Output value of the tree.
    #     """
    #     leaf_index = self.traverse_tree(x)  # Find the leaf node for the input
    #     return self.leaf_vals[leaf_index]  # Return the value at the leaf node

    def _resize_arrays(self):
        """
        Resize the vars, split, and leaf_vals arrays by doubling their length.
        """
        new_length = len(self.vars) * 2

        # Resize vars array
        new_vars = np.full(new_length, -2, dtype=int)
        new_vars[:len(self.vars)] = self.vars
        self.vars = new_vars

        # Resize split array
        new_thresholds = np.full(new_length, np.nan, dtype=float)
        new_thresholds[:len(self.thresholds)] = self.thresholds
        self.thresholds = new_thresholds

        # Resize leaf_vals array
        new_leaf_vals = np.full(new_length, np.nan, dtype=float)
        new_leaf_vals[:len(self.leaf_vals)] = self.leaf_vals
        self.leaf_vals = new_leaf_vals

        # Resize n_vals array
        new_n_vals = np.full(new_length, -2, dtype=int)
        new_n_vals[:len(self.n_vals)] = self.n_vals
        self.n_vals = new_n_vals


    def split_leaf(self, leaf_id: int, var: int, split_threshold: float, left_val: float=np.nan, 
                   right_val: float=np.nan, left_n: int=-2, right_n: int=-2):
        """
        Split a leaf node into two child nodes.

        Parameters:
        - leaf_id: int
            Index of the leaf node to split.
        - var: int
            Variable to use for the split.
        - split_threshold: float
            Threshold value for the split.
        - left_val: float
            Value to assign to the left child node.
        - right_val: float
            Value to assign to the right child node.
        """
        # Check if the node is already a leaf
        if self.vars[leaf_id] != -1:
            raise ValueError("Node is not a leaf and cannot be split.")

        # Check if the index overflows and resize arrays if necessary
        left_child = leaf_id * 2 + 1
        right_child = leaf_id * 2 + 2

        if left_child >= len(self.vars) or right_child >= len(self.vars):
            self._resize_arrays()

        # Assign the split variable and threshold to the leaf node
        self.vars[leaf_id] = var
        self.thresholds[leaf_id] = split_threshold

        # Initialize the new leaf nodes
        self.vars[left_child] = -1
        self.vars[right_child] = -1

        # Assign the provided values to the new leaf nodes
        self.leaf_vals[left_child] = left_val
        self.leaf_vals[right_child] = right_val

        self.n_vals[left_child] = left_n
        self.n_vals[right_child] = right_n

    def prune_split(self, node_id: int):
        """
        Prune a terminal split node, turning it back into a leaf.

        Parameters:
        - node_id: int
            Index of the split node to prune.
        """
        # Check if the node is a split node
        if not self.is_terminal_split_node(node_id):
            raise ValueError("Node is not a terminal split node and cannot be pruned.")

        # Turn the split node into a leaf
        self.vars[node_id] = -1
        self.thresholds[node_id] = np.nan

        left_child = node_id * 2 + 1
        right_child = node_id * 2 + 2
        self.vars[left_child] = -2
        self.vars[right_child] = -2

    def set_leaf_value(self, node_id, leaf_val):
        if self.is_leaf(node_id):
            self.leaf_vals[node_id] = leaf_val
        else:
            raise ValueError("Not a leaf")

    def is_leaf(self, node_id):
        return self.vars[node_id] == -1

    def is_split_node(self, node_id):
        return self.vars[node_id] not in [-1, -2]
    
    def is_terminal_split_node(self, node_id):
        return self.is_split_node(node_id) \
            and self.is_leaf(node_id * 2 + 1) \
            and self.is_leaf(node_id * 2 + 2)
    
    def get_n_leaves(self):
        return len([i for i in range(len(self.vars)) if self.is_leaf(i)])

    def get_random_terminal_split(self, generator: np.random.Generator) -> int:
        """
        Get a random terminal split node.

        Returns:
        - int
            Index of the random terminal split node.
        """
        # Find all terminal split nodes (nodes with two leaf children)
        terminal_splits = [
            i for i in range(len(self.vars))
            if self.is_terminal_split_node(i)
        ]

        if not terminal_splits:
            raise ValueError("No terminal split nodes found.")

        # Return a random terminal split node
        return generator.choice(terminal_splits)

    def get_random_leaf(self, generator: np.random.Generator) -> int:
        """
        Get a random leaf node.

        Returns:
        - int
            Index of the random leaf node.
        """
        # Find all leaf nodes
        leaf_nodes = [
            i for i in range(len(self.vars))
            if self.is_leaf(i)  # It's a leaf node
        ]

        if not leaf_nodes:
            raise ValueError("No leaf nodes found.")

        # Return a random leaf node
        return generator.choice(leaf_nodes)

    def get_random_split(self, generator: np.random.Generator) -> int:
        """
        Get a random split node.

        Returns:
        - int
            Index of the random split node.
        """
        # Find all split nodes
        split_nodes = [
            i for i in range(len(self.vars))
            if self.is_split_node(i)  # It's a split node
        ]

        if not split_nodes:
            raise ValueError("No split nodes found.")

        # Return a random split node
        return generator.choice(split_nodes)
    
    def __str__(self):
        """
        Return a string representation of the TreeParams object.
        """
        return self._print_tree()
        
    def __repr__(self):
        return f"TreeParams(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals}, n_vals={self.n_vals})"
    
    def _print_tree(self, node_id=0, prefix=""):
        pprefix = prefix + "\t"
        if self.vars[node_id] == -1: # Leaf node
            return prefix + self._print_node(node_id)
        else:
            left_idx = node_id * 2 + 1
            right_idx = node_id * 2 + 2
            return (
                prefix
                + self._print_node(node_id)
                + "\n"
                + self._print_tree(left_idx, pprefix)
                + "\n"
                + self._print_tree(right_idx, pprefix)
            )
        
    def _print_node(self, node_id):
        if self.vars[node_id] == -1:
            return f"Val: {self.leaf_vals[node_id]:0.3f} (leaf, n = {self.n_vals[node_id]})"
        else:
            return f"X_{self.vars[node_id]} <= {self.thresholds[node_id]:0.3f}" + \
                f" (split, n = {self.n_vals[node_id]})"

class BARTParams:
    """
    Represents the parameters of the BART model.
    """
    def __init__(self, trees: list, sigma2: float, prior, X : np.ndarray, y : np.ndarray):
        """
        Initialize the BART parameters.

        Parameters:
        - trees: list<TreeParams>
            List of trees in the model.
        - n_trees: int
            Number of trees.
        - sigma2: float
            Noise variance.
        """
        # Fixed params
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.prior = prior

        # Mutable params
        self.trees = copy.deepcopy(trees)
        self.n_trees = len(self.trees)
        self.sigma2 = sigma2
        self.noise_ratio = None

    @property
    def residuals(self):
        return self.y - self.evaluate(self.X)

    def copy(self):
        return BARTParams(X=self.X, y=self.y, prior=self.prior, trees=self.trees, sigma2=self.sigma2)

    def evaluate(self, X: np.ndarray, holdout: list = None) -> float:
        """
        Evaluate the BART model for a given input by summing the outputs of all trees.

        Parameters:
        - x: np.ndarray
            Input data points (2D array).
        - holdout: list<int>
            Indices of trees to exclude from evaluation (optional).

        Returns:
        - float
            Sum of the outputs of all trees.
        """
        total_output = np.zeros(X.shape[0])

        # Iterate over all trees
        for i, tree in enumerate(self.trees):
            if holdout is None or i not in holdout:  # Skip trees in the holdout list
                total_output += tree.evaluate(X)  # Add the tree's output to the total

        return total_output
    
    def get_leaf_indicators(self, tree_ids):
        ordinal_encoding = np.zeros((self.n, len(tree_ids)), dtype=int)
        for col, tree_id in enumerate(tree_ids):
            ordinal_encoding[:, col] = self.trees[tree_id].traverse_tree(self.X)
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        leaf_indicators = one_hot_encoder.fit_transform(ordinal_encoding)
        return leaf_indicators

    def get_log_prior(self, tree_ids):
        """
        Compute the ratio of priors for a given move.

        Parameters:
        tree_ids : array-like
            The IDs of the trees to hold out for evaluation.

        Returns:
        - float
            Prior ratio.
        """
        return np.sum([self.prior.tree_log_prior(self.trees[tree_id]) 
                       for tree_id in tree_ids])

    def get_log_marginal_lkhd(self, tree_ids):
        """
        Calculate the log marginal likelihood for the given tree IDs.

        Parameters:
        -----------
        tree_ids : array-like
            The IDs of the trees to hold out for evaluation.

        Returns:
        --------
        float
            The log marginal likelihood value.

        Notes:
        ------
        This function computes the log marginal likelihood by performing the following steps:
        1. Calculate the residuals by subtracting the evaluated predictions from the actual values.
        2. Obtain the leaf indicators for the given tree IDs.
        3. Perform Singular Value Decomposition (SVD) on the leaf indicators.
        4. Compute the log determinant using the singular values and noise ratio.
        5. Calculate the coefficients and projections of the residuals onto the left singular vectors.
        6. Compute the least squares residuals and ridge bias.
        7. Return the negative half of the sum of the log determinant and the normalized residuals and bias.
        """
        resids = self.residuals
        leaf_indicators = self.get_leaf_indicators(tree_ids)
        U, S, _ = np.linalg.svd(leaf_indicators)
        logdet = np.sum(np.log(S ** 2 / self.noise_ratio + 1))
        resid_u_coefs = U.T @ resids
        resids_u = U @ resid_u_coefs
        ls_resids = np.sum((resids - resids_u) ** 2)
        ridge_bias = np.sum(resid_u_coefs ** 2 / (S ** 2 / self.noise_ratio + 1))
        return - (logdet + (ls_resids + ridge_bias) / self.sigma2) / 2

    def resample_sigma2(self):
        """
        Sample the noise variance.

        Returns:
        - float
            Sampled noise variance.
        """
        alpha, beta = self.prior.sigma2_prior_icdf(self.X, self.y, "linear")
        resids = self.residuals
        posterior_alpha = alpha + (self.n / 2.)
        posterior_beta = beta + (0.5 * (np.sum(np.square(resids))))
        posterior_theta = 1/posterior_beta
        # here np uses the scale (theta parameterization) instead of the rate (beta parameterization)
        sigma_2_posterior = np.power(np.random.gamma(posterior_alpha, posterior_theta), -0.5)
        return sigma_2_posterior

    def resample_leaf_params(self, tree_ids):
        """
        Sample the leaf parameters for the given tree IDs.

        Parameters:
        tree_ids : array-like
            The IDs of the trees to hold out for evaluation.

        Returns:
        - np.ndarray
            Sampled leaf parameters.
        """
        residuals = self.y - self.evaluate(self.X, tree_ids)
        leaf_indicators = self.get_leaf_indicators(tree_ids)
        leaf_averages = np.linalg.lstsq(leaf_indicators, residuals, rcond=None)[0]
        likihood_mean = leaf_indicators @ leaf_averages
        k = 2 # default value recommended in the BART paper
        prior_var = 0.5 * (1 / k * np.sqrt(self.n_trees))
        n = self.n
        likihood_var = (self.resample_sigma2() ** 2) / n
        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        return posterior_mean + (np.random.normal(size=len(self.y)) * np.power(posterior_variance / self.n_trees, 0.5))