import numpy as np
from sklearn.preprocessing import OneHotEncoder

from priors import BARTPrior
from moves import Move

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
        self.vars = np.full(3, np.nan, dtype=int)
        self.vars[0] = -1
        self.thresholds = np.full(3, np.nan, dtype=float)
        self.leaf_vals = np.full(3, np.nan, dtype=float)

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
        new_length = len(self.var) * 2

        # Resize vars array
        new_var = np.full(new_length, np.nan, dtype=int)
        new_var[:len(self.var)] = self.var
        self.var = new_var

        # Resize split array
        new_split = np.full(new_length, np.nan, dtype=float)
        new_split[:len(self.split)] = self.split
        self.split = new_split

        # Resize leaf_vals array
        new_leaf_vals = np.zeros(new_length, dtype=float)
        new_leaf_vals[:len(self.leaf_vals)] = self.leaf_vals
        self.leaf_vals = new_leaf_vals


    def split_leaf(self, leaf_index: int, var: int, split_threshold: float, left_val: float=np.nan, right_val: float=np.nan):
        """
        Split a leaf node into two child nodes.

        Parameters:
        - leaf_index: int
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
        if self.vars[leaf_index] != -1:
            raise ValueError("Node is not a leaf and cannot be split.")

        # Check if the index overflows and resize arrays if necessary
        left_child = leaf_index * 2 + 1
        right_child = leaf_index * 2 + 2

        if left_child >= len(self.var) or right_child >= len(self.var):
            self._resize_arrays()

        # Assign the split variable and threshold to the leaf node
        self.vars[leaf_index] = var
        self.thresholds[leaf_index] = split_threshold

        # Initialize the new leaf nodes
        self.vars[left_child] = -1
        self.vars[right_child] = -1

        # Assign the provided values to the new leaf nodes
        self.leaf_vals[left_child] = left_val
        self.leaf_vals[right_child] = right_val

    def prune_split(self, split_index: int):
        """
        Prune a terminal split node, turning it back into a leaf.

        Parameters:
        - split_index: int
            Index of the split node to prune.
        """
        # Check if the node is a split node
        if self.vars[split_index] == -1:
            raise ValueError("Node is already a leaf and cannot be pruned.")

        # Turn the split node into a leaf
        self.vars[split_index] = -1
        self.thresholds[split_index] = np.nan

        left_child = leaf_index * 2 + 1
        right_child = leaf_index * 2 + 2
        self.vars[left_child] = np.nan
        self.vars[right_child] = np.nan

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
            if self.vars[i] != -1  # It's a split node
            and self.vars[i * 2 + 1] == -1  # Left child is a leaf
            and self.vars[i * 2 + 2] == -1  # Right child is a leaf
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
            if self.vars[i] == -1  # It's a leaf node
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
            if self.vars[i] is not None  # It's a split node
        ]

        if not split_nodes:
            raise ValueError("No split nodes found.")

        # Return a random split node
        return generator.choice(split_nodes)
    
    def __str__(self):
        return f"TreeParams(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals})"
        
    def __repr__(self):
        """
        Return a string representation of the TreeParams object.
        """
        return f"TreeParams(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals})"


class BARTParams:
    """
    Represents the parameters of the BART model.
    """
    def __init__(self, trees: list, sigma2: float, prior: BARTPrior, X : np.ndarray, y : np.ndarray):
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
        self.X = X
        self.y = y
        self.prior = prior
        self.n, self.p = X.shape
        self.trees = copy.deepcopy(trees)
        self.n_trees = len(self.trees)
        self.sigma2 = sigma2
        self.noise_ratio = None

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
        - move: Move
            The move to compute the prior ratio for.

        Returns:
        - float
            Prior ratio.
        """
        return np.sum([self.prior.tree_log_prior(self.trees[tree_id]) 
                       for tree_id in tree_ids])

    def get_log_marginal_lkhd(self, tree_ids):
        leaf_indicators = self.get_leaf_indicators(tree_ids)
        U, S, _ = svd(leaf_indicators)
        logdet = np.sum(np.log(S ** 2 / self.noise_ratio + 1))
        r_U_coefs = U.T @ resids
        r_U = U @ y_U_coefs
        ls_resids = np.sum((resids - r_U) ** 2)
        ridge_bias = np.sum(r_U_coefs ** 2 / (S ** 2 / self.noise_ratio + 1))
        return - (logdet + (ls_resids + ridge_bias) / self.params.sigma2) / 2

    def sample_sigma2(self):
        """
        Sample the noise variance.

        Returns:
        - float
            Sampled noise variance.
        """
        pass

    def sample_leaf_params(self, tree_index: int):
        """
        Sample the leaf parameters for a given tree.

        Parameters:
        - tree_index: int
            Index of the tree.

        Returns:
        - np.ndarray
            Sampled leaf parameters.
        """
        pass