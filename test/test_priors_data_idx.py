import unittest
import numpy as np
from bart_playground.params import Tree, Parameters
from bart_playground.priors import (
    TreesPrior, GlobalParamPrior, BARTLikelihood, ComprehensivePrior,
    _single_tree_resample_leaf_vals, _single_tree_log_marginal_lkhd_numba
)
from bart_playground.util import Dataset

class TestPriorsDataIndices(unittest.TestCase):
    
    def setUp(self):
        """Setup test data"""
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 4
        self.n_trees = 5
        
        # Generate synthetic data
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = (2 * self.X[:, 0] + 1.5 * self.X[:, 1] + 
                 np.random.normal(0, 0.5, self.n_samples)).astype(np.float32)
        
        # Create dataset
        self.dataset = Dataset(self.X, self.y)
        
        # Create trees with leaf_data_indices
        self.trees = []
        for i in range(self.n_trees):
            tree = Tree.new(self.X)
            # Split some trees to create non-trivial structures
            if i % 2 == 0 and len(tree.leaf_data_indices[0]) > 2:
                tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
            self.trees.append(tree)
        
        # Create global parameters
        self.global_params = {"eps_sigma2": np.array([1.0])}
        
        # Create Parameters object
        self.params = Parameters(self.trees, self.global_params)
        
        # Create priors
        self.tree_prior = TreesPrior(n_trees=self.n_trees, generator=np.random.default_rng(42))
        self.global_prior = GlobalParamPrior(generator=np.random.default_rng(42))
        self.likelihood = BARTLikelihood(self.tree_prior.f_sigma2)

    def test_single_tree_resample_leaf_vals_with_data_indices(self):
        """Test resampling leaf values using leaf_data_indices"""
        tree_id = 0
        tree = self.trees[tree_id]
        
        # Compute residuals
        residuals = self.y - self.params.evaluate(all_except=[tree_id])
        
        # Test the new method with leaf_data_indices
        leaf_nodes = list(tree.leaf_data_indices.keys())
        leaf_resid_sums = np.zeros(len(leaf_nodes), dtype=np.float32)
        leaf_n_samples = np.zeros(len(leaf_nodes), dtype=np.int32)
        
        for i, leaf_id in enumerate(leaf_nodes):
            data_indices = tree.leaf_data_indices[leaf_id]
            leaf_resid_sums[i] = np.sum(residuals[data_indices])
            leaf_n_samples[i] = len(data_indices)
        
        # Call numba function
        new_leaf_vals = _single_tree_resample_leaf_vals(
            leaf_resid_sums,
            leaf_n_samples,
            eps_sigma2=self.global_params["eps_sigma2"][0],
            f_sigma2=self.tree_prior.f_sigma2,
            random_normal_p=np.random.standard_normal(len(leaf_nodes))
        )
        
        # Check output shape and type
        self.assertEqual(len(new_leaf_vals), len(leaf_nodes))
        self.assertTrue(np.isfinite(new_leaf_vals).all())

    def test_trees_prior_resample_leaf_vals_single_tree(self):
        """Test TreesPrior.resample_leaf_vals with single tree using leaf_data_indices"""
        tree_id = 0
        
        # Resample leaf values
        new_leaf_vals = self.tree_prior.resample_leaf_vals(self.params, self.y, [tree_id])
        
        # Check output
        expected_n_leaves = len(self.trees[tree_id].leaf_data_indices)
        self.assertEqual(len(new_leaf_vals), expected_n_leaves)
        self.assertTrue(np.isfinite(new_leaf_vals).all())

    def test_trees_prior_resample_leaf_vals_multiple_trees(self):
        """Test TreesPrior.resample_leaf_vals with multiple trees"""
        tree_ids = [0, 1, 2]
        
        # Resample leaf values
        new_leaf_vals = self.tree_prior.resample_leaf_vals(self.params, self.y, tree_ids)
        
        # Check output
        total_leaves = sum(len(self.trees[i].leaf_data_indices) for i in tree_ids)
        self.assertEqual(len(new_leaf_vals), total_leaves)
        self.assertTrue(np.isfinite(new_leaf_vals).all())

    def test_single_tree_log_marginal_lkhd_with_data_indices(self):
        """Test log marginal likelihood calculation using leaf_data_indices"""
        tree_id = 0
        tree = self.trees[tree_id]
        
        # Compute residuals
        resids = self.y - self.params.evaluate(all_except=[tree_id])
        
        # Test the new method with leaf_data_indices
        leaf_nodes = list(tree.leaf_data_indices.keys())
        leaf_resid_sums = np.zeros(len(leaf_nodes), dtype=np.float32)
        leaf_n_samples = np.zeros(len(leaf_nodes), dtype=np.int32)
        
        for i, leaf_id in enumerate(leaf_nodes):
            data_indices = tree.leaf_data_indices[leaf_id]
            leaf_resid_sums[i] = np.sum(resids[data_indices])
            leaf_n_samples[i] = len(data_indices)
        
        total_resid_sq = np.sum(resids ** 2)
        
        # Call numba function
        log_lkhd = _single_tree_log_marginal_lkhd_numba(
            leaf_resid_sums,
            leaf_n_samples,
            total_resid_sq,
            self.global_params["eps_sigma2"][0],
            self.tree_prior.f_sigma2
        )
        
        # Check output
        self.assertTrue(np.isfinite(log_lkhd))
        self.assertIsInstance(log_lkhd, (float, np.float32, np.float64))

    def test_bart_likelihood_single_tree(self):
        """Test BARTLikelihood.trees_log_marginal_lkhd with single tree"""
        tree_id = 0
        
        # Calculate log marginal likelihood
        log_lkhd = self.likelihood.trees_log_marginal_lkhd(self.params, self.y, [tree_id])
        
        # Check output
        self.assertTrue(np.isfinite(log_lkhd))
        self.assertIsInstance(log_lkhd, (float, np.float32, np.float64))

    def test_data_conservation_in_leaf_data_indices(self):
        """Test that leaf_data_indices correctly accounts for all data points"""
        for i, tree in enumerate(self.trees):
            all_data_indices = []
            for leaf_id, data_indices in tree.leaf_data_indices.items():
                all_data_indices.extend(data_indices)
            
            # Convert to integers and sort for comparison
            all_data_indices = sorted([int(x) for x in all_data_indices])
            expected_indices = list(range(self.n_samples))
            
            self.assertEqual(all_data_indices, expected_indices,
                           f"Tree {i}: Data indices don't match expected range")

    def test_tree_with_deeper_splits(self):
        """Test functionality with deeper tree structures"""
        tree = Tree.new(self.X)
        
        # Create deeper splits
        tree.split_leaf(0, 0, 0.0, -1.0, 1.0)  # Split root
        if len(tree.leaf_data_indices.get(1, [])) > 1:
            tree.split_leaf(1, 1, 0.0, -2.0, 0.0)  # Split left child
        
        # Test leaf value resampling
        params_single = Parameters([tree], self.global_params)
        new_leaf_vals = self.tree_prior.resample_leaf_vals(params_single, self.y, [0])
        
        # Check that we have the right number of leaf values
        expected_n_leaves = len(tree.leaf_data_indices)
        self.assertEqual(len(new_leaf_vals), expected_n_leaves)
        
        # Test likelihood calculation
        log_lkhd = self.likelihood.trees_log_marginal_lkhd(params_single, self.y, [0])
        self.assertTrue(np.isfinite(log_lkhd))

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty tree_ids list
        with self.assertRaises((IndexError, ValueError)):
            self.tree_prior.resample_leaf_vals(self.params, self.y, [])
        
        # Test with invalid tree_id
        with self.assertRaises(IndexError):
            self.tree_prior.resample_leaf_vals(self.params, self.y, [999])


if __name__ == '__main__':
    unittest.main()