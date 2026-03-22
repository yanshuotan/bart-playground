import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bart_playground import Parameters, Tree
from bart_playground import ComprehensivePrior
from bart_playground import Dataset
from bart_playground import Grow
from bart_playground.util import DefaultPreprocessor

class TestPrior(unittest.TestCase):
    def setUp(self):
        X, y = np.random.rand(100, 5), np.random.rand(100)
        dataset = Dataset(X, y)
        self.trees = [Tree.new(dataX=dataset.X) for i in range(5)]
        self.tree_params = Parameters(self.trees, None, None)
        self.alpha = 0.5
        self.beta = 0.5

    def test_default_prior(self):
        rng = np.random.default_rng(42)
        prior = ComprehensivePrior(
            n_trees=len(self.trees), tree_alpha=self.alpha, tree_beta=self.beta, f_k=2.0,
            eps_q=0.9, eps_nu=3.0, specification="linear", generator=rng,
            dirichlet_prior=False, quick_decay=False, s_alpha=2.0
        )
        # empty tree
        empty_prior_first_tree = prior.tree_prior.trees_log_prior(self.tree_params, np.array([0]))
        self.assertEqual(empty_prior_first_tree, np.log(1-self.alpha), f"The log prior for an empty tree should be {np.log(1-self.alpha)}.")
        empty_prior_all_trees = prior.tree_prior.trees_log_prior(self.tree_params, np.arange(len(self.trees)))
        self.assertEqual(empty_prior_all_trees, len(self.trees) * np.log(1-self.alpha), f"The log prior for an empty tree should be {len(self.trees) *np.log(1- self.alpha)}.")
        # # make one split
        for i in range(len(self.trees)):
            self.trees[i].split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)
        one_split_prior = prior.tree_prior.trees_log_prior(self.tree_params, np.array([0]))
        # now the probability is alpha * (1-2^(-beta))^2
        one_split_prob = self.alpha * (1 - self.alpha * (2)**(-self.beta))**2
        log_one_split_prob = round(np.log(one_split_prob), 5)
        one_split_prior = round(float(one_split_prior), 5)
        self.assertEqual(one_split_prior, log_one_split_prob, f"The log prior for a tree with one split should be {log_one_split_prob}.")
        # do one split for rest of the trees

        all_split_prior = prior.tree_prior.trees_log_prior(self.tree_params, np.arange(len(self.trees)))
        # now the probability is alpha * (1-2^(-beta))^2 * (1-2^(-beta))^(n_trees)
        log_all_split_prob = 5 * np.log(one_split_prob)
        log_all_split_prob = round(float(log_all_split_prob), 5)
        all_split_prior = round(float(all_split_prior), 5)
        self.assertEqual(all_split_prior, log_all_split_prob, f"The log prior for a tree with one split should be {log_all_split_prob}.")

class TestPrior2(unittest.TestCase):
    def setUp(self):
        """
        Set up a default prior instance and mock dataset.
        """
        rng = np.random.default_rng(42)
        self.prior = ComprehensivePrior(
            n_trees=200, tree_alpha=0.95, tree_beta=2.0, f_k=2.0,
            eps_q=0.9, eps_nu=3.0, specification="linear", generator=rng,
            dirichlet_prior=False, quick_decay=False, s_alpha=2.0
        )
        self.mock_data = MagicMock(spec=Dataset)
        self.mock_data.X = np.random.randn(100, 5)
        self.mock_data.y = np.random.randn(100)
        self.mock_data.n = 100

    def _create_real_params(self, n_trees=1, n_samples=100, n_features=5, random_seed=42):
        """
        Helper method to create real Parameters object for testing.
        Creates Dataset, Trees, and Parameters with proper global_params structure.
        """
        rng = np.random.default_rng(random_seed)
        X, y = rng.standard_normal((n_samples, n_features)).astype(np.float32), rng.standard_normal(n_samples).astype(np.float32)
        dataset = Dataset(X, y)
        trees = [Tree.new(dataX=dataset.X) for _ in range(n_trees)]
        global_params = {"eps_sigma2": np.array([1.0], dtype=np.float32)}
        return Parameters(trees, global_params, cache=None), X, y

    def test_fit(self):
        """
        Test fitting the prior to a dataset.
        """
        self.prior.global_prior.fit_hyperparameters(self.mock_data)
        self.assertIsNotNone(self.prior.global_prior.eps_lambda)

    def test_init_global_params(self):
        """
        Test initializing global parameters.
        """
        self.prior.global_prior.fit_hyperparameters(self.mock_data)
        params = self.prior.global_prior.init_global_params(self.mock_data)
        self.assertIn("eps_sigma2", params)
        self.assertTrue(params["eps_sigma2"] > 0)

    def test_resample_global_params(self):
        """
        Test resampling global parameters.
        """
        self.prior.global_prior.fit_hyperparameters(self.mock_data)
        mock_params = MagicMock(spec=Parameters)
        mock_params.evaluate.return_value = np.zeros(100)

        params = self.prior.global_prior.resample_global_params(mock_params, self.mock_data.y)
        self.assertIn("eps_sigma2", params)
        self.assertTrue(params["eps_sigma2"] > 0)


    def test_resample_leaf_vals(self):
        """
        Test resampling leaf values.
        """
        params, X, y = self._create_real_params(n_trees=1, random_seed=42)
        # Add a split to create leaf structure
        params.trees[0].split_leaf(0, var=0, threshold=0.0, left_val=1.0, right_val=-1.0)
        
        leaf_vals = self.prior.tree_prior.resample_leaf_vals(params, y, tree_ids=[0])
        
        # Verify result shape matches number of leaves
        self.assertEqual(leaf_vals.shape[0], len(params.trees[0].leaves))
        self.assertTrue(np.all(np.isfinite(leaf_vals)))



    def test_trees_log_prior(self):
        """
        Test computing log prior for trees.
        """
        mock_params = MagicMock(spec=Parameters)
        mock_tree = MagicMock()
        mock_tree.vars = np.array([0, -1, 1, -2])
        mock_params.trees = [mock_tree]
        
        log_prior = self.prior.tree_prior.trees_log_prior(mock_params, [0])
        self.assertIsInstance(log_prior, float)

    def test_trees_log_marginal_lkhd(self):
        """
        Test computing log marginal likelihood for trees using real objects.
        """
        params, X, y = self._create_real_params(n_trees=1, random_seed=42)
        # Add a split to create leaf structure with leaf_ids
        params.trees[0].split_leaf(0, var=0, threshold=0.0, left_val=1.0, right_val=-1.0)
        
        log_lkhd = self.prior.likelihood.trees_log_marginal_lkhd(params, y, [0])
        
        # Verify result is a finite float
        self.assertIsInstance(log_lkhd, float)
        self.assertTrue(np.isfinite(log_lkhd))

    def test_trees_log_prior_ratio(self):
        """
        Test computing log prior ratio.
        """
        mock_move = MagicMock()
        mock_move.current = MagicMock()
        mock_move.proposed = MagicMock()
        mock_move.trees_changed = [0]

        self.prior.tree_prior.trees_log_prior = MagicMock(side_effect=[-1.0, -0.5])
        ratio = self.prior.tree_prior.trees_log_prior_ratio(mock_move)
        self.assertAlmostEqual(ratio, 0.5)

    def test_trees_log_mh_ratio(self):
        """
        Test computing MH ratio components using real Parameters and Move objects.
        """
        params, X, y = self._create_real_params(n_trees=1, random_seed=42)
        
        # Create real Move object
        thresholds = DefaultPreprocessor.test_thresholds(X)
        move = Grow(params, trees_changed=[0], possible_thresholds=thresholds)
        move.propose(np.random.default_rng(42))
        
        # Test the component methods
        prior_ratio = self.prior.tree_prior.trees_log_prior_ratio(move)
        lkhd_ratio = self.prior.likelihood.trees_log_marginal_lkhd_ratio(
            move, y, marginalize=False
        )
        
        # Verify methods exist and return finite values
        self.assertIsInstance(prior_ratio, (int, float))
        self.assertIsInstance(lkhd_ratio, (int, float))
        self.assertTrue(np.isfinite(prior_ratio))
        self.assertTrue(np.isfinite(lkhd_ratio))




if __name__ == "__main__":
    unittest.main()
