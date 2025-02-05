import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bart_playground import Parameters, Tree
from bart_playground import DefaultPrior
from bart_playground import Dataset

class TestPrior(unittest.TestCase):
    def setUp(self):
        X, y = np.random.rand(100, 5), np.random.rand(100)
        dataset = Dataset(X, y, None)
        self.trees = [Tree(data=dataset) for i in range(5)]
        self.tree_params = Parameters(self.trees, None, dataset)
        self.alpha = 0.5
        self.beta = 0.5

    def test_default_prior(self):
        prior = DefaultPrior(n_trees=len(self.trees), tree_alpha=self.alpha, tree_beta=self.beta)
        # empty tree
        empty_prior_first_tree = prior.trees_log_prior(self.tree_params, np.array([0]))
        self.assertEqual(empty_prior_first_tree, np.log(1-self.alpha), f"The log prior for an empty tree should be {np.log(1-self.alpha)}.")
        empty_prior_all_trees = prior.trees_log_prior(self.tree_params, np.arange(len(self.trees)))
        self.assertEqual(empty_prior_all_trees, len(self.trees) * np.log(1-self.alpha), f"The log prior for an empty tree should be {len(self.trees) *np.log(1- self.alpha)}.")
        # # make one split
        for i in range(len(self.trees)):
            self.trees[i].split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)
        one_split_prior = prior.trees_log_prior(self.tree_params, np.array([0]))
        # now the probability is alpha * (1-2^(-beta))^2
        one_split_prob = self.alpha * (1 - self.alpha * (2)**(-self.beta))**2
        log_one_split_prob = np.log(one_split_prob).round(5)
        one_split_prior = one_split_prior.round(5)
        self.assertEqual(one_split_prior, log_one_split_prob, f"The log prior for a tree with one split should be {log_one_split_prob}.")
        # do one split for rest of the trees

        all_split_prior = prior.trees_log_prior(self.tree_params, np.arange(len(self.trees)))
        # now the probability is alpha * (1-2^(-beta))^2 * (1-2^(-beta))^(n_trees)
        log_all_split_prob = 5 * np.log(one_split_prob)
        log_all_split_prob = log_all_split_prob.round(5)
        all_split_prior = all_split_prior.round(5)
        self.assertEqual(all_split_prior, log_all_split_prob, f"The log prior for a tree with one split should be {log_all_split_prob}.")

class TestPrior2(unittest.TestCase):
    def setUp(self):
        """
        Set up a default prior instance and mock dataset.
        """
        self.prior = DefaultPrior()
        self.mock_data = MagicMock(spec=Dataset)
        self.mock_data.X = np.random.randn(100, 5)
        self.mock_data.y = np.random.randn(100)
        self.mock_data.n = 100

    def test_fit(self):
        """
        Test fitting the prior to a dataset.
        """
        self.prior.fit(self.mock_data)
        self.assertIsNotNone(self.prior.eps_lambda)

    def test_init_global_params(self):
        """
        Test initializing global parameters.
        """
        self.prior.fit(self.mock_data)
        params = self.prior.init_global_params(self.mock_data)
        self.assertIn("eps_sigma2", params)
        self.assertTrue(params["eps_sigma2"] > 0)

    def test_resample_global_params(self):
        """
        Test resampling global parameters.
        """
        self.prior.fit(self.mock_data)
        mock_params = MagicMock(spec=Parameters)
        mock_params.data = self.mock_data
        mock_params.evaluate.return_value = np.zeros(100)

        params = self.prior.resample_global_params(mock_params)
        self.assertIn("eps_sigma2", params)
        self.assertTrue(params["eps_sigma2"] > 0)


    @patch("bart_playground.priors.sqrtm")
    def test_resample_leaf_vals(self, mock_sqrtm):
        """
        Test resampling leaf values.
        """
        mock_params = MagicMock(spec=Parameters)
        mock_params.data = self.mock_data
        mock_params.data.y = np.random.randn(100)
        mock_params.global_params = {"eps_sigma2": 1.0}
        mock_params.evaluate.return_value = np.zeros(100)
        mock_params.leaf_basis.return_value = np.random.randn(100, 5)

        mock_sqrtm.return_value = np.eye(5)

        leaf_vals = self.prior.resample_leaf_vals(mock_params, [0])
        self.assertEqual(leaf_vals.shape[0], 5)



    def test_trees_log_prior(self):
        """
        Test computing log prior for trees.
        """
        mock_params = MagicMock(spec=Parameters)
        mock_tree = MagicMock()
        mock_tree.vars = np.array([0, -1, 1, -2])
        mock_params.trees = [mock_tree]
        
        log_prior = self.prior.trees_log_prior(mock_params, [0])
        self.assertIsInstance(log_prior, float)

    def test_trees_log_marginal_lkhd(self):
        """
        Test computing log marginal likelihood for trees.
        """
        mock_params = MagicMock(spec=Parameters)
        mock_params.data = self.mock_data
        mock_params.global_params = {"eps_sigma2": 1.0}
        mock_params.evaluate.return_value = np.zeros(100)
        mock_params.leaf_basis.return_value = np.random.randn(100, 5)

        log_lkhd = self.prior.trees_log_marginal_lkhd(mock_params, [0])
        self.assertIsInstance(log_lkhd, float)

    def test_trees_log_prior_ratio(self):
        """
        Test computing log prior ratio.
        """
        mock_move = MagicMock()
        mock_move.current = MagicMock()
        mock_move.proposed = MagicMock()
        mock_move.trees_changed = [0]

        self.prior.trees_log_prior = MagicMock(side_effect=[-1.0, -0.5])
        ratio = self.prior.trees_log_prior_ratio(mock_move)
        self.assertAlmostEqual(ratio, 0.5)

    def test_trees_log_mh_ratio(self):
        """
        Test computing MH ratio.
        """
        mock_move = MagicMock()
        mock_move.current = MagicMock()
        mock_move.proposed = MagicMock()
        mock_move.trees_changed = [0]

        self.prior.trees_log_prior_ratio = MagicMock(return_value=-0.3)
        self.prior.trees_log_marginal_lkhd_ratio = MagicMock(return_value=0.7)

        mh_ratio = self.prior.trees_log_mh_ratio(mock_move)
        self.assertAlmostEqual(mh_ratio, 0.4)


if __name__ == "__main__":
    unittest.main()
