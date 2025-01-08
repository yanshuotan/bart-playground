import unittest

import numpy as np

from src.params import Parameters, Tree
from src.priors import DefaultPrior
from src.util import Dataset

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




if __name__ == "__main__":
    unittest.main()
