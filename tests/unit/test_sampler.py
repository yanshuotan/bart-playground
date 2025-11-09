import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Import the classes and constants to be tested from the samplers module
from bart_playground import TemperatureSchedule
from bart_playground.samplers import DefaultSampler, default_proposal_probs
from bart_playground.params import Parameters, Tree
from bart_playground.priors import ComprehensivePrior

from bart_playground.moves import all_moves
from bart_playground.util import Dataset, DefaultPreprocessor

class TestSamplers(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        X, y = np.random.rand(100, 5), np.random.rand(100)
        self.dataset = Dataset(X, y)

        super().__init__(methodName)

    def test_temperature_schedule_default(self):
        """
        Test default temperature schedule
        """
        ts = TemperatureSchedule()
        self.assertEqual(ts(10), 1)

    def test_temperature_schedule_custom(self):
        """
        Test custom temperature schedule
        """
        ts = TemperatureSchedule(lambda t: t * 2)
        self.assertEqual(ts(5), 10)

    def test_add_data(self):
        """
        Test the add_data method
        """
        prior = MagicMock()
        sampler = DefaultSampler(prior, default_proposal_probs, np.random.default_rng())
        data = MagicMock()
        sampler.add_data(data)
        self.assertIs(sampler.data, data)

    def test_run_without_data(self):
        """
        Test run() without data
        The run method should raise AssertionError when data is None.
        """
        prior = MagicMock()
        sampler = DefaultSampler(prior, default_proposal_probs, np.random.default_rng())
        with self.assertRaises(AssertionError):
            sampler.run(1)

    @patch("bart_playground.samplers.Parameters")
    @patch("bart_playground.samplers.Tree.new")
    def test_get_init_state(self, mock_tree_new, mock_parameters):
        """
        Test the get_init_state method
        It calls get_init_state and verifies that Tree() is called for correct times and that Parameters() is called with the correct arguments.
        """
        # Set up a mock prior 
        prior = MagicMock()
        prior.tree_prior.n_trees = 3
        prior.global_prior.init_global_params.return_value = "initial_global_params"
        
        sampler = DefaultSampler(prior, default_proposal_probs, np.random.default_rng())
        data = MagicMock()
        sampler.add_data(data)
        
        state = sampler.get_init_state()
        
        # Check that Tree is called 3 times (corresponding to 3 trees)
        self.assertEqual(mock_tree_new.call_count, 3)
        # Check the arguments passed during the Parameters call
        args, _ = mock_parameters.call_args
        trees_arg = args[0]
        self.assertEqual(len(trees_arg), 3)
        self.assertEqual(args[1], "initial_global_params")
        # Check the returned state
        self.assertEqual(state, mock_parameters.return_value)

    def test_sample_move(self):
        """
        Test the sample_move method
        It verifies that sample_move returns a tuple of (move_str, move_cls) where
        move_str is a valid move name and move_cls is the corresponding class.
        """
        prior = MagicMock()
        generator = MagicMock()
        generator.choice = MagicMock(return_value=np.array(["grow"] * 100))
        sampler = DefaultSampler(prior, default_proposal_probs, generator)
        
        move_str, move_cls = sampler.sample_move()
        self.assertIsInstance(move_str, str)
        self.assertEqual(move_str, "grow")
        self.assertIn(move_str, all_moves.keys())
        self.assertEqual(move_cls, all_moves["grow"])
        self.assertTrue(callable(move_cls))

    @patch("bart_playground.samplers.all_moves")
    def test_one_iter_acceptance(self, mock_all_moves):
        """
        Test the acceptance branch in the one_iter method
        It sets prior.n_trees to 1 and makes prior.trees_log_mh_ratio return 0, so that exp(0)=1, always accepting.
        It configures a dummy generator so that choice returns "grow" and uniform returns 0.5 (below 1).
        It calls one_iter and verifies that mock_move.propose is called, prior.resample_leaf_vals is called and mock_move.proposed.update_leaf_vals is called, and that global_params is updated to "updated_global".
        """
        prior = MagicMock()
        prior.n_trees = 1
        prior.tree_prior.n_trees = 1
        prior.tree_prior.resample_leaf_vals.return_value = np.array([0.5], dtype=np.float32)
        prior.global_prior.resample_global_params.return_value = {"eps_sigma2": 1.0}
        prior.tree_prior.trees_log_prior_ratio.return_value = 0
        prior.likelihood.trees_log_marginal_lkhd_ratio.return_value = 0

        trees = [Tree.new(dataX=self.dataset.X) for _ in range(1)]
        current = Parameters(trees, {"eps_sigma2": 1.0})
        proposed_trees = [Tree.new(dataX=self.dataset.X) for _ in range(1)]
        proposed_state = Parameters(proposed_trees, {"eps_sigma2": 1.0})

        mock_move = MagicMock()
        mock_move.proposed = proposed_state
        mock_move.current = current
        mock_move.log_tran_ratio = 0
        mock_move.propose.return_value = True
        mock_move_function = MagicMock(return_value=mock_move)
        mock_all_moves.__getitem__.return_value = mock_move_function

        dummy_gen = MagicMock()
        dummy_gen.choice = MagicMock(return_value=np.array(["grow"] * 100))
        dummy_gen.uniform = MagicMock(return_value=0.5)

        sampler = DefaultSampler(prior, default_proposal_probs, dummy_gen)
        sampler.add_data(self.dataset)
        sampler.add_thresholds(DefaultPreprocessor.test_thresholds(self.dataset.X))
        temp = sampler.temp_schedule(0)

        with patch.object(proposed_state, 'update_leaf_vals') as mock_update_leaf_vals:
            new_state = sampler.one_iter(current, temp)
            
            # Check that Move.propose is called
            mock_move.propose.assert_called_once_with(dummy_gen)
            # Check that prior.resample_leaf_vals is correctly called
            prior.tree_prior.resample_leaf_vals.assert_called_once()
            # Check that Move.proposed.update_leaf_vals is called
            mock_update_leaf_vals.assert_called_once_with([0], np.array([0.5], dtype=np.float32))
            # Check that prior.resample_global_params is called
            prior.global_prior.resample_global_params.assert_called_once()
            self.assertEqual(new_state.global_params, {"eps_sigma2": 1.0})

    @patch("bart_playground.samplers.all_moves")
    def test_one_iter_rejection(self, mock_all_moves):
        """
        Test the rejection branch in the one_iter method
        It sets prior.trees_log_mh_ratio to return -10, so that exp(-10) is nearly 0, always rejecting.
        It creates a mock move similar to before, but mock_move.proposed.update_leaf_vals should not be called.
        It verifies that global_params is updated via prior.resample_global_params.
        """
        prior = MagicMock()
        prior.n_trees = 1
        prior.tree_prior.n_trees = 1
        prior.global_prior.resample_global_params.return_value = {"eps_sigma2": 1.0}
        prior.tree_prior.trees_log_prior_ratio.return_value = -5
        prior.likelihood.trees_log_marginal_lkhd_ratio.return_value = -5

        trees = [Tree.new(dataX=self.dataset.X) for _ in range(1)]
        current = Parameters(trees, {"eps_sigma2": 1.0})

        mock_move = MagicMock()
        mock_move.proposed = MagicMock()
        mock_move.current = current
        mock_move.log_tran_ratio = 0
        mock_move.propose.return_value = True
        mock_move_function = MagicMock(return_value=mock_move)
        mock_all_moves.__getitem__.return_value = mock_move_function

        dummy_gen = MagicMock()
        dummy_gen.choice = MagicMock(return_value=np.array(["grow"] * 100))
        dummy_gen.uniform = MagicMock(return_value=0.5)

        sampler = DefaultSampler(prior, default_proposal_probs, dummy_gen)
        sampler.add_data(self.dataset)
        sampler.add_thresholds(DefaultPreprocessor.test_thresholds(self.dataset.X))
        temp = sampler.temp_schedule(0)

        new_state = sampler.one_iter(current, temp)
        mock_move.propose.assert_called_once_with(dummy_gen)
        # In the rejection branch, update_leaf_vals should not be called
        mock_move.proposed.update_leaf_vals.assert_not_called()
        prior.global_prior.resample_global_params.assert_called_once()
        self.assertEqual(new_state.global_params, {"eps_sigma2": 1.0})

    @patch("bart_playground.samplers.all_moves")
    def test_run_method(self, mock_all_moves):
        """
        Test the run method
        It calls run with 2 iterations and verifies that sampler.n_iter equals 2 and that the length of sampler.trace equals 2 + 1 = 3.
        """
        prior = MagicMock()
        prior.n_trees = 1
        prior.tree_prior.n_trees = 1
        prior.tree_prior.resample_leaf_vals.return_value = np.array([0.5], dtype=np.float32)
        prior.global_prior.resample_global_params.return_value = {"eps_sigma2": 1.0}
        prior.tree_prior.trees_log_prior_ratio.return_value = 0
        prior.likelihood.trees_log_marginal_lkhd_ratio.return_value = 0

        trees = [Tree.new(dataX=self.dataset.X) for _ in range(prior.tree_prior.n_trees)]
        initial_state = Parameters(trees, {"eps_sigma2": 1.0})
        proposed_state = Parameters([Tree.new(dataX=self.dataset.X) for _ in range(1)], {"eps_sigma2": 1.0})

        mock_move = MagicMock()
        mock_move.proposed = proposed_state
        mock_move.current = initial_state
        mock_move.log_tran_ratio = 0
        mock_move.propose.return_value = True
        mock_move_function = MagicMock(return_value=mock_move)
        mock_all_moves.__getitem__.return_value = mock_move_function

        dummy_gen = MagicMock()
        dummy_gen.choice = MagicMock(return_value=np.array(["grow"] * 100))
        dummy_gen.uniform = MagicMock(return_value=0.5)

        sampler = DefaultSampler(prior, default_proposal_probs, dummy_gen)
        sampler.add_data(self.dataset)
        sampler.add_thresholds(DefaultPreprocessor.test_thresholds(self.dataset.X))

        sampler.get_init_state = MagicMock(return_value=initial_state)

        sampler.run(2, progress_bar=False)
        self.assertEqual(sampler.n_iter, 2)
        self.assertEqual(len(sampler.trace), 3)

class TestSamplers2(unittest.TestCase):

    def setUp(self):
        X, y = np.random.rand(100, 5), np.random.rand(100)
        self.dataset = Dataset(X, y)
        self.trees = [Tree.new(dataX=self.dataset.X) for _ in range(5)]
        self.tree_params = Parameters(self.trees, None, None)
        # Ensure every tree has at least one split using Parameters
        for tree in self.tree_params.trees:
            tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)
        self.alpha = 0.5
        self.beta = 0.5
        self.prior = ComprehensivePrior(n_trees=len(self.trees), tree_alpha=self.alpha, tree_beta=self.beta)
        # self.prior.fit(self.dataset)
        self.generator = np.random.default_rng(42)
        self.temp_schedule = TemperatureSchedule()
        self.sampler = DefaultSampler(
            prior=self.prior, 
            proposal_probs={"grow": 0.5, "prune": 0.5},
            generator=self.generator,
            temp_schedule=self.temp_schedule
        )
        self.sampler.add_data(self.dataset)
        self.sampler.add_thresholds(DefaultPreprocessor.test_thresholds(self.dataset.X))
    
    def test_add_data(self):
        self.assertIsNotNone(self.sampler.data)
        self.assertEqual(self.sampler.data.X.shape, (100, 5))
        self.assertEqual(self.sampler.data.y.shape, (100,))
    
    def test_get_init_state(self):
        init_state = self.sampler.get_init_state()
        self.assertIsInstance(init_state, Parameters)
        self.assertEqual(len(init_state.trees), self.prior.tree_prior.n_trees)
    
    def test_run(self):
        self.sampler.run(5)
        self.assertEqual(len(self.sampler.trace), 6)
    
    def test_sample_move(self):
        move_str, move_cls = self.sampler.sample_move()
        self.assertIsInstance(move_str, str)
        self.assertIn(move_str, all_moves.keys())
        self.assertEqual(move_cls, all_moves[move_str])

    def test_trees_log_mh_ratio(self):
        """
        Test computing MH ratio.
        """
        mock_move = MagicMock()
        mock_move.current = MagicMock()
        mock_move.proposed = MagicMock()
        mock_move.trees_changed = [0]
        mock_move.log_tran_ratio = 0

        self.sampler.tree_prior = MagicMock()
        self.sampler.tree_prior.trees_log_prior_ratio = MagicMock(return_value=-0.3)
        self.sampler.likelihood = MagicMock()
        self.sampler.likelihood.trees_log_marginal_lkhd_ratio = MagicMock(return_value=0.7)
        self.sampler.proposals = {"grow": 0.5, "prune": 0.5}

        temp = 1.0
        move_key = "grow"
        mh_ratio = self.sampler.log_mh_ratio(mock_move, temp, move_key)
        # Expected: (-0.3 + 0.7) / 1.0 + 0 + log(0.5) - log(0.5) = 0.4
        self.assertAlmostEqual(mh_ratio, 0.4)

if __name__ == '__main__':
    unittest.main()
