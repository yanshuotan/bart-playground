import unittest
import numpy as np
from bart_playground import Tree

class TestTreeDataIndices(unittest.TestCase):
    
    def setUp(self):
        """Setup test data and tree"""
        np.random.seed(42)
        self.n_samples = 20
        self.n_features = 3
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.tree = Tree.new(self.X)
    
    def test_initial_leaf_data_indices(self):
        """Test that initial tree has all data in root node"""
        self.assertIsNotNone(self.tree.leaf_data_indices)
        self.assertEqual(len(self.tree.leaf_data_indices), 1)
        self.assertIn(0, self.tree.leaf_data_indices)
        self.assertEqual(self.tree.leaf_data_indices[0], list(range(self.n_samples)))
        
    def test_split_leaf_data_indices(self):
        """Test that split_leaf correctly redistributes data indices"""
        # Split root node
        var = 0
        threshold = 0.0
        is_valid = self.tree.split_leaf(0, var, threshold, -1.0, 1.0)
        
        self.assertTrue(is_valid)
        
        # Check that root node is no longer in leaf_data_indices
        self.assertNotIn(0, self.tree.leaf_data_indices)
        
        # Check that children nodes are in leaf_data_indices
        left_child = 1
        right_child = 2
        self.assertIn(left_child, self.tree.leaf_data_indices)
        self.assertIn(right_child, self.tree.leaf_data_indices)
        
        # Verify data redistribution
        left_indices = self.tree.leaf_data_indices[left_child]
        right_indices = self.tree.leaf_data_indices[right_child]
        
        # All original data should be distributed to children
        all_redistributed = sorted(left_indices + right_indices)
        self.assertEqual(all_redistributed, list(range(self.n_samples)))
        
        # Verify split criterion
        for idx in left_indices:
            self.assertLessEqual(self.X[idx, var], threshold)
        for idx in right_indices:
            self.assertGreater(self.X[idx, var], threshold)
        
        # Check that both children have data (valid split)
        self.assertGreater(len(left_indices), 0)
        self.assertGreater(len(right_indices), 0)
        
    def test_multiple_splits(self):
        """Test multiple consecutive splits"""
        # First split
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        # Split left child
        left_child = 1
        if len(self.tree.leaf_data_indices[left_child]) > 1:
            self.tree.split_leaf(left_child, 1, 0.0, -2.0, 0.0)
            
            # Check structure
            self.assertNotIn(left_child, self.tree.leaf_data_indices)
            
            left_left = 3  # 1*2+1
            left_right = 4  # 1*2+2
            
            if left_left in self.tree.leaf_data_indices and left_right in self.tree.leaf_data_indices:
                # Verify data conservation
                all_data = []
                for leaf_id in self.tree.leaf_data_indices:
                    all_data.extend(self.tree.leaf_data_indices[leaf_id])
                
                self.assertEqual(sorted(all_data), list(range(self.n_samples)))
    
    def test_prune_split_data_indices(self):
        """Test that prune_split correctly merges data indices"""
        # Split root
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        # Store original data distribution
        original_left = self.tree.leaf_data_indices[1].copy()
        original_right = self.tree.leaf_data_indices[2].copy()
        
        # Prune the split
        self.tree.prune_split(0)
        
        # Check that root is back as a leaf with all data
        self.assertIn(0, self.tree.leaf_data_indices)
        self.assertEqual(len(self.tree.leaf_data_indices), 1)
        
        # All data should be back in root
        merged_data = sorted(self.tree.leaf_data_indices[0])
        self.assertEqual(merged_data, list(range(self.n_samples)))
        
        # Verify it's the union of original left and right
        expected_data = sorted(original_left + original_right)
        self.assertEqual(merged_data, expected_data)
        
    def test_change_split_data_indices(self):
        """Test that change_split correctly redistributes data"""
        # Split root
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        # Change the split
        new_var = 1
        new_threshold = 0.5
        is_valid = self.tree.change_split(0, new_var, new_threshold)
        
        if is_valid:
            # Verify new split criterion
            left_indices = self.tree.leaf_data_indices[1]
            right_indices = self.tree.leaf_data_indices[2]
            
            for idx in left_indices:
                self.assertLessEqual(self.X[idx, new_var], new_threshold)
            for idx in right_indices:
                self.assertGreater(self.X[idx, new_var], new_threshold)
                
            # All data should still be accounted for
            all_data = sorted(left_indices + right_indices)
            self.assertEqual(all_data, list(range(self.n_samples)))
    
    def test_consistency_with_n_array(self):
        """Test that leaf_data_indices is consistent with n array"""
        # Split root
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        # Check consistency
        for leaf_id in self.tree.leaf_data_indices:
            expected_n = len(self.tree.leaf_data_indices[leaf_id])
            actual_n = self.tree.n[leaf_id]
            self.assertEqual(actual_n, expected_n, 
                           f"Mismatch at leaf {leaf_id}: n={actual_n}, len(indices)={expected_n}")
    
    def test_deep_tree_data_indices(self):
        """Test data indices with a deeper tree structure"""
        # Build a small tree: root -> split -> two leaves, one of which splits again
        
        # Level 0: split root
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        # Level 1: split right child if it has enough data
        right_child = 2
        if len(self.tree.leaf_data_indices[right_child]) > 1:
            self.tree.split_leaf(right_child, 1, 0.0, -2.0, 2.0)
            
            # Check all leaves have data
            total_data = 0
            for leaf_id in self.tree.leaf_data_indices:
                self.assertGreater(len(self.tree.leaf_data_indices[leaf_id]), 0)
                total_data += len(self.tree.leaf_data_indices[leaf_id])
            
            # Total data should equal original sample count
            self.assertEqual(total_data, self.n_samples)

    def test_deep_tree_structure(self):
        """Test deep tree structure with high-numbered leaf nodes"""
        # Use larger dataset to ensure splits are possible
        np.random.seed(123)
        large_X = np.random.randn(50, 3).astype(np.float32)
        tree = Tree.new(large_X)
        
        # Build deeper tree structure systematically
        # Level 0: Split root (0) -> leaves [1, 2]
        self.assertTrue(tree.split_leaf(0, 0, 0.0, -1.0, 1.0))
        
        # Level 1: Split both children -> leaves [3, 4, 5, 6]
        if len(tree.leaf_data_indices[1]) > 1:
            self.assertTrue(tree.split_leaf(1, 1, 0.0, -2.0, 0.0))
        if len(tree.leaf_data_indices[2]) > 1:
            self.assertTrue(tree.split_leaf(2, 2, 0.0, 2.0, 3.0))
        
        # Level 2: Split one more leaf to get higher numbers
        current_leaves = sorted(tree.leaf_data_indices.keys())
        if len(current_leaves) >= 3 and len(tree.leaf_data_indices[current_leaves[2]]) > 1:
            leaf_to_split = current_leaves[2]  # Pick the third leaf
            self.assertTrue(tree.split_leaf(leaf_to_split, 0, 0.5, 10.0, 20.0))
        
        # Verify we have high-numbered leaves
        final_leaves = sorted(tree.leaf_data_indices.keys())
        max_leaf = max(final_leaves)
        self.assertGreaterEqual(max_leaf, 5, f"Expected leaf >= 5, got max leaf {max_leaf}")
        
        # Verify data conservation
        total_data = sum(len(tree.leaf_data_indices[leaf]) for leaf in final_leaves)
        self.assertEqual(total_data, 50)
        
        # Verify all leaves have data
        for leaf_id in final_leaves:
            self.assertGreater(len(tree.leaf_data_indices[leaf_id]), 0, 
                            f"Leaf {leaf_id} has no data")
        
        # Test that evaluation works on deep tree
        predictions = tree.evaluate(large_X)
        self.assertEqual(len(predictions), 50)
        self.assertTrue(np.isfinite(predictions).all())
        
        print(f"Deep tree created with leaves: {final_leaves}")
        print(f"Max leaf node: {max_leaf}")
    
    def test_data_indices_no_duplicates(self):
        """Test that no data index appears in multiple leaves"""
        # Create a tree with multiple splits
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        if len(self.tree.leaf_data_indices[1]) > 1:
            self.tree.split_leaf(1, 1, 0.0, -2.0, 0.0)
        
        # Collect all data indices
        all_indices = []
        for leaf_id in self.tree.leaf_data_indices:
            leaf_indices = self.tree.leaf_data_indices[leaf_id]
            all_indices.extend(leaf_indices)
        
        # Check no duplicates
        self.assertEqual(len(all_indices), len(set(all_indices)))
        
        # Check all original indices are present
        self.assertEqual(sorted(all_indices), list(range(self.n_samples)))
    
    def test_edge_case_empty_split(self):
        """Test behavior when a split would result in empty children"""
        # Try to split with a threshold that puts all data on one side
        extreme_threshold = self.X[:, 0].max() + 1.0
        
        # This should return False (invalid split)
        is_valid = self.tree.split_leaf(0, 0, extreme_threshold, -1.0, 1.0)
        
        # The split should be invalid, and tree should remain unchanged
        self.assertFalse(is_valid)
    
    def test_tree_copy_preserves_data_indices(self):
        """Test that tree copying preserves leaf_data_indices"""
        # Split the tree
        self.tree.split_leaf(0, 0, 0.0, -1.0, 1.0)
        
        # Copy the tree
        copied_tree = self.tree.copy()
        
        # Check that leaf_data_indices is preserved
        self.assertEqual(len(copied_tree.leaf_data_indices), len(self.tree.leaf_data_indices))
        
        for leaf_id in self.tree.leaf_data_indices:
            self.assertIn(leaf_id, copied_tree.leaf_data_indices)
            self.assertEqual(copied_tree.leaf_data_indices[leaf_id], 
                           self.tree.leaf_data_indices[leaf_id])
        
        # Verify independence (modifying copy shouldn't affect original)
        if 1 in copied_tree.leaf_data_indices and len(copied_tree.leaf_data_indices[1]) > 1:
            copied_tree.split_leaf(1, 1, 0.0, -2.0, 0.0)
            
            # Original should be unchanged
            self.assertNotEqual(len(self.tree.leaf_data_indices), len(copied_tree.leaf_data_indices))

    def test_evaluate_after_split(self):
        """Test tree evaluation after splitting"""
        # Split the tree
        var = 0
        threshold = 0.0
        left_val = -1.0
        right_val = 1.0
        
        self.tree.split_leaf(0, var, threshold, left_val, right_val)
        
        # Evaluate the tree
        predictions = self.tree.evaluate(self.X)
        
        # Check predictions match split criterion
        for i, pred in enumerate(predictions):
            if self.X[i, var] <= threshold:
                self.assertEqual(pred, left_val)
            else:
                self.assertEqual(pred, right_val)


if __name__ == '__main__':
    unittest.main()