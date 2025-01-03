import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.params import Tree
import numpy as np

# Test tree traversal functionality
def test_traverse_tree():
    """Test that traverse_tree correctly assigns nodes."""
    tree = Tree()
    tree.vars = np.array([0, -1, -1], dtype=int)
    tree.thresholds = np.array([0.5, np.nan, np.nan])
    tree.leaf_vals = np.array([np.nan, 1.0, -1.0])

    X_test = np.array([
        [0.4],  # Should go to the left child (<= 0.5)
        [0.6],  # Should go to the right child (> 0.5)
    ])

    node_ids = tree.traverse_tree(X_test)
    assert node_ids[0] == 1, "Sample 1 should be assigned to the left leaf node!"
    assert node_ids[1] == 2, "Sample 2 should be assigned to the right leaf node!"
    print("test_traverse_tree passed!")

# Test tree evaluation functionality
def test_evaluate():
    """Test that evaluate returns correct leaf values."""
    tree = Tree()
    tree.vars = np.array([0, -1, -1], dtype=int)
    tree.thresholds = np.array([0.5, np.nan, np.nan])
    tree.leaf_vals = np.array([np.nan, 1.0, -1.0])

    X_test = np.array([
        [0.4],  # Should evaluate to 1.0
        [0.6],  # Should evaluate to -1.0
    ])

    outputs = tree.evaluate(X_test)
    assert outputs[0] == 1.0, "Evaluation failed for sample 1!"
    assert outputs[1] == -1.0, "Evaluation failed for sample 2!"
    print("test_evaluate passed!")

# Test split_leaf functionality
def test_split_leaf():
    """Test that split_leaf correctly updates the tree structure."""
    tree = Tree()
    split_index = 0  # Root node index
    var = 0  # Variable to split on
    threshold = 0.5  # Split threshold
    left_val = 1.0  # Value for the left leaf node
    right_val = -1.0  # Value for the right leaf node

    tree.split_leaf(split_index, var, threshold, left_val, right_val)

    assert tree.vars[split_index] == var, "Root node variable was not set correctly!"
    assert tree.thresholds[split_index] == threshold, "Root node threshold was not set correctly!"
    assert tree.vars[split_index * 2 + 1] == -1, "Left child was not set as a leaf!"
    assert tree.vars[split_index * 2 + 2] == -1, "Right child was not set as a leaf!"
    assert tree.leaf_vals[split_index * 2 + 1] == left_val, "Left leaf value was not set correctly!"
    assert tree.leaf_vals[split_index * 2 + 2] == right_val, "Right leaf value was not set correctly!"
    print("test_split_leaf passed!")

# Test prune_split functionality
def test_prune_split():
    """Test that prune_split correctly prunes a split node into a leaf."""
    tree = Tree()
    tree.vars = np.array([0, -1, -1, -2, -2, -2, -2, -2], dtype=int)
    tree.thresholds = np.array([0.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    tree.leaf_vals = np.array([np.nan, 1.0, -1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    tree.n = np.array([-2, 100, 50, -2, -2, -2, -2, -2], dtype=int)

    prune_index = 0
    tree.prune_split(prune_index)

    assert tree.vars[prune_index] == -1, "Pruned node is not a leaf!"
    assert np.isnan(tree.thresholds[prune_index]), "Threshold of pruned node was not cleared!"
    assert tree.vars[1] == -2, "Left child was not cleared after pruning!"
    assert tree.vars[2] == -2, "Right child was not cleared after pruning!"
    print("test_prune_split passed!")

# Test set_leaf_value functionality
def test_set_leaf_value():
    """Test that set_leaf_value updates the leaf value correctly."""
    tree = Tree()
    split_index = 0  # Root node index
    var = 0  # Variable to split on
    threshold = 0.5  # Split threshold
    left_val = 1.0  # Initial value for the left leaf node
    right_val = -1.0  # Initial value for the right leaf node

    tree.split_leaf(split_index, var, threshold, left_val, right_val)

    new_left_val = 2.0
    left_leaf_index = split_index * 2 + 1
    tree.set_leaf_value(left_leaf_index, new_left_val)

    assert tree.is_leaf(left_leaf_index), "The specified node is not a leaf!"
    assert tree.leaf_vals[left_leaf_index] == new_left_val, "The leaf value was not updated correctly!"
    print("test_set_leaf_value passed!")

# Test is_leaf functionality
def test_is_leaf():
    """Test that is_leaf correctly identifies leaf nodes."""
    tree = Tree()
    tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
    tree.split_leaf(2, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)

    assert tree.is_leaf(1), "Node 1 should be a leaf but is not!"
    assert tree.is_leaf(5), "Node 5 should be a leaf but is not!"
    assert tree.is_leaf(6), "Node 6 should be a leaf but is not!"
    assert not tree.is_leaf(0), "Node 0 should not be a leaf but is incorrectly marked as one!"
    assert not tree.is_leaf(2), "Node 2 should not be a leaf but is incorrectly marked as one!"
    print("test_is_leaf passed!")

# Test is_split_node functionality
def test_is_split_node():
    """Test that is_split_node correctly identifies split nodes."""
    tree = Tree()
    tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
    tree.split_leaf(2, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)

    assert tree.is_split_node(0), "Node 0 should be a split node but is not!"
    assert tree.is_split_node(2), "Node 2 should be a split node but is not!"
    assert not tree.is_split_node(1), "Node 1 should not be a split node but is incorrectly marked as one!"
    assert not tree.is_split_node(5), "Node 5 should not be a split node but is incorrectly marked as one!"
    print("test_is_split_node passed!")

# Test is_terminal_split_node functionality
def test_is_terminal_split_node():
    """Test that is_terminal_split_node correctly identifies terminal split nodes."""
    tree = Tree()
    tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
    tree.split_leaf(2, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)

    assert not tree.is_terminal_split_node(0), "Node 0 should not be a terminal split node but is incorrectly marked as one!"
    assert tree.is_terminal_split_node(2), "Node 2 should be a terminal split node but is not!"
    print("test_is_terminal_split_node passed!")

# Test get_n_leaves functionality
def test_get_n_leaves():
    """Test that get_n_leaves returns the correct number of leaf nodes."""
    tree = Tree()
    tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
    tree.split_leaf(2, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)

    expected_n_leaves = 3  # Nodes 1, 5, and 6 are leaves
    actual_n_leaves = tree.get_n_leaves()

    assert actual_n_leaves == expected_n_leaves, f"Expected {expected_n_leaves} leaves, but got {actual_n_leaves}!"
    print(f"test_get_n_leaves passed!")

# Test get_random_terminal_split functionality
def test_get_random_terminal_split():
    """Test that get_random_terminal_split correctly selects a terminal split node."""
    tree = Tree()
    tree.vars = np.array([0, 1, 1, -1, -1, -1, -1], dtype=int)
    tree.thresholds = np.array([0.5, 0.3, 0.7, np.nan, np.nan, np.nan, np.nan])
    tree.leaf_vals = np.array([np.nan, np.nan, np.nan, 1.0, -1.0, 2.0, -2.0])
    rng = np.random.default_rng(seed=42)

    try:
        random_terminal_split = tree.rand_terminal_split_node(rng)
        assert tree.vars[random_terminal_split] != -1, "Selected node is not a split!"
        assert tree.vars[random_terminal_split * 2 + 1] == -1, "Left child is not a leaf!"
        assert tree.vars[random_terminal_split * 2 + 2] == -1, "Right child is not a leaf!"
        print("test_get_random_terminal_split passed!")
    except ValueError as e:
        print(f"Error: {e}")
        print("No terminal splits available.")

# Test get_random_leaf functionality
def test_get_random_leaf():
    """Test that get_random_leaf correctly selects a leaf node."""
    tree = Tree()
    tree.vars = np.array([0, -1, -1], dtype=int)
    tree.thresholds = np.array([0.5, np.nan, np.nan])
    tree.leaf_vals = np.array([np.nan, 1.0, -1.0])
    rng = np.random.default_rng(seed=0)

    random_leaf = tree.rand_leaf_node(rng)
    assert tree.vars[random_leaf] == -1, "Selected node is not a leaf!"
    print("test_get_random_leaf passed!")

# Test get_random_split functionality
def test_get_random_split():
    """Test that get_random_split correctly selects a split node."""
    tree = Tree()
    tree.vars = np.array([0, -1, 1, -2, -2, -1, -1, -2], dtype=int)
    tree.thresholds = np.array([0.5, np.nan, 0.7, np.nan, np.nan, np.nan, np.nan, np.nan])
    tree.leaf_vals = np.array([np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, -2.0, np.nan])
    rng = np.random.default_rng(seed=42)

    try:
        random_split = tree.rand_split_node(rng)
        assert tree.is_split_node(random_split), f"Node {random_split} is not a split node!"
        print("test_get_random_split passed!")
    except ValueError as e:
        print(f"Error: {e}")
        print("No split nodes available.")


if __name__ == "__main__":
    test_traverse_tree()
    test_evaluate()
    test_split_leaf()
    test_prune_split()
    test_set_leaf_value()
    test_is_leaf()
    test_is_split_node()
    test_is_terminal_split_node()
    test_get_n_leaves()
    test_get_random_terminal_split()
    test_get_random_leaf()
    test_get_random_split()
