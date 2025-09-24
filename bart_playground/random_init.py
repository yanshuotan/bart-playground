import numpy as np
from .params import Tree

def create_random_init_trees(n_trees, dataX, possible_thresholds=None, generator=None, dirichlet_s=None):
    """
    Create a list of trees with random initial splits.
    
    Parameters:
    - n_trees: Number of trees to create
    - dataX: Input data matrix
    - possible_thresholds: Dictionary of possible threshold values
    - generator: Random number generator
    - dirichlet_s: Dirichlet probabilities for variable selection
    
    Returns:
    - List of Tree objects with random initial splits
    """
    
    if generator is None:
        generator = np.random.default_rng()
    
    trees = []
    for _ in range(n_trees):
        tree = Tree.new_with_random_split(
            dataX=dataX,
            possible_thresholds=possible_thresholds,
            generator=generator,
            dirichlet_s=dirichlet_s
        )
        trees.append(tree)
    return trees