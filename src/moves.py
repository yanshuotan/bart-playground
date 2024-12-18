import copy

from params import BARTParams

class Move:
    """
    Base class for moves in the BART sampler.
    """
    def __init__(self, random_state: int, current : BARTParams, trees_changed: np.ndarray):
        """
        Initialize the move.

        Parameters:
        - random_state: int
            Random state for reproducibility.
        - current: BARTParams
            Current state of the BART model.
        - trees_changed: np.ndarray
            Indices of trees that were changed.
        """
        self.random_state = random_state
        self.current = current
        self.proposed = None
        self.trees_changed = trees_changed

    def propose(self):
        """
        Propose a new state.
        """

class Grow(Move):
    """
    Move to grow a tree.
    """
    def __init__(self, random_state: int, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(random_state, current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, var, threshold, generator):
        self.proposed = copy.deepcopy(self.current)
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = tree.get_random_leaf(generator)
        tree.split_leaf(node_id, var, threshold)

class Prune(Move):
    """
    Move to prune a tree.
    """
    def __init__(self, random_state: int, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(random_state, current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, generator):
        self.proposed = copy.deepcopy(self.current)
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = tree.get_random_terminal_split(generator)
        tree.prune_split(node_id)

class Change(Move):
    """
    Move to change a tree.
    """
    def __init__(self, random_state: int, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(random_state, current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, var, threshold, generator):
        self.proposed = copy.deepcopy(self.current)
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = tree.get_random_split(generator)
        tree.vars[node_id] = var
        tree.thresholds[node_id] = threshold


class Swap(Move):
    """
    Move to swap two trees.
    """
    pass