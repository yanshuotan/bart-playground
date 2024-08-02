class Node:

    def __init__(self, split: Split, depth: int, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self.depth = depth
        self._split = split
        self._left_child = left_child
        self._right_child = right_child

    @property
    def data(self) -> Data:
        return self.split.data

    @property
    def left_child(self) -> 'TreeNode':
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        return self._right_child

    @property
    def split(self):
        return self._split