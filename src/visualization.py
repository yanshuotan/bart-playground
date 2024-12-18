from graphviz import Digraph

def visualize_tree(tree_structure, tree_params, filename: str = "tree", format: str = "png"):
    """
    Visualize a tree structure using Graphviz.

    Parameters:
    - tree_structure: TreeStructure
        The structure of the tree to visualize.
    - tree_params: TreeParams
        The parameters of the tree (e.g., leaf values).
    - filename: str
        The name of the output file (without extension).
    - format: str
        The format of the output file (e.g., "png", "pdf").

    Returns:
    - graphviz.Digraph
        The Graphviz object representing the tree.
    """
    dot = Digraph(comment="Tree Visualization", format=format)

    # Helper function to recursively add nodes and edges
    def add_nodes_edges(node_id, is_leaf, var, split, left_child=None, right_child=None):
        if is_leaf:
            # Add leaf node with its value
            leaf_value = tree_params.leaf_vals[node_id]
            dot.node(str(node_id), f"Leaf\nValue: {leaf_value:.2f}", shape="box")
        else:
            # Add split node with variable and split value
            dot.node(str(node_id), f"Var: {var}\nSplit: {split:.2f}")
            # Recursively add left and right children
            add_nodes_edges(left_child, tree_structure.var[left_child] is None,
                            tree_structure.var[left_child], tree_structure.split[left_child],
                            left_child=left_child * 2 + 1, right_child=left_child * 2 + 2)
            add_nodes_edges(right_child, tree_structure.var[right_child] is None,
                            tree_structure.var[right_child], tree_structure.split[right_child],
                            left_child=right_child * 2 + 1, right_child=right_child * 2 + 2)
            # Add edges to children
            dot.edge(str(node_id), str(left_child), label="Left")
            dot.edge(str(node_id), str(right_child), label="Right")

    # Start with the root node (node_id = 0)
    add_nodes_edges(0, tree_structure.var[0] is None, tree_structure.var[0], tree_structure.split[0])

    # Render and save the tree visualization
    dot.render(filename, cleanup=True)
    return dot