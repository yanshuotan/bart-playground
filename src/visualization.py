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
    def add_nodes_edges(node_id):
        if node_id >= len(tree_structure.var) or tree_structure.var[node_id] == -2:
            # If node_id is out of range or invalid, skip processing
            return

        if tree_structure.var[node_id] == -1:
            # Add leaf node with its value
            leaf_value = tree_params.leaf_vals[node_id]
            dot.node(str(node_id), f"Leaf\nValue: {leaf_value:.2f}", shape="box")
        else:
            # Add split node with variable and split value
            var = tree_structure.var[node_id]
            split = tree_structure.split[node_id]
            dot.node(str(node_id), f"Var: X_{var}\nSplit: {split:.2f}")

            # Recursively add left and right children
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2

            add_nodes_edges(left_child)
            add_nodes_edges(right_child)

            # Add edges to children
            if left_child < len(tree_structure.var) and tree_structure.var[left_child] != -2:
                dot.edge(str(node_id), str(left_child), label="Left")
            if right_child < len(tree_structure.var) and tree_structure.var[right_child] != -2:
                dot.edge(str(node_id), str(right_child), label="Right")

    # Start with the root node (node_id = 0)
    add_nodes_edges(0)

    # Render and save the tree visualization
    dot.render(filename, cleanup=True)
    return dot
