import numpy as np
import json
from .params import Tree
from typing import Optional, Tuple


def fit_and_init_trees(
    X, y, dataX, model=None, params=None, n_estimators=100, debug=False, **fit_kwargs
):
    import json
    import xgboost as xgb
    from .params import Tree

    if model is None:
        params = params or {}
        model = xgb.XGBRegressor(n_estimators=n_estimators, **params)
        model.fit(X, y, **fit_kwargs)
    booster = model.get_booster()

    dumps = booster.get_dump(dump_format='json')
    init_trees = []
    for i, tree_json in enumerate(dumps):
        if debug:
            print(f"--- XGBoost JSON tree {i} ---\n{tree_json}\n")
        parsed = json.loads(tree_json)
        t = _xgb_json_to_tree(parsed, dataX, debug=debug)
        if debug:
            print(f"+++ Converted BART Tree {i} +++\n{t}\n")
        init_trees.append(t)

    return model, init_trees


def _subtree_leaf_value(node: dict) -> float:
    """Return a stable fallback value when an imported split is invalid."""
    if 'leaf' in node:
        return float(node['leaf'])

    leaf_values = [_subtree_leaf_value(child) for child in node.get('children', [])]
    if not leaf_values:
        return 0.0
    return float(np.mean(leaf_values))


def _xgb_threshold_to_bart_threshold(threshold: float, dtype) -> float:
    """
    XGBoost routes numeric splits with a strict '<' comparison, while Tree uses
    '<='. Step one representable value lower so equal values follow XGBoost's
    right branch under BART traversal.
    """
    if not np.isfinite(threshold):
        return float(threshold)

    float_dtype = dtype if np.issubdtype(dtype, np.floating) else np.float32
    threshold_arr = np.asarray(threshold, dtype=float_dtype)
    lower_arr = np.nextafter(
        threshold_arr,
        np.asarray(-np.inf, dtype=float_dtype),
    )
    return float(lower_arr.item())


def _collapse_to_leaf(t: Tree, node_id: int, leaf_value: float):
    if t.is_split_node(node_id):
        t.prune_split(node_id, recursive=True)
    t.set_leaf_value(node_id, leaf_value)
    t.update_outputs()


def _xgb_json_to_tree(node: dict, dataX: np.ndarray, debug: bool = False) -> Tree:
    """
    Recursively convert an XGBoost JSON tree into a BART Tree via heap indexing.
    """
    t = Tree.new(dataX)
    mapping: dict = {node['nodeid']: 0}

    def recurse(n: dict):
        old_id = n['nodeid']
        idx = mapping[old_id]
        # Leaf
        if 'leaf' in n:
            t.set_leaf_value(idx, float(n['leaf']))
        else:
            # Internal split
            feat = n.get('split_feature', n.get('split'))
            var = int(str(feat).lstrip('f'))
            raw_thr = float(n.get('split_condition', n.get('threshold', n.get('split'))))
            thr = _xgb_threshold_to_bart_threshold(raw_thr, dataX.dtype)
            is_valid = t.split_leaf(idx, var, thr)
            if not is_valid:
                leaf_value = _subtree_leaf_value(n)
                _collapse_to_leaf(t, idx, leaf_value)
                if debug:
                    print(
                        "[xgb_init] collapsed invalid split "
                        f"node={idx}, var={var}, raw_threshold={raw_thr}, "
                        f"bart_threshold={thr}, leaf_value={leaf_value}"
                    )
                return
            # Map children
            left_old, right_old = n['yes'], n['no']
            mapping[left_old] = idx * 2 + 1
            mapping[right_old] = idx * 2 + 2
            for child in n.get('children', []):
                recurse(child)

    recurse(node)
    t.update_outputs()
    return t