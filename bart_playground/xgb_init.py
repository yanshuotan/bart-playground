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
        t = _xgb_json_to_tree(parsed, dataX)
        if debug:
            print(f"+++ Converted BART Tree {i} +++\n{t}\n")
        init_trees.append(t)

    return model, init_trees


def _xgb_json_to_tree(node: dict, dataX: np.ndarray) -> Tree:
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
            thr = float(n.get('split_condition', n.get('threshold', n.get('split'))))
            t.split_leaf(idx, var, thr)
            # Map children
            left_old, right_old = n['yes'], n['no']
            mapping[left_old] = idx * 2 + 1
            mapping[right_old] = idx * 2 + 2
            for child in n.get('children', []):
                recurse(child)

    recurse(node)
    t.update_outputs()
    return t