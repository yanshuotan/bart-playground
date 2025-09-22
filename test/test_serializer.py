import numpy as np

from bart_playground.serializer import (
    NDArrayDTO,
    tree_to_dto,
    dto_to_tree,
    params_to_dto,
    dto_to_params,
    tree_to_json,
    tree_from_json,
    params_to_json,
    params_from_json,
    _to_builtin,
)
from bart_playground.params import Tree, Parameters


def assert_array_equal_strict_dtype(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    if np.issubdtype(a.dtype, np.floating):
        np.testing.assert_allclose(a, b, equal_nan=True)
    else:
        assert np.array_equal(a, b)


def assert_array_equal_loose_int_dtype(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape
    # allow int dtype normalization (e.g., int64 -> int32)
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, equal_nan=True)
    else:
        assert np.array_equal(a, b)


def assert_tree_equal(t1: Tree, t2: Tree, *, check_dataX: bool):
    if check_dataX:
        assert t2.dataX is not None
        assert_array_equal_strict_dtype(t1.dataX, t2.dataX)
    else:
        assert t2.dataX is None

    assert_array_equal_loose_int_dtype(t1.vars, t2.vars)
    assert_array_equal_strict_dtype(t1.thresholds, t2.thresholds)
    assert_array_equal_strict_dtype(t1.leaf_vals, t2.leaf_vals)

    if t1.n is None:
        assert t2.n is None
    else:
        assert_array_equal_loose_int_dtype(t1.n, t2.n)

    if t1.leaf_ids is None:
        assert t2.leaf_ids is None
    else:
        assert_array_equal_loose_int_dtype(t1.leaf_ids, t2.leaf_ids)

    if t1.evals is None:
        assert t2.evals is None
    else:
        assert_array_equal_strict_dtype(t1.evals, t2.evals)

    assert str(t1.float_dtype) == str(t2.float_dtype)


def test_ndarraydto_roundtrip_various_dtypes_and_views():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64, np.int32):
        arr = (rng.standard_normal((5, 4)).astype(dtype) if np.issubdtype(dtype, np.floating)
               else np.arange(20, dtype=dtype).reshape(5, 4))
        noncontig = arr[:, ::2]
        for original in (arr, noncontig):
            dto = NDArrayDTO.from_array(original)
            restored = dto.to_array()
            assert_array_equal_strict_dtype(original, restored)


def _build_sample_tree(with_data=True) -> Tree:
    rng = np.random.default_rng(42)
    dataX = rng.standard_normal((10, 3)).astype(np.float32) if with_data else None
    tree = Tree.new(dataX) if with_data else Tree.new(None)

    if with_data:
        # Introduce a split for non-trivial caches/structure
        is_valid = tree.split_leaf(node_id=0, var=1, threshold=np.float32(0.0), left_val=np.float32(1.5), right_val=np.float32(-2.0))
        assert is_valid is True
        tree.update_outputs()
    return tree


def test_tree_dto_roundtrip_with_and_without_dataX():
    tree = _build_sample_tree(with_data=True)

    dto_with = tree_to_dto(tree, include_dataX=True)
    restored_with = dto_to_tree(dto_with)
    assert_tree_equal(tree, restored_with, check_dataX=True)

    dto_wo = tree_to_dto(tree, include_dataX=False)
    restored_wo = dto_to_tree(dto_wo)
    assert_tree_equal(tree, restored_wo, check_dataX=False)


def test_tree_json_helpers():
    tree = _build_sample_tree(with_data=True)
    s = tree_to_json(tree, include_dataX=True)
    restored = tree_from_json(s)
    assert_tree_equal(tree, restored, check_dataX=True)


def _build_sample_params() -> Parameters:
    t1 = _build_sample_tree(with_data=True)
    t2 = _build_sample_tree(with_data=True)
    global_params = {
        "alpha": np.float64(0.3),
        "beta": np.int32(2),
        "arr_small": np.array([1, 2], dtype=np.int64),
        "nested": {"x": np.float32(1.0), "y": [np.float64(2.0), np.int32(3)]},
    }
    params = Parameters([t1, t2], global_params=global_params, cache=None)
    # Ensure cache initialized
    assert params.cache is not None
    return params


def assert_params_equivalent(p1: Parameters, p2: Parameters):
    assert p1.n_trees == p2.n_trees
    for a, b in zip(p1.trees, p2.trees):
        # dataX may or may not be present depending on serialization flags; compare structure/caches only
        assert_tree_equal(a, b, check_dataX=b.dataX is not None)

    # global_params are normalized to builtins by serializer
    assert p2.global_params == _to_builtin(p1.global_params)


def test_params_dto_roundtrip_with_and_without_cache():
    params = _build_sample_params()

    # include_cache=True should preserve cache exactly
    dto_with = params_to_dto(params, include_dataX=False, include_cache=True)
    restored_with = dto_to_params(dto_with)
    assert_params_equivalent(params, restored_with)
    assert_array_equal_strict_dtype(params.cache, restored_with.cache)

    # include_cache=False should recompute cache equal to original
    dto_wo = params_to_dto(params, include_dataX=False, include_cache=False)
    restored_wo = dto_to_params(dto_wo)
    assert_params_equivalent(params, restored_wo)
    assert_array_equal_strict_dtype(params.cache, restored_wo.cache)


def test_params_json_helpers_with_and_without_cache():
    params = _build_sample_params()

    # include_cache=True
    s_with = params_to_json(params, include_dataX=False, include_cache=True)
    restored_with = params_from_json(s_with)
    assert_params_equivalent(params, restored_with)
    assert_array_equal_strict_dtype(params.cache, restored_with.cache)

    # include_cache=False
    s_wo = params_to_json(params, include_dataX=False, include_cache=False)
    restored_wo = params_from_json(s_wo)
    assert_params_equivalent(params, restored_wo)
    assert_array_equal_strict_dtype(params.cache, restored_wo.cache)


