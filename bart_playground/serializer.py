"""
Pydantic DTOs and helpers to serialize/deserialize Tree and Parameters.

This module provides JSON-friendly Data Transfer Objects (DTOs) for the
`Tree` and `Parameters` classes defined in `bart_playground.params`, along with
helpers to convert to/from those runtime objects.

Design notes:
- NumPy arrays are encoded as base64 with explicit dtype and shape to preserve
  exact types and avoid large JSON lists. This keeps payloads compact and
  lossless for float32/float64/int32 arrays used by the codebase.
- `dataX` can be optionally included/excluded when serializing to control
  payload size. Cached arrays (`n`, `leaf_ids`, `evals`, `cache`) are
  serialized if present.

Usage:
- Create JSON from Parameters: `to_json(params)`
- Restore Parameters from JSON: `from_json(json_str)`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import is_dataclass, asdict
import base64
import json

import numpy as np
from pydantic import BaseModel

# Local runtime classes
from .params import Tree, Parameters
from .bart import DefaultBART, BART
if TYPE_CHECKING:
    from .mcbart import MultiChainBART
from .util import DefaultPreprocessor


class NDArrayDTO(BaseModel):
    """Lossless encoding for a NumPy array as dtype + shape + base64 data."""

    shape: Tuple[int, ...]
    dtype: str
    data: str  # base64-encoded little-endian bytes from `arr.tobytes()`

    @staticmethod
    def from_array(arr: np.ndarray) -> "NDArrayDTO":
        # Ensure contiguous memory for stable tobytes
        a = np.ascontiguousarray(arr)
        data_b64 = base64.b64encode(a.tobytes()).decode("ascii")
        return NDArrayDTO(shape=a.shape, dtype=str(a.dtype), data=data_b64)

    def to_array(self) -> np.ndarray:
        raw = base64.b64decode(self.data.encode("ascii"))
        arr = np.frombuffer(raw, dtype=np.dtype(self.dtype)).copy()
        if self.shape:
            arr = arr.reshape(self.shape)
        return arr


class TreeDTO(BaseModel):
    """Serializable state for a single Tree."""

    # Optional dataset
    dataX: Optional[NDArrayDTO] = None

    # Core structure
    vars: NDArrayDTO
    thresholds: NDArrayDTO
    leaf_vals: NDArrayDTO

    # Optional caches
    n: Optional[NDArrayDTO] = None
    leaf_ids: Optional[NDArrayDTO] = None
    evals: Optional[NDArrayDTO] = None

    # Redundant but handy for quick inspection
    float_dtype: str


class ParametersDTO(BaseModel):
    """Serializable state for full Parameters object."""

    trees: List[TreeDTO]
    global_params: Dict[str, Any]
    cache: Optional[NDArrayDTO] = None
    float_dtype: str


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and dataclasses to builtin JSON types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        # Prefer lists for small arrays in params dict; for large data use NDArrayDTO explicitly elsewhere
        return obj.tolist()
    if is_dataclass(obj):
        return _to_builtin(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def tree_to_dto(tree: Tree, *, include_dataX: bool = False) -> TreeDTO:
    """Convert a runtime Tree into a TreeDTO."""
    return TreeDTO(
        dataX=NDArrayDTO.from_array(tree.dataX) if (include_dataX and tree.dataX is not None) else None,
        vars=NDArrayDTO.from_array(tree.vars),
        thresholds=NDArrayDTO.from_array(tree.thresholds),
        leaf_vals=NDArrayDTO.from_array(tree.leaf_vals),
        n=NDArrayDTO.from_array(tree.n) if getattr(tree, "n", None) is not None else None,
        leaf_ids=NDArrayDTO.from_array(tree.leaf_ids) if getattr(tree, "leaf_ids", None) is not None else None,
        evals=NDArrayDTO.from_array(tree.evals) if getattr(tree, "evals", None) is not None else None,
        float_dtype=str(tree.float_dtype),
    )


def dto_to_tree(dto: TreeDTO) -> Tree:
    """Convert a TreeDTO back to a runtime Tree object."""
    dataX = dto.dataX.to_array() if dto.dataX is not None else None
    vars_arr = dto.vars.to_array().astype(np.int32, copy=False)
    thresholds_arr = dto.thresholds.to_array()
    leaf_vals_arr = dto.leaf_vals.to_array()

    n_arr = dto.n.to_array().astype(np.int32, copy=False) if dto.n is not None else None
    leaf_ids_arr = (
        dto.leaf_ids.to_array().astype(np.int32, copy=False) if dto.leaf_ids is not None else None
    )
    evals_arr = dto.evals.to_array() if dto.evals is not None else None

    tree = Tree(
        dataX=dataX,
        vars=vars_arr,
        thresholds=thresholds_arr,
        leaf_vals=leaf_vals_arr,
        n=n_arr,
        leaf_ids=leaf_ids_arr,
        evals=evals_arr,
    )
    return tree


def params_to_dto(params: Parameters, *, include_dataX: bool = False, include_cache: bool = True) -> ParametersDTO:
    """Convert runtime Parameters to DTO.

    include_dataX controls whether each tree's data matrix is serialized.
    include_cache controls serialization of the aggregated cache in Parameters.
    """
    return ParametersDTO(
        trees=[tree_to_dto(t, include_dataX=include_dataX) for t in params.trees],
        global_params=_to_builtin(params.global_params),
        cache=NDArrayDTO.from_array(params.cache) if (include_cache and getattr(params, "cache", None) is not None) else None,
        float_dtype=str(params.float_dtype),
    )


def dto_to_params(dto: ParametersDTO) -> Parameters:
    """Convert a ParametersDTO back into a runtime Parameters instance."""
    trees = [dto_to_tree(t) for t in dto.trees]
    cache_arr = dto.cache.to_array() if dto.cache is not None else None
    # global_params are basic types already
    params = Parameters(trees=trees, global_params=dict(dto.global_params), cache=cache_arr)
    return params


# Convenience JSON helpers
def tree_to_json(tree: Tree, *, include_dataX: bool = False) -> str:
    return tree_to_dto(tree, include_dataX=include_dataX).model_dump_json()


def tree_from_json(s: str) -> Tree:
    dto = TreeDTO.model_validate_json(s)
    return dto_to_tree(dto)


def params_to_json(params: Parameters, *, include_dataX: bool = False, include_cache: bool = True) -> str:
    dto = params_to_dto(params, include_dataX=include_dataX, include_cache=include_cache)
    # Use pydantic's JSON dump for consistency
    return dto.model_dump_json()


def params_from_json(s: str) -> Parameters:
    dto = ParametersDTO.model_validate_json(s)
    return dto_to_params(dto)


__all__ = [
    "NDArrayDTO",
    "TreeDTO",
    "ParametersDTO",
    "tree_to_dto",
    "dto_to_tree",
    "params_to_dto",
    "dto_to_params",
    "tree_to_json",
    "tree_from_json",
    "params_to_json",
    "params_from_json",
]

# -----------------------
# BART/DefaultBART support
# -----------------------


class RNGStateDTO(BaseModel):
    bit_generator: str
    state: Dict[str, Any]

    @staticmethod
    def from_generator(gen: np.random.Generator) -> "RNGStateDTO":
        st = gen.bit_generator.state
        return RNGStateDTO(bit_generator=st["bit_generator"], state=st)

    def to_generator(self) -> np.random.Generator:
        # Construct the appropriate BitGenerator and wrap in a Generator
        bitgen_cls = getattr(np.random, self.bit_generator)
        bg = bitgen_cls()
        bg.state = self.state
        return np.random.Generator(bg)


class DefaultPreprocessorDTO(BaseModel):
    kind: str = "default"
    max_bins: int
    y_min: float
    y_max: float

    @staticmethod
    def from_preproc(p: DefaultPreprocessor) -> "DefaultPreprocessorDTO":
        # y_min/max exist after fit
        y_min = getattr(p, "y_min", 0.0)
        y_max = getattr(p, "y_max", 0.0)
        return DefaultPreprocessorDTO(max_bins=p.max_bins, y_min=float(y_min), y_max=float(y_max))

    def to_preproc(self) -> DefaultPreprocessor:
        p = DefaultPreprocessor(max_bins=self.max_bins)
        # populate y_min/max so backtransform_y works
        p.y_min = np.array(self.y_min).item()
        p.y_max = np.array(self.y_max).item()
        # thresholds not needed for posterior_sample
        return p


class DefaultBARTDTO(BaseModel):
    cls: str = "DefaultBART"
    ndpost: int
    nskip: int
    preprocessor: DefaultPreprocessorDTO
    rng: RNGStateDTO
    # Trace of Parameter states post burn-in or full trace depending on usage
    trace: List[ParametersDTO]


def default_bart_to_dto(model: DefaultBART, *, include_dataX: bool = False, include_cache: bool = True) -> DefaultBARTDTO:
    return DefaultBARTDTO(
        ndpost=int(model.ndpost),
        nskip=int(model.nskip),
        preprocessor=DefaultPreprocessorDTO.from_preproc(model.preprocessor),
        rng=RNGStateDTO.from_generator(model.sampler.generator),
        trace=[params_to_dto(s, include_dataX=include_dataX, include_cache=include_cache) for s in model.trace],
    )


def dto_to_default_bart(dto: DefaultBARTDTO) -> DefaultBART:
    # Start from a vanilla DefaultBART then overwrite fields
    m = DefaultBART()
    m.ndpost = int(dto.ndpost)
    m.nskip = int(dto.nskip)
    m.preprocessor = dto.preprocessor.to_preproc()
    # Replace RNG state on the sampler
    m.sampler.generator = dto.rng.to_generator()
    # Rebuild trace
    m.trace = [dto_to_params(s) for s in dto.trace]
    m.is_fitted = True
    return m


def bart_to_json(model: BART, *, include_dataX: bool = False, include_cache: bool = True) -> str:
    if isinstance(model, DefaultBART):
        dto = default_bart_to_dto(model, include_dataX=include_dataX, include_cache=include_cache)
        return dto.model_dump_json()
    else:
        raise NotImplementedError("Only DefaultBART is supported in this serializer. If you want generic BART/other variants, let me know.")


def bart_from_json(s: str) -> BART:
    # Peek into JSON to route by cls
    raw = json.loads(s)
    cls = raw.get("cls", "DefaultBART")
    if cls == "DefaultBART":
        dto = DefaultBARTDTO.model_validate(raw)
        return dto_to_default_bart(dto)
    else:
        raise NotImplementedError(f"Unsupported BART cls '{cls}'. Only DefaultBART is implemented.")


__all__ += [
    "RNGStateDTO",
    "DefaultPreprocessorDTO",
    "DefaultBARTDTO",
    "default_bart_to_dto",
    "dto_to_default_bart",
    "bart_to_json",
    "bart_from_json",
]

# -----------------------
# MultiChainBART (DefaultBART chains)
# -----------------------


def multichain_to_json(model: MultiChainBART) -> str:
    """Serialize a MultiChainBART instance directly to JSON.

    Embeds each chain's JSON (from `collect_model_json`) in the payload to avoid
    Python object materialization and redundant conversions.
    """
    chains_json = model.collect_model_json()
    payload = {
        "cls": "MultiChainBART",
        "bart_cls": "DefaultBART",
        "n_ensembles": int(getattr(model, "n_ensembles")),
        "rng": RNGStateDTO.from_generator(getattr(model, "rng")).model_dump(),
        # Store chains as nested JSON objects (not strings)
        "chains": [json.loads(s) for s in chains_json],
    }
    return json.dumps(payload)


def multichain_from_json(s: str) -> "MultiChainDefaultBARTPortable":
    raw = json.loads(s)
    if raw.get("cls", "MultiChainBART") != "MultiChainBART":
        raise ValueError("JSON does not represent a MultiChainBART payload")
    if raw.get("bart_cls", "DefaultBART") != "DefaultBART":
        raise NotImplementedError("Only DefaultBART chains are supported")

    rng = RNGStateDTO.model_validate(raw["rng"]).to_generator()
    chains_raw = raw.get("chains", [])
    # Rebuild DefaultBART chains from nested dicts
    chains = [dto_to_default_bart(DefaultBARTDTO.model_validate(c)) for c in chains_raw]
    return MultiChainDefaultBARTPortable(chains=chains, rng=rng)


class MultiChainDefaultBARTPortable:
    """
    Lightweight, Ray-free multi-chain container holding DefaultBART chains.
    Implements posterior_sample compatible with MultiChainBART's logic.
    """

    def __init__(self, chains: List[DefaultBART], rng: Optional[np.random.Generator] = None):
        self.chains = chains
        self.n_ensembles = len(chains)
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def _trace_length(self) -> int:
        # Mirror MultiChainBART API used by agents for scheduling
        if not self.chains:
            return 0
        # Use first chain's trace length as representative (all typically share ndpost)
        return self.chains[0]._trace_length

    def posterior_sample(self, X, schedule):
        idx = self.rng.integers(0, self.n_ensembles)
        return self.chains[idx].posterior_sample(X, schedule)

    def posterior_f(self, X):
        import numpy as _np
        return _np.concatenate([c.posterior_f(X) for c in self.chains], axis=1)

    def predict(self, X):
        import numpy as _np
        preds = _np.array([c.predict(X) for c in self.chains])
        return _np.mean(preds, axis=0)

    def posterior_predict(self, X):
        import numpy as _np
        preds = [c.posterior_predict(X) for c in self.chains]
        return _np.concatenate(preds, axis=1)


__all__ += [
    "multichain_to_json",
    "multichain_from_json",
    "MultiChainDefaultBARTPortable",
]
