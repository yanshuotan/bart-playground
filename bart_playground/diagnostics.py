import numpy as np
from typing import Any, Dict, List, Tuple, Optional

import arviz as az
import pandas as pd
from .mcbart import MultiChainBART
from .samplers import Sampler

def _actor_collect_values(model: Any, key: str, X: Optional[np.ndarray] = None) -> List[Any]:
    """
    Collect per-iteration scalar series for a single chain.

    If X is provided, returns a vector per draw (f(X) per row, backtransformed when applicable).
    """
    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before diagnostics.")
    trace = getattr(model, "trace", None)
    if trace is None or len(getattr(model, 'range_post', [])) == 0:
        raise ValueError("Empty trace; run sampling first.")
    # If no X provided, extract scalar from global parameters
    if X is None:
        return [
            state.global_params[key].item()
            for state in trace
        ]

    # Otherwise, compute f(x) at each draw
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    vals: List[Any] = []
    for k in model.range_post:
        y_eval = model.predict_trace(int(k), X_arr, backtransform=True)
        vals.append(np.asarray(y_eval).reshape(-1))
    return vals

def _collect_chain_series(model: Any, key: str = "eps_sigma2", X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, int]:
    """
    Collect per-iteration series per chain.

    Returns
    -------
    series : np.ndarray
        Shape (n_chains, n_draws, output_dim) array.
        - When X is None: output_dim == 1 (scalar global parameter per draw)
        - When X is not None: output_dim equals the length of the prediction vector,
          which is n_samples if model outputs scalars, or n_samples * output_features
          if model outputs vectors (flattened to 1D)
    n_chains : int
    n_draws : int
    """
    # MultiChainBART exposes actors and per-actor models via collect_model_states()
    if isinstance(model, MultiChainBART):
        per_chain: List[List[Any]] = model.collect(_actor_collect_values, key, X)
    else:
        per_chain = [_actor_collect_values(model, key, X)]

    # Ensure equal length across chains (truncate to min length if necessary)
    min_len = min(len(v) for v in per_chain)
    if min_len == 0:
        raise ValueError("No post-burn-in draws available in at least one chain.")
    per_chain = [v[:min_len] for v in per_chain]
    # Build array
    if X is not None:
        # Expect list[list[np.ndarray]] with consistent last-dim
        # Determine vector length
        per_chain_norm: List[List[np.ndarray]] = [
            [np.asarray(arr).reshape(-1) for arr in chain_vals]
            for chain_vals in per_chain
        ]
        series = np.stack([np.stack(chain_vals, axis=0) for chain_vals in per_chain_norm], axis=0)
    else:
        series2d = np.asarray(per_chain, dtype=float)
        series = series2d[:, :, None]  # shape (n_chains, n_draws, 1)
    return series, series.shape[0], series.shape[1]

def _actor_collect_move_counts(model: Any) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Return (selected, success, accepted) move count dictionaries from the in-actor sampler.
    """
    sampler: Sampler = model.sampler
    sel = dict(getattr(sampler, "move_selected_counts", {}))
    suc = dict(getattr(sampler, "move_success_counts", {}))
    acc = dict(getattr(sampler, "move_accepted_counts", {}))
    # Ensure plain ints
    sel = {k: int(v) for k, v in sel.items()}
    suc = {k: int(v) for k, v in suc.items()}
    acc = {k: int(v) for k, v in acc.items()}
    return sel, suc, acc

from dataclasses import dataclass, asdict
@dataclass
class MoveAcceptance:
    selected: int
    proposed: int
    accepted: int

    @property
    def acc_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed > 0 else np.nan
    
    @property
    def prop_rate(self) -> float:
        return self.proposed / self.selected if self.selected > 0 else np.nan

    def __post_init__(self):
        self.selected = int(self.selected)
        self.proposed = int(self.proposed)
        self.accepted = int(self.accepted)

    def combine(self, other: 'MoveAcceptance') -> 'MoveAcceptance':
        return MoveAcceptance(
            selected=self.selected + other.selected,
            proposed=self.proposed + other.proposed,
            accepted=self.accepted + other.accepted,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "acc_rate": float(self.acc_rate), "prop_rate": float(self.prop_rate)}

def _collect_move_acceptance(model: Any) -> Dict[str, MoveAcceptance]:
    """
    Collect per-move selection/success/acceptance counts and rates.

    For single-chain models, reads `model.sampler.move_*_counts`.
    For multi-chain, aggregates these dicts across chains.

    Returns a dictionary with keys per move and 'overall' with:
    { move: MoveAcceptance, 'overall': MoveAcceptance }
    where proposed == success in the sampler terminology.
    """

    agg_sel: Dict[str, int] = {}
    agg_suc: Dict[str, int] = {}
    agg_acc: Dict[str, int] = {}

    if isinstance(model, MultiChainBART):
        counts_list = model.collect(_actor_collect_move_counts)
        for sel, suc, acc in counts_list:
            for k, v in sel.items():
                agg_sel[k] = agg_sel.get(k, 0) + int(v)
            for k, v in suc.items():
                agg_suc[k] = agg_suc.get(k, 0) + int(v)
            for k, v in acc.items():
                agg_acc[k] = agg_acc.get(k, 0) + int(v)
    else:
        sel, suc, acc = _actor_collect_move_counts(model)
        agg_sel, agg_suc, agg_acc = sel, suc, acc

    # Compute per-move rates
    result: Dict[str, MoveAcceptance] = {}
    all_moves = set(agg_sel) | set(agg_suc) | set(agg_acc)
    total_selected = 0
    total_proposed = 0
    total_accepted = 0
    for mv in sorted(all_moves):
        sel = int(agg_sel.get(mv, 0))
        suc = int(agg_suc.get(mv, 0))  # proposed
        acc = int(agg_acc.get(mv, 0))
        total_selected += sel
        total_proposed += suc
        total_accepted += acc
        result[mv] = MoveAcceptance(selected=sel, proposed=suc, accepted=acc)

    result["overall"] = MoveAcceptance(selected=total_selected, proposed=total_proposed, accepted=total_accepted)
    return result

def compute_diagnostics(model: Any, key: Optional[str] = None, X: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute MCMC diagnostics for a fitted BART-like model.

    Metrics:
    - R-hat (rank normalized, split) via ArviZ
    - bulk ESS via ArviZ
    - MCSE (mean MC standard error)
    - Move acceptance statistics from the sampler

    Parameters
    ----------
    model : Any
        A fitted `DefaultBART`-like object or `MultiChainBART`.
    key : str, optional
        Name of the scalar in `global_params` to use (default is 'eps_sigma2' if not provided).
        Ignored when X is provided.
    X : np.ndarray, optional
        If provided, compute diagnostics for f(X) evaluated at each draw without aggregation (per-row diagnostics).

    Returns
    -------
    dict
        {
          'meta': { 'n_chains', 'n_draws' },
          'metrics': pandas.DataFrame with columns [...metrics...],
          'acceptance': { per-move stats and 'overall' }
        }
    """
    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before diagnostics.")
    if len(getattr(model, 'range_post', [])) == 0:
        raise ValueError("Empty trace; run sampling first.")

    # Default key (only relevant when X is None)
    # TODO: maybe not necessary?
    use_key = "eps_sigma2" if (X is None and key is None) else (key if key is not None else "eps_sigma2") 
    series, n_chains, n_draws = _collect_chain_series(model, key=use_key, X=X)

    # Create InferenceData with one observed variable named after the key
    var_name = (use_key if X is None else "f_x")
    posterior = {var_name: series} # Shape (chains, draws, k)
    idata = az.from_dict(posterior=posterior)

    # R-hat and ESS (bulk)
    rhat_da: Any = az.rhat(idata, method="rank")
    ess_da: Any = az.ess(idata, method="bulk")
    rhat_arr = np.asarray(rhat_da[var_name].values)
    ess_arr = np.asarray(ess_da[var_name].values)

    # MCSE of the mean for the scalar series; ArviZ mcse returns array per var
    mcse_da: Any = az.mcse(idata)
    mcse_arr = np.asarray(mcse_da[var_name].values)
    # Compute SD across chained draws per variable: (chains*draws, k)
    flat = series.reshape(-1, series.shape[-1]).astype(np.float64, copy=False)
    sd_vec = np.std(flat, axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mcse_over_sd_vec = np.where(sd_vec > 0, mcse_arr / sd_vec, np.nan)

    acceptance = _collect_move_acceptance(model)

    metrics_df = pd.DataFrame({
        "rhat": rhat_arr.reshape(-1),
        "ess_bulk": ess_arr.reshape(-1),
        "mcse_mean": mcse_arr.reshape(-1),
        "mcse_over_sd": mcse_over_sd_vec.reshape(-1),
    })

    return {
        "meta": {
            "n_chains": int(n_chains),
            "n_draws": int(n_draws)
        },
        "metrics": metrics_df,
        "acceptance": acceptance,
    }

__all__ = [
    "compute_diagnostics",
]
