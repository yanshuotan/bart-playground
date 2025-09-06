import numpy as np
from typing import Any, Dict, List, Tuple

import arviz as az
from .params import Parameters
from .mcbart import MultiChainBART
from .samplers import Sampler

def _extract_scalar_series_from_state(state: Parameters, key: str = "eps_sigma2") -> float:
    """
    Extract a scalar from a sampler state.
    """
    return state.global_params[key].item()

def _collect_single_chain_values(model: Any, key: str = "eps_sigma2") -> List[float]:
    """
    Collect per-iteration scalar series for a single chain.
    """
    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before diagnostics.")
    trace = getattr(model, "trace", None)
    if trace is None or len(trace) == 0:
        raise ValueError("Empty trace; run sampling first.")
    return [
        _extract_scalar_series_from_state(state, key=key)
        for state in trace
    ]

def _collect_chain_series(model: Any, key: str = "eps_sigma2") -> Tuple[np.ndarray, int, int]:
    """
    Collect per-iteration scalar series per chain.

    Returns
    -------
    series : np.ndarray
        Shape (n_chains, n_draws) array of scalar values.
    n_chains : int
    n_draws : int
    """
    # MultiChainBART exposes actors and per-actor models via collect_model_states()
    if isinstance(model, MultiChainBART):
        chains = model.collect_model_states()
        per_chain: List[List[float]] = []
        for chain_model in chains:
            vals = _collect_single_chain_values(chain_model, key=key)
            per_chain.append(vals)
        # Ensure equal length across chains (truncate to min length if necessary)
        min_len = min(len(v) for v in per_chain)
        if min_len == 0:
            raise ValueError("No post-burn-in draws available in at least one chain.")
        per_chain = [v[:min_len] for v in per_chain]
        series = np.asarray(per_chain, dtype=float)
        return series, series.shape[0], series.shape[1]

    # DefaultBART / single-chain
    vals = _collect_single_chain_values(model, key=key)
    series = np.asarray(vals, dtype=float)[None, :]  # shape (1, draws)
    return series, 1, series.shape[1]

def _collect_move_acceptance(model: Any) -> Dict[str, Dict[str, float]]:
    """
    Collect per-move selection/success/acceptance counts and rates.

    For single-chain models, reads `model.sampler.move_*_counts`.
    For multi-chain, aggregates these dicts across chains.

    Returns a dictionary with keys per move and 'overall' with:
    { move: {selected, proposed, accepted, acc_rate, prop_rate}, 'overall': {...} }
    where proposed == success in the sampler terminology.
    """
    def read_counts(sampler: Sampler) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        sel = getattr(sampler, "move_selected_counts", {})
        suc = getattr(sampler, "move_success_counts", {})
        acc = getattr(sampler, "move_accepted_counts", {})
        return dict(sel), dict(suc), dict(acc)

    agg_sel: Dict[str, int] = {}
    agg_suc: Dict[str, int] = {}
    agg_acc: Dict[str, int] = {}

    if isinstance(model, MultiChainBART):
        chains = model.collect_model_states()
        for chain_model in chains:
            sel, suc, acc = read_counts(chain_model.sampler)
            for k, v in sel.items():
                agg_sel[k] = agg_sel.get(k, 0) + int(v)
            for k, v in suc.items():
                agg_suc[k] = agg_suc.get(k, 0) + int(v)
            for k, v in acc.items():
                agg_acc[k] = agg_acc.get(k, 0) + int(v)
    else:
        sel, suc, acc = read_counts(model.sampler)
        agg_sel, agg_suc, agg_acc = sel, suc, acc

    # Compute per-move rates
    result: Dict[str, Dict[str, float]] = {}
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
        acc_rate = (acc / suc) if suc > 0 else np.nan
        prop_rate = (suc / sel) if sel > 0 else np.nan
        result[mv] = {
            "selected": float(sel),
            "proposed": float(suc),
            "accepted": float(acc),
            "acc_rate": float(acc_rate) if np.isfinite(acc_rate) else np.nan,
            "prop_rate": float(prop_rate) if np.isfinite(prop_rate) else np.nan,
        }

    overall_acc_rate = (total_accepted / total_proposed) if total_proposed > 0 else np.nan
    overall_prop_rate = (total_proposed / total_selected) if total_selected > 0 else np.nan
    result["overall"] = {
        "selected": float(total_selected),
        "proposed": float(total_proposed),
        "accepted": float(total_accepted),
        "acc_rate": float(overall_acc_rate) if np.isfinite(overall_acc_rate) else np.nan,
        "prop_rate": float(overall_prop_rate) if np.isfinite(overall_prop_rate) else np.nan,
    }
    return result


def compute_diagnostics(model: Any, key: str = "eps_sigma2") -> Dict[str, Any]:
    """
    Compute MCMC diagnostics for a fitted BART-like model.

    Metrics:
    - R-hat (rank normalized, split) via ArviZ
    - bulk ESS via ArviZ
    - MCSE (mean MC standard error) for the scalar series
    - Move acceptance statistics from the sampler

    Parameters
    ----------
    model : Any
        A fitted `DefaultBART`-like object or `MultiChainBART`.
    key : str
        Name of the scalar in `global_params` to use (default: 'eps_sigma2').

    Returns
    -------
    dict
        {
          'n_chains', 'n_draws',
          'rhat', 'ess_bulk', 'mcse_mean', 'mcse_over_sd',
          'acceptance': { per-move stats and 'overall' }
        }
    """
    series, n_chains, n_draws = _collect_chain_series(model, key=key)

    # Create InferenceData with one observed variable named after the key
    posterior = {key: series[:, :, None]}  # add dummy dim for shape (chains, draws, 1)
    idata = az.from_dict(posterior=posterior)

    # R-hat and ESS (bulk)
    rhat = float(az.rhat(idata, method="rank")[key].values.squeeze())
    ess_bulk = float(az.ess(idata, method="bulk")[key].values.squeeze())

    # MCSE of the mean for the scalar series; ArviZ mcse returns array per var
    mcse_mean = float(az.mcse(idata)[key].values.squeeze())
    flat = series.reshape(-1).astype(np.float64, copy=False)
    sd = float(np.std(flat, ddof=1)) if flat.size > 1 else float("nan")
    mcse_over_sd = float(mcse_mean / sd) if sd > 0 and np.isfinite(mcse_mean) else float("nan")

    acceptance = _collect_move_acceptance(model)

    return {
        "n_chains": int(n_chains),
        "n_draws": int(n_draws),
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "mcse_mean": mcse_mean,
        "mcse_over_sd": mcse_over_sd,
        "acceptance": acceptance,
    }


__all__ = [
    "compute_diagnostics",
]


