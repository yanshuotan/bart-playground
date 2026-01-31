"""
Diagnostics functionality for BART Thompson Sampling agents.

This module contains the DiagnosticsMixin class which provides methods for
computing MCMC diagnostics and feature inclusion statistics for BART models.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from bart_playground.diagnostics import compute_diagnostics, MoveAcceptance

if TYPE_CHECKING:
    # Avoid circular imports while providing type hints
    from bart_playground.bandit.agents.bart_ts_agents import BARTTSAgent

logger = logging.getLogger(__name__)


class DiagnosticsMixin:
    """
    Mixin to handle diagnostics and feature inclusion logic.
    
    This mixin provides methods for:
    - Computing chain-level MCMC diagnostics
    - Computing probe-level diagnostics
    - Computing feature inclusion statistics
    
    It requires the host class to have:
      - self.model, self.models
      - self.is_model_fitted, self.n_post
      - self.is_logistic, self.separate_models
      - self.encoder, self.n_arms, self.n_features
      - self._preprocess_probes()
    """

    def diagnostics_chain(self: "BARTTSAgent", key: str = "eps_sigma2") -> Dict[str, Any]:
        """
        Compute chain-level MCMC diagnostics for the underlying model.
        
        This method computes diagnostics for a global parameter (default: eps_sigma2)
        across all MCMC chains.
        
        Parameters:
            key (str): Name of the global parameter to diagnose (default: "eps_sigma2")
            
        Returns:
            Dict[str, Any]: Dictionary containing diagnostic results with keys:
                - 'meta': Metadata about the chains
                - 'metrics': Diagnostic metrics (R-hat, ESS, etc.)
                - 'acceptance': Move acceptance statistics
                Returns empty dict if model is not fitted or has no posterior samples
        """
        if not self.is_model_fitted or self.n_post <= 0:
            return {}
        return compute_diagnostics(self.model, key=key)

    def diagnostics_probes(self: "BARTTSAgent", X_probes: np.ndarray) -> Dict[str, Any]:
        """
        Compute f(X) diagnostics for probe observations.
        
        This method evaluates the BART function at specific probe points and computes
        MCMC diagnostics for each (probe, arm) combination.
        
        Parameters:
            X_probes (np.ndarray): Probe covariates, shape (n_probes, n_features)
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'meta': Metadata including n_probes, n_arms, n_post
                - 'metrics': DataFrame with diagnostics for each (probe, arm) pair
                - 'acceptance': Aggregated move acceptance statistics
                
        Raises:
            NotImplementedError: If called on a logistic BART agent
            ValueError: If model is not fitted or has no post-burn-in draws
        """
        if self.is_logistic:
            raise NotImplementedError("diagnostics_probes: logistic handling not implemented yet")

        # Preprocess probes and check posterior draws
        X_probes, n_probes, n_arms, n_post = self._preprocess_probes(X_probes)

        diag: dict[str, Any] = {"meta": {}, "metrics": {}, "acceptance": {}}

        if not self.separate_models:
            # Build rows for all (probe, arm) pairs
            encoded_list: List[np.ndarray] = []
            for p in range(n_probes):
                x = X_probes[p, :]
                X_enc = self.encoder.encode(x, arm=-1)  # (n_arms, combined_dim)
                encoded_list.append(X_enc)
            X_all = np.vstack(encoded_list)  # (n_probes*n_arms, combined_dim)

            raw_diag = compute_diagnostics(self.model, X=X_all)
            # raw_diag["metrics"] is a pandas DataFrame with columns [...metrics...]
            metrics_df = raw_diag["metrics"]
            # Build tidy DataFrame with probe/arm columns
            if len(metrics_df) != n_probes * n_arms:
                logger.warning("diagnostics_probes: unexpected metrics length vs n_probes*n_arms")
            probes = np.repeat(np.arange(n_probes, dtype=int), n_arms)
            arms = np.tile(np.arange(n_arms, dtype=int), n_probes)
            out_df = pd.DataFrame({"probe_idx": probes, "arm": arms})
            out_df = pd.concat([out_df, metrics_df], axis=1)

            diag["meta"]["model"] = [raw_diag["meta"]]
            diag["metrics"] = out_df
            diag["acceptance"] = raw_diag["acceptance"]
        else:
            # Separate models: run per-arm and build tidy DataFrame with probe/arm columns
            df_list: List[pd.DataFrame] = []
            diag["meta"]["model"] = []
            for a in range(n_arms):
                model_a = self.models[a]
                diag_a = compute_diagnostics(model_a, X=X_probes)
                diag["meta"]["model"].append(diag_a["meta"])
                metrics_df_a = diag_a["metrics"]
                df_a = pd.DataFrame({"probe_idx": np.arange(n_probes, dtype=int), "arm": a})
                df_a = pd.concat([df_a, metrics_df_a], axis=1)
                df_list.append(df_a)
                for move, move_acceptance in diag_a["acceptance"].items():
                    if move not in diag["acceptance"]:
                        diag["acceptance"][move] = MoveAcceptance(selected=0, proposed=0, accepted=0)
                    diag["acceptance"][move] = diag["acceptance"][move].combine(move_acceptance)
            # This dataframe is of different sort order than the one from the non-separate models
            # This is fine for most purposes; the caller should not rely on the order
            diag["metrics"] = pd.concat(df_list, ignore_index=True)

        diag["meta"]["n_probes"] = n_probes
        diag["meta"]["n_arms"] = n_arms
        diag["meta"]["n_post"] = n_post

        return diag

    def feature_inclusion(self: "BARTTSAgent") -> Dict[str, Any]:
        """
        Compute per-feature inclusion statistics based on the current posterior.
        
        This method calculates how frequently each feature appears in the tree splits
        across all posterior samples, averaged across arms.
        
        Only supports:
          * Regression BART (non-logistic)
          * Separate models encoding (one model per arm, original covariates)
          
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'meta': Metadata including n_features, n_arms, encoding
                - 'metrics': DataFrame with feature_idx and inclusion frequency
                
        Raises:
            NotImplementedError: If called on logistic BART or non-separate encoding
        """
        if self.is_logistic:
            raise NotImplementedError(
                "feature_inclusion is only implemented for regression BARTTSAgent."
            )
        if not self.separate_models:
            raise NotImplementedError(
                "feature_inclusion currently only supports encoding='separate'."
            )

        # In the separate-models setup, we have one model per arm, each seeing the
        # original covariates. We compute a per-arm inclusion frequency and then
        # average across arms to get a single per-feature vector.
        P = self.n_features
        per_arm_inclusion: List[np.ndarray] = []

        for arm in range(self.n_arms):
            freq = self.models[arm].feature_inclusion_frequency("split")
            per_arm_inclusion.append(freq)

        stacked = np.stack(per_arm_inclusion, axis=0)  # (n_arms, P)
        inclusion = np.mean(stacked, axis=0)  # (P,)

        feature_idx = np.arange(P, dtype=int)
        df = pd.DataFrame(
            {
                "feature_idx": feature_idx,
                "inclusion": inclusion,
            }
        )

        meta: Dict[str, Any] = {
            "n_features": P,
            "n_arms": self.n_arms,
            "encoding": self.encoding,
            "n_post": int(self.n_post),
        }

        return {"meta": meta, "metrics": df}

