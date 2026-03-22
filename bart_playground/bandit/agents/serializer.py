"""
Agent-level serializer for bandit agents.

Supports round-trippable serialize/deserialize for choose_arm behavior.

Scope:
- DefaultBARTTSAgent
- SillyAgent
- LinearAgentStable

Notes:
- For BART models, we delegate to bart_playground.serializer:
  - Single-chain DefaultBART via bart_to_json/bart_from_json
  - Portable multi-chain via multichain_to_json/multichain_from_json
- update_state is not supported.
"""

from typing import Any, List, Optional, Union, Literal
import numpy as np
from pydantic import BaseModel

# Agent classes
from .basic_agents import SillyAgent, LinearAgentStable, AgentType
from .bart_ts_agents import DefaultBARTTSAgent

# BART model serializer helpers
from bart_playground.serializer import (
    bart_to_json,
    bart_from_json,
    multichain_to_json,
    multichain_from_json,
    RNGStateDTO,
)
class SillyAgentDTO(BaseModel):
    kind: Literal["SillyAgent"]
    n_arms: int
    n_features: int
    random_state: Optional[int] = None
    rng: RNGStateDTO


class LinearAgentStableDTO(BaseModel):
    kind: Literal["LinearAgentStable"]
    n_arms: int
    n_features: int
    agent_type: Literal["ts", "ucb"]
    v: float
    alpha: float
    # Per-arm matrices
    B: List[List[List[float]]]
    m2_r: List[List[List[float]]]
    L: List[List[List[float]]]
    rng: RNGStateDTO


class DefaultBARTTSAgentDTO(BaseModel):
    kind: Literal["DefaultBARTTSAgent"]
    n_arms: int
    n_features: int
    # Agent config/state
    initial_random_selections: int
    random_state: int
    encoding: str
    t: int
    is_model_fitted: bool
    rng: RNGStateDTO
    warmstart_arms: List[int]

    # Serialized DefaultBART or MultiChain (portable) model JSON
    bart_json: Optional[str] = None
    multichain_json: Optional[str] = None


AgentDTO = Union[SillyAgentDTO, LinearAgentStableDTO, DefaultBARTTSAgentDTO]


def serialize_agent(agent: Any) -> str:
    """Serialize supported agents to a JSON string."""
    # SillyAgent
    if isinstance(agent, SillyAgent):
        dto = SillyAgentDTO(
            kind="SillyAgent",
            n_arms=agent.n_arms,
            n_features=agent.n_features,
            random_state=agent.random_state,
            rng=RNGStateDTO.from_generator(agent.rng),
        )
        return dto.model_dump_json()

    # LinearAgentStable
    if isinstance(agent, LinearAgentStable):
        dto = LinearAgentStableDTO(
            kind="LinearAgentStable",
            n_arms=agent.n_arms,
            n_features=agent.n_features,
            agent_type="ts" if agent.agent_type.name.lower() == "ts" else "ucb",
            v=float(agent.v),
            alpha=float(agent.alpha),
            B=[m.tolist() for m in agent.B],
            m2_r=[m.tolist() for m in agent.m2_r],
            L=[m.tolist() for m in agent.L],
            rng=RNGStateDTO.from_generator(agent.rng),
        )
        return dto.model_dump_json()

    # DefaultBARTTSAgent (supports DefaultBART and MultiChain)
    if isinstance(agent, DefaultBARTTSAgent):
        model = agent.model
        bart_json_val: Optional[str] = None
        multichain_json_val: Optional[str] = None
        if hasattr(model, 'collect_model_json'):
            # Ray-backed MultiChainBART implements collect_model_json()
            multichain_json_val = multichain_to_json(model)  # type: ignore[arg-type]
        else:
            # Single-chain DefaultBART
            try:
                bart_json_val = bart_to_json(model, include_dataX=False, include_cache=True)
            except Exception as e:
                raise NotImplementedError(f"Unsupported BART model type for agent serialization: {type(model)}") from e

        dto = DefaultBARTTSAgentDTO(
            kind="DefaultBARTTSAgent",
            n_arms=agent.n_arms,
            n_features=agent.n_features,
            initial_random_selections=agent.initial_random_selections,
            random_state=agent.random_state,
            encoding=agent.encoding,
            t=agent.t,
            is_model_fitted=agent.is_model_fitted,
            rng=RNGStateDTO.from_generator(agent.rng),
            warmstart_arms=agent._warmstart_arms,
            bart_json=bart_json_val,
            multichain_json=multichain_json_val,
        )
        return dto.model_dump_json()

    raise TypeError(f"Unsupported agent type for serialization: {type(agent)}")


def deserialize_agent(s: str) -> Any:
    """Deserialize an agent JSON string back into a live agent instance."""
    import json
    payload = json.loads(s)
    kind = payload.get("kind")

    # SillyAgent
    if kind == "SillyAgent":
        dto = SillyAgentDTO.model_validate(payload)
        agent = SillyAgent(n_arms=dto.n_arms, n_features=dto.n_features, random_state=dto.random_state)
        agent.rng = dto.rng.to_generator()
        return agent

    # LinearAgentStable
    if kind == "LinearAgentStable":
        dto = LinearAgentStableDTO.model_validate(payload)
        agent = LinearAgentStable(
            agent_type=AgentType.TS if dto.agent_type == "ts" else AgentType.UCB,
            n_arms=dto.n_arms,
            n_features=dto.n_features - 1,  # ctor expects raw features; class adds +1 intercept
            v=dto.v,
            alpha=dto.alpha,
        )
        # Restore matrices
        agent.B = [np.array(m, dtype=float) for m in dto.B]
        agent.m2_r = [np.array(m, dtype=float) for m in dto.m2_r]
        agent.L = [np.array(m, dtype=float) for m in dto.L]
        agent.rng = dto.rng.to_generator()
        return agent

    # DefaultBARTTSAgent
    if kind == "DefaultBARTTSAgent":
        dto = DefaultBARTTSAgentDTO.model_validate(payload)
        # Build agent (single-chain DefaultBART) with lightweight ctor
        agent = DefaultBARTTSAgent(
            n_arms=dto.n_arms,
            n_features=dto.n_features,
            initial_random_selections=dto.initial_random_selections,
            random_state=dto.random_state,
            encoding=dto.encoding,
            n_chains=1,
        )

        # Restore agent state
        agent.t = dto.t
        agent.is_model_fitted = dto.is_model_fitted
        agent.rng = dto.rng.to_generator()
        agent._warmstart_arms = dto.warmstart_arms

        # Restore model from JSON
        if dto.bart_json is not None:
            model = bart_from_json(dto.bart_json)
        elif dto.multichain_json is not None:
            model = multichain_from_json(dto.multichain_json)
        else:
            raise ValueError("Neither bart_json nor multichain_json provided")
        agent.model = model

        return agent

    raise ValueError(f"Unknown agent kind: {kind}")


__all__ = [
    "serialize_agent",
    "deserialize_agent",
]
