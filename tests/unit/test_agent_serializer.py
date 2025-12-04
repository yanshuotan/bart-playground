import numpy as np

from bart_playground.bandit.agents.basic_agents import SillyAgent, LinearAgentStable, AgentType
from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent
from bart_playground.bandit.agents.serializer import serialize_agent, deserialize_agent


def _lockstep_compare_choices(agent_a, agent_b, x, n_steps=5):
    for _ in range(n_steps):
        a = agent_a.choose_arm(x)
        b = agent_b.choose_arm(x)
        assert a == b


def test_silly_agent_choose_arm_roundtrip():
    agent = SillyAgent(n_arms=5, n_features=3, random_state=123)

    # Serialize before any draws to capture initial RNG state
    s = serialize_agent(agent)
    agent2 = deserialize_agent(s)

    x = np.zeros((3,), dtype=float)
    _lockstep_compare_choices(agent, agent2, x, n_steps=10)


def test_linear_agent_stable_choose_arm_roundtrip():
    agent = LinearAgentStable(agent_type=AgentType.TS, n_arms=4, n_features=6, v=0.5, alpha=1.0, random_state=2024)

    s = serialize_agent(agent)
    agent2 = deserialize_agent(s)

    x = np.linspace(0.0, 1.0, 6, dtype=float)
    _lockstep_compare_choices(agent, agent2, x, n_steps=10)


def test_default_bart_agent_choose_arm_roundtrip():
    # Keep early phase to use RNG-only path for speed and determinism
    agent = DefaultBARTTSAgent(
        n_arms=3,
        n_features=4,
        initial_random_selections=10,
        random_state=7,
        encoding='multi',
        n_chains=1,
        bart_kwargs={"ndpost": 10, "nskip": 5},
    )

    s = serialize_agent(agent)
    agent2 = deserialize_agent(s)

    x = np.arange(4, dtype=float)
    _lockstep_compare_choices(agent, agent2, x, n_steps=10)


def test_default_bart_agent_multichain_choose_arm_roundtrip():
    # Use real MultiChainBART via n_chains > 1; we stay in the initial RNG phase
    agent = DefaultBARTTSAgent(
        n_arms=3,
        n_features=4,
        initial_random_selections=10,
        random_state=11,
        encoding='multi',
        n_chains=2,
        bart_kwargs={"ndpost": 10, "nskip": 5},
    )

    s = serialize_agent(agent)
    agent2 = deserialize_agent(s)

    x = np.arange(4, dtype=float)
    _lockstep_compare_choices(agent, agent2, x, n_steps=10)


