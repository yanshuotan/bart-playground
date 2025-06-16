# %% [markdown]
# # Comparing Bandit Agents
# 
# This notebook compares the performance of different bandit agents.

# %% [markdown]
# ### Imports

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os, sys

from bart_playground.bandit.sim_util import *
from compare_agents import (
    compare_agents_across_scenarios, print_summary_results, plot_comparison_results
)
# from bart_playground.bandit.rome.rome_scenarios import HomogeneousScenario, NonlinearScenario

# %%
# Define experiment parameters
from bart_playground.bandit.bcf_agent import BCFAgent, BCFAgentPSOff
from bart_playground.bandit.basic_agents import SillyAgent, LinearTSAgent
from bart_playground.bandit.ensemble_agent import EnsembleAgent
from bart_playground.bandit.me_agents import HierTSAgent, LinearTSAgent2, LinearUCBAgent, METSAgent
from bart_playground.bandit.bart_agent import BARTAgent, LogisticBARTAgent, DefaultBARTAgent, MultiChainBARTAgent
# from bart_playground.bandit.neural_ts_agent import NeuralTSDiagAgent

import multiprocessing

from bart_playground.bart import LogisticBART

cores =  multiprocessing.cpu_count() - 1

# %%
from typing import List, Tuple, Any
from bart_playground.bandit.agent import BanditAgent

# %% [markdown]
# ## Tunable Parameters

# %%
# Create test scenarios
np.random.seed(0)
    
scenarios = {
    # "Linear": LinearScenario(P=4, K=3, sigma2=1.0),
    # "LinearOffset": LinearOffsetScenario(P=4, K=3, sigma2=1.0),
    # "Offset": OffsetScenario(P=4, K=3, sigma2=1.0),
    # "Offset2": OffsetScenario(P=4, K=3, sigma2=0.1),
    # "Friedman": FriedmanScenario(P=5, K=3, sigma2=1.0, lambda_val=15),
    # "Sparse": FriedmanScenario(P=50, K=3, sigma2=1.0, lambda_val=5),
    
    ## "Isolet": OpenMLScenario('isolet', version=1),
    "Magic": OpenMLScenario('MagicTelescope', version=1),
    "Adult": OpenMLScenario('adult', version=2),
    "Shuttle": OpenMLScenario('shuttle', version=1),
    ## "Letter": OpenMLScenario('letter', version=1),
    "Mushroom": OpenMLScenario('mushroom', version=1),
    "Covertype": OpenMLScenario('covertype', version=3),
    "MNIST": OpenMLScenario('mnist_784', version=1),
}

args = sys.argv[1:]
if len(args) == 0:
    print("No scenarios specified, using Mushroom.")
    args = ['Mushroom']
scenarios = {k: v for k, v in scenarios.items() if k in args} if args else scenarios

rep_dataset = list(scenarios.keys())[0]
log_encoding = 'native' if rep_dataset in ['Adult', 'Magic', 'Mushroom'] else 'multi' 

np.random.seed(0)

# import torch
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(42)

# %%
from bart_playground.bandit.bart_agent import BARTAgent, LogisticBARTAgent, MultiChainBARTAgent
from bart_playground.bandit.ensemble_agent import EnsembleAgent
from bart_playground.bart import DefaultBART, LogisticBART

def class_to_agents(sim, scenario_K, scenario_P, agent_classes: List[Any]) -> List[BanditAgent]:
    """
    Convert agent classes to instances.
    """
        # Create agents with different seeds for this simulation
    sim_agents = []
    for agent_cls in agent_classes:
        if agent_cls == BCFAgent:
            agent = BCFAgent(
                n_arms=scenario_K,
                n_features=scenario_P,
                nskip=100,
                ndpost=100,
                nadd=3,
                nbatch=1,
                random_state=1000 + sim
            )
        elif agent_cls == BARTAgent:
            agent = BARTAgent(
                n_arms=scenario_K,
                n_features=scenario_P,
                nskip=50,
                ndpost=50,
                nadd=5,
                random_state=1000 + sim,
                encoding='multi'
            )
        elif agent_cls == LogisticBARTAgent:
            agent = LogisticBARTAgent(
                n_arms=scenario_K,
                n_features=scenario_P,
                nskip=50,
                ndpost=50,
                nadd=1,
                random_state=1000 + sim,
                encoding=log_encoding
            )
        elif agent_cls == MultiChainBARTAgent:
            agent = MultiChainBARTAgent(
                bart_class=LogisticBART,
                n_arms=scenario_K,
                n_features=scenario_P,
                n_ensembles=4,
                nskip=50,
                ndpost=50,
                nadd=2,
                random_state=1000 + sim,
                encoding=log_encoding
            )
        elif agent_cls == EnsembleAgent:
            agent = EnsembleAgent(
                n_arms=scenario_K,
                n_features=scenario_P,
                bcf_kwargs = dict(nskip=100,
                ndpost=10,
                nadd=2,
                random_state=1000 + sim),
                linear_ts_kwargs = dict(v=1)
            )
        elif agent_cls == BCFAgentPSOff:
            agent = BCFAgentPSOff(
                n_arms=scenario_K,
                n_features=scenario_P,
                nskip=100,
                ndpost=10,
                nadd=2,
                nbatch=1,
                random_state=1000 + sim
            )
        elif agent_cls == LinearTSAgent:
            agent = LinearTSAgent(
                n_arms=scenario_K,
                n_features=scenario_P,
                v = 1
            )
        # elif agent_cls == RoMEAgent:
        #     agent = RoMEAgent(
        #         n_arms=scenario_K,
        #         n_features=scenario_P,
        #         featurize=_featurize,
        #         t_max=n_draws,
        #         pool_users=False
        #     )
        # elif agent_cls == StandardTSAgent:
        #     agent = StandardTSAgent(
        #         n_arms=scenario_K,
        #         n_features=scenario_P,
        #         featurize=_featurize
        #     )
        # elif agent_cls == ActionCenteredTSAgent:
        #     agent = ActionCenteredTSAgent(
        #         n_arms=scenario_K,
        #         n_features=scenario_P,
        #         featurize=_featurize
        #     )
        # elif agent_cls == IntelligentPoolingAgent:
        #     agent = IntelligentPoolingAgent(
        #         n_arms=scenario_K,
        #         n_features=scenario_P,
        #         featurize=_featurize,
        #         t_max=n_draws
        #     )
        else:
            # For other agents, just initialize with standard params
            agent = agent_cls(
                n_arms=scenario_K,
                n_features=scenario_P
            )
        sim_agents.append(agent)
    
    return sim_agents

# %%
n_simulations = 3  # Number of simulations per scenario
max_draws = 10000      # Number of draws per simulation

def call_func():
    agent_dict = {
        "LinearTS": LinearTSAgent,
        "LinearTS2": LinearTSAgent2,
        # "LogisticBART": LogisticBARTAgent,
        "MultiChainBART": MultiChainBARTAgent,
        # "Neural": NeuralTSDiagAgent,
        # "MBCF+Linear": BCFAgentPSOff,
        # "METS": METSAgent,
        # "HierTS": HierTSAgent
    }
    agent_classes = list(agent_dict.values())
    agent_names = list(agent_dict.keys())
    return compare_agents_across_scenarios(
        scenarios=scenarios,
        n_simulations=n_simulations,
        max_draws=max_draws,
        agent_classes=agent_classes,
        agent_names=agent_names,
        parallel=False,
        class_to_agents=class_to_agents
    )

# %%
results = call_func()

# %% [markdown]
# ## Summary Results

# %%
print_summary_results(results)

# %%
key, value = next(iter(results.items()))
internal_key, internal_value = next(iter(value['times'].items()))
n_draws = internal_value.shape[1]

# %% [markdown]
# ## Visualize Results
# 
# Finally, let's visualize the cumulative regret for each agent across scenarios.

# %%
# Create results directory if it doesn't exist
results_dir = "./results/agent_comparison_final"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

import pickle
appendix_name = list(scenarios.keys())[0]  # Use the first scenario name as appendix
result_filename = os.path.join(results_dir, f"result_{appendix_name}.pkl")
pickle.dump(results, open(result_filename, "wb"))

# %%
results = pickle.load(file=open(result_filename, "rb"))

# Plot results and save to file
plot_comparison_results(
    results=results,
    save_path=f"{results_dir}/agent_comparison_results_{appendix_name}.png",
    show_time=False
)
