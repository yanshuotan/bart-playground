# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: bartpg
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparing Bandit Agents
#
# The notebook (`compare.ipynb`) compares the performance of different bandit agents. The `compare.py` script is for command-line execution. They are automatically paired using **jupytext**. 

# %% [markdown]
# ### Imports

# %%
import warnings, logging
warnings.filterwarnings('ignore')

import numpy as np
import os, sys
from typing import List
import multiprocessing

cores =  multiprocessing.cpu_count() - 1

from bart_playground.bandit.sim_util import *
from compare_agents import (
    AgentSpec, compare_agents_across_scenarios, print_summary_results, plot_comparison_results,
    _ca_logger
)
# from bart_playground.bandit.rome.rome_scenarios import HomogeneousScenario, NonlinearScenario

# %%
from bart_playground.bandit.bcf_agent import BCFAgent, BCFAgentPSOff
from bart_playground.bandit.basic_agents import SillyAgent, LinearTSAgent
from bart_playground.bandit.ensemble_agent import EnsembleAgent
from bart_playground.bandit.me_agents import HierTSAgent, LinearTSAgent2, LinearUCBAgent, METSAgent
from bart_playground.bandit.bart_agent import BARTAgent, LogisticBARTAgent, DefaultBARTAgent, MultiChainBARTAgent
from bart_playground.bart import LogisticBART
# from bart_playground.bandit.neural_ts_agent import NeuralTSDiagAgent

# %% [markdown]
# ## Tunable Parameters

# %%
# Create test scenarios
scenario_factories = {
    # "Linear": lambda: LinearScenario(P=4, K=3, sigma2=1.0),
    # "LinearOffset": lambda: LinearOffsetScenario(P=4, K=3, sigma2=1.0),
    # "Offset": lambda: OffsetScenario(P=4, K=3, sigma2=1.0),
    # "Offset2": lambda: OffsetScenario(P=4, K=3, sigma2=0.1),
    # "Friedman": lambda: FriedmanScenario(P=5, K=3, sigma2=1.0, lambda_val=15),
    # "Sparse": lambda: FriedmanScenario(P=50, K=3, sigma2=1.0, lambda_val=5),
    ## "Isolet": lambda: OpenMLScenario('isolet', version=1),
    "Magic": lambda: OpenMLScenario('MagicTelescope', version=1),
    "Adult": lambda: OpenMLScenario('adult', version=2),
    "Shuttle": lambda: OpenMLScenario('shuttle', version=1),
    ## "Letter": lambda: OpenMLScenario('letter', version=1),
    "Mushroom": lambda: OpenMLScenario('mushroom', version=1),
    "Covertype": lambda: OpenMLScenario('covertype', version=3),
    "MNIST": lambda: OpenMLScenario('mnist_784', version=1),
}

# Parse command-line arguments
args = sys.argv[1:]

# Determine which scenarios to run; default to Mushroom if none or invalid
if len(args) == 0 or not any(arg in scenario_factories for arg in args):
    print("No valid scenarios specified, using Mushroom.")
    args = ['Mushroom']

selected_keys = [k for k in scenario_factories if k in args]
scenarios = {key: scenario_factories[key]() for key in selected_keys}

rep_dataset = selected_keys[0]
log_encoding = 'native' if rep_dataset in ['Adult', 'Magic', 'Mushroom'] else 'multi' 

# import torch
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(42)

# %%
from bart_playground.bandit.bart_agent import BARTAgent, LogisticBARTAgent, MultiChainBARTAgent
from bart_playground.bandit.ensemble_agent import EnsembleAgent
from bart_playground.bart import DefaultBART, LogisticBART

all_agent_specs: List[AgentSpec] = [
    ("BCF",        BCFAgent,        {'nskip':100, 'ndpost':100, 'nadd':3, 'nbatch':1, 'random_state':1000}),
    ("BART",       BARTAgent,       {'nskip':50,  'ndpost':50,  'nadd':5,                   'random_state':1000, 'encoding':'multi'}),
    ("LogisticBART", LogisticBARTAgent,{
        'nskip':50,
        'ndpost':50,
        'nadd':1,
        'random_state':1000,
        'encoding':log_encoding
    }),
    ("MCBART",     MultiChainBARTAgent,{
        'bart_class': LogisticBART,
        'n_ensembles': 4,
        'n_trees': 50,
        'nskip': 50,
        'ndpost': 50,
        'nadd': 2,
        'random_state': 1000,
        'encoding': log_encoding
    }),
    ("Ensemble",   EnsembleAgent,   {
        'bcf_kwargs':       dict(nskip=100, ndpost=10, nadd=2, random_state=1000),
        'linear_ts_kwargs': dict(v=1)
    }),
    ("BCF_PSOff",  BCFAgentPSOff,   {'nskip':100, 'ndpost':10, 'nadd':2, 'nbatch':1, 'random_state':1000}),
    ("LinearTS",   LinearTSAgent,   {'v':1}),
    ("LinearTS2",  LinearTSAgent2,  {}),
    # commented out for on-demand testing:
    # ("RoME",             RoMEAgent,              {'featurize':_featurize, 't_max':n_draws, 'pool_users':False}),
    # ("StandardTS",       StandardTSAgent,        {'featurize':_featurize}),
    # ("ActionCenteredTS", ActionCenteredTSAgent,  {'featurize':_featurize}),
    # ("IntelligentPooling", IntelligentPoolingAgent, {'featurize':_featurize, 't_max':n_draws}),
]
# %%
# Filter agents to include only those we want to test
agent_specs = [agent for agent in all_agent_specs if agent[0] in ["MCBART", "LinearTS"]]

# %%
n_simulations = 3  # Number of simulations per scenario
max_draws = 200      # Number of draws per simulation

results_dir = "./results/agent_comparison"
os.makedirs(results_dir, exist_ok=True)

def call_func():
    return compare_agents_across_scenarios(
        scenarios=scenarios,
        agent_specs=agent_specs,
        sim_indices=[0, 1],
        max_draws=max_draws,
        parallel=False,
        save_dir=results_dir,
        log_to_file=True
    )

# %%
results = call_func()

# %% [markdown]
# ## Summary Results

# %%
print_summary_results(results)

# %% [markdown]
# ## Visualize Results
#
# Finally, let's visualize the cumulative regret for each agent across scenarios.

# %%
import pickle
appendix_name = list(scenarios.keys())[0]  # Use the first scenario name as appendix
result_filename = os.path.join(results_dir, f"result_{appendix_name}.pkl")
pickle.dump(results, open(result_filename, "wb"))
_ca_logger.info(f"Results saved to {result_filename}")

# %%
results = pickle.load(file=open(result_filename, "rb"))

# Plot results and save to file
plot_comparison_results(
    results=results,
    save_loc=f"{results_dir}/agent_comparison_results_{appendix_name}.png"
)
_ca_logger.info(f"Plot saved to {results_dir}/agent_comparison_results_{appendix_name}.png")
