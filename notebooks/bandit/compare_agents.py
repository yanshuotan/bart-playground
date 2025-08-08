import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Union, Iterator, Type
from tqdm import tqdm
from datetime import datetime
import ray
import logging
import pandas as pd

from bart_playground.bandit.sim_util import simulate, Scenario, _sim_logger
from bart_playground.bandit.ope import instantiate_agents

def add_logging_file(save_dir: str):
    now = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("bandit_simulator")
    f_handler = logging.FileHandler(f"{log_dir}/sim_{now}.log", mode="w", encoding="utf-8")

    f_handler.setLevel(logging.DEBUG)
    f_fmt = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    f_formatter = logging.Formatter(f_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    f_handler.setFormatter(f_formatter)
    logger.addHandler(f_handler)

AgentSpec = Tuple[str, Type, Dict[str, Any]]

@ray.remote
def _run_single_simulation_remote(sim, scenario, agent_specs, n_draws):
    """
    Remote function to run a single simulation with the given scenario and agents.
    """
    return _run_single_simulation(sim, scenario, agent_specs, n_draws)

def _run_single_simulation(sim, scenario, agent_specs, n_draws):
    """
    Run a single simulation with the given scenario and agents.
    
    Args:
        sim (int): Simulation number (used for random seed)
        scenario (Scenario): The scenario instance to use for simulation
        agent_specs (List[AgentSpec]): Specifications for each agent
        n_draws (int): Number of draws per simulation
        
    Returns:
        Tuple: (sim_index, regrets, computation_times)
    """
    # Create agents with different seeds for this simulation
    sim_agents = instantiate_agents(agent_specs, scenario.K, scenario.P, sim)

    _sim_logger.debug(f"Shuffling scenario for simulation {sim} with random state {sim}...")
    scenario.shuffle(random_state=sim)

    _sim_logger.debug(f"Scenario random generator state: {scenario.rng_state}")
    # Run simulation
    cum_regrets, time_agents = simulate(scenario, sim_agents, n_draws=n_draws, agent_names=[name for name, _, _ in agent_specs])
    
    # Return results for this simulation
    return sim, cum_regrets, time_agents

def generate_simulation_data_for_agents(scenario_name: str, scenario: Scenario, agent_specs: List[AgentSpec], sim_indices: List[int], n_draws: int = 500, parallel=True, save_dir: str = "test_results") -> Dict[str, Any]:
    """
    Generate simulation data for multiple agents on a given scenario.
    
    Args:
        scenario_name (str): Name of the scenario
        scenario (Scenario): The scenario instance to use for simulation
        agent_specs (List[AgentSpec]): Specifications for each agent
        sim_indices (List[int]): List of simulation indices to run
        n_draws (int): Number of draws per simulation
        parallel (bool): Whether to run simulations in parallel using Ray
        save_dir (str): Directory to save intermediate simulation results
        
    Returns:
        Dict: Dictionary containing simulation results
    """
    internal_name = scenario.__class__.__name__
    # LogisticBART is not suitable for continuous scenarios (i.e. non-OpenMLScenario)
    if internal_name != "OpenMLScenario" and internal_name != "DrinkLessScenario":
        agent_specs = [spec for spec in agent_specs if not spec[0].startswith("Logistic")]
        
    agent_names = [name for name, _, _ in agent_specs]
    n_simulations = len(sim_indices)
    all_regrets = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    all_times = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    
    _sim_logger.debug("Scenario information:")
    _sim_logger.debug(f"  Name: {scenario_name}. Internal name: {internal_name}")
    if hasattr(scenario, 'dataset_name'):
        _sim_logger.debug(f"    Dataset: {scenario.dataset_name}")
    _sim_logger.debug(f"  K (arms): {scenario.K}")
    _sim_logger.debug(f"  P (features): {scenario.P}")
    _sim_logger.debug(f"  Max draws: {scenario.max_draws}")
    _sim_logger.debug("Agent names and specs:")
    for name, cls, kwargs in agent_specs:
        _sim_logger.debug(f"  {name}: {cls.__name__}, kwargs: {kwargs}")

    if parallel:
        # Run simulations in parallel
        _sim_logger.info(f"Running {n_simulations} simulations in parallel using Ray...")
        results_futures = [
            _run_single_simulation_remote.remote(sim, scenario, agent_specs, n_draws)
            for sim in sim_indices
        ]
        # Wait for all tasks to complete and retrieve the results.
        results = [ray.get(future) for future in tqdm(results_futures, desc="Simulating")]
    else:
        # Run simulations sequentially
        results = []
        _sim_logger.info(f"Sequentially running simulation.")
        for i, sim in enumerate(sim_indices):
            _sim_logger.info(f"Progress: {i}/{n_simulations}. Current sim index: {sim}. ")
            _sim_logger.debug(f"Set np.random state: {sim}")
            np.random.seed(sim)  # Set random seed for reproducibility
            result : Tuple[int, np.ndarray, np.ndarray] = _run_single_simulation(sim, scenario, agent_specs, n_draws)
                
            sim_id, cum_regrets, time_agents = result
            serializable_result = {
                "scenario_name": scenario_name,
                "sim_id": sim_id,
                "agent_names": agent_names,
                "cum_regrets": cum_regrets.tolist(),
                "time_agents": time_agents.tolist()
            }
            if save_dir is not None:
                file_path = os.path.join(save_dir, f"{scenario_name}_sim{sim}_{datetime.now().strftime('%m%d_%H%M')}.json")
                with open(file_path, "w") as f:
                    json.dump(serializable_result, f, indent=4) 
                _sim_logger.info(f"Saved results for simulation {sim} of {scenario_name} in {file_path}.")

            results.append(result)
    
    for result_idx, (sim, cum_regrets, time_agents) in enumerate(results):
        # Store results
        for i, name in enumerate(agent_names):
            all_regrets[name][result_idx, :] = cum_regrets[:, i]
            all_times[name][result_idx, :] = time_agents[:, i]

    _sim_logger.debug("Simulation data generation completed successfully.")

    return {
        "scenario_name": scenario_name,
        "sim_id": sim_id,
        'regrets': all_regrets,
        'times': all_times
    }

def compare_agents_across_scenarios(scenarios: Dict[str, Scenario],
    agent_specs: List[AgentSpec], sim_indices: Union[List[int], int, None] = None, 
    max_draws: int = 500, parallel: bool = False, save_dir: str = "test_results", log_to_file: bool = True) -> Dict[str, Dict]:
    """
    Compare multiple agents across different scenarios.
    
    Args:
        scenarios (Dict[str, Scenario]): Dictionary of scenario name to scenario instance
        sim_indices (List[int]): List of simulation indices to run (default: [0,1,2,...,7]). Note that this list will be used as random seeds.
        max_draws (int): Max number of draws per simulation
        parallel (bool): Whether to run simulations in parallel using Ray
        agent_specs (List[AgentSpec]): Specifications for each agent
        save_dir (str): Directory to save intermediate simulation results
        log_to_file (bool): Whether to log detailed information to a file
    Returns:
        Dict: Results for each scenario and agent
    """
    if sim_indices is None:
        sim_indices = 8  # Default to 8 simulations if not provided
    if isinstance(sim_indices, int):
        sim_indices = list(range(sim_indices))
    
    from bart_playground.bandit.sim_util import setup_logging
    _ca_logger = setup_logging()
    if log_to_file:
        add_logging_file(save_dir)
        _ca_logger.info(f"Logging detailed information to {save_dir}/logs.")
    if parallel:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)
        _ca_logger.info("Ray initialized for parallel simulations.")
    
    results = {}
    
    try:
        for scenario_name, scenario in scenarios.items():
            _ca_logger.info(f"Evaluating {scenario_name} scenario...")
            n_draws = int(min(max_draws, scenario.max_draws))
            scenario_results = generate_simulation_data_for_agents(
                scenario_name=scenario_name,
                scenario=scenario,
                agent_specs=agent_specs,
                sim_indices=sim_indices,
                n_draws=n_draws,
                parallel=parallel,
                save_dir=save_dir
            )
            results[scenario_name] = scenario_results
    finally:
        # Shut down Ray when all work is finished.
        if parallel and ray.is_initialized():
            ray.shutdown()
            _ca_logger.info("Ray has been shut down.")
    
    return results

def plot_comparison_results(results: Dict[str, Dict], save_loc: str = None, show_random: bool = True, target: str = "regrets") -> None:
    """
    Plot comparison results across all scenarios and agents.
    
    Args:
        results (Dict): Results from compare_agents_across_scenarios
        save_loc (str): Location to save the plot (if None, will not save)
    """
    n_scenarios = len(results)
    n_rows = (n_scenarios + 1) // 2  # Ceiling division for number of rows
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 12 * n_rows))
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = [axes]
    
    # Plot each scenario
    scenario_idx = 0
    for scenario_name, scenario_results in results.items():
        row = scenario_idx // 2
        col = scenario_idx % 2
        
        ax = axes[row][col]
        
        colors = plt.get_cmap('nipy_spectral')(np.linspace(0, 0.9, len(scenario_results[target].keys())))
        ax.set_prop_cycle(color=colors)
        
        targeted_perf = scenario_results[target]
        if target == 'times':
            # For times, we need to cumulate it
            for agent_name, agent_times in targeted_perf.items():
                targeted_perf[agent_name] = np.cumsum(agent_times, axis=1)
        
        # For each agent
        for agent_name, agent_regrets in targeted_perf.items():
            if not show_random and agent_name.startswith("Random"):
                continue
            mean_regret = np.mean(agent_regrets, axis=0)
            sd_regret = np.std(agent_regrets, axis=0)
            lower_ci = mean_regret - sd_regret # np.percentile(agent_regrets, 25, axis=0)
            upper_ci = mean_regret + sd_regret # np.percentile(agent_regrets, 75, axis=0)
            
            n_draws = mean_regret.shape[0]
            # Plot mean regret
            ax.plot(range(n_draws), mean_regret, label=f"{agent_name}", linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(range(n_draws), lower_ci, upper_ci, alpha=0.2)

        ax.set_title(f"{scenario_name} Scenario", fontsize=14)
        ax.set_xlabel("Draw", fontsize=12)
        ax.set_ylabel(f"Cumulative {target}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        scenario_idx += 1
    
    # If there's an odd number of scenarios, remove the empty subplot
    if n_scenarios % 2 == 1:
        fig.delaxes(axes[-1][-1])
    
    plt.tight_layout()
    
    if save_loc:
        plt.savefig(save_loc, dpi=300)
    
    plt.show()

def print_summary_results(results: Dict[str, Dict]) -> None:
    """
    Display summary tables of final cumulative regrets and total computation times
    (mean ± std) for each scenario and agent.
    """
    for scenario, data in results.items():
        print(f"\n=== {scenario} Scenario ===")
        regrets = data['regrets']
        times = data['times']

        # Final regrets table
        rows = []
        for agent, arr in regrets.items():
            final = arr[:, -1]
            rows.append({
                'Agent': agent,
                'Mean Regret': np.mean(final),
                'Std Regret': np.std(final),
            })
        df_reg = pd.DataFrame(rows)
        print("\nFinal cumulative regrets (mean ± std):")
        print(df_reg.to_string(index=False, float_format="{:.2f}".format))

        # Computation times table
        rows = []
        for agent, arr in times.items():
            total = arr.sum(axis=1)
            rows.append({
                'Agent': agent,
                'Mean Time (s)': np.mean(total),
                'Std Time (s)': np.std(total),
            })
        df_time = pd.DataFrame(rows)
        print("\nAverage computation times (seconds):")
        print(df_time.to_string(index=False, float_format="{:.4f}".format))

        print("\n" + "=" * 40)

_rgx = None

def _parse_one(name: str) -> dict:
    m = _rgx.match(name)
    if not m:          # invalid → all None
        return {
            'Agent': name, 'Model': None, 'MC': None, 'Refresh': None,
            'Iter': None, 'Tree': None
        }

    return {
        'Agent': name,
        'Model': f"{'Logistic ' if m.group('log') else ''}{m.group('cat')}",
        'MC':    'yes' if m.group('mc') else 'no',
        'Refresh': 'yes' if m.group('refresh') else 'no',
        'Iter':  m.group('iter') or '1x',
        'Tree':  m.group('tree') or '1x',
    }
    
def _parse_agents(names):
    import re
    from collections import defaultdict
    
    global _rgx
    if _rgx is None:
        _rgx = re.compile(
            r'^(?P<log>Logistic)?'          # optional “Logistic”
            r'(?P<mc>MC)?'                  # optional “MC”
            r'(?P<refresh>Refresh)?'      # optional "Refresh"
            r'BART(?P<cat>[mos])'           # required “BARTm|o|s”
            r'(?:_iter(?P<iter>0\.5x|2x))?' # optional “_iter0.5x|2x”
            r'(?:_tree(?P<tree>0\.5x|2x))?' # optional “_tree0.5x|2x”
            r'$'
        )

    rows       = [_parse_one(n) for n in names]
    originals  = [r.copy() for r in rows]              # freeze originals
    props      = ('Model', 'MC', 'Iter', 'Tree', 'Refresh')

    for p in props:
        # group by the OTHER three properties
        groups = defaultdict(set)                      # key → {values of p}
        for r in originals:
            if r[p] is None:                           # skip invalid rows
                continue
            key = tuple(r[q] for q in props if q != p)
            groups[key].add(r[p])

        for i, r in enumerate(originals):              # decide Non-contrast
            if r[p] is None:
                continue
            key = tuple(r[q] for q in props if q != p)
            if len(groups[key]) == 1:                  # no contrasting peer
                rows[i][p] = 'Non-contrast'

    return rows

def _iter_relative_regrets(results: Dict[str, Dict], target: str
                           ) -> Iterator[Tuple[str, str, np.ndarray, float]]:
    for scenario, data in results.items():
        regrets = data[target]
        rnd = regrets.get("LinearTS")
        if rnd is None:
            continue
        rnd_mean = rnd[:, -1].mean()
        for agent, arr in regrets.items():
            if target == "times": # do not normalize times
                rel_arr = np.cumsum(arr, axis=1)  # cumulative times
            else:
                rel_arr  = arr / rnd_mean        # full path-wise regrets
            final_arr = rel_arr[:, -1]
            rel_mean = final_arr.mean() # / final_arr.std() # mean of final draw
            yield scenario, agent, rel_arr, rel_mean


def print_relative_performance(results: Dict[str, Dict], target: str = "regrets"):
    records = [
    {"Scenario": sc, "Agent": ag, "RelMean": rm}
    for sc, ag, _rel_arr, rm in _iter_relative_regrets(results, target=target)
    ]

    df = pd.DataFrame(records)
    
    from scipy.stats import hmean
    # now aggregate across scenarios
    summary = (
        df
        .groupby('Agent')['RelMean']
        .agg(
            MeanRelMean = hmean,   # average RelMean
            StdRelMean  = 'std',    # std dev of those RelMean’s
            N           = 'count'   # number of scenarios
        )
        .reset_index()
    )
    # standard error
    summary['SE_RelMean'] = summary['StdRelMean'] / np.sqrt(summary['N'])
    summary = summary.sort_values('MeanRelMean')

    # pretty‐print
    header = f"{'Agent':<25s} {'Mean':>8s} {'SE':>8s} {'Std (stability)':>8s}"
    sep    = "-" * len(header)
    print("\nAgent performance (relative to LinearTS):")
    print(header)
    print(sep)
    for _, row in summary.iterrows():
        print(f"{row.Agent:<25s} "
              f"{row.MeanRelMean:8.3f} "
              f"{row.SE_RelMean:8.3f} "
              f"{row.StdRelMean:8.3f}")
        
def add_average_scenario(results: Dict[str, Dict]) -> Dict[str, Dict]:
    from copy import deepcopy

    new_results = deepcopy(results)

    # collect normalised arrays per agent
    per_agent = {}
    max_iter = max(len(data['regrets']) for data in results.values())
    for _sc, ag, rel_arr, _rm in _iter_relative_regrets(results, target="regrets"):
        # if this agent is not yet in per_agent, start its list
        if ag not in per_agent:
            per_agent[ag] = []
        # append the normalized array
        per_agent[ag].append(rel_arr[:, :max_iter])
        
    # build the synthetic “Average” scenario
    new_results["Average"] = {
        "scenario_name": "Average",
        "sim_id": None,
        "regrets": {ag: np.vstack(arrs)  # shape: [runs × draws]
                    for ag, arrs in per_agent.items()},
        "times": {}
    }
    return new_results

def plot_print_total_regret_factors(results: Dict[str, Dict]) -> None:
    draw_index = -1  # Use the last draw (all cumulative regrets) for boxplots
    results = add_average_scenario(results)
    
    factor_cols = ['Model', 'Tree', 'Iter', 'MC', 'Refresh']
    from collections import defaultdict
    cat_means = {c: defaultdict(dict) for c in factor_cols}

    for scenario, data in results.items():
        records = []
        for agent, arr in data['regrets'].items():
            for val in arr[:, draw_index]:
                records.append({'Agent': agent, 'Regret': val})
        df_long = pd.DataFrame(records)

        # parse_agents on the list of agent names
        meta = (
            pd.DataFrame(_parse_agents(list(data['regrets'].keys())))
              .set_index('Agent')
        )
        df_long = df_long.join(meta, on='Agent')

        # Plot by each factor
        for col in factor_cols:
            plt.figure(figsize=(8, 5))
            df_plot = df_long.copy()
            df_plot[col] = df_plot[col].fillna(df_plot['Agent'])
            df_plot.boxplot(column='Regret', by=col, showmeans=True)

            plt.title(f"{scenario} - Regret by {col}")
            plt.suptitle('')
            plt.ylabel('Regret')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"results/boxplots/{scenario}_{col}_regret_boxplot.png", dpi=300)
            plt.close('all')

            # collect means for printable table
            valid = df_long[df_long[col].notna()]  # drop agents without this factor
            # compute mean regret per category within the factor
            grp = (
                valid.groupby(col)['Regret']
                     .mean()
                     .to_dict()           # {category: mean}
            )
            for category, mean_val in grp.items():
                if category == "Non-contrast":
                    continue
                cat_means[col][category][scenario] = mean_val

    for col in factor_cols:
        df_tbl = pd.DataFrame(cat_means[col])           # rows=categories, cols=scenarios
        df_tbl = df_tbl.sort_index()                    # nicer ordering
        print(f"\nMean relative regret by \"{col}\" category (rows) and scenario (columns):")
        print(df_tbl.to_string(float_format="%.3f"))

def print_topn_agents_per_scenario(results: Dict[str, Dict], top_n: int = 5):
    # collect top-n agents per scenario
    top_per_scenario = {}
    for scenario, data in results.items():
        means = {
            agent: arr[:, -1].mean()
            for agent, arr in data['regrets'].items()
        }
        # get the best top_n agent names
        best_agents = [agent for agent, _ in sorted(means.items(), key=lambda x: x[1])[:top_n]]
        top_per_scenario[scenario] = best_agents

    # build a table: rows are ranks 1..top_n, columns are scenarios
    df_top = pd.DataFrame(
        top_per_scenario,
        index=[f"Rank {i+1}" for i in range(top_n)]
    )
    print("\nTop {} agents per scenario:".format(top_n))
    print(df_top.to_string())


from typing import Dict
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations

def print_pairwise_win_tie_lose(results: Dict[str, Dict], alpha: float = 0.05):
    normed = add_average_scenario(results)
    
    # Collect all unique agents across all scenarios
    all_agents = set()
    for data in normed.values():
        all_agents.update(data['regrets'].keys())
    agents = sorted(all_agents)
    
    # Accumulator for totals
    totals = {a: {'win': 0, 'tie': 0, 'lose': 0} for a in agents}
    
    for scenario, data in normed.items():
        # Initialize per-scenario counters
        counts = {a: {'win': 0, 'tie': 0, 'lose': 0} for a in agents}
        # Cache final-draw arrays for agents present in this scenario
        final_vals = {a: arr[:, -1] for a, arr in data['regrets'].items()}
        
        # All pairs
        for a1, a2 in combinations(agents, 2):
            if a1 in final_vals and a2 in final_vals:
                v1, v2 = final_vals[a1], final_vals[a2]
                stat, p = ttest_ind(v1, v2, equal_var=False)
                if p < alpha:
                    # Significant difference (lower mean regret is better)
                    if v1.mean() < v2.mean():
                        counts[a1]['win'] += 1
                        counts[a2]['lose'] += 1
                    else:
                        counts[a2]['win'] += 1
                        counts[a1]['lose'] += 1
                else:
                    # No significant difference
                    counts[a1]['tie'] += 1
                    counts[a2]['tie'] += 1
            else:
                # Not both present: consider as tie
                counts[a1]['tie'] += 1
                counts[a2]['tie'] += 1
        
        # Accumulate totals
        for a in agents:
            for key in ('win', 'tie', 'lose'):
                totals[a][key] += counts[a][key]
    
    # Prepare and print grand-total
    df_tot = pd.DataFrame.from_dict(totals, orient='index')
    df_tot = (
        df_tot
        .assign(score=df_tot['win'] * 3 + df_tot['tie'])
        .sort_values(by='score', ascending=False)
    )
    print("\nTotals across all scenarios (sorted by win*3+tie):")
    print(df_tot[['win', 'tie', 'lose']].to_string())
    