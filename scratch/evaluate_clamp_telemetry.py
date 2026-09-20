import os
import pickle
import numpy as np
import pandas as pd

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

bank_file = "checkpoints/test_bank_50.pkl"
with open(bank_file, "rb") as f:
    bank_data = pickle.load(f)

seen_scenarios = bank_data["seen_35"]
channel = ChannelModel()

results = []

total_clamp_triggers = 0
total_clamp_evals = 0
all_clamp_ratios = []

print(f"=== Evaluating Geometric Slack Clamp Telemetry Across 35 Seen Seeds ===")

for idx, sc in enumerate(seen_scenarios):
    seed = sc["seed"]
    num_drones = sc["num_drones"]
    comm_range = sc["comm_range"]
    fail_agent_id = sc["fail_agent_id"]
    fail_timestep = sc["fail_timestep"]
    
    # Evaluate DARA
    config_dara = ExperimentConfig(num_drones=num_drones, comm_range=comm_range, seed=seed)
    sim_dara = SwarmSimulator(config_dara)
    scenario_dara = DisasterRelayScenario(sim_dara)
    g_a, g_b = scenario_dara.setup_scenario(endpoint_a_pos=sc["endpoint_a_pos"], endpoint_b_pos=sc["endpoint_b_pos"])
    
    inj_dara = FailureInjector(sim_dara)
    topo_dara = SwarmTopologyManager(sim_dara)
    policy_dara = DARAHeuristicPolicy(sim_dara)
    
    failed_dara = False
    reconn_dara = None
    
    for t in range(150):
        if t == fail_timestep:
            inj_dara.fail_agent(fail_agent_id)
            failed_dara = True
            
        acc = policy_dara.compute_control_forces()
        sim_dara.step(accelerations=acc)
        
        G = topo_dara.build_graph()
        if failed_dara and reconn_dara is None:
            if topo_dara.has_path(g_a, g_b, graph=G):
                reconn_dara = t - fail_timestep
                
    G_dara_final = topo_dara.build_graph()
    conn_dara = topo_dara.has_path(g_a, g_b, graph=G_dara_final)
    sum_rate_dara = channel.compute_network_throughput(sim_dara, G_dara_final)
    
    # Evaluate Greedy DARA
    config_greedy = ExperimentConfig(num_drones=num_drones, comm_range=comm_range, seed=seed)
    sim_greedy = SwarmSimulator(config_greedy)
    scenario_greedy = DisasterRelayScenario(sim_greedy)
    g_a, g_b = scenario_greedy.setup_scenario(endpoint_a_pos=sc["endpoint_a_pos"], endpoint_b_pos=sc["endpoint_b_pos"])
    
    inj_greedy = FailureInjector(sim_greedy)
    topo_greedy = SwarmTopologyManager(sim_greedy)
    policy_greedy = GreedyDARAHeuristicPolicy(sim_greedy, channel_model=channel)
    policy_greedy.reset_clamp_stats()
    
    failed_greedy = False
    reconn_greedy = None
    
    for t in range(150):
        if t == fail_timestep:
            inj_greedy.fail_agent(fail_agent_id)
            failed_greedy = True
            
        acc = policy_greedy.compute_control_forces()
        sim_greedy.step(accelerations=acc)
        
        G = topo_greedy.build_graph()
        if failed_greedy and reconn_greedy is None:
            if topo_greedy.has_path(g_a, g_b, graph=G):
                reconn_greedy = t - fail_timestep
                
    G_greedy_final = topo_greedy.build_graph()
    conn_greedy = topo_greedy.has_path(g_a, g_b, graph=G_greedy_final)
    sum_rate_greedy = channel.compute_network_throughput(sim_greedy, G_greedy_final)
    
    total_clamp_triggers += policy_greedy.clamp_trigger_count
    total_clamp_evals += policy_greedy.total_eval_count
    all_clamp_ratios.extend(policy_greedy.clamp_ratios)
    
    results.append({
        "seed": seed,
        "num_drones": num_drones,
        "fail_agent_id": fail_agent_id,
        "dara_conn": conn_dara,
        "dara_time": reconn_dara if reconn_dara is not None else np.nan,
        "dara_sum_rate": sum_rate_dara,
        "greedy_conn": conn_greedy,
        "greedy_time": reconn_greedy if reconn_greedy is not None else np.nan,
        "greedy_sum_rate": sum_rate_greedy,
        "clamp_triggers": policy_greedy.clamp_trigger_count,
        "total_evals": policy_greedy.total_eval_count
    })

df_res = pd.DataFrame(results)

print("\n=== CLAMP TELEMETRY RESULTS ===")
print(f"1. Total tick-relay evaluations: {total_clamp_evals}")
print(f"2. Total tick-relay pairs triggering clamp (|snr_force| > 0.8 * min_slack): {total_clamp_triggers}")
if total_clamp_evals > 0:
    print(f"3. Trigger percentage: {total_clamp_triggers / total_clamp_evals * 100.0:.2f}%")

if all_clamp_ratios:
    mean_ratio = float(np.mean(all_clamp_ratios))
    min_ratio = float(np.min(all_clamp_ratios))
    max_ratio = float(np.max(all_clamp_ratios))
    print(f"4. Clamp ratio (|snr_clamped| / |snr_unclamped|) stats:")
    print(f"   - Mean ratio: {mean_ratio:.4f}")
    print(f"   - Min ratio : {min_ratio:.4f}")
    print(f"   - Max ratio : {max_ratio:.4f}")

print("\n=== SUMMARY METRICS SIDE-BY-SIDE ===")
dara_successes = sum(df_res["dara_conn"])
greedy_successes = sum(df_res["greedy_conn"])

dara_mean_time = df_res[df_res["dara_conn"]]["dara_time"].mean()
greedy_mean_time = df_res[df_res["greedy_conn"]]["greedy_time"].mean()

dara_mean_rate = df_res[df_res["dara_conn"]]["dara_sum_rate"].mean()
greedy_mean_rate = df_res[df_res["greedy_conn"]]["greedy_sum_rate"].mean()

print(f"Metric                           | Heuristic DARA | Fixed Clamped Greedy DARA")
print(f"-----------------------------------------------------------------------------")
print(f"Reconnection Success Rate        | {dara_successes}/35 ({dara_successes/35*100:.1f}%) | {greedy_successes}/35 ({greedy_successes/35*100:.1f}%)")
print(f"Mean Time-to-Reconnect (Ticks)   | {dara_mean_time:.2f} t       | {greedy_mean_time:.2f} t")
print(f"Mean Sum-Rate (Reconnected)      | {dara_mean_rate:.2f} Mbps | {greedy_mean_rate:.2f} Mbps")
