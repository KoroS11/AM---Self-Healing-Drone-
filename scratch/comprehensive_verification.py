import os
import pickle
import hashlib
import pathlib
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

def run_35_seeds_experiment(k_tp=1.0, alpha1=1.0, alpha2=1.0):
    bank_file = "checkpoints/test_bank_50.pkl"
    with open(bank_file, "rb") as f:
        bank_data = pickle.load(f)
    seen_scenarios = bank_data["seen_35"]
    channel = ChannelModel()
    
    rows = []
    
    for sc in seen_scenarios:
        seed = sc["seed"]
        num_drones = sc["num_drones"]
        comm_range = sc["comm_range"]
        fail_agent_id = sc["fail_agent_id"]
        fail_timestep = sc["fail_timestep"]
        
        # 1. Heuristic DARA
        cfg_dara = ExperimentConfig(num_drones=num_drones, comm_range=comm_range, seed=seed)
        sim_dara = SwarmSimulator(cfg_dara)
        scen_dara = DisasterRelayScenario(sim_dara)
        g_a, g_b = scen_dara.setup_scenario(endpoint_a_pos=sc["endpoint_a_pos"], endpoint_b_pos=sc["endpoint_b_pos"])
        inj_dara = FailureInjector(sim_dara)
        topo_dara = SwarmTopologyManager(sim_dara)
        pol_dara = DARAHeuristicPolicy(sim_dara)
        
        reconn_dara = None
        for t in range(150):
            if t == fail_timestep:
                inj_dara.fail_agent(fail_agent_id)
            acc = pol_dara.compute_control_forces()
            sim_dara.step(accelerations=acc)
            G = topo_dara.build_graph()
            if t > fail_timestep and reconn_dara is None:
                if topo_dara.has_path(g_a, g_b, graph=G):
                    reconn_dara = t - fail_timestep
                    
        G_dara = topo_dara.build_graph()
        conn_dara = topo_dara.has_path(g_a, g_b, graph=G_dara)
        sr_dara = channel.compute_network_throughput(sim_dara, G_dara)
        
        # 2. Lexicographic Greedy DARA
        cfg_greedy = ExperimentConfig(num_drones=num_drones, comm_range=comm_range, seed=seed)
        sim_greedy = SwarmSimulator(cfg_greedy)
        scen_greedy = DisasterRelayScenario(sim_greedy)
        g_a, g_b = scen_greedy.setup_scenario(endpoint_a_pos=sc["endpoint_a_pos"], endpoint_b_pos=sc["endpoint_b_pos"])
        inj_greedy = FailureInjector(sim_greedy)
        topo_greedy = SwarmTopologyManager(sim_greedy)
        pol_greedy = GreedyDARAHeuristicPolicy(sim_greedy, channel_model=channel, k_tp=k_tp, alpha1=alpha1, alpha2=alpha2)
        
        reconn_greedy = None
        for t in range(150):
            if t == fail_timestep:
                inj_greedy.fail_agent(fail_agent_id)
            acc = pol_greedy.compute_control_forces()
            sim_greedy.step(accelerations=acc)
            G = topo_greedy.build_graph()
            if t > fail_timestep and reconn_greedy is None:
                if topo_greedy.has_path(g_a, g_b, graph=G):
                    reconn_greedy = t - fail_timestep
                    
        G_greedy = topo_greedy.build_graph()
        conn_greedy = topo_greedy.has_path(g_a, g_b, graph=G_greedy)
        sr_greedy = channel.compute_network_throughput(sim_greedy, G_greedy)
        
        rows.append({
            "seed": seed,
            "num_drones": num_drones,
            "fail_agent_id": fail_agent_id,
            "dara_conn": conn_dara,
            "dara_time": reconn_dara,
            "dara_sum_rate": sr_dara,
            "greedy_conn": conn_greedy,
            "greedy_time": reconn_greedy,
            "greedy_sum_rate": sr_greedy,
            "delta_mbps": sr_greedy - sr_dara,
            "pct_delta": ((sr_greedy - sr_dara) / sr_dara) * 100.0,
            "ticks_pure_dara": pol_greedy.ticks_pure_dara,
            "ticks_polishing": pol_greedy.ticks_throughput_polishing,
            "infeasible_solves": pol_greedy.infeasible_solve_count,
            "total_solves": pol_greedy.total_solve_count
        })
        
    return pd.DataFrame(rows)

print("=== 1. DETERMINISM VERIFICATION: RUNNING TWO IDENTICAL PASSES ===")
print("Executing Pass 1...")
df_pass1 = run_35_seeds_experiment(k_tp=1.0)
print("Executing Pass 2...")
df_pass2 = run_35_seeds_experiment(k_tp=1.0)

# Check bit-level equality
dara_time_eq = np.array_equal(df_pass1["dara_time"].values, df_pass2["dara_time"].values)
dara_rate_eq = np.array_equal(df_pass1["dara_sum_rate"].values, df_pass2["dara_sum_rate"].values)
greedy_time_eq = np.array_equal(df_pass1["greedy_time"].values, df_pass2["greedy_time"].values)
greedy_rate_eq = np.array_equal(df_pass1["greedy_sum_rate"].values, df_pass2["greedy_sum_rate"].values)

all_bit_identical = dara_time_eq and dara_rate_eq and greedy_time_eq and greedy_rate_eq
max_abs_diff_dara = np.max(np.abs(df_pass1["dara_sum_rate"].values - df_pass2["dara_sum_rate"].values))
max_abs_diff_greedy = np.max(np.abs(df_pass1["greedy_sum_rate"].values - df_pass2["greedy_sum_rate"].values))

print(f"\nDeterminism Results:")
print(f"  - DARA Time Bit-Identical:        {dara_time_eq}")
print(f"  - DARA Sum-Rate Bit-Identical:    {dara_rate_eq} (Max abs diff: {max_abs_diff_dara})")
print(f"  - Greedy Time Bit-Identical:      {greedy_time_eq}")
print(f"  - Greedy Sum-Rate Bit-Identical:  {greedy_rate_eq} (Max abs diff: {max_abs_diff_greedy})")
print(f"  - Overall Determinism:            {'PASSED (100% BIT-IDENTICAL)' if all_bit_identical else 'FAILED'}")

print("\n=== 2. RAW PER-SEED TABLE (ALL 35 SEEDS) ===")
display_cols = ["seed", "num_drones", "fail_agent_id", "dara_time", "dara_sum_rate", "greedy_time", "greedy_sum_rate", "delta_mbps", "pct_delta", "ticks_polishing"]
print(df_pass1[display_cols].to_string(index=False))

print("\n=== 3. STATISTICAL SIGNIFICANCE (PAIRED WILCOXON SIGNED-RANK TEST) ===")
dara_rates = df_pass1["dara_sum_rate"].values
greedy_rates = df_pass1["greedy_sum_rate"].values
deltas = df_pass1["delta_mbps"].values

# Wilcoxon signed-rank test (one-sided greater and two-sided)
res_greater = stats.wilcoxon(greedy_rates, dara_rates, alternative="greater")
res_two_sided = stats.wilcoxon(greedy_rates, dara_rates, alternative="two-sided")

# Paired t-test for additional parametric validation
t_test_res = stats.ttest_rel(greedy_rates, dara_rates)

print(f"Sample Size:                     N = {len(deltas)}")
print(f"Mean Delta (Greedy - DARA):      {np.mean(deltas):+.6f} Mbps ({np.mean(df_pass1['pct_delta']):+.4f}%)")
print(f"Median Delta:                    {np.median(deltas):+.6f} Mbps ({np.median(df_pass1['pct_delta']):+.4f}%)")
print(f"Standard Deviation of Delta:     {np.std(deltas, ddof=1):.6f} Mbps")
print(f"Min Delta:                       {np.min(deltas):+.6f} Mbps")
print(f"Max Delta:                       {np.max(deltas):+.6f} Mbps")
print(f"Number of Seeds with Delta > 0:  {np.sum(deltas > 0)} / 35 ({np.sum(deltas > 0)/35*100:.1f}%)")
print(f"Number of Seeds with Delta == 0: {np.sum(deltas == 0)} / 35")
print(f"Number of Seeds with Delta < 0:  {np.sum(deltas < 0)} / 35")
print(f"\nWilcoxon Signed-Rank Test (alternative='greater'):")
print(f"  - Statistic W:                 {res_greater.statistic}")
print(f"  - p-value:                     {res_greater.pvalue:.4e} (p < 0.001: {res_greater.pvalue < 0.001})")
print(f"\nWilcoxon Signed-Rank Test (alternative='two-sided'):")
print(f"  - Statistic W:                 {res_two_sided.statistic}")
print(f"  - p-value:                     {res_two_sided.pvalue:.4e}")
print(f"\nPaired t-test (two-sided):")
print(f"  - t-statistic:                 {t_test_res.statistic:.4f}")
print(f"  - p-value:                     {t_test_res.pvalue:.4e}")

print("\n=== 4. CONTROL EXPERIMENT (k_tp = 0.0: CBF-QP FILTERING ZERO THROUGHPUT TERM) ===")
print("Executing Control Pass with k_tp=0.0...")
df_control = run_35_seeds_experiment(k_tp=0.0)

control_dara_diff = np.abs(df_control["greedy_sum_rate"].values - df_control["dara_sum_rate"].values)
control_max_diff = np.max(control_dara_diff)
control_mean_diff = np.mean(control_dara_diff)
control_exact_match = np.allclose(df_control["greedy_sum_rate"].values, df_control["dara_sum_rate"].values, atol=1e-5)

print(f"Control Results (k_tp=0.0 vs DARA Baseline):")
print(f"  - Max Absolute Difference:     {control_max_diff:.8f} Mbps")
print(f"  - Mean Absolute Difference:    {control_mean_diff:.8f} Mbps")
print(f"  - Exact Match (atol=1e-5):     {control_exact_match}")
print(f"  - Number of Seeds Identical:   {np.sum(np.isclose(df_control['greedy_sum_rate'].values, df_control['dara_sum_rate'].values, atol=1e-5))} / 35")

if not control_exact_match:
    print("\nPer-seed differences in control run:")
    diff_df = pd.DataFrame({
        "seed": df_control["seed"],
        "dara_rate": df_control["dara_sum_rate"],
        "control_rate": df_control["greedy_sum_rate"],
        "diff": df_control["greedy_sum_rate"] - df_control["dara_sum_rate"]
    })
    print(diff_df[np.abs(diff_df["diff"]) > 1e-6].to_string(index=False))
