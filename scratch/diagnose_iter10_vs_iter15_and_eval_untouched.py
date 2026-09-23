import os
import pickle
import torch
import numpy as np
import pandas as pd

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.scenarios.test_bank import TestBankGenerator
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from train_pipeline import TrainPipeline

def rollout_with_reward_breakdown(actor, scenario_config, device="cpu", max_steps=150):
    config = ExperimentConfig(
        num_drones=scenario_config["num_drones"],
        comm_range=scenario_config["comm_range"],
        seed=scenario_config["seed"]
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario(
        endpoint_a_pos=scenario_config.get("endpoint_a_pos", (0.0, 0.0)),
        endpoint_b_pos=scenario_config.get("endpoint_b_pos", (200.0, 0.0))
    )
    injector = FailureInjector(sim)
    policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()
    
    comm_range = scenario_config["comm_range"]
    lambda_bonus = 500.0
    lambda_conn = 1.0
    lambda_disc_step = -0.1
    lambda_margin = 0.20
    lambda_vel = 0.05
    lambda_acc = 0.002
    
    ever_reconnected = False
    reconnected_step = None
    failed = False
    
    comp_bonus = 0.0
    comp_conn = 0.0
    comp_margin = 0.0
    comp_vel = 0.0
    comp_acc = 0.0
    
    for t in range(max_steps):
        if t == scenario_config["fail_timestep"] and not failed:
            injector.fail_agent(scenario_config["fail_agent_id"])
            failed = True
            
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        is_conn = topo.has_path(g_a, g_b, graph=G)
        
        is_first_reconn = False
        if failed and is_conn and not ever_reconnected:
            is_first_reconn = True
            ever_reconnected = True
            reconnected_step = t - scenario_config["fail_timestep"]
            
        # 1. Bonus
        r_b = lambda_bonus if is_first_reconn else 0.0
        # 2. Conn
        r_c = lambda_conn if is_conn else lambda_disc_step
        # 3 & 4. Margin & Vel
        r_m = 0.0
        r_v = 0.0
        if is_conn:
            mobile_mask = (sim.statuses == AgentStatus.ACTIVE) & (sim.max_speeds > 0.0)
            if np.any(mobile_mask):
                v_sq = np.mean(np.sum(sim.velocities[mobile_mask] ** 2, axis=1))
                r_v = -lambda_vel * float(v_sq)
                
            pos = sim.positions
            active_mask = sim.statuses == AgentStatus.ACTIVE
            active_indices = np.where(active_mask)[0]
            if len(active_indices) >= 2:
                diffs = pos[active_indices, None, :] - pos[None, active_indices, :]
                dists = np.linalg.norm(diffs, axis=-1)
                i_upper, j_upper = np.triu_indices(len(active_indices), k=1)
                edge_dists = dists[i_upper, j_upper]
                valid_edges = edge_dists[edge_dists <= comm_range]
                if len(valid_edges) > 0:
                    margins = (comm_range - valid_edges) / comm_range
                    r_m = lambda_margin * float(np.mean(margins))
        # 5. Acc
        r_a = 0.0
        if acc is not None:
            mobile_mask = (sim.statuses == AgentStatus.ACTIVE) & (sim.max_speeds > 0.0)
            if np.any(mobile_mask):
                a_sq = np.mean(np.sum(acc[mobile_mask] ** 2, axis=1))
                r_a = -lambda_acc * float(a_sq)
                
        comp_bonus += r_b
        comp_conn += r_c
        comp_margin += r_m
        comp_vel += r_v
        comp_acc += r_a

    G_final = topo.build_graph()
    final_conn = topo.has_path(g_a, g_b, graph=G_final)
    sum_rate = channel.compute_network_throughput(sim, G_final)
    
    return {
        "seed": scenario_config["seed"],
        "reconnected": ever_reconnected,
        "reconnect_time": reconnected_step,
        "final_connected": final_conn,
        "sum_rate_mbps": sum_rate,
        "r_bonus": comp_bonus,
        "r_conn": comp_conn,
        "r_margin": comp_margin,
        "r_vel": comp_vel,
        "r_acc": comp_acc,
        "total_reward": comp_bonus + comp_conn + comp_margin + comp_vel + comp_acc
    }

def main():
    pipeline = TrainPipeline(output_dir="checkpoints")
    device = pipeline.device
    all_50 = pipeline.bank_data["all_50"]
    
    # Load Actor 10 and Actor 15
    actor_10 = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_10.load_state_dict(torch.load("checkpoints/checkpoint_B_iter_10.pt", map_location=device), strict=False)
    actor_10.eval()
    
    actor_15 = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_15.load_state_dict(torch.load("checkpoints/checkpoint_B_iter_15.pt", map_location=device), strict=False)
    actor_15.eval()
    
    print("==================================================================")
    print(" PART 1: DIAGNOSIS OF ITERATION 10 -> ITERATION 15 REGRESSION")
    print("==================================================================")
    
    records_10 = [rollout_with_reward_breakdown(actor_10, sc, device=device) for sc in all_50]
    records_15 = [rollout_with_reward_breakdown(actor_15, sc, device=device) for sc in all_50]
    
    df_10 = pd.DataFrame(records_10)
    df_15 = pd.DataFrame(records_15)
    
    print("\n--- Aggregate Reward Component Comparison (Full 50 Scenarios) ---")
    metrics = ["r_bonus", "r_conn", "r_margin", "r_vel", "r_acc", "total_reward", "reconnect_time", "sum_rate_mbps"]
    agg_table = []
    for m in metrics:
        v10 = df_10[m].mean()
        v15 = df_15[m].mean()
        diff = v15 - v10
        agg_table.append({
            "Component": m,
            "Iter 10 Mean": v10,
            "Iter 15 Mean": v15,
            "Delta (15 - 10)": diff
        })
    df_agg = pd.DataFrame(agg_table)
    print(df_agg.to_string(index=False))
    
    print("\n--- Per-Scenario Pass/Fail Delta (Seeds where Iter 10 != Iter 15) ---")
    delta_seeds = []
    for i, sc in enumerate(all_50):
        s10 = df_10.iloc[i]["reconnected"]
        s15 = df_15.iloc[i]["reconnected"]
        if s10 != s15:
            delta_seeds.append({
                "Seed": sc["seed"],
                "Iter 10 Pass": s10,
                "Iter 10 Time": df_10.iloc[i]["reconnect_time"],
                "Iter 10 Reward": df_10.iloc[i]["total_reward"],
                "Iter 15 Pass": s15,
                "Iter 15 Time": df_15.iloc[i]["reconnect_time"],
                "Iter 15 Reward": df_15.iloc[i]["total_reward"],
                "Num Drones": sc["num_drones"],
                "Failed Agent": sc["fail_agent_id"]
            })
    df_delta = pd.DataFrame(delta_seeds)
    print(df_delta.to_string(index=False))
    df_delta.to_csv("scratch/iter10_vs_iter15_delta_seeds.csv", index=False)
    
    print("\n==================================================================")
    print(" PART 2: UNTOUCHED CONFIRMATION SUITE (SEEDS 1060..1069)")
    print("==================================================================")
    
    tb_gen = TestBankGenerator()
    untouched_10 = [tb_gen.generate_scenario_config(s) for s in range(1060, 1070)]
    
    actor_A = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_A.load_state_dict(torch.load("checkpoints/checkpoint_A_best_k2.pt", map_location=device), strict=False)
    actor_A.eval()
    
    actor_5 = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_5.load_state_dict(torch.load("checkpoints/checkpoint_B_iter_5.pt", map_location=device), strict=False)
    actor_5.eval()
    
    eval_A = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor_A, device=str(device)), untouched_10)
    eval_5 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor_5, device=str(device)), untouched_10)
    eval_10 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor_10, device=str(device)), untouched_10)
    eval_15 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor_15, device=str(device)), untouched_10)
    
    print(f"Untouched Seeds 1060..1069 Results (Zero Pre-Selection Leakage):")
    print(f"  Checkpoint A (Warm-Start):  Success = {eval_A['success_rate']*100:.1f}% ({int(eval_A['success_rate']*10)}/10), Mean Time = {eval_A['mean_time']:.2f} t, SR = {eval_A['mean_sum_rate']:.2f} Mbps")
    print(f"  Checkpoint B (Iteration 5):  Success = {eval_5['success_rate']*100:.1f}% ({int(eval_5['success_rate']*10)}/10), Mean Time = {eval_5['mean_time']:.2f} t, SR = {eval_5['mean_sum_rate']:.2f} Mbps")
    print(f"  Checkpoint B (Iteration 10): Success = {eval_10['success_rate']*100:.1f}% ({int(eval_10['success_rate']*10)}/10), Mean Time = {eval_10['mean_time']:.2f} t, SR = {eval_10['mean_sum_rate']:.2f} Mbps")
    print(f"  Checkpoint B (Iteration 15): Success = {eval_15['success_rate']*100:.1f}% ({int(eval_15['success_rate']*10)}/10), Mean Time = {eval_15['mean_time']:.2f} t, SR = {eval_15['mean_sum_rate']:.2f} Mbps")
    
    untouched_summary = [
        {"Model": "Checkpoint A Baseline", "Success": eval_A['success_rate'], "Mean Time": eval_A['mean_time'], "Sum-Rate": eval_A['mean_sum_rate']},
        {"Model": "Checkpoint B (Iter 5)", "Success": eval_5['success_rate'], "Mean Time": eval_5['mean_time'], "Sum-Rate": eval_5['mean_sum_rate']},
        {"Model": "Checkpoint B (Iter 10)", "Success": eval_10['success_rate'], "Mean Time": eval_10['mean_time'], "Sum-Rate": eval_10['mean_sum_rate']},
        {"Model": "Checkpoint B (Iter 15)", "Success": eval_15['success_rate'], "Mean Time": eval_15['mean_time'], "Sum-Rate": eval_15['mean_sum_rate']}
    ]
    pd.DataFrame(untouched_summary).to_csv("scratch/untouched_seeds_1060_1069_results.csv", index=False)
    print("\nSaved untouched evaluation to scratch/untouched_seeds_1060_1069_results.csv")

if __name__ == "__main__":
    main()
