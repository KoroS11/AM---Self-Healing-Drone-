import os
import sys
import pickle
import hashlib
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.scenarios.test_bank import TestBankGenerator
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Unknown ({e})"

def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def evaluate_closed_loop_scenarios(actor, scenarios, device="cpu", max_steps=150, heterogeneous_overrides=None):
    actor.eval()
    channel = ChannelModel()
    results = []
    
    for sc in scenarios:
        config = ExperimentConfig(
            num_drones=sc["num_drones"],
            comm_range=sc["comm_range"],
            seed=sc["seed"]
        )
        sim = SwarmSimulator(config)
        scenario = DisasterRelayScenario(sim)
        g_a, g_b = scenario.setup_scenario(
            endpoint_a_pos=sc.get("endpoint_a_pos", (0.0, 0.0)),
            endpoint_b_pos=sc.get("endpoint_b_pos", (200.0, 0.0))
        )
        
        # Apply custom heterogeneous comm ranges if specified
        if heterogeneous_overrides and sc["seed"] in heterogeneous_overrides:
            for agent_id, r_val in heterogeneous_overrides[sc["seed"]].items():
                sim.comm_ranges[agent_id] = r_val
        elif "custom_comm_ranges" in sc:
            for agent_id, r_val in sc["custom_comm_ranges"].items():
                sim.comm_ranges[agent_id] = r_val
                
        injector = FailureInjector(sim)
        policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
        topo = SwarmTopologyManager(sim)
        
        failures = sc.get("failures", [])
        if not failures and "fail_timestep" in sc:
            failures = [{"step": sc["fail_timestep"], "agent_id": sc["fail_agent_id"]}]
            
        last_fail_step = max([f["step"] for f in failures]) if failures else 0
        
        ever_reconnected = False
        reconnected_rel_step = None
        reconnected_abs_step = None
        disconnections_after_reconnect = 0
        previously_reconnected = False
        
        for t in range(max_steps):
            for f in failures:
                if t == f["step"]:
                    injector.fail_agent(f["agent_id"])
                    
            acc = policy.compute_control_forces()
            sim.step(accelerations=acc)
            
            G = topo.build_graph()
            is_conn = topo.has_path(g_a, g_b, graph=G)
            
            if t >= last_fail_step:
                if is_conn:
                    if not ever_reconnected:
                        ever_reconnected = True
                        reconnected_abs_step = t
                        reconnected_rel_step = t - last_fail_step
                    previously_reconnected = True
                else:
                    if previously_reconnected:
                        disconnections_after_reconnect += 1
                        
        G_final = topo.build_graph()
        final_conn = topo.has_path(g_a, g_b, graph=G_final)
        sum_rate = channel.compute_network_throughput(sim, G_final)
        
        failure_mode = "Success (Stable)"
        if not final_conn:
            if not ever_reconnected:
                failure_mode = "Failed to Reconnect (Insufficient Span/Reach)"
            elif disconnections_after_reconnect > 0:
                failure_mode = f"Reconnect-then-Drift ({disconnections_after_reconnect} post-reconnect disconnects)"
            else:
                failure_mode = "Disconnected at Step 150"
        elif disconnections_after_reconnect > 0:
            failure_mode = f"Success with Oscillation ({disconnections_after_reconnect} transient drops)"
            
        results.append({
            "category": sc.get("category", "Benchmark"),
            "desc": sc.get("desc", f"N={sc['num_drones']}, Seed={sc['seed']}"),
            "seed": sc["seed"],
            "num_drones": sc["num_drones"],
            "comm_range": sc["comm_range"],
            "reconnected": final_conn,
            "reconnect_rel_time": reconnected_rel_step,
            "reconnect_abs_time": reconnected_abs_step,
            "transient_drops": disconnections_after_reconnect,
            "sum_rate": sum_rate,
            "failure_mode": failure_mode
        })
        
    return pd.DataFrame(results)

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    
    b_prime_path = "checkpoints/checkpoint_B_prime_best.pt"
    if not os.path.exists(b_prime_path):
        # Fallback to iteration 10 or latest if best not saved
        candidates = [f"checkpoints/checkpoint_B_prime_iter_{i}.pt" for i in range(15, 0, -1)]
        for c in candidates:
            if os.path.exists(c):
                b_prime_path = c
                break
                
    commit_hash = get_git_commit_hash()
    b_prime_md5 = compute_file_md5(b_prime_path)
    
    print("==================================================================")
    print(" B-PRIME COMPLETE VERIFICATION & 4-STEP PROTOCOL BATTERY")
    print("==================================================================")
    print(f"Git Commit Hash:         {commit_hash}")
    print(f"B-Prime Checkpoint Path: {b_prime_path}")
    print(f"B-Prime Checkpoint MD5:  {b_prime_md5}")
    print("------------------------------------------------------------------")
    
    # Load B-Prime Model
    actor_B_prime = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0).to(device)
    raw_state = torch.load(b_prime_path, map_location=device)
    actor_state = raw_state["actor_state_dict"] if "actor_state_dict" in raw_state else raw_state
    actor_B_prime.load_state_dict(actor_state, strict=True)
    actor_B_prime.eval()
    
    # Load Checkpoint B Best for baseline comparison
    b_best_path = "checkpoints/checkpoint_B_best.pt"
    actor_B_best = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0).to(device)
    raw_state_b = torch.load(b_best_path, map_location=device)
    actor_state_b = raw_state_b["actor_state_dict"] if "actor_state_dict" in raw_state_b else raw_state_b
    
    # Wrap in zero-init state for comparison
    actor_b_dict = actor_B_best.state_dict()
    for k, v in actor_state_b.items():
        if k == "node_encoder.0.weight" and v.shape[1] == 8:
            w_new = torch.zeros((64, 9), dtype=v.dtype)
            w_new[:, :8] = v
            w_new[:, 8] = 0.0
            actor_b_dict[k] = w_new
        elif k == "edge_encoder.0.weight" and v.shape[1] == 4:
            w_new = torch.zeros((32, 5), dtype=v.dtype)
            w_new[:, :4] = v
            w_new[:, 4] = 0.0
            actor_b_dict[k] = w_new
        elif k in actor_b_dict:
            actor_b_dict[k] = v
    actor_B_best.load_state_dict(actor_b_dict, strict=True)
    actor_B_best.eval()
    
    # -------------------------------------------------------------
    # 1. 50-Scenario Benchmark Suite (Seen 35 + Held-Out 15)
    # -------------------------------------------------------------
    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank_50 = pickle.load(f)
    seen_35 = bank_50["seen_35"]
    held_out_15 = bank_50["held_out_15"]
    all_50 = bank_50["all_50"]
    
    df_seen_bp = evaluate_closed_loop_scenarios(actor_B_prime, seen_35, device=device)
    df_held_bp = evaluate_closed_loop_scenarios(actor_B_prime, held_out_15, device=device)
    df_all_bp = pd.concat([df_seen_bp, df_held_bp], ignore_index=True)
    
    df_seen_b = evaluate_closed_loop_scenarios(actor_B_best, seen_35, device=device)
    df_held_b = evaluate_closed_loop_scenarios(actor_B_best, held_out_15, device=device)
    df_all_b = pd.concat([df_seen_b, df_held_b], ignore_index=True)
    
    print("\n[1. Full Checkpoint B 50-Scenario Gate Regression Check]")
    print(f"Seen 35 Scenarios:     B-Prime = {df_seen_bp['reconnected'].sum()}/35 ({df_seen_bp['reconnected'].mean()*100:.1f}%), Mean Time = {df_seen_bp.loc[df_seen_bp['reconnected'], 'reconnect_rel_time'].mean():.2f}t | Checkpoint B = {df_seen_b['reconnected'].sum()}/35 ({df_seen_b['reconnected'].mean()*100:.1f}%), Mean Time = {df_seen_b.loc[df_seen_b['reconnected'], 'reconnect_rel_time'].mean():.2f}t")
    print(f"Held-Out 15 Scenarios: B-Prime = {df_held_bp['reconnected'].sum()}/15 ({df_held_bp['reconnected'].mean()*100:.1f}%), Mean Time = {df_held_bp.loc[df_held_bp['reconnected'], 'reconnect_rel_time'].mean():.2f}t | Checkpoint B = {df_held_b['reconnected'].sum()}/15 ({df_held_b['reconnected'].mean()*100:.1f}%), Mean Time = {df_held_b.loc[df_held_b['reconnected'], 'reconnect_rel_time'].mean():.2f}t")
    print(f"All 50 Scenarios:      B-Prime = {df_all_bp['reconnected'].sum()}/50 ({df_all_bp['reconnected'].mean()*100:.1f}%), Mean Time = {df_all_bp.loc[df_all_bp['reconnected'], 'reconnect_rel_time'].mean():.2f}t | Checkpoint B = {df_all_b['reconnected'].sum()}/50 ({df_all_b['reconnected'].mean()*100:.1f}%), Mean Time = {df_all_b.loc[df_all_b['reconnected'], 'reconnect_rel_time'].mean():.2f}t")
    
    # -------------------------------------------------------------
    # 2. Category 11: Comm-Range Heterogeneity Evaluation
    # -------------------------------------------------------------
    cat11_scenarios = [
        {
            "category": "11. Comm-Range Heterogeneity",
            "desc": "N=9, d=170m, 1 Degraded Relay (Agent 4 Rc=22.0m vs 28.0m), Fail Agent 6 at t=10",
            "seed": 2111, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (170.0, 0.0),
            "custom_comm_ranges": {4: 22.0},
            "failures": [{"step": 10, "agent_id": 6}]
        },
        {
            "category": "11. Comm-Range Heterogeneity",
            "desc": "N=9, d=165m, 2 Degraded Relays (Agents 3, 6 Rc=22.0m vs 28.0m), Fail Agent 5 at t=10",
            "seed": 2112, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (165.0, 0.0),
            "custom_comm_ranges": {3: 22.0, 6: 22.0},
            "failures": [{"step": 10, "agent_id": 5}]
        },
        {
            "category": "11. Comm-Range Heterogeneity",
            "desc": "N=11, d=180m, 2 Degraded Relays (Agents 4, 8 Rc=22.0m vs 28.0m), Fail Agent 6 at t=10",
            "seed": 2113, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "custom_comm_ranges": {4: 22.0, 8: 22.0},
            "failures": [{"step": 10, "agent_id": 6}]
        },
        {
            "category": "11. Comm-Range Heterogeneity",
            "desc": "N=11, d=185m, 1 Severely Degraded Relay (Agent 5 Rc=20.0m vs 28.0m), Fail Agent 7 at t=10",
            "seed": 2114, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (185.0, 0.0),
            "custom_comm_ranges": {5: 20.0},
            "failures": [{"step": 10, "agent_id": 7}]
        }
    ]
    
    df_c11_bp = evaluate_closed_loop_scenarios(actor_B_prime, cat11_scenarios, device=device)
    df_c11_b = evaluate_closed_loop_scenarios(actor_B_best, cat11_scenarios, device=device)
    
    print("\n====================================================================================================")
    print(" CATEGORY 11 (COMM-RANGE HETEROGENEITY) SIDE-BY-SIDE EVALUATION")
    print(f" Success Rate: B-Prime = {df_c11_bp['reconnected'].sum()}/4 ({df_c11_bp['reconnected'].mean()*100:.1f}%) | Checkpoint B = {df_c11_b['reconnected'].sum()}/4 ({df_c11_b['reconnected'].mean()*100:.1f}%)")
    print("====================================================================================================")
    
    for i in range(len(cat11_scenarios)):
        row_bp = df_c11_bp.iloc[i]
        row_b = df_c11_b.iloc[i]
        
        t_bp = f"{row_bp['reconnect_rel_time']:.1f}t" if pd.notna(row_bp['reconnect_rel_time']) else "—"
        t_b = f"{row_b['reconnect_rel_time']:.1f}t" if pd.notna(row_b['reconnect_rel_time']) else "—"
        
        print(f"Config: {row_bp['desc']}")
        print(f"  * Checkpoint B:       Reconn={row_b['reconnected']} | Time={t_b} | Drops={row_b['transient_drops']} | Rate={row_b['sum_rate']:.2f} Mbps | Mode={row_b['failure_mode']}")
        print(f"  * Checkpoint B-Prime: Reconn={row_bp['reconnected']} | Time={t_bp} | Drops={row_bp['transient_drops']} | Rate={row_bp['sum_rate']:.2f} Mbps | Mode={row_bp['failure_mode']}")
        print("----------------------------------------------------------------------------------------------------")
        
    # -------------------------------------------------------------
    # 3. Four-Step Verification Protocol
    # -------------------------------------------------------------
    print("\n==================================================================")
    print(" 4-STEP VERIFICATION PROTOCOL ON B-PRIME")
    print("==================================================================")
    
    # Step 1: Baseline Audit (Already printed above)
    print("\n[Protocol Step 1: Baseline Audit against Checkpoint B]")
    print(f"50-Scenario Success: B-Prime = {df_all_bp['reconnected'].mean()*100:.1f}% vs Checkpoint B = {df_all_b['reconnected'].mean()*100:.1f}%")
    print(f"50-Scenario Mean Time: B-Prime = {df_all_bp.loc[df_all_bp['reconnected'], 'reconnect_rel_time'].mean():.2f}t vs Checkpoint B = {df_all_b.loc[df_all_b['reconnected'], 'reconnect_rel_time'].mean():.2f}t")
    
    # Step 2: Bit-Identical Determinism Rerun
    print("\n[Protocol Step 2: Determinism Rerun (Bit-Identical Check)]")
    df_all_bp_run2 = evaluate_closed_loop_scenarios(actor_B_prime, all_50, device=device)
    time_diffs = np.abs(df_all_bp["reconnect_rel_time"].fillna(-1).values - df_all_bp_run2["reconnect_rel_time"].fillna(-1).values)
    rate_diffs = np.abs(df_all_bp["sum_rate"].values - df_all_bp_run2["sum_rate"].values)
    print(f"Max Absolute Reconnect Time Delta between Run 1 and Run 2: {np.max(time_diffs):.10e}")
    print(f"Max Absolute Sum-Rate Delta between Run 1 and Run 2:       {np.max(rate_diffs):.10e}")
    assert np.max(time_diffs) == 0.0 and np.max(rate_diffs) == 0.0, "Determinism failed!"
    print("Status: 100% BIT-IDENTICAL DETERMINISM CONFIRMED.")
    
    # Step 3: Per-Seed Distribution & Significance Test
    print("\n[Protocol Step 3: Per-Seed Distribution & Paired Significance Test]")
    time_delta = df_all_bp["reconnect_rel_time"].values - df_all_b["reconnect_rel_time"].values
    valid_mask = ~np.isnan(time_delta)
    if np.any(valid_mask):
        w_stat, p_val = stats.wilcoxon(time_delta[valid_mask]) if np.any(time_delta[valid_mask] != 0) else (0.0, 1.0)
        print(f"Mean Per-Seed Time Delta (B-Prime vs B): {np.mean(time_delta[valid_mask]):.4f} ticks")
        print(f"Wilcoxon Signed-Rank Test: W = {w_stat}, p = {p_val:.6e}")
        
    # Step 4: Zero-Effect Control Run (All Agents Forced to Nominal Rc=28.0m on Cat 11)
    print("\n[Protocol Step 4: Zero-Effect Control Run (All Agents Forced to Nominal Rc=28.0m)]")
    cat11_nominal = []
    for sc in cat11_scenarios:
        sc_nom = sc.copy()
        if "custom_comm_ranges" in sc_nom:
            del sc_nom["custom_comm_ranges"]
        cat11_nominal.append(sc_nom)
        
    df_c11_control_bp = evaluate_closed_loop_scenarios(actor_B_prime, cat11_nominal, device=device)
    df_c11_control_b = evaluate_closed_loop_scenarios(actor_B_best, cat11_nominal, device=device)
    
    print(f"B-Prime Nominal Control: {df_c11_control_bp['reconnected'].sum()}/4 ({df_c11_control_bp['reconnected'].mean()*100:.1f}%), Mean Time = {df_c11_control_bp.loc[df_c11_control_bp['reconnected'], 'reconnect_rel_time'].mean():.2f}t")
    print(f"B Best Nominal Control:  {df_c11_control_b['reconnected'].sum()}/4 ({df_c11_control_b['reconnected'].mean()*100:.1f}%), Mean Time = {df_c11_control_b.loc[df_c11_control_b['reconnected'], 'reconnect_rel_time'].mean():.2f}t")
    
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
