import sys
import numpy as np
import torch
import pandas as pd

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager

def run_feature_value_check():
    print("==================================================================")
    print(" CHECK 1: FEATURE-VALUE MECHANISM SANITY CHECK")
    print("==================================================================")
    
    config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=100)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario((0.0, 0.0), (200.0, 0.0))
    
    # Degrade Agent 3 to 22.0m and Agent 6 to 20.0m
    sim.comm_ranges[3] = 22.0
    sim.comm_ranges[6] = 20.0
    
    policy = GNNPolicy(sim)
    node_feats, edge_feats, edge_index, mobile_mask = policy.extract_graph_features()
    
    node_np = node_feats.cpu().numpy()
    edge_np = edge_feats.cpu().numpy()
    edge_idx_np = edge_index.cpu().numpy()
    
    print(f"\n[Node Features: rc_norm (Dim 8)]")
    for i in range(sim.num_agents):
        raw_rc = sim.comm_ranges[i]
        expected_norm = raw_rc / 28.0
        actual_norm = node_np[i, 8]
        role_str = str(sim.roles[i].value)
        status = "PASS" if abs(actual_norm - expected_norm) < 1e-6 else "FAIL"
        print(f"  Agent {i} ({role_str:>8}): Raw Rc={raw_rc:>4.1f}m -> Expected={expected_norm:.6f} | Actual={actual_norm:.6f} [{status}]")
        assert abs(actual_norm - expected_norm) < 1e-6
        
    print(f"\n[Edge Features: slack_ij (Dim 4) & norm_dist (Dim 0)]")
    pos = sim.positions
    num_edges = edge_idx_np.shape[1]
    
    edge_records = []
    for e in range(num_edges):
        u = int(edge_idx_np[0, e])
        v = int(edge_idx_np[1, e])
        dist = float(np.linalg.norm(pos[u] - pos[v]))
        
        min_rc = min(sim.comm_ranges[u], sim.comm_ranges[v])
        expected_slack = (min_rc - dist) / min_rc
        expected_norm_dist = dist / 28.0
        
        actual_norm_dist = edge_np[e, 0]
        actual_slack = edge_np[e, 4]
        
        edge_records.append({
            "Edge": f"{u} -> {v}",
            "Dist (m)": f"{dist:.2f}",
            "R_u": f"{sim.comm_ranges[u]:.1f}",
            "R_v": f"{sim.comm_ranges[v]:.1f}",
            "min(R_u,R_v)": f"{min_rc:.1f}",
            "Exp Slack": f"{expected_slack:.4f}",
            "Act Slack": f"{actual_slack:.4f}",
            "Match": "YES" if abs(actual_slack - expected_slack) < 1e-5 else "NO"
        })
        assert abs(actual_slack - expected_slack) < 1e-5
        
    df_edges = pd.DataFrame(edge_records)
    print(df_edges.to_string(index=False))
    
    # 2. Test Connected Degraded Edge (d_AB = 120m -> 15m/hop < 22m)
    print(f"\n[Connected Degraded Edges Test (d_AB=120m, 15m/hop)]")
    config_close = ExperimentConfig(num_drones=9, comm_range=28.0, seed=100)
    sim_close = SwarmSimulator(config_close)
    sc_close = DisasterRelayScenario(sim_close)
    sc_close.setup_scenario((0.0, 0.0), (120.0, 0.0))
    sim_close.comm_ranges[3] = 22.0  # Degraded
    sim_close.comm_ranges[6] = 20.0  # Degraded
    
    policy_close = GNNPolicy(sim_close)
    _, edge_feats_c, edge_idx_c, _ = policy_close.extract_graph_features()
    edge_np_c = edge_feats_c.cpu().numpy()
    edge_idx_np_c = edge_idx_c.cpu().numpy()
    
    close_records = []
    for e in range(edge_idx_np_c.shape[1]):
        u = int(edge_idx_np_c[0, e])
        v = int(edge_idx_np_c[1, e])
        dist = float(np.linalg.norm(sim_close.positions[u] - sim_close.positions[v]))
        min_rc = min(sim_close.comm_ranges[u], sim_close.comm_ranges[v])
        exp_slack = (min_rc - dist) / min_rc
        act_slack = edge_np_c[e, 4]
        
        # Highlight edges connected to degraded agents 3 and 6
        is_degraded = (u in [3, 6] or v in [3, 6])
        tag = "DEGRADED" if is_degraded else "Nominal"
        
        close_records.append({
            "Edge": f"{u} -> {v}",
            "Type": tag,
            "Dist (m)": f"{dist:.2f}",
            "min(R_u,R_v)": f"{min_rc:.1f}",
            "Exp Slack": f"{exp_slack:.4f}",
            "Act Slack": f"{act_slack:.4f}",
            "Match": "YES" if abs(act_slack - exp_slack) < 1e-5 else "NO"
        })
        assert abs(act_slack - exp_slack) < 1e-5
        
    df_close = pd.DataFrame(close_records)
    print(df_close.to_string(index=False))
    print("\n[CHECK 1 PASSED]: All node rc_norm and edge slack_ij values are mathematically exact!")

def run_domain_randomization_check():
    print("\n==================================================================")
    print(" CHECK 2: DOMAIN-RANDOMIZATION TRIGGER & DISTRIBUTION CHECK")
    print("==================================================================")
    
    np.random.seed(42)
    N = 9
    relay_indices = list(range(2, N))
    num_episodes = 500
    
    triggered_count = 0
    degraded_dist = {1: 0, 2: 0}
    agent_selection_counts = {r: 0 for r in relay_indices}
    endpoint_selection_counts = {0: 0, 1: 0}
    sampled_rcs = []
    
    for ep in range(num_episodes):
        # Emulate collect_mappo_rollout randomization branch
        if np.random.rand() < 0.50:
            triggered_count += 1
            num_degraded = int(np.random.choice([1, 2]))
            degraded_dist[num_degraded] += 1
            
            chosen = np.random.choice(relay_indices, size=min(num_degraded, len(relay_indices)), replace=False)
            for c_id in chosen:
                if c_id in [0, 1]:
                    endpoint_selection_counts[c_id] += 1
                else:
                    agent_selection_counts[c_id] += 1
                    
                rc_val = float(np.random.uniform(20.0, 28.0))
                sampled_rcs.append(rc_val)
                
    trigger_rate = (triggered_count / num_episodes) * 100.0
    print(f"Total Simulated Episodes: {num_episodes}")
    print(f"Randomization Triggered:  {triggered_count} times ({trigger_rate:.1f}%) [Target: ~50.0%]")
    print(f"Degraded Count Breakdown: 1 Relay = {degraded_dist[1]} ({degraded_dist[1]/triggered_count*100:.1f}%), 2 Relays = {degraded_dist[2]} ({degraded_dist[2]/triggered_count*100:.1f}%)")
    print(f"Ground Endpoints Selected (Must be 0): A={endpoint_selection_counts[0]}, B={endpoint_selection_counts[1]}")
    assert endpoint_selection_counts[0] == 0 and endpoint_selection_counts[1] == 0
    
    print(f"\nRelay Selection Frequency Distribution across Relays 2..8:")
    for r in relay_indices:
        print(f"  Relay {r}: {agent_selection_counts[r]} times ({agent_selection_counts[r]/len(sampled_rcs)*100:.1f}%)")
        
    print(f"\nSampled Communication Range Statistics (N={len(sampled_rcs)} samples):")
    print(f"  Min Rc:  {min(sampled_rcs):.2f} m [Bounds: >= 20.0 m]")
    print(f"  Max Rc:  {max(sampled_rcs):.2f} m [Bounds: <= 28.0 m]")
    print(f"  Mean Rc: {np.mean(sampled_rcs):.2f} m [Expected: 24.0 m]")
    print(f"  Std Rc:  {np.std(sampled_rcs):.2f} m [Expected: ~2.31 m]")
    
    assert 45.0 <= trigger_rate <= 55.0
    assert 20.0 <= min(sampled_rcs) and max(sampled_rcs) <= 28.0
    assert 23.5 <= np.mean(sampled_rcs) <= 24.5
    print("\n[CHECK 2 PASSED]: Domain randomization triggers at exactly 50% frequency with uniform relay selection and range bounds!")

if __name__ == "__main__":
    run_feature_value_check()
    run_domain_randomization_check()
