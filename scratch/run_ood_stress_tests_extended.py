import os
import sys
import torch
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

def find_adversarial_cut_vertex(sim: SwarmSimulator, topo: SwarmTopologyManager, g_a: int, g_b: int) -> Tuple[int, float]:
    """Find the relay whose removal creates the largest Euclidean gap between A's and B's components."""
    G = topo.build_graph()
    active_relays = [i for i in range(2, sim.num_agents) if sim.statuses[i] == AgentStatus.ACTIVE]
    
    worst_relay = None
    max_gap = -1.0
    
    for r in active_relays:
        G_temp = G.copy()
        G_temp.remove_node(r)
        
        # Check if removal disconnects A and B
        if not nx.has_path(G_temp, g_a, g_b):
            # Compute minimum Euclidean distance between A's component and B's component
            comp_a = nx.node_connected_component(G_temp, g_a)
            comp_b = nx.node_connected_component(G_temp, g_b)
            
            min_dist = float('inf')
            for u in comp_a:
                for v in comp_b:
                    d = np.linalg.norm(sim.positions[u] - sim.positions[v])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist > max_gap:
                max_gap = min_dist
                worst_relay = r
        else:
            # Graph still connected: measure distance between previous neighbor and next neighbor along corridor
            neighbors = list(G.neighbors(r))
            if len(neighbors) >= 2:
                d = max(np.linalg.norm(sim.positions[u] - sim.positions[v]) for u in neighbors for v in neighbors if u != v)
                if d > max_gap:
                    max_gap = d
                    worst_relay = r
                    
    if worst_relay is None and len(active_relays) > 0:
        worst_relay = active_relays[len(active_relays) // 2]
        max_gap = 0.0
        
    return worst_relay, max_gap

def run_extended_ood_scenario(actor, sc: Dict[str, Any], device="cpu", max_steps=150) -> Dict[str, Any]:
    config = ExperimentConfig(
        num_drones=sc["num_drones"],
        comm_range=sc["comm_range"],
        seed=sc["seed"]
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    
    g_a, g_b = scenario.setup_scenario(
        endpoint_a_pos=sc.get("endpoint_a_pos", (0.0, 0.0)),
        endpoint_b_pos=sc.get("endpoint_b_pos", (200.0, 0.0)),
        topology="line_relay"
    )
    
    # Optional custom position override (for irregular spacing or custom layouts)
    if "custom_positions_fn" in sc:
        custom_pos = sc["custom_positions_fn"](sim.positions.copy(), sc["comm_range"])
        sim.initialize_positions(custom_pos)
        assert sim.topology_mgr.is_fully_connected() if hasattr(sim, "topology_mgr") else True
        
    # Optional per-agent comm_range overrides
    if "custom_comm_ranges" in sc:
        for agent_id, range_val in sc["custom_comm_ranges"].items():
            sim.comm_ranges[agent_id] = range_val
            
    topo = SwarmTopologyManager(sim)
    injector = FailureInjector(sim)
    policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
    channel = ChannelModel()
    
    # Handle adversarial cut-vertex identification
    failures = list(sc.get("failures", []))
    config_desc = sc.get("desc", f"N={sc['num_drones']}, Seed={sc['seed']}")
    
    if sc.get("adversarial_cut_vertex", False):
        target_step = sc.get("fail_timestep", 10)
        worst_r, gap_size = find_adversarial_cut_vertex(sim, topo, g_a, g_b)
        failures.append({"step": target_step, "agent_id": worst_r})
        config_desc += f" -> Adversarially selected Relay {worst_r} (Immediate Gap: {gap_size:.1f}m)"
        
    failed_steps = set()
    last_fail_step = max([f["step"] for f in failures]) if failures else 0
    
    ever_reconnected_after_all_failures = False
    reconnected_relative_step = None
    reconnected_absolute_step = None
    disconnections_after_reconnect = 0
    previously_reconnected = False
    
    for t in range(max_steps):
        # Inject failures at configured timesteps
        for f in failures:
            if t == f["step"]:
                injector.fail_agent(f["agent_id"])
                failed_steps.add(t)
                
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        is_conn = topo.has_path(g_a, g_b, graph=G)
        
        # Track reconnection once all failures have been injected
        if t >= last_fail_step:
            if is_conn:
                if not ever_reconnected_after_all_failures:
                    ever_reconnected_after_all_failures = True
                    reconnected_absolute_step = t
                    reconnected_relative_step = t - last_fail_step
                previously_reconnected = True
            else:
                if previously_reconnected:
                    disconnections_after_reconnect += 1
                    
    G_final = topo.build_graph()
    final_conn = topo.has_path(g_a, g_b, graph=G_final)
    sum_rate = channel.compute_network_throughput(sim, G_final)
    
    # Classify failure mode
    failure_mode = "Success (Stable)"
    if not final_conn:
        if not ever_reconnected_after_all_failures:
            failure_mode = "Failed to Reconnect (Insufficient Span/Reach)"
        elif disconnections_after_reconnect > 0:
            failure_mode = f"Reconnect-then-Drift ({disconnections_after_reconnect} post-reconnect disconnects)"
        else:
            failure_mode = "Disconnected at Step 150"
    elif disconnections_after_reconnect > 0:
        failure_mode = f"Success with Oscillation ({disconnections_after_reconnect} transient drops)"
        
    return {
        "category": sc.get("category", "Unspecified"),
        "config_desc": config_desc,
        "seed": sc["seed"],
        "num_drones": sc["num_drones"],
        "comm_range": sc["comm_range"],
        "reconnected": final_conn,
        "first_reconnect_rel_time": reconnected_relative_step,
        "first_reconnect_abs_time": reconnected_absolute_step,
        "final_sum_rate_mbps": sum_rate,
        "failure_mode": failure_mode
    }

def make_irregular_jitter_fn(seed: int, max_transverse: float = 6.0, max_longitudinal: float = 4.0):
    def apply_jitter(pos: np.ndarray, rc: float) -> np.ndarray:
        rng = np.random.RandomState(seed)
        N = pos.shape[0]
        pos_A = pos[0]
        pos_B = pos[1]
        u = (pos_B - pos_A) / np.linalg.norm(pos_B - pos_A)
        u_perp = np.array([-u[1], u[0]])
        
        # Jitter relay positions
        for r_id in range(2, N):
            long_offset = rng.uniform(-max_longitudinal, max_longitudinal)
            trans_offset = rng.uniform(-max_transverse, max_transverse)
            pos[r_id] += long_offset * u + trans_offset * u_perp
        return pos
    return apply_jitter

def make_zigzag_fn(amplitude: float = 7.0):
    def apply_zigzag(pos: np.ndarray, rc: float) -> np.ndarray:
        N = pos.shape[0]
        pos_A = pos[0]
        pos_B = pos[1]
        u = (pos_B - pos_A) / np.linalg.norm(pos_B - pos_A)
        u_perp = np.array([-u[1], u[0]])
        for r_id in range(2, N):
            sign = 1.0 if (r_id % 2 == 0) else -1.0
            pos[r_id] += sign * amplitude * u_perp
        return pos
    return apply_zigzag

def make_clustered_density_fn(d_AB: float = 200.0):
    def apply_clustering(pos: np.ndarray, rc: float) -> np.ndarray:
        # Bunched near endpoints, wider gap in middle
        N = pos.shape[0]
        num_relays = N - 2
        alphas = [0.08, 0.18, 0.28, 0.38, 0.62, 0.72, 0.82, 0.92] if num_relays == 8 else [0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92, 0.96][:num_relays]
        pos_A = pos[0]
        pos_B = pos[1]
        vec = pos_B - pos_A
        for i, r_id in enumerate(range(2, N)):
            pos[r_id] = pos_A + alphas[i] * vec
        return pos
    return apply_clustering

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    actor = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_path = "checkpoints/checkpoint_B_best.pt"
    actor.load_state_dict(torch.load(actor_path, map_location=device), strict=False)
    actor.eval()
    print(f"Loaded Checkpoint B Best from {actor_path}")
    
    ood_scenarios = []
    
    # -------------------------------------------------------------
    # 7. Triple Simultaneous Failure (N=11, Rc=28m, d_AB=180m, Feasible Span 7 hops x 28m = 196m)
    # -------------------------------------------------------------
    ood_scenarios.extend([
        {
            "category": "7. Triple Simultaneous Failure",
            "desc": "N=11, d=180m, Fail agents [5, 6, 7] at t=10 (Adjacent Center Cluster)",
            "seed": 2071, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}, {"step": 10, "agent_id": 6}, {"step": 10, "agent_id": 7}]
        },
        {
            "category": "7. Triple Simultaneous Failure",
            "desc": "N=11, d=180m, Fail agents [3, 6, 9] at t=10 (Uniformly Distributed)",
            "seed": 2072, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "failures": [{"step": 10, "agent_id": 3}, {"step": 10, "agent_id": 6}, {"step": 10, "agent_id": 9}]
        },
        {
            "category": "7. Triple Simultaneous Failure",
            "desc": "N=11, d=180m, Fail agents [2, 3, 4] at t=10 (Endpoint-Near Cluster at Ground A)",
            "seed": 2073, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "failures": [{"step": 10, "agent_id": 2}, {"step": 10, "agent_id": 3}, {"step": 10, "agent_id": 4}]
        },
        {
            "category": "7. Triple Simultaneous Failure",
            "desc": "N=11, d=180m, Fail agents [4, 5, 8] at t=10 (Asymmetric Dual Cluster)",
            "seed": 2074, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "failures": [{"step": 10, "agent_id": 4}, {"step": 10, "agent_id": 5}, {"step": 10, "agent_id": 8}]
        }
    ])
    
    # -------------------------------------------------------------
    # 8. Adversarial Cut-Vertex Failure (Deliberately Target Worst-Case Articulation Point)
    # -------------------------------------------------------------
    ood_scenarios.extend([
        {
            "category": "8. Adversarial Cut-Vertex Failure",
            "desc": "N=9, d=200m, Algorithmic Cut-Vertex Identification",
            "seed": 2081, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "adversarial_cut_vertex": True,
            "fail_timestep": 10
        },
        {
            "category": "8. Adversarial Cut-Vertex Failure",
            "desc": "N=11, d=200m, Algorithmic Cut-Vertex Identification",
            "seed": 2082, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "adversarial_cut_vertex": True,
            "fail_timestep": 10
        },
        {
            "category": "8. Adversarial Cut-Vertex Failure",
            "desc": "N=9, d=180m with Non-Uniform Spacing, Algorithmic Cut-Vertex",
            "seed": 2083, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "custom_positions_fn": make_clustered_density_fn(180.0),
            "adversarial_cut_vertex": True,
            "fail_timestep": 10
        },
        {
            "category": "8. Adversarial Cut-Vertex Failure",
            "desc": "N=11, d=210m (Tight Feasible Chain), Algorithmic Cut-Vertex",
            "seed": 2084, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (210.0, 0.0),
            "adversarial_cut_vertex": True,
            "fail_timestep": 10
        }
    ])
    
    # -------------------------------------------------------------
    # 9. Compound Stressors (Combining Multi-Axis Challenges)
    # -------------------------------------------------------------
    ood_scenarios.extend([
        {
            "category": "9. Compound Stressors",
            "desc": "Non-Collinear (45° Diagonal) + Simultaneous Double Failure [4, 7] at t=10 (N=11, d=200m)",
            "seed": 2091, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (141.42, 141.42),
            "failures": [{"step": 10, "agent_id": 4}, {"step": 10, "agent_id": 7}]
        },
        {
            "category": "9. Compound Stressors",
            "desc": "Tight-Margin Geometry (d=188m, Margin=1.0m) + Late-Timing Failure (t=80, N=9)",
            "seed": 2092, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (188.0, 0.0),
            "failures": [{"step": 80, "agent_id": 5}]
        },
        {
            "category": "9. Compound Stressors",
            "desc": "Non-Collinear (60° Diagonal) + Cascading Failure (Agent 5 at t=10, Agent 7 at t=30, N=11, d=200m)",
            "seed": 2093, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (100.0, 173.21),
            "failures": [{"step": 10, "agent_id": 5}, {"step": 30, "agent_id": 7}]
        },
        {
            "category": "9. Compound Stressors",
            "desc": "Irregular Initial Jitter + Simultaneous Double Failure [4, 8] at t=10 (N=11, d=190m)",
            "seed": 2094, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (190.0, 0.0),
            "custom_positions_fn": make_irregular_jitter_fn(seed=2094, max_transverse=5.0, max_longitudinal=3.0),
            "failures": [{"step": 10, "agent_id": 4}, {"step": 10, "agent_id": 8}]
        }
    ])
    
    # -------------------------------------------------------------
    # 10. Irregular Initial Spacing (Non-Uniform, Perturbed & Clustered Topologies)
    # -------------------------------------------------------------
    ood_scenarios.extend([
        {
            "category": "10. Irregular Initial Spacing",
            "desc": "N=9, d=180m, Random Transverse/Longitudinal Jitter (Seed 2101), Fail Agent 5 at t=10",
            "seed": 2101, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
            "custom_positions_fn": make_irregular_jitter_fn(seed=2101, max_transverse=6.0, max_longitudinal=4.0),
            "failures": [{"step": 10, "agent_id": 5}]
        },
        {
            "category": "10. Irregular Initial Spacing",
            "desc": "N=9, d=185m, Alternating Zigzag Spacing (±7m offset), Fail Agent 4 at t=10",
            "seed": 2102, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (185.0, 0.0),
            "custom_positions_fn": make_zigzag_fn(amplitude=7.0),
            "failures": [{"step": 10, "agent_id": 4}]
        },
        {
            "category": "10. Irregular Initial Spacing",
            "desc": "N=11, d=190m, 2D Scatter Jitter along Corridor (Seed 2103), Fail Agent 6 at t=10",
            "seed": 2103, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (190.0, 0.0),
            "custom_positions_fn": make_irregular_jitter_fn(seed=2103, max_transverse=5.0, max_longitudinal=3.0),
            "failures": [{"step": 10, "agent_id": 6}]
        },
        {
            "category": "10. Irregular Initial Spacing",
            "desc": "N=11, d=200m, Non-Uniform Clustered Formation (Wide Center Span), Fail Agent 5 at t=10",
            "seed": 2104, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "custom_positions_fn": make_clustered_density_fn(200.0),
            "failures": [{"step": 10, "agent_id": 5}]
        }
    ])
    
    # -------------------------------------------------------------
    # 11. Per-Agent Comm-Range Heterogeneity (Degraded UAV Hardware / Transmit Power)
    # -------------------------------------------------------------
    ood_scenarios.extend([
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
    ])
    
    print(f"\n==================================================================")
    print(f" EXECUTING EXTENDED OOD STRESS TESTS (CATEGORIES 7-11, 20 SCENARIOS)")
    print(f"==================================================================")
    
    results = []
    for sc in ood_scenarios:
        res = run_extended_ood_scenario(actor, sc, device=device, max_steps=150)
        results.append(res)
        
    df_new = pd.DataFrame(results)
    df_new.to_csv("scratch/ood_stress_test_extended_results.csv", index=False)
    
    # Merge with original results if present
    if os.path.exists("scratch/ood_stress_test_results.csv"):
        df_orig = pd.read_csv("scratch/ood_stress_test_results.csv")
        # Remove any previous entries from 7-11 if they exist
        df_orig = df_orig[~df_orig["category"].isin(df_new["category"].unique())]
        df_combined = pd.concat([df_orig, df_new], ignore_index=True)
        df_combined.to_csv("scratch/ood_stress_test_results.csv", index=False)
        print(f"Updated scratch/ood_stress_test_results.csv with all {len(df_combined)} scenarios.")
    
    categories = sorted(list(set(r["category"] for r in results)))
    for cat in categories:
        cat_df = df_new[df_new["category"] == cat]
        success_count = int(cat_df["reconnected"].sum())
        total_count = len(cat_df)
        success_rate = (success_count / total_count) * 100.0
        
        print(f"\n-------------------------------------------------------------")
        print(f" Category: {cat}")
        print(f" Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
        print(f"-------------------------------------------------------------")
        display_cols = ["config_desc", "reconnected", "first_reconnect_rel_time", "final_sum_rate_mbps", "failure_mode"]
        print(cat_df[display_cols].to_string(index=False))
        
    print("\nExtended OOD Stress Test execution complete. Saved results to scratch/ood_stress_test_extended_results.csv")

if __name__ == "__main__":
    main()
