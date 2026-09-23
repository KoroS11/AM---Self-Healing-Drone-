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
        
        if not nx.has_path(G_temp, g_a, g_b):
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

def run_single_eval(actor, sc: Dict[str, Any], device="cpu", max_steps=150) -> Dict[str, Any]:
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
    
    # Optional custom position override
    if "custom_positions_fn" in sc:
        custom_pos = sc["custom_positions_fn"](sim.positions.copy(), sc["comm_range"])
        sim.initialize_positions(custom_pos)
        
    # Optional per-agent comm_range overrides
    if "custom_comm_ranges" in sc:
        for agent_id, range_val in sc["custom_comm_ranges"].items():
            sim.comm_ranges[agent_id] = range_val
            
    topo = SwarmTopologyManager(sim)
    injector = FailureInjector(sim)
    policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
    channel = ChannelModel()
    
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
        for f in failures:
            if t == f["step"]:
                injector.fail_agent(f["agent_id"])
                failed_steps.add(t)
                
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        is_conn = topo.has_path(g_a, g_b, graph=G)
        
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

def get_all_scenarios() -> List[Dict[str, Any]]:
    all_sc = []
    
    # -------------------------------------------------------------
    # 1. Simultaneous Double Failure
    # -------------------------------------------------------------
    all_sc.extend([
        {
            "category": "1. Simultaneous Double Failure",
            "desc": "N=9, d=200m, Fail agents [3, 6] at t=10 (Separated)",
            "seed": 2001, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 3}, {"step": 10, "agent_id": 6}]
        },
        {
            "category": "1. Simultaneous Double Failure",
            "desc": "N=9, d=200m, Fail agents [4, 5] at t=10 (Adjacent)",
            "seed": 2002, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 4}, {"step": 10, "agent_id": 5}]
        },
        {
            "category": "1. Simultaneous Double Failure",
            "desc": "N=11, d=200m, Fail agents [3, 7] at t=10 (Separated)",
            "seed": 2003, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 3}, {"step": 10, "agent_id": 7}]
        },
        {
            "category": "1. Simultaneous Double Failure",
            "desc": "N=11, d=200m, Fail agents [5, 6] at t=10 (Adjacent)",
            "seed": 2004, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}, {"step": 10, "agent_id": 6}]
        }
    ])
    
    # -------------------------------------------------------------
    # 2. Cascading Failure
    # -------------------------------------------------------------
    all_sc.extend([
        {
            "category": "2. Cascading Failure",
            "desc": "N=9, d=200m, Fail agent 4 at t=10, Fail agent 6 at t=30",
            "seed": 2011, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 4}, {"step": 30, "agent_id": 6}]
        },
        {
            "category": "2. Cascading Failure",
            "desc": "N=9, d=200m, Fail agent 3 at t=10, Fail agent 5 at t=30",
            "seed": 2012, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 3}, {"step": 30, "agent_id": 5}]
        },
        {
            "category": "2. Cascading Failure",
            "desc": "N=11, d=200m, Fail agent 5 at t=10, Fail agent 7 at t=30",
            "seed": 2013, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}, {"step": 30, "agent_id": 7}]
        },
        {
            "category": "2. Cascading Failure",
            "desc": "N=11, d=200m, Fail agent 4 at t=10, Fail agent 3 at t=30 (Adjacent)",
            "seed": 2014, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 10, "agent_id": 4}, {"step": 30, "agent_id": 3}]
        }
    ])
    
    # -------------------------------------------------------------
    # 3. Scale Test (Large Swarm)
    # -------------------------------------------------------------
    all_sc.extend([
        {
            "category": "3. Scale Test (Large Swarm)",
            "desc": "N=15, d=350m (14 hops, Rc=28m), Fail agent 7 at t=10 (Center)",
            "seed": 2021, "num_drones": 15, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (350.0, 0.0),
            "failures": [{"step": 10, "agent_id": 7}]
        },
        {
            "category": "3. Scale Test (Large Swarm)",
            "desc": "N=15, d=350m (14 hops, Rc=28m), Fail agent 4 at t=10 (Endpoint-near)",
            "seed": 2022, "num_drones": 15, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (350.0, 0.0),
            "failures": [{"step": 10, "agent_id": 4}]
        },
        {
            "category": "3. Scale Test (Large Swarm)",
            "desc": "N=21, d=500m (20 hops, Rc=28m), Fail agent 10 at t=10 (Center)",
            "seed": 2023, "num_drones": 21, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (500.0, 0.0),
            "failures": [{"step": 10, "agent_id": 10}]
        },
        {
            "category": "3. Scale Test (Large Swarm)",
            "desc": "N=21, d=500m (20 hops, Rc=28m), Fail agent 5 at t=10 (Endpoint-near)",
            "seed": 2024, "num_drones": 21, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (500.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}]
        }
    ])
    
    # -------------------------------------------------------------
    # 4. Non-Collinear Corridor
    # -------------------------------------------------------------
    all_sc.extend([
        {
            "category": "4. Non-Collinear Corridor",
            "desc": "N=9, d=200m, 45-deg Diagonal B=(141.4, 141.4), Fail agent 5 at t=10",
            "seed": 2031, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (141.42, 141.42),
            "failures": [{"step": 10, "agent_id": 5}]
        },
        {
            "category": "4. Non-Collinear Corridor",
            "desc": "N=9, d=200m, 53.1-deg Offset B=(120.0, 160.0), Fail agent 4 at t=10",
            "seed": 2032, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (120.0, 160.0),
            "failures": [{"step": 10, "agent_id": 4}]
        },
        {
            "category": "4. Non-Collinear Corridor",
            "desc": "N=11, d=200m, 45-deg Diagonal B=(141.4, 141.4), Fail agent 6 at t=10",
            "seed": 2033, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (141.42, 141.42),
            "failures": [{"step": 10, "agent_id": 6}]
        },
        {
            "category": "4. Non-Collinear Corridor",
            "desc": "N=11, d=200m, 60-deg Offset B=(100.0, 173.2), Fail agent 5 at t=10",
            "seed": 2034, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (100.0, 173.21),
            "failures": [{"step": 10, "agent_id": 5}]
        }
    ])
    
    # -------------------------------------------------------------
    # 5. Failure Timing Extremes
    # -------------------------------------------------------------
    all_sc.extend([
        {
            "category": "5. Failure Timing Extremes",
            "desc": "N=9, d=200m, Extreme Early Failure at t=1, Fail agent 5",
            "seed": 2041, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 1, "agent_id": 5}]
        },
        {
            "category": "5. Failure Timing Extremes",
            "desc": "N=11, d=200m, Extreme Early Failure at t=1, Fail agent 6",
            "seed": 2042, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 1, "agent_id": 6}]
        },
        {
            "category": "5. Failure Timing Extremes",
            "desc": "N=9, d=200m, Extreme Late Failure at t=140 (10 ticks to cutoff), Fail agent 5",
            "seed": 2043, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 140, "agent_id": 5}]
        },
        {
            "category": "5. Failure Timing Extremes",
            "desc": "N=11, d=200m, Extreme Late Failure at t=140 (10 ticks to cutoff), Fail agent 6",
            "seed": 2044, "num_drones": 11, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (200.0, 0.0),
            "failures": [{"step": 140, "agent_id": 6}]
        }
    ])
    
    # -------------------------------------------------------------
    # 6. Tight-Margin Geometry
    # -------------------------------------------------------------
    all_sc.extend([
        {
            "category": "6. Tight-Margin Geometry",
            "desc": "N=9, d=185m (7 hops remaining, 26.43m/hop, margin=1.57m), Fail agent 5 at t=10",
            "seed": 2051, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (185.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}]
        },
        {
            "category": "6. Tight-Margin Geometry",
            "desc": "N=9, d=189m (7 hops remaining, 27.00m/hop, margin=1.00m), Fail agent 4 at t=10",
            "seed": 2052, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (189.0, 0.0),
            "failures": [{"step": 10, "agent_id": 4}]
        },
        {
            "category": "6. Tight-Margin Geometry",
            "desc": "N=9, d=192m (7 hops remaining, 27.43m/hop, margin=0.57m), Fail agent 5 at t=10",
            "seed": 2053, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (192.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}]
        },
        {
            "category": "6. Tight-Margin Geometry",
            "desc": "N=9, d=195m (7 hops remaining, 27.86m/hop, margin=0.14m [Extreme Razor]), Fail agent 5 at t=10",
            "seed": 2054, "num_drones": 9, "comm_range": 28.0,
            "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (195.0, 0.0),
            "failures": [{"step": 10, "agent_id": 5}]
        }
    ])
    
    # -------------------------------------------------------------
    # 7. Triple Simultaneous Failure
    # -------------------------------------------------------------
    all_sc.extend([
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
    # 8. Adversarial Cut-Vertex Failure
    # -------------------------------------------------------------
    all_sc.extend([
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
    # 9. Compound Stressors
    # -------------------------------------------------------------
    all_sc.extend([
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
    # 10. Irregular Initial Spacing
    # -------------------------------------------------------------
    all_sc.extend([
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
    # 11. Per-Agent Comm-Range Heterogeneity
    # -------------------------------------------------------------
    all_sc.extend([
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
    
    return all_sc

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    
    actor_A_path = "checkpoints/checkpoint_A_best_k2.pt"
    actor_B_path = "checkpoints/checkpoint_B_best.pt"
    
    actor_A = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_A.load_state_dict(torch.load(actor_A_path, map_location=device), strict=False)
    actor_A.eval()
    print(f"Loaded Checkpoint A (Pre-RL BC/DAgger) from {actor_A_path}")
    
    actor_B = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_B.load_state_dict(torch.load(actor_B_path, map_location=device), strict=False)
    actor_B.eval()
    print(f"Loaded Checkpoint B (Post-RL Rotation-Invariant Best) from {actor_B_path}")
    
    scenarios = get_all_scenarios()
    print(f"\n==================================================================")
    print(f" EXECUTING COMPLETE 11-CATEGORY STRESS BATTERY (44 SCENARIOS)")
    print(f" APPLES-TO-APPLES COMPARISON: CHECKPOINT A vs. CHECKPOINT B")
    print(f"==================================================================")
    
    records = []
    
    for sc in scenarios:
        res_A = run_single_eval(actor_A, sc, device=device, max_steps=150)
        res_B = run_single_eval(actor_B, sc, device=device, max_steps=150)
        
        records.append({
            "category": sc["category"],
            "config_desc": res_B["config_desc"],
            "seed": sc["seed"],
            "num_drones": sc["num_drones"],
            "comm_range": sc["comm_range"],
            
            # Checkpoint A Metrics
            "A_reconnected": res_A["reconnected"],
            "A_first_reconnect_rel_time": res_A["first_reconnect_rel_time"],
            "A_final_sum_rate_mbps": res_A["final_sum_rate_mbps"],
            "A_failure_mode": res_A["failure_mode"],
            
            # Checkpoint B Metrics
            "B_reconnected": res_B["reconnected"],
            "B_first_reconnect_rel_time": res_B["first_reconnect_rel_time"],
            "B_final_sum_rate_mbps": res_B["final_sum_rate_mbps"],
            "B_failure_mode": res_B["failure_mode"],
        })
        
    df = pd.DataFrame(records)
    df.to_csv("scratch/checkpoint_A_vs_B_stress_battery_results.csv", index=False)
    print(f"Saved side-by-side results to scratch/checkpoint_A_vs_B_stress_battery_results.csv\n")
    
    categories = sorted(list(set(r["category"] for r in records)))
    
    for cat in categories:
        cat_df = df[df["category"] == cat]
        succ_A = int(cat_df["A_reconnected"].sum())
        succ_B = int(cat_df["B_reconnected"].sum())
        tot = len(cat_df)
        
        print(f"====================================================================================================")
        print(f" CATEGORY: {cat}")
        print(f" Success Rate: Checkpoint A = {succ_A}/{tot} ({succ_A/tot*100:.1f}%) | Checkpoint B = {succ_B}/{tot} ({succ_B/tot*100:.1f}%)")
        print(f"====================================================================================================")
        
        for idx, row in cat_df.iterrows():
            t_A_str = f"{row['A_first_reconnect_rel_time']:.1f}t" if pd.notna(row['A_first_reconnect_rel_time']) else "—"
            t_B_str = f"{row['B_first_reconnect_rel_time']:.1f}t" if pd.notna(row['B_first_reconnect_rel_time']) else "—"
            
            print(f"Config: {row['config_desc']}")
            print(f"  * Checkpoint A: Reconn={row['A_reconnected']} | Time={t_A_str} | Rate={row['A_final_sum_rate_mbps']:.2f} Mbps | Mode={row['A_failure_mode']}")
            print(f"  * Checkpoint B: Reconn={row['B_reconnected']} | Time={t_B_str} | Rate={row['B_final_sum_rate_mbps']:.2f} Mbps | Mode={row['B_failure_mode']}")
            print(f"----------------------------------------------------------------------------------------------------")
        print()

if __name__ == "__main__":
    main()
