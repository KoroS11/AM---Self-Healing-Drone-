import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

def run_ood_scenario(actor, sc: Dict[str, Any], device="cpu", max_steps=150) -> Dict[str, Any]:
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
    injector = FailureInjector(sim)
    policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()
    
    # Handle single or multiple scheduled failures
    failures = sc.get("failures", [])
    if not failures and "fail_timestep" in sc:
        failures = [{"step": sc["fail_timestep"], "agent_id": sc["fail_agent_id"]}]
        
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
        "config_desc": sc.get("desc", f"N={sc['num_drones']}, Seed={sc['seed']}"),
        "seed": sc["seed"],
        "num_drones": sc["num_drones"],
        "comm_range": sc["comm_range"],
        "reconnected": final_conn,
        "first_reconnect_rel_time": reconnected_relative_step,
        "first_reconnect_abs_time": reconnected_absolute_step,
        "final_sum_rate_mbps": sum_rate,
        "failure_mode": failure_mode
    }

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    actor = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_path = "checkpoints/checkpoint_B_best.pt"
    actor.load_state_dict(torch.load(actor_path, map_location=device), strict=False)
    actor.eval()
    print(f"Loaded Checkpoint B Best (Iteration 10) from {actor_path}")
    
    # Define OOD Stress Test Scenarios across 6 Categories
    ood_scenarios = []
    
    # -------------------------------------------------------------
    # 1. Simultaneous Double Failure
    # -------------------------------------------------------------
    ood_scenarios.extend([
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
    ood_scenarios.extend([
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
    # 3. Scale Test (N=15 and N=21)
    # -------------------------------------------------------------
    ood_scenarios.extend([
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
    # 4. Non-Collinear Corridor (Diagonal / 2D Offsets)
    # -------------------------------------------------------------
    ood_scenarios.extend([
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
    ood_scenarios.extend([
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
    # 6. Tight-Margin Geometry (Geometric Feasibility Limit)
    # -------------------------------------------------------------
    ood_scenarios.extend([
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
    
    print(f"\n==================================================================")
    print(f" EXECUTING OUT-OF-DISTRIBUTION (OOD) STRESS TESTS (24 SCENARIOS)")
    print(f"==================================================================")
    
    results = []
    for sc in ood_scenarios:
        res = run_ood_scenario(actor, sc, device=device, max_steps=150)
        results.append(res)
        
    df_all = pd.DataFrame(results)
    df_all.to_csv("scratch/ood_stress_test_results.csv", index=False)
    
    categories = sorted(list(set(r["category"] for r in results)))
    for cat in categories:
        cat_df = df_all[df_all["category"] == cat]
        success_count = int(cat_df["reconnected"].sum())
        total_count = len(cat_df)
        success_rate = (success_count / total_count) * 100.0
        
        print(f"\n-------------------------------------------------------------")
        print(f" Category: {cat}")
        print(f" Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
        print(f"-------------------------------------------------------------")
        display_cols = ["config_desc", "reconnected", "first_reconnect_rel_time", "final_sum_rate_mbps", "failure_mode"]
        print(cat_df[display_cols].to_string(index=False))
        
    print("\nOOD Stress Test execution complete. Saved results to scratch/ood_stress_test_results.csv")

if __name__ == "__main__":
    main()
