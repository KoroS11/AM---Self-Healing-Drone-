import pickle
import torch
import numpy as np

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager

def analyze_extended_horizon():
    print("=== PART 1: EXTENDED HORIZON (300 TICKS) FOR SEEDS 1000 & 1023 ===")
    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank = pickle.load(f)
        
    actor = ActorGNN(k_hops=2, max_accel=5.0)
    actor.load_state_dict(torch.load("checkpoints/checkpoint_A_best_k2.pt", map_location="cpu"))
    actor.eval()
    
    scenarios_by_seed = {sc["seed"]: sc for sc in bank["all_50"]}
    
    for seed in [1000, 1023]:
        sc = scenarios_by_seed[seed]
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
        policy = GNNPolicy(sim, actor_gnn=actor, device="cpu")
        topo = SwarmTopologyManager(sim)
        
        conn_history = []
        mean_velocities = []
        max_displacements = []
        
        init_positions = np.copy(sim.positions)
        
        for t in range(300):
            if t == sc["fail_timestep"]:
                injector.fail_agent(sc["fail_agent_id"])
            acc = policy.compute_control_forces()
            sim.step(accelerations=acc)
            G = topo.build_graph()
            has_path = topo.has_path(g_a, g_b, graph=G)
            conn_history.append(has_path)
            
            # Active mobile speeds
            mobile_mask = (sim.statuses == AgentStatus.ACTIVE) & (sim.max_speeds > 0.0)
            speeds = np.linalg.norm(sim.velocities[mobile_mask], axis=1)
            mean_velocities.append(np.mean(speeds) if len(speeds) > 0 else 0.0)
            
        conn_arr = np.array(conn_history)
        
        # Connected segments post failure
        segments = []
        cur_start = None
        for t in range(sc["fail_timestep"], 300):
            c = conn_arr[t]
            if c:
                if cur_start is None:
                    cur_start = t
            else:
                if cur_start is not None:
                    segments.append((cur_start, t - 1))
                    cur_start = None
        if cur_start is not None:
            segments.append((cur_start, 299))
            
        total_conn_300 = int(np.sum(conn_arr[sc["fail_timestep"]:]))
        total_possible = 300 - sc["fail_timestep"]
        
        print(f"\n-------------------------------------------------------")
        print(f"Seed {seed} (N={sc['num_drones']}, fail_id={sc['fail_agent_id']} at t={sc['fail_timestep']}):")
        print(f"  Connected Ticks (post-failure): {total_conn_300} / {total_possible} ({total_conn_300/total_possible*100:.1f}%)")
        print(f"  Connected Segments [t_start, t_end]: {segments}")
        print(f"  Terminal Connectivity at t=299: {conn_arr[299]}")
        print(f"  Mean Active Speed at t=75: {mean_velocities[75]:.4f} m/s | at t=150: {mean_velocities[150]:.4f} m/s | at t=299: {mean_velocities[299]:.4f} m/s")
        
        # Timeline summary in 50-tick blocks
        print(f"  Timeline in 50-tick windows:")
        for b in range(0, 300, 50):
            block_conn = conn_arr[b:b+50]
            print(f"    Ticks {b:3d}..{b+49:3d}: {np.sum(block_conn):2d}/50 connected (Status: {'CONNECTED' if np.all(block_conn) else ('DISCONNECTED' if not np.any(block_conn) else 'PARTIAL/OSCILLATING')})")

if __name__ == "__main__":
    analyze_extended_horizon()
