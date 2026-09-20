import pickle
import torch
import numpy as np

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager

def inspect_failed_seeds():
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
        
        print(f"\n=======================================================")
        print(f"DIAGNOSTIC TRACE FOR SEED {seed} (drones={sc['num_drones']}, fail_id={sc['fail_agent_id']}, fail_t={sc['fail_timestep']})")
        print(f"=======================================================")
        
        conn_history = []
        for t in range(150):
            if t == sc["fail_timestep"]:
                injector.fail_agent(sc["fail_agent_id"])
            acc = policy.compute_control_forces()
            sim.step(accelerations=acc)
            G = topo.build_graph()
            has_path = topo.has_path(g_a, g_b, graph=G)
            conn_history.append((t, has_path))
            
        # Analyze intervals of connectivity
        connected_ticks = [t for t, c in conn_history if c and t >= sc["fail_timestep"]]
        print(f"Total post-failure ticks connected: {len(connected_ticks)} / {150 - sc['fail_timestep']}")
        if connected_ticks:
            first_t = connected_ticks[0]
            last_t = connected_ticks[-1]
            reconn_time = first_t - sc["fail_timestep"]
            print(f"First post-failure reconnection at step t={first_t} (time_to_reconnect = {reconn_time} ticks)")
            print(f"Last connected step: t={last_t}")
            
            # Find contiguous connected segments
            segments = []
            cur_start = None
            for t, c in conn_history:
                if t < sc["fail_timestep"]:
                    continue
                if c:
                    if cur_start is None:
                        cur_start = t
                else:
                    if cur_start is not None:
                        segments.append((cur_start, t - 1))
                        cur_start = None
            if cur_start is not None:
                segments.append((cur_start, 149))
                
            print(f"Connected segments [t_start, t_end]: {segments}")
            print(f"Final tick (t=149) connected: {conn_history[-1][1]}")
            
            # Print state around disconnect
            for s_start, s_end in segments:
                if s_end < 149:
                    print(f"  --> Disconnected at t={s_end + 1} (stayed connected for {s_end - s_start + 1} ticks from t={s_start} to t={s_end})")
        else:
            print("NEVER connected post-failure.")

if __name__ == "__main__":
    inspect_failed_seeds()
