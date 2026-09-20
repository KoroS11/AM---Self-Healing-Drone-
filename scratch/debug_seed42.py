import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.graph.topology import SwarmTopologyManager

def trace_run(strategy_cls, strategy_name):
    config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    scenario.setup_scenario(endpoint_a_pos=(0.0, 0.0), endpoint_b_pos=(200.0, 0.0))
    
    injector = FailureInjector(sim)
    topo = SwarmTopologyManager(sim)
    policy = strategy_cls(sim)
    
    print(f"\n--- Tracing {strategy_name} (Seed 42, N=9, AB=200m) ---")
    print(f"Initial Ground A: {sim.positions[0]}, Ground B: {sim.positions[8]}")
    print("Initial relay positions:")
    for i in range(1, 8):
        print(f"  Relay {i}: {sim.positions[i]}")
        
    failed = False
    for t in range(250):
        if t == 20:
            print(f"\n[Step 20] Injecting failure into agent 3 (position: {sim.positions[3]})")
            injector.fail_agent(3)
            failed = True
            
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        has_path = topo.has_path(0, 8, graph=G)
        
        if failed and (t % 20 == 0 or has_path):
            print(f"  [Step {t:3d}] Path A->B: {has_path:5s} | Active relays: {sum(sim.statuses == 'active') - 2}")
            # Print pairwise distances along active relays
            pos = sim.positions
            active_relays = [r for r in range(1, 8) if sim.statuses[r] == "active"]
            chain = [0] + active_relays + [8]
            dists = [np.linalg.norm(pos[chain[k]] - pos[chain[k+1]]) for k in range(len(chain)-1)]
            max_dist = max(dists)
            print(f"            Chain distances: {[round(d, 2) for d in dists]} (max: {max_dist:.2f}m)")
            
            if has_path:
                print(f"==> Reconnected at step {t}! (time_to_reconnect = {t - 20} ticks)")
                break

trace_run(DARAHeuristicPolicy, "DARA")
trace_run(GreedyDARAHeuristicPolicy, "Greedy DARA")
