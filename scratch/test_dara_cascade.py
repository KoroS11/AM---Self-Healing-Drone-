import numpy as np
import networkx as nx
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy

def test_cascade():
    config = ExperimentConfig(num_drones=10, comm_range=28.0, dt=0.1, max_speed=10.0, max_accel=5.0, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    
    g_a, g_b = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    
    for _ in range(10):
        sim.step()
        
    injector.fail_agent(5)
    sim.step()
    
    policy = DARAHeuristicPolicy(sim, r_safe=5.0, k_repair=5.0, k_rep=4.0, max_cascade_depth=5)
    
    for t in range(120):
        G = top_mgr.build_graph()
        frag = injector.check_fragmentation(top_mgr, g_a, g_b, graph=G)
        if not frag["is_fragmented"]:
            print(f"RECONNECTED AT STEP {t}! (Components={frag['num_components']}, lambda2={frag['algebraic_connectivity']:.4f})")
            break
        forces = policy.compute_control_forces(graph=G)
        sim.step(accelerations=forces)
        
    G_final = top_mgr.build_graph()
    frag = injector.check_fragmentation(top_mgr, g_a, g_b, graph=G_final)
    print(f"Final frag={frag['is_fragmented']}, comps={frag['num_components']}, path={frag['path_exists']}")
    for u, v in G_final.edges():
        d = np.linalg.norm(sim.positions[u] - sim.positions[v])
        print(f"  Edge ({u:2d} <-> {v:2d}): distance = {d:.2f} m")

if __name__ == "__main__":
    test_cascade()
