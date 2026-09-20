from typing import Optional
import numpy as np
import networkx as nx
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus, RecoveryStrategy
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.base import BasePolicy

class MidpointRelaxationPolicy(BasePolicy):
    """Pure Local Midpoint Relaxation Policy — Zero pos_a, pos_b, f_pos, or M."""
    def __init__(self, simulator: SwarmSimulator, k_repair: float = 5.0, k_damp: float = 3.0):
        super().__init__(simulator, strategy=RecoveryStrategy.HEURISTIC_DARA)
        self.k_repair = k_repair
        self.k_damp = k_damp
        self.topology_mgr = SwarmTopologyManager(simulator)

    def compute_control_forces(self, graph: Optional[nx.Graph] = None) -> np.ndarray:
        N = self.sim.num_agents
        accelerations = np.zeros((N, 2), dtype=np.float64)
        
        G = graph if graph is not None else self.topology_mgr.build_graph()
        
        active_indices = [
            i for i in range(N) if self.sim.statuses[i] == AgentStatus.ACTIVE
        ]
        
        # Sort active nodes along the relative 1D axis to identify 1-hop chain neighbors
        pos = self.sim.positions
        projections = [pos[i, 0] for i in active_indices]  # local relative ordering
        sorted_order = np.argsort(projections)
        chain = [active_indices[k] for k in sorted_order]
        
        # Midpoint relaxation: for interior mobile relays, target = 0.5 * (pos[left] + pos[right])
        for idx in range(1, len(chain) - 1):
            curr_node = chain[idx]
            if self.sim.max_speeds[curr_node] > 0.0:  # mobile relay
                left_node = chain[idx - 1]
                right_node = chain[idx + 1]
                
                target_pos = 0.5 * (pos[left_node] + pos[right_node])
                dir_vec = target_pos - pos[curr_node]
                dist = np.linalg.norm(dir_vec)
                if dist > 1e-3:
                    accelerations[curr_node] = self.k_repair * (dir_vec / dist) - self.k_damp * self.sim.velocities[curr_node]
                    
        return accelerations

def test_relaxation():
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
    
    policy = MidpointRelaxationPolicy(sim)
    
    reconnected = False
    for t in range(120):
        G = top_mgr.build_graph()
        frag = injector.check_fragmentation(top_mgr, g_a, g_b, graph=G)
        if not frag["is_fragmented"]:
            print(f"RECONNECTED AT STEP {t}! (Components={frag['num_components']}, lambda2={frag['algebraic_connectivity']:.4f})")
            reconnected = True
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
    test_relaxation()
