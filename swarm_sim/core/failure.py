from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import networkx as nx
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.utils.enums import AgentRole, AgentStatus

if TYPE_CHECKING:
    from swarm_sim.graph.topology import SwarmTopologyManager

class FailureInjector:
    """Manages node failure injection, ground endpoint protection, and fragmentation checks.
    
    Tick Ordering Rule:
        Failure injection methods (or injector.step(t)) MUST execute BEFORE SwarmSimulator.step()
        in the tick loop. This guarantees that an agent failing at timestep t has its velocity
        zeroed BEFORE kinematic integration occurs, so it takes zero motion on the tick of failure.
    """
    def __init__(self, simulator: SwarmSimulator):
        self.sim = simulator

    def fail_agent(self, agent_id: int, force: bool = False) -> None:
        """Fail a single agent by ID. Zeroes velocity immediately.
        
        Raises ValueError if agent_id is a Ground Endpoint (GROUND_A or GROUND_B) and force=False.
        """
        role = self.sim.roles[agent_id]
        if role in (AgentRole.GROUND_A, AgentRole.GROUND_B) and not force:
            raise ValueError(f"Cannot fail ground endpoint agent_id={agent_id} (role={role.value}) unless force=True.")
            
        self.sim.statuses[agent_id] = AgentStatus.FAILED
        self.sim.velocities[agent_id] = [0.0, 0.0]

    def fail_agents(self, agent_ids: List[int], force: bool = False) -> List[int]:
        """Fail multiple agents by ID, skipping protected ground endpoints if force=False.
        
        Returns list of agent_ids that were actually failed.
        """
        actually_failed = []
        for agent_id in agent_ids:
            role = self.sim.roles[agent_id]
            if role in (AgentRole.GROUND_A, AgentRole.GROUND_B) and not force:
                continue
            self.fail_agent(agent_id, force=force)
            actually_failed.append(agent_id)
        return actually_failed

    def fail_spatial_zone(self, center: Tuple[float, float], radius: float, force: bool = False) -> List[int]:
        """Fail all active agents within Euclidean distance radius of center.
        
        Returns list of newly failed agent IDs.
        """
        active_indices = np.where(self.sim.statuses == AgentStatus.ACTIVE)[0]
        center_arr = np.array(center, dtype=np.float64)
        failed_ids = []
        
        for idx in active_indices:
            dist = np.linalg.norm(self.sim.positions[idx] - center_arr)
            if dist <= radius:
                role = self.sim.roles[idx]
                if role in (AgentRole.GROUND_A, AgentRole.GROUND_B) and not force:
                    continue
                self.fail_agent(idx, force=force)
                failed_ids.append(idx)
                
        return failed_ids

    def step(self, current_timestep: int) -> List[int]:
        """Process configured scheduled failures for current_timestep.
        
        MUST be called BEFORE SwarmSimulator.step() in simulation loop.
        Returns list of newly failed agent IDs.
        """
        config = self.sim.config
        if config.failure_timestep is None or current_timestep != config.failure_timestep:
            return []
            
        if config.failure_count <= 0:
            return []
            
        # Select active relay agents for failure
        active_relay_indices = [
            i for i in range(self.sim.num_agents)
            if self.sim.statuses[i] == AgentStatus.ACTIVE and self.sim.roles[i] == AgentRole.RELAY
        ]
        
        count = min(config.failure_count, len(active_relay_indices))
        if count <= 0:
            return []
            
        chosen_ids = list(np.random.choice(active_relay_indices, size=count, replace=False))
        return self.fail_agents(chosen_ids, force=False)

    def check_fragmentation(
        self,
        topology_mgr: SwarmTopologyManager,
        ground_a_id: int,
        ground_b_id: int,
        graph: Optional[nx.Graph] = None
    ) -> Dict[str, Any]:
        """Evaluate network fragmentation status in a single pass.
        
        Accepts optional pre-built NetworkX graph to preserve NFR-1 performance.
        """
        G = graph if graph is not None else topology_mgr.build_graph()
        path_exists = topology_mgr.has_path(ground_a_id, ground_b_id, graph=G)
        num_components = topology_mgr.get_num_connected_components(graph=G)
        alg_conn = topology_mgr.get_algebraic_connectivity(graph=G)
        
        is_fragmented = (not path_exists) or (num_components > 1)
        
        return {
            "is_fragmented": is_fragmented,
            "path_exists": path_exists,
            "num_components": num_components,
            "algebraic_connectivity": alg_conn
        }
