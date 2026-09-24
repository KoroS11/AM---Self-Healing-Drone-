from typing import Tuple
import numpy as np
from scipy.spatial.distance import pdist
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.utils.enums import AgentRole

def project_point_onto_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return nearest point on line segment AB to point P."""
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-9:
        return a.copy()
    t = np.dot(p - a, ab) / ab_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    return a + t_clamped * ab

class DisasterRelayScenario:
    """Disaster-zone relay chain scenario generator."""
    def __init__(self, simulator: SwarmSimulator):
        self.sim = simulator
        self.topology_mgr = SwarmTopologyManager(simulator)

    def setup_scenario(
        self,
        endpoint_a_pos: Tuple[float, float],
        endpoint_b_pos: Tuple[float, float],
        topology: str = "line_relay",
        max_retries: int = 100
    ) -> Tuple[int, int]:
        """Setup ground endpoints and relay UAV topology with guaranteed initial connectivity.
        
        Returns (ground_a_id, ground_b_id).
        """
        N = self.sim.num_agents
        assert N >= 2, "Scenario requires at least 2 agents (Ground A and Ground B)"
        
        ground_a_id = 0
        ground_b_id = 1
        relay_ids = list(range(2, N))
        num_relays = len(relay_ids)
        
        positions = np.zeros((N, 2), dtype=np.float64)
        positions[ground_a_id] = endpoint_a_pos
        positions[ground_b_id] = endpoint_b_pos
        
        roles = [AgentRole.RELAY] * N
        roles[ground_a_id] = AgentRole.GROUND_A
        roles[ground_b_id] = AgentRole.GROUND_B
        
        # Enforce stationary ground endpoints by setting max_speed = 0.0 and zero velocity
        self.sim.max_speeds[ground_a_id] = 0.0
        self.sim.max_speeds[ground_b_id] = 0.0
        self.sim.velocities[ground_a_id] = [0.0, 0.0]
        self.sim.velocities[ground_b_id] = [0.0, 0.0]
        
        rc = self.sim.config.comm_range
        pos_A = np.array(endpoint_a_pos)
        pos_B = np.array(endpoint_b_pos)
        ab_dist = np.linalg.norm(pos_B - pos_A)
        
        # Compute unit vector along AB direction and perpendicular vector
        if ab_dist > 1e-6:
            u = (pos_B - pos_A) / ab_dist
        else:
            u = np.array([1.0, 0.0])
        u_perp = np.array([-u[1], u[0]])
        
        if topology == "line_relay":
            if num_relays > 0:
                for i, r_id in enumerate(relay_ids):
                    alpha = (i + 1) / (num_relays + 1)
                    positions[r_id] = pos_A + alpha * (pos_B - pos_A)
            self.sim.initialize_positions(positions, roles)
            
        elif topology == "grid_lattice":
            spacing = 0.8 * rc
            midpoint = (pos_A + pos_B) / 2.0
            
            if ab_dist <= rc:
                # Direct range bypass with extra relays in centered square grid
                grid_side = int(np.ceil(np.sqrt(num_relays))) if num_relays > 0 else 1
                for i, r_id in enumerate(relay_ids):
                    col = i % grid_side
                    row = i // grid_side
                    alpha = (col - (grid_side - 1) / 2.0) * spacing
                    beta = (row - (grid_side - 1) / 2.0) * spacing
                    positions[r_id] = midpoint + alpha * u + beta * u_perp
                self.sim.initialize_positions(positions, roles)
            else:
                # Bridging corridor required
                num_cols = int(np.ceil(ab_dist / spacing))
                min_required_relays = max(1, num_cols)
                
                if num_relays < min_required_relays:
                    raise ValueError(
                        f"grid_lattice needs at least {min_required_relays} relay agents to form a bridging "
                        f"corridor over a distance of {ab_dist:.1f} with comm_range {rc:.1f}; got {num_relays}"
                    )
                    
                num_rows = max(1, int(np.ceil(num_relays / num_cols)))
                
                for i, r_id in enumerate(relay_ids):
                    col = i % num_cols
                    row = i // num_cols
                    
                    alpha = (col + 1) / (num_cols + 1) * ab_dist
                    beta = (row - (num_rows - 1) / 2.0) * spacing
                    positions[r_id] = pos_A + alpha * u + beta * u_perp
                self.sim.initialize_positions(positions, roles)
            
        elif topology == "random_scatter":
            box_min = np.minimum(pos_A, pos_B) - rc * 0.5
            box_max = np.maximum(pos_A, pos_B) + rc * 0.5
            
            success = False
            for attempt in range(max_retries):
                for r_id in relay_ids:
                    positions[r_id] = np.random.uniform(box_min, box_max)
                self.sim.initialize_positions(positions, roles)
                
                if self.topology_mgr.is_fully_connected() and self.topology_mgr.has_path(ground_a_id, ground_b_id):
                    success = True
                    break
                    
            if not success:
                # Segment-projection fallback: project and space relay nodes along line segment A-B
                for i, r_id in enumerate(relay_ids):
                    alpha = (i + 1) / (num_relays + 1)
                    # Add slight random jitter (up to 0.2*rc) off-axis to maintain scatter aesthetic while guaranteeing connectivity
                    jitter = (np.random.rand(2) - 0.5) * (rc * 0.3)
                    positions[r_id] = pos_A + alpha * (pos_B - pos_A) + jitter
                self.sim.initialize_positions(positions, roles)
                
            if not self.topology_mgr.is_fully_connected():
                # If jitter broke connectivity, fallback to strict line placement along AB
                for i, r_id in enumerate(relay_ids):
                    alpha = (i + 1) / (num_relays + 1)
                    positions[r_id] = pos_A + alpha * (pos_B - pos_A)
                self.sim.initialize_positions(positions, roles)
        else:
            raise ValueError(f"Unknown topology '{topology}'")
            
        # Post-setup assertions matching requirements
        assert self.topology_mgr.has_path(ground_a_id, ground_b_id), "Initial scenario failed end-to-end path assertion A-to-B!"
        assert self.topology_mgr.is_fully_connected(), "Initial scenario graph is not fully connected!"
        
        # Fast scipy.spatial.distance.pdist spatial spread check
        if ab_dist > 1e-6 and N > 1:
            max_pair_dist = float(pdist(self.sim.positions).max())
            assert max_pair_dist >= 0.8 * ab_dist, f"Scenario failed minimum spatial spread threshold ({max_pair_dist:.2f} < {0.8 * ab_dist:.2f})!"
            
        return ground_a_id, ground_b_id
