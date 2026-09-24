from typing import Optional, List, Tuple, Any
import numpy as np
from scipy.spatial import cKDTree
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.agent import UAVAgent

class SwarmSimulator:
    """Vectorized UAV Swarm Simulator.
    
    Owns canonical agent states as 2D/1D NumPy matrices for high performance (>20 Hz at N>=100).
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.num_agents = config.num_drones
        self.dt = config.dt
        
        # Canonical vectorized state matrices
        self.positions = np.zeros((self.num_agents, 2), dtype=np.float64)
        self.velocities = np.zeros((self.num_agents, 2), dtype=np.float64)
        self.roles = np.full(self.num_agents, AgentRole.RELAY, dtype=object)
        self.statuses = np.full(self.num_agents, AgentStatus.ACTIVE, dtype=object)
        self.comm_ranges = np.full(self.num_agents, config.comm_range, dtype=np.float64)
        self.max_speeds = np.full(self.num_agents, config.max_speed, dtype=np.float64)
        self.max_accels = np.full(self.num_agents, config.max_accel, dtype=np.float64)

        if config.seed is not None:
            np.random.seed(config.seed)

    def initialize_positions(self, positions: np.ndarray, roles: Optional[List[AgentRole]] = None) -> None:
        """Initialize position matrix and optional agent roles."""
        assert positions.shape == (self.num_agents, 2), f"Expected shape ({self.num_agents}, 2), got {positions.shape}"
        self.positions[:] = positions
        if roles is not None:
            assert len(roles) == self.num_agents, f"Expected {self.num_agents} roles, got {len(roles)}"
            self.roles[:] = roles

    def step(
        self,
        accelerations: Optional[np.ndarray] = None,
        policy: Optional[Any] = None
    ) -> None:
        """Execute one simulation step using pure NumPy vectorization.
        
        Precedence Rule:
            - Raises ValueError if BOTH accelerations and policy are supplied.
            - If policy is supplied, accelerations = policy.compute_control_forces().
        """
        if accelerations is not None and policy is not None:
            raise ValueError("Cannot supply both accelerations and policy to step(). Choose one.")
            
        if policy is not None:
            accelerations = policy.compute_control_forces()

        # Active AND mobile agents (ground endpoints have max_speed == 0.0 and zero velocity)
        mobile_mask = (self.statuses == AgentStatus.ACTIVE) & (self.max_speeds > 0.0)
        
        # 1. Update velocity with acceleration if supplied
        if accelerations is not None:
            self.velocities[mobile_mask] += accelerations[mobile_mask] * self.dt

        # 2. Vectorized velocity clamping using np.maximum for safe division
        active_mobile_indices = np.where(mobile_mask)[0]
        if active_mobile_indices.size > 0:
            vel = self.velocities[active_mobile_indices]
            speeds = np.linalg.norm(vel, axis=1, keepdims=True)
            safe_speeds = np.maximum(speeds, 1e-9)
            scale = np.minimum(1.0, self.max_speeds[active_mobile_indices, None] / safe_speeds)
            self.velocities[active_mobile_indices] = vel * scale

        # 3. Vectorized position integration (FR-1.2)
        self.positions[mobile_mask] += self.velocities[mobile_mask] * self.dt

        # 4. Optional spatial boundary clamping
        if self.config.bounds is not None:
            min_x, max_x, min_y, max_y = self.config.bounds
            self.positions[mobile_mask, 0] = np.clip(self.positions[mobile_mask, 0], min_x, max_x)
            self.positions[mobile_mask, 1] = np.clip(self.positions[mobile_mask, 1], min_y, max_y)

    def get_agent(self, agent_id: int) -> UAVAgent:
        """Get a read-only proxy view for state inspection of a specific agent."""
        assert 0 <= agent_id < self.num_agents, f"Agent ID {agent_id} out of bounds"
        return UAVAgent(self, agent_id)

    def get_spatial_tree(self) -> Tuple[cKDTree, np.ndarray]:
        """Build SciPy cKDTree over active agent positions.
        
        Returns:
            tree: cKDTree built on active positions
            active_indices: 1D array mapping local tree indices back to canonical agent IDs
        """
        active_indices = np.where(self.statuses == AgentStatus.ACTIVE)[0]
        if active_indices.size == 0:
            active_positions = np.empty((0, 2), dtype=np.float64)
        else:
            active_positions = self.positions[active_indices]
            
        tree = cKDTree(active_positions)
        return tree, active_indices

    def get_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """Find pairs of active agents within communication range using cKDTree query_pairs()."""
        tree, active_indices = self.get_spatial_tree()
        if active_indices.size < 2:
            return []
            
        active_ranges = self.comm_ranges[active_indices]
        max_r = float(np.max(active_ranges))
        pairs = tree.query_pairs(r=max_r)
        
        # Fast-path when all active agents share identical communication range
        if np.all(active_ranges == max_r):
            return [(int(active_indices[i]), int(active_indices[j])) for i, j in pairs]
            
        # Heterogeneous comm ranges: edge exists if distance <= min(R_i, R_j)
        valid_pairs = []
        for i_local, j_local in pairs:
            idx_i = int(active_indices[i_local])
            idx_j = int(active_indices[j_local])
            effective_r = min(self.comm_ranges[idx_i], self.comm_ranges[idx_j])
            dist = np.linalg.norm(self.positions[idx_i] - self.positions[idx_j])
            if dist <= effective_r:
                valid_pairs.append((idx_i, idx_j))
        return valid_pairs
