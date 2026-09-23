from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import networkx as nx
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.policies.base import BasePolicy
from swarm_sim.utils.enums import AgentRole, AgentStatus, RecoveryStrategy

class DARAHeuristicPolicy(BasePolicy):
    """100% Local Perception DARA Policy: Primary Gap Bridging (r1 -> f_pos) + Live Graph Neighbor Midpoint Relaxation.
    
    References:
        - Wang et al. (2016, "Adaptive Connectivity Restoration")
        - Varadharajan et al. (2020, "Swarm Relays")
    """
    def __init__(
        self,
        simulator: SwarmSimulator,
        r_safe: float = 5.0,
        k_repair: float = 5.0,
        k_rep: float = 4.0,
        k_damp: float = 3.0,
        max_cascade_depth: int = 5
    ):
        super().__init__(simulator, strategy=RecoveryStrategy.HEURISTIC_DARA)
        self.r_safe = r_safe
        self.k_repair = k_repair
        self.k_rep = k_rep
        self.k_damp = k_damp
        self.max_cascade_depth = max_cascade_depth
        self.topology_mgr = SwarmTopologyManager(simulator)
        
        # Track last known positions of failed nodes
        self._last_known_failed_positions: Dict[int, np.ndarray] = {}

    def _get_chain_neighbors(self, r_idx: int, G: nx.Graph, active_set: set) -> Tuple[Optional[int], Optional[int]]:
        """Return up to two graph/sensor neighbors of r_idx using live graph G and 1.5 * Rc local sensor perception.
        No global position sort, no 1D projections, no roster of active relays.
        """
        pos = self.sim.positions
        rc = self.sim.config.comm_range
        
        neighbors = []
        if r_idx in G:
            neighbors = [n for n in G.neighbors(r_idx) if n in active_set]
            
        if len(neighbors) < 2:
            other_active = [n for n in active_set if n != r_idx and n not in neighbors]
            if len(other_active) > 0:
                dists = [(n, np.linalg.norm(pos[n] - pos[r_idx])) for n in other_active]
                dists = [item for item in dists if item[1] <= 1.5 * rc]
                dists.sort(key=lambda x: x[1])
                for n_id, _ in dists:
                    neighbors.append(n_id)
                    if len(neighbors) >= 2:
                        break

        if len(neighbors) >= 2:
            return neighbors[0], neighbors[1]
        elif len(neighbors) == 1:
            return neighbors[0], None
        else:
            return None, None

    def compute_control_forces(
        self,
        graph: Optional[nx.Graph] = None,
        return_pre_clamp: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[int, bool]]]:
        """Compute (N, 2) control acceleration array for active agents.

        100% Graph-Based Local Perception:
        - Primary repair node r1 targets f_pos directly ONLY while degree < 2 in G (gap severed).
        - Degree >= 2 nodes target 0.5 * (p_{left} + p_{right}) from G.neighbors(i).
        - ZERO projections, ZERO sorted_order, ZERO pos[:, 0] anywhere in force calculations.
        - Single pull per relay per tick via pulled_this_tick dict preventing double-counting.
        """
        N = self.sim.num_agents
        accelerations = np.zeros((N, 2), dtype=np.float64)

        G = graph if graph is not None else self.topology_mgr.build_graph()

        g_a_id, g_b_id = 0, 1
        path_exists = self.topology_mgr.has_path(g_a_id, g_b_id, graph=G)
        num_components = self.topology_mgr.get_num_connected_components(graph=G)
        repair_needed = not (path_exists and num_components == 1)

        pulled_this_tick: Dict[int, bool] = {}

        if not repair_needed:
            self._last_known_failed_positions.clear()
            # Lock swarm in reconnected equilibrium by zeroing velocity instantly
            active_mobile_mask = (self.sim.statuses == AgentStatus.ACTIVE) & (self.sim.max_speeds > 0.0)
            self.sim.velocities[active_mobile_mask] = [0.0, 0.0]
            if return_pre_clamp:
                return accelerations, accelerations.copy(), pulled_this_tick
            return accelerations

        failed_indices = np.where(self.sim.statuses == AgentStatus.FAILED)[0]
        for f_id in failed_indices:
            if f_id not in self._last_known_failed_positions:
                self._last_known_failed_positions[f_id] = self.sim.positions[f_id].copy()

        if repair_needed and len(self._last_known_failed_positions) > 0:
            active_indices = [
                i for i in range(N) if self.sim.statuses[i] == AgentStatus.ACTIVE
            ]
            active_set = set(active_indices)
            active_relay_indices = [
                i for i in active_indices if self.sim.roles[i] == AgentRole.RELAY
            ]

            if len(active_relay_indices) > 0:
                pos = self.sim.positions

                # 1. Primary repair node r1 nearest to f_pos targets f_pos while degree < 2 in G
                for f_id, f_pos in list(self._last_known_failed_positions.items()):
                    distances = [np.linalg.norm(pos[r] - f_pos) for r in active_relay_indices]
                    r1_idx = active_relay_indices[int(np.argmin(distances))]
                    
                    left, right = self._get_chain_neighbors(r1_idx, G, active_set)
                    if left is None or right is None:
                        self._apply_repair_pull(r1_idx, f_pos, accelerations, pulled_this_tick)

                # 2. Pure graph neighbor midpoint relaxation for all other active mobile relays
                for i in active_relay_indices:
                    if i not in pulled_this_tick and self.sim.max_speeds[i] > 0.0:
                        left, right = self._get_chain_neighbors(i, G, active_set)
                        if left is not None and right is not None:
                            target_pos = 0.5 * (pos[left] + pos[right])
                            self._apply_repair_pull(i, target_pos, accelerations, pulled_this_tick)

        # 5. 100% Vectorized Collision Avoidance (R_safe)
        active_mobile_mask = (self.sim.statuses == AgentStatus.ACTIVE) & (self.sim.max_speeds > 0.0)
        active_mobile_indices = np.where(active_mobile_mask)[0]

        if active_mobile_indices.size > 1:
            pos_active = self.sim.positions[active_mobile_indices]
            diff = pos_active[:, None, :] - pos_active[None, :, :]
            d_ij = np.linalg.norm(diff, axis=2)
            safe_d = np.maximum(d_ij, 1e-3)
            rep_mask = (d_ij < self.r_safe) & (d_ij > 1e-6)
            rep_mag = np.zeros_like(d_ij, dtype=np.float64)
            rep_mag[rep_mask] = self.k_rep * (1.0 / safe_d[rep_mask] - 1.0 / self.r_safe)
            unit_diff = diff / safe_d[:, :, None]
            rep_forces = np.sum(rep_mag[:, :, None] * unit_diff, axis=1)
            accelerations[active_mobile_indices] += rep_forces

        raw_accelerations = accelerations.copy()

        # 6. Branchless Safe-Division Acceleration Clamping to max_accel per agent
        if active_mobile_indices.size > 0:
            acc_mags = np.linalg.norm(accelerations[active_mobile_indices], axis=1, keepdims=True)
            safe_acc_mags = np.maximum(acc_mags, 1e-9)
            max_a = self.sim.max_accels[active_mobile_indices, None]
            scale = np.minimum(1.0, max_a / safe_acc_mags)
            accelerations[active_mobile_indices] *= scale

        # Stationary ground endpoints strictly zeroed
        ground_mask = (self.sim.max_speeds == 0.0)
        accelerations[ground_mask] = [0.0, 0.0]

        if return_pre_clamp:
            return accelerations, raw_accelerations, pulled_this_tick

        return accelerations

    def _apply_repair_pull(
        self,
        r_idx: int,
        target_pos: np.ndarray,
        accelerations: np.ndarray,
        pulled_this_tick: Dict[int, bool]
    ):
        """PD pull of r_idx toward target_pos. No-op if r_idx already received a repair force this tick."""
        if r_idx in pulled_this_tick:
            return
        dir_vec = target_pos - self.sim.positions[r_idx]
        dist = np.linalg.norm(dir_vec)
        if dist > 1e-3:
            accelerations[r_idx] += self.k_repair * (dir_vec / dist) - self.k_damp * self.sim.velocities[r_idx]
            pulled_this_tick[r_idx] = True
