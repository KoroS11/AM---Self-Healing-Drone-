from typing import Optional, Dict, Tuple, Union, Any
import numpy as np
import torch
import networkx as nx
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.policies.base import BasePolicy
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.utils.enums import AgentRole, AgentStatus, RecoveryStrategy

class GNNPolicy(BasePolicy):
    """GNN Policy Wrapper connecting PyTorch ActorGNN model to SwarmSimulator.
    
    Provides 100% interface compatibility with SwarmSimulator.step(policy=...) and step(accelerations=...).
    """
    def __init__(
        self,
        simulator: SwarmSimulator,
        actor_gnn: Optional[ActorGNN] = None,
        channel_model: Optional[ChannelModel] = None,
        k_hops: int = 1,
        device: Optional[str] = None
    ):
        super().__init__(simulator, strategy=RecoveryStrategy.LEARNED_GNN)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.actor_gnn = actor_gnn if actor_gnn is not None else ActorGNN(
            k_hops=k_hops, max_accel=simulator.config.max_accel
        )
        self.actor_gnn.to(self.device)
        self.actor_gnn.eval()
        
        self.channel_model = channel_model if channel_model is not None else ChannelModel()
        self.topology_mgr = SwarmTopologyManager(simulator)
        
        # Track last known failed positions
        self._last_known_failed_positions: Dict[int, np.ndarray] = {}

    def get_corridor_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Ground A position, corridor unit vector (u_par), and perpendicular unit vector (u_perp)."""
        sim = self.sim
        pos = sim.positions
        ground_a_indices = np.where(sim.roles == AgentRole.GROUND_A)[0]
        ground_b_indices = np.where(sim.roles == AgentRole.GROUND_B)[0]
        if len(ground_a_indices) > 0 and len(ground_b_indices) > 0:
            pos_A = pos[ground_a_indices[0]]
            pos_B = pos[ground_b_indices[0]]
            ab_vec = pos_B - pos_A
            ab_dist = max(np.linalg.norm(ab_vec), 1e-6)
            u_par = ab_vec / ab_dist
            u_perp = np.array([-u_par[1], u_par[0]])
        else:
            pos_A = np.array([0.0, 0.0])
            u_par = np.array([1.0, 0.0])
            u_perp = np.array([0.0, 1.0])
        return pos_A, u_par, u_perp

    def corridor_to_world_forces(self, corridor_forces: np.ndarray) -> np.ndarray:
        """Transform corridor-relative 2D forces [a_par, a_perp] to world-frame forces."""
        _, u_par, u_perp = self.get_corridor_frame()
        world_forces = corridor_forces[:, 0:1] * u_par[None, :] + corridor_forces[:, 1:2] * u_perp[None, :]
        return world_forces

    def world_to_corridor_forces(self, world_forces: np.ndarray) -> np.ndarray:
        """Transform world-frame 2D forces to corridor-relative forces [a_par, a_perp]."""
        _, u_par, u_perp = self.get_corridor_frame()
        a_par = np.sum(world_forces * u_par[None, :], axis=1, keepdims=True)
        a_perp = np.sum(world_forces * u_perp[None, :], axis=1, keepdims=True)
        return np.concatenate([a_par, a_perp], axis=1)


    def extract_graph_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract rotation-invariant node features (N, 8), edge features (E, 4), edge index (2, E), and mobile mask (N,)."""
        sim = self.sim
        N = sim.num_agents
        pos = sim.positions
        vel = sim.velocities
        rc = sim.config.comm_range
        
        pos_A, u_par, u_perp = self.get_corridor_frame()
        
        # Update failed positions tracking
        failed_indices = np.where(sim.statuses == AgentStatus.FAILED)[0]
        for f_id in failed_indices:
            if f_id not in self._last_known_failed_positions:
                self._last_known_failed_positions[f_id] = pos[f_id].copy()
                
        G = self.topology_mgr.build_graph()
        
        # Determine bounds
        bounds_x = 300.0
        bounds_y = 300.0
        if sim.config.bounds is not None:
            min_x, max_x, min_y, max_y = sim.config.bounds
            bounds_x = max(abs(min_x), abs(max_x), 1.0)
            bounds_y = max(abs(min_y), abs(max_y), 1.0)
            
        # Build node feature array (N, 9) in corridor-relative frame
        node_feats = np.zeros((N, 9), dtype=np.float32)
        for i in range(N):
            role_val = 0.0
            if sim.roles[i] == AgentRole.GROUND_A:
                role_val = 1.0
            elif sim.roles[i] == AgentRole.GROUND_B:
                role_val = 2.0
                
            deg = float(G.degree(i)) / float(max(N - 1, 1)) if i in G else 0.0
            
            # check if agent i had any neighbor that failed
            has_lost_neighbor = 0.0
            if self._last_known_failed_positions:
                dists = [np.linalg.norm(pos[i] - f_pos) for f_pos in self._last_known_failed_positions.values()]
                if min(dists) <= 1.5 * rc:
                    has_lost_neighbor = 1.0
                    
            # Transform position relative to Ground A and rotate into corridor frame
            p_rel = pos[i] - pos_A
            p_par = float(np.dot(p_rel, u_par))
            p_perp = float(np.dot(p_rel, u_perp))
            
            # Rotate velocity into corridor frame
            v_par = float(np.dot(vel[i], u_par))
            v_perp = float(np.dot(vel[i], u_perp))
            max_sp = max(sim.max_speeds[i], 1e-3)
            rc_norm = float(sim.comm_ranges[i]) / 28.0
            
            node_feats[i] = [
                p_par / bounds_x,
                p_perp / bounds_y,
                v_par / max_sp,
                v_perp / max_sp,
                role_val,
                1.0 if sim.statuses[i] == AgentStatus.ACTIVE else 0.0,
                deg,
                has_lost_neighbor,
                rc_norm
            ]
            
        # Build edge features and edge index
        edges = list(G.edges())
        num_edges = len(edges) * 2  # Bidirectional
        
        if num_edges == 0:
            edge_feats = np.zeros((0, 5), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_feats_list = []
            src_list = []
            dst_list = []
            
            for u, v in edges:
                for src_node, dst_node in [(u, v), (v, u)]:
                    src_list.append(src_node)
                    dst_list.append(dst_node)
                    
                    dist = float(np.linalg.norm(pos[src_node] - pos[dst_node]))
                    is_backhaul = (
                        sim.roles[src_node].value.startswith("ground") or
                        sim.roles[dst_node].value.startswith("ground")
                    )
                    snr, rate = self.channel_model.get_link_throughput(
                        pos[src_node], pos[dst_node], is_backhaul=is_backhaul
                    )
                    
                    dir_vec = pos[dst_node] - pos[src_node]
                    dir_norm = max(np.linalg.norm(dir_vec), 1e-3)
                    dir_u = dir_vec / dir_norm
                    
                    # Project edge unit direction onto corridor unit vector
                    dir_par = float(np.dot(dir_u, u_par))
                    
                    min_rc_ij = min(float(sim.comm_ranges[src_node]), float(sim.comm_ranges[dst_node]))
                    slack_ij = (min_rc_ij - dist) / max(min_rc_ij, 1e-3)
                    
                    edge_feats_list.append([
                        dist / 28.0,
                        rate / 50.0,  # Normalized rate
                        1.0 if is_backhaul else 0.0,
                        dir_par,
                        slack_ij
                    ])
                    
            edge_feats = np.array(edge_feats_list, dtype=np.float32)
            edge_index = np.array([src_list, dst_list], dtype=np.int64)
            
        mobile_mask = (sim.statuses == AgentStatus.ACTIVE) & (sim.max_speeds > 0.0)
        
        node_tensor = torch.from_numpy(node_feats).to(self.device)
        edge_tensor = torch.from_numpy(edge_feats).to(self.device)
        edge_idx_tensor = torch.from_numpy(edge_index).to(self.device)
        mobile_mask_tensor = torch.from_numpy(mobile_mask).to(self.device)
        
        return node_tensor, edge_tensor, edge_idx_tensor, mobile_mask_tensor

    def compute_control_forces(
        self,
        return_pre_clamp: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[int, bool]]]:
        """Compute control forces using PyTorch ActorGNN forward pass and rotate to world frame."""
        node_t, edge_t, edge_idx_t, mobile_mask_t = self.extract_graph_features()
        
        with torch.no_grad():
            actions_t = self.actor_gnn(node_t, edge_t, edge_idx_t, mobile_mask_t)
            
        corridor_acc = actions_t.cpu().numpy()
        world_acc = self.corridor_to_world_forces(corridor_acc)
        
        if return_pre_clamp:
            pulled_this_tick = {i: True for i in np.where(mobile_mask_t.cpu().numpy())[0]}
            return world_acc, world_acc.copy(), pulled_this_tick
            
        return world_acc
