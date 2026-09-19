from typing import Optional, Dict, Tuple, Union, Any
import numpy as np
import scipy.sparse as sp
import qpsolvers
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.utils.enums import AgentRole, AgentStatus, RecoveryStrategy

class GreedyDARAHeuristicPolicy(DARAHeuristicPolicy):
    """Lexicographic Greedy Throughput-Aware DARA Policy.
    
    Lexicographic Priority:
    1. Primary Objective (Global Connectivity): When the network is disconnected (no A<->B path),
       executes pure DARAHeuristicPolicy to restore end-to-end connectivity with zero SNR interference
       and zero CBF restriction.
    2. Secondary Objective (Throughput Optimization): Once the network is connected (A<->B path exists),
       adds local SNR throughput gradient nudges to DARA forces and passes them through a Second-Order
       CBF-QP safety filter to guarantee inter-relay distances remain <= R_c.
    """
    def __init__(
        self,
        simulator: SwarmSimulator,
        channel_model: Optional[ChannelModel] = None,
        k_tp: float = 1.0,
        delta_s: float = 0.5,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        **kwargs: Any
    ):
        super().__init__(simulator, **kwargs)
        self.strategy = RecoveryStrategy.GREEDY_DARA
        self.channel_model = channel_model if channel_model is not None else ChannelModel()
        self.k_tp = k_tp
        self.delta_s = delta_s
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset execution mode and QP solver telemetry counters."""
        self.ticks_pure_dara = 0
        self.ticks_throughput_polishing = 0
        self.infeasible_solve_count = 0
        self.total_solve_count = 0

    def _get_ground_endpoints(self) -> Tuple[Optional[int], Optional[int]]:
        """Identify ground endpoint node IDs."""
        g_a = None
        g_b = None
        for idx, role in enumerate(self.sim.roles):
            role_val = getattr(role, "value", str(role))
            if role == AgentRole.GROUND_A or role_val == "ground_a":
                g_a = idx
            elif role == AgentRole.GROUND_B or role_val == "ground_b":
                g_b = idx
        return g_a, g_b

    def compute_control_forces(
        self,
        return_pre_clamp: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[int, bool]]]:
        """Compute control forces with lexicographic switching between pure DARA and CBF-QP polishing."""
        G = self.topology_mgr.build_graph()
        g_a, g_b = self._get_ground_endpoints()

        # Check end-to-end connectivity between ground endpoints A and B
        if g_a is not None and g_b is not None:
            is_connected = self.topology_mgr.has_path(g_a, g_b, graph=G)
        else:
            is_connected = self.topology_mgr.is_fully_connected(graph=G)

        # Mode 1: Disconnected -> Pure DARA (zero SNR term, zero CBF filter)
        if not is_connected:
            self.ticks_pure_dara += 1
            return super().compute_control_forces(return_pre_clamp=return_pre_clamp)

        # Mode 2: Connected -> Throughput-Polishing with CBF-QP Safety Filter
        self.ticks_throughput_polishing += 1
        res = super().compute_control_forces(return_pre_clamp=True)
        accelerations, raw_accelerations, pulled_this_tick = res

        pos = self.sim.positions
        vel = self.sim.velocities
        rc = self.sim.config.comm_range

        # Identify active mobile relays
        active_mobile_mask = (self.sim.statuses == AgentStatus.ACTIVE) & (self.sim.max_speeds > 0.0)
        active_mobile_indices = np.where(active_mobile_mask)[0]
        M = len(active_mobile_indices)

        if M > 0:
            idx_map = {idx: k for k, idx in enumerate(active_mobile_indices)}
            u_desired = accelerations[active_mobile_indices].copy()

            # Compute local SNR throughput gradient for each active mobile relay
            for k_m, i in enumerate(active_mobile_indices):
                neighbors = list(G.neighbors(i)) if i in G else []
                if not neighbors:
                    continue

                curr_pos = pos[i]
                curr_snr = sum(
                    self.channel_model.get_link_throughput(
                        curr_pos,
                        pos[n],
                        is_backhaul=(self.sim.roles[n].value.startswith("ground"))
                    )[0] for n in neighbors
                )

                grad_x = 0.0
                grad_y = 0.0

                for dx, dy in [(self.delta_s, 0.0), (-self.delta_s, 0.0)]:
                    cand_pos = curr_pos + np.array([dx, dy])
                    cand_snr = sum(
                        self.channel_model.get_link_throughput(
                            cand_pos,
                            pos[n],
                            is_backhaul=(self.sim.roles[n].value.startswith("ground"))
                        )[0] for n in neighbors
                    )
                    grad_x += (cand_snr - curr_snr) / dx

                for dx, dy in [(0.0, self.delta_s), (0.0, -self.delta_s)]:
                    cand_pos = curr_pos + np.array([dx, dy])
                    cand_snr = sum(
                        self.channel_model.get_link_throughput(
                            cand_pos,
                            pos[n],
                            is_backhaul=(self.sim.roles[n].value.startswith("ground"))
                        )[0] for n in neighbors
                    )
                    grad_y += (cand_snr - curr_snr) / dy

                grad_vec = np.array([grad_x, grad_y])
                grad_norm = np.linalg.norm(grad_vec)
                if grad_norm > 1e-3:
                    u_desired[k_m] += self.k_tp * (grad_vec / grad_norm)

            # Formulate Second-Order CBF QP:
            # min_u 1/2 ||u - u_desired||^2  s.t.  A_cbf u <= b_cbf,  ||u_i|| <= a_max
            P = sp.eye(2 * M, format="csc")
            q = -u_desired.reshape(2 * M)

            A_rows = []
            b_vals = []

            # Add CBF constraints for all active edges (i, j) with ||p_i - p_j|| <= R_c
            for u_node, v_node in G.edges():
                p_ij = pos[u_node] - pos[v_node]
                v_ij = vel[u_node] - vel[v_node]
                dist_sq = float(np.dot(p_ij, p_ij))
                if dist_sq > rc**2:
                    continue

                h_ij = rc**2 - dist_sq
                h_dot = -2.0 * float(np.dot(p_ij, v_ij))
                
                # Second-order CBF constraint: psi_dot + alpha2 * h_ij >= 0
                rhs = -2.0 * float(np.dot(v_ij, v_ij)) + self.alpha1 * h_dot + self.alpha2 * h_ij

                row = np.zeros(2 * M, dtype=np.float64)
                if u_node in idx_map:
                    k_u = idx_map[u_node]
                    row[2*k_u : 2*k_u+2] += 2.0 * p_ij
                if v_node in idx_map:
                    k_v = idx_map[v_node]
                    row[2*k_v : 2*k_v+2] -= 2.0 * p_ij

                A_rows.append(row)
                b_vals.append(rhs)

            # Polyhedral 16-gon acceleration bounds for ||u_m||_2 <= a_max
            K = 16
            angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
            for k_m, idx in enumerate(active_mobile_indices):
                a_max = float(self.sim.max_accels[idx])
                for theta in angles:
                    row = np.zeros(2 * M, dtype=np.float64)
                    row[2*k_m] = np.cos(theta)
                    row[2*k_m + 1] = np.sin(theta)
                    A_rows.append(row)
                    b_vals.append(a_max)

            G_mat = sp.csc_matrix(np.array(A_rows)) if A_rows else None
            h_vec = np.array(b_vals, dtype=np.float64) if b_vals else None

            lb = np.zeros(2 * M, dtype=np.float64)
            ub = np.zeros(2 * M, dtype=np.float64)
            for k_m, idx in enumerate(active_mobile_indices):
                a_max = float(self.sim.max_accels[idx])
                lb[2*k_m : 2*k_m+2] = -a_max
                ub[2*k_m : 2*k_m+2] = a_max

            self.total_solve_count += 1
            sol = qpsolvers.solve_qp(P, q, G=G_mat, h=h_vec, lb=lb, ub=ub, solver="osqp")

            if sol is not None:
                u_opt = sol.reshape((M, 2))
                accelerations[active_mobile_indices] = u_opt
            else:
                self.infeasible_solve_count += 1
                accelerations[active_mobile_indices] = raw_accelerations[active_mobile_indices]

        # Re-clamp acceleration magnitudes to max_accel
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
