import os, sys, math, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import qpsolvers
import torch

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from scratch.evaluate_b_prime_full_battery import build_all_scenarios


class ParametricCBFGNNPolicy(GNNPolicy):
    def __init__(
        self,
        simulator: SwarmSimulator,
        actor_gnn: ActorGNN,
        alpha1: float = 3.0,
        alpha2: float = 1.5,
        barrier_margin: float = 0.05,
        device: str = "cpu"
    ):
        super().__init__(simulator, actor_gnn=actor_gnn, device=device)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.barrier_margin = barrier_margin
        self.cbf_active_count = 0
        self.qp_infeasible_count = 0

    def _is_network_connected(self) -> bool:
        G = self.topology_mgr.build_graph()
        g_a = 0
        g_b = 1
        for idx, role in enumerate(self.sim.roles):
            if role == AgentRole.GROUND_A or str(role) == "ground_a":
                g_a = idx
            elif role == AgentRole.GROUND_B or str(role) == "ground_b":
                g_b = idx
        return self.topology_mgr.has_path(g_a, g_b, graph=G)

    def apply_cbf_qp(self, raw_world_acc: np.ndarray) -> np.ndarray:
        sim = self.sim
        pos = sim.positions
        vel = sim.velocities
        
        active_mobile_mask = (sim.statuses == AgentStatus.ACTIVE) & (sim.max_speeds > 0.0)
        active_mobile_indices = np.where(active_mobile_mask)[0]
        M = len(active_mobile_indices)
        if M == 0:
            return raw_world_acc

        idx_map = {idx: k for k, idx in enumerate(active_mobile_indices)}
        u_desired = raw_world_acc[active_mobile_indices].copy()

        G = self.topology_mgr.build_graph()
        edges = list(G.edges())

        P = sp.eye(2 * M, format="csc")
        q = -u_desired.reshape(2 * M)

        A_rows = []
        b_vals = []

        for u_node, v_node in edges:
            rc_eff = min(float(sim.comm_ranges[u_node]), float(sim.comm_ranges[v_node]))
            r_safe = max(rc_eff - self.barrier_margin, 1.0)
            
            p_ij = pos[u_node] - pos[v_node]
            v_ij = vel[u_node] - vel[v_node]
            dist_sq = float(np.dot(p_ij, p_ij))
            
            h_ij = r_safe**2 - dist_sq
            h_dot = -2.0 * float(np.dot(p_ij, v_ij))

            rhs = -2.0 * float(np.dot(v_ij, v_ij)) + self.alpha1 * h_dot + self.alpha2 * h_ij

            row = np.zeros(2 * M, dtype=np.float64)
            has_mobile = False
            if u_node in idx_map:
                k_u = idx_map[u_node]
                row[2*k_u : 2*k_u+2] += 2.0 * p_ij
                has_mobile = True
            if v_node in idx_map:
                k_v = idx_map[v_node]
                row[2*k_v : 2*k_v+2] -= 2.0 * p_ij
                has_mobile = True

            if has_mobile:
                A_rows.append(row)
                b_vals.append(rhs)

        K_poly = 16
        angles = np.linspace(0, 2 * np.pi, K_poly, endpoint=False)
        for k_m, idx in enumerate(active_mobile_indices):
            a_max = float(sim.max_accels[idx])
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
            a_max = float(sim.max_accels[idx])
            lb[2*k_m : 2*k_m+2] = -a_max
            ub[2*k_m : 2*k_m+2] = a_max

        sol = qpsolvers.solve_qp(P, q, G=G_mat, h=h_vec, lb=lb, ub=ub, solver="osqp")

        filtered_world_acc = raw_world_acc.copy()
        if sol is not None:
            u_opt = sol.reshape((M, 2))
            filtered_world_acc[active_mobile_indices] = u_opt
            self.cbf_active_count += 1
        else:
            self.qp_infeasible_count += 1
            filtered_world_acc[active_mobile_indices] = u_desired

        acc_mags = np.linalg.norm(filtered_world_acc[active_mobile_indices], axis=1, keepdims=True)
        safe_mags = np.maximum(acc_mags, 1e-9)
        max_a = sim.max_accels[active_mobile_indices, None]
        scale = np.minimum(1.0, max_a / safe_mags)
        filtered_world_acc[active_mobile_indices] *= scale

        return filtered_world_acc

    def compute_control_forces(self, return_pre_clamp: bool = False):
        raw_world_acc = super().compute_control_forces(return_pre_clamp=False)
        if not self._is_network_connected():
            final_acc = raw_world_acc
        else:
            final_acc = self.apply_cbf_qp(raw_world_acc)
            
        if return_pre_clamp:
            mobile_mask_t = (self.sim.statuses == AgentStatus.ACTIVE) & (self.sim.max_speeds > 0.0)
            pulled_this_tick = {i: True for i in np.where(mobile_mask_t)[0]}
            return final_acc, raw_world_acc, pulled_this_tick
        return final_acc


def test_margin_on_cat6(margins=[0.0, 0.02, 0.05, 0.10]):
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load(bp_path, map_location="cpu", weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    all_scenarios = build_all_scenarios()
    cat6_scenarios = all_scenarios["6. Tight-Margin Geometry"]

    print("=" * 100)
    print("TESTING BARRIER MARGIN EPSILON SWEEP ON CATEGORY 6")
    print("=" * 100)

    for eps in margins:
        print(f"\n--- BARRIER MARGIN epsilon = {eps:.2f}m ---")
        for sc_idx, sc in enumerate(cat6_scenarios):
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
            policy = ParametricCBFGNNPolicy(sim, actor_gnn=actor_bp, alpha1=3.0, alpha2=1.5, barrier_margin=eps, device="cpu")
            topo = SwarmTopologyManager(sim)

            failures = list(sc.get("failures", []))
            last_fail_step = max([f["step"] for f in failures]) if failures else 0

            first_conn = None
            drops = 0
            prev_c = False
            conn_hist = []
            min_slack_over_run = 999.0
            min_slack_tick = None
            min_slack_link = None

            for t in range(300):
                for f in failures:
                    if t == f["step"]:
                        injector.fail_agent(f["agent_id"])

                acc = policy.compute_control_forces()
                sim.step(accelerations=acc)

                G = topo.build_graph()
                g_a_id = 0
                g_b_id = 1
                for idx, role in enumerate(sim.roles):
                    if role == AgentRole.GROUND_A or str(role) == "ground_a":
                        g_a_id = idx
                    elif role == AgentRole.GROUND_B or str(role) == "ground_b":
                        g_b_id = idx

                is_conn = topo.has_path(g_a_id, g_b_id, graph=G)
                conn_hist.append(is_conn)

                if t >= last_fail_step:
                    if is_conn:
                        if first_conn is None:
                            first_conn = t
                        prev_c = True
                    else:
                        if prev_c:
                            drops += 1
                        prev_c = False

                # Active agent chain ordering along X
                active_mask = (sim.statuses == AgentStatus.ACTIVE)
                active_indices = np.where(active_mask)[0]
                sorted_by_x = sorted(active_indices, key=lambda a: sim.positions[a, 0])

                if t >= 70:
                    for k in range(len(sorted_by_x) - 1):
                        u = sorted_by_x[k]
                        v = sorted_by_x[k+1]
                        d = float(np.linalg.norm(sim.positions[u] - sim.positions[v]))
                        slack = 28.0 - d
                        if slack < min_slack_over_run:
                            min_slack_over_run = slack
                            min_slack_tick = t
                            min_slack_link = (u, v)

            k20 = all(conn_hist[280:300])
            c300 = conn_hist[299]
            c150 = conn_hist[149]
            print(f"  Sc{sc_idx+1} ({sc['desc']}): FirstReconn={first_conn}, Drops={drops}, Conn@150={c150}, Conn@300={c300}, K20={k20}, MinSlack(t>=70)={min_slack_over_run:+.4f}m @ {min_slack_link} (t={min_slack_tick})")


if __name__ == "__main__":
    test_margin_on_cat6()
