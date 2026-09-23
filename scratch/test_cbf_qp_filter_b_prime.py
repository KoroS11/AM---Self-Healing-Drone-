"""
TEST CBF-QP FILTER ON CHECKPOINT B-PRIME
========================================
Applies Second-Order CBF-QP filter on live edges to GNNPolicy's raw accelerations.
Evaluates Category 11 (all 4 scenarios) and benchmark scenarios.
"""
import os, sys, math, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import qpsolvers
import torch
import networkx as nx

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel


class CBFFilteredGNNPolicy(GNNPolicy):
    """GNN Policy with per-tick Second-Order CBF-QP safety filter on all live communication edges."""
    def __init__(
        self,
        simulator: SwarmSimulator,
        actor_gnn: ActorGNN,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        barrier_margin: float = 0.5,  # Buffer in meters from Rc
        device: str = "cpu"
    ):
        super().__init__(simulator, actor_gnn=actor_gnn, device=device)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.barrier_margin = barrier_margin
        self.cbf_active_count = 0
        self.qp_infeasible_count = 0

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

        # Build current graph of LIVE edges
        G = self.topology_mgr.build_graph()
        edges = list(G.edges())

        # Formulate QP: min 1/2 ||u - u_desired||^2
        P = sp.eye(2 * M, format="csc")
        q = -u_desired.reshape(2 * M)

        A_rows = []
        b_vals = []

        # Second-order CBF constraints for each currently live edge (u, v)
        for u_node, v_node in edges:
            rc_eff = min(float(sim.comm_ranges[u_node]), float(sim.comm_ranges[v_node]))
            # Effective safe radius with barrier margin
            r_safe = max(rc_eff - self.barrier_margin, 1.0)
            
            p_ij = pos[u_node] - pos[v_node]
            v_ij = vel[u_node] - vel[v_node]
            dist_sq = float(np.dot(p_ij, p_ij))
            
            # h(x) = r_safe^2 - ||p_ij||^2 >= 0
            h_ij = r_safe**2 - dist_sq
            h_dot = -2.0 * float(np.dot(p_ij, v_ij))

            # Second-order CBF: ddot{h} + alpha1 * dot{h} + alpha2 * h >= 0
            # ddot{h} = -2 ||v_ij||^2 - 2 p_ij^T (a_u - a_v)
            # => 2 p_ij^T (a_u - a_v) <= -2 ||v_ij||^2 + alpha1 * h_dot + alpha2 * h_ij
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

        # Polyhedral 16-gon acceleration bounds for ||u_m||_2 <= a_max
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
            # Fallback to raw acceleration
            filtered_world_acc[active_mobile_indices] = u_desired

        # Magnitude clamp
        acc_mags = np.linalg.norm(filtered_world_acc[active_mobile_indices], axis=1, keepdims=True)
        safe_mags = np.maximum(acc_mags, 1e-9)
        max_a = sim.max_accels[active_mobile_indices, None]
        scale = np.minimum(1.0, max_a / safe_mags)
        filtered_world_acc[active_mobile_indices] *= scale

        return filtered_world_acc

    def compute_control_forces(
        self,
        return_pre_clamp: bool = False
    ):
        raw_world_acc = super().compute_control_forces(return_pre_clamp=False)
        filtered_acc = self.apply_cbf_qp(raw_world_acc)
        if return_pre_clamp:
            mobile_mask_t = (self.sim.statuses == AgentStatus.ACTIVE) & (self.sim.max_speeds > 0.0)
            pulled_this_tick = {i: True for i in np.where(mobile_mask_t)[0]}
            return filtered_acc, raw_world_acc, pulled_this_tick
        return filtered_acc


def run_scenario_with_policy(policy, sc, max_steps=300):
    sim = policy.sim
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()
    injector = FailureInjector(sim)

    failures = list(sc.get("failures", []))
    last_fail_step = max([f["step"] for f in failures]) if failures else 0

    records = []

    for t in range(max_steps):
        for f in failures:
            if t == f["step"]:
                injector.fail_agent(f["agent_id"])

        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)

        G = topo.build_graph()
        g_a = 0
        g_b = 1
        for idx, role in enumerate(sim.roles):
            if role == AgentRole.GROUND_A or str(role) == "ground_a":
                g_a = idx
            elif role == AgentRole.GROUND_B or str(role) == "ground_b":
                g_b = idx

        is_conn = topo.has_path(g_a, g_b, graph=G)
        sum_rate = channel.compute_network_throughput(sim, G)

        active_ids = [i for i in range(sim.num_agents) if sim.statuses[i] == AgentStatus.ACTIVE]
        pos_A = sim.positions[g_a]
        pos_B = sim.positions[g_b]
        u_ab = (pos_B - pos_A) / np.linalg.norm(pos_B - pos_A)
        active_ordered = sorted(active_ids, key=lambda i: np.dot(sim.positions[i] - pos_A, u_ab))

        min_slack = 999.0
        max_link_dist = 0.0
        for k in range(len(active_ordered) - 1):
            u_id = active_ordered[k]
            v_id = active_ordered[k+1]
            d_uv = np.linalg.norm(sim.positions[u_id] - sim.positions[v_id])
            rc_eff = min(sim.comm_ranges[u_id], sim.comm_ranges[v_id])
            slack = rc_eff - d_uv
            if slack < min_slack:
                min_slack = slack
            if d_uv > max_link_dist:
                max_link_dist = d_uv

        mobile_ids = [i for i in active_ids if sim.max_speeds[i] > 0]
        speeds = [np.linalg.norm(sim.velocities[i]) for i in mobile_ids] if mobile_ids else [0.0]

        records.append({
            "step": t,
            "connected": is_conn,
            "sum_rate": sum_rate,
            "min_slack": min_slack,
            "max_link_dist": max_link_dist,
            "mean_speed": np.mean(speeds),
            "max_speed": np.max(speeds)
        })

    return records


CAT11_SCENARIOS = [
    {
        "id": 1, "desc": "N=9, d=170m, Agent 4 Rc=22m, Fail 6@t=10",
        "seed": 2111, "num_drones": 9, "comm_range": 28.0,
        "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (170.0, 0.0),
        "custom_comm_ranges": {4: 22.0}, "failures": [{"step": 10, "agent_id": 6}]
    },
    {
        "id": 2, "desc": "N=9, d=165m, Agents 3,6 Rc=22m, Fail 5@t=10",
        "seed": 2112, "num_drones": 9, "comm_range": 28.0,
        "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (165.0, 0.0),
        "custom_comm_ranges": {3: 22.0, 6: 22.0}, "failures": [{"step": 10, "agent_id": 5}]
    },
    {
        "id": 3, "desc": "N=11, d=180m, Agents 4,8 Rc=22m, Fail 6@t=10",
        "seed": 2113, "num_drones": 11, "comm_range": 28.0,
        "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (180.0, 0.0),
        "custom_comm_ranges": {4: 22.0, 8: 22.0}, "failures": [{"step": 10, "agent_id": 6}]
    },
    {
        "id": 4, "desc": "N=11, d=185m, Agent 5 Rc=20m, Fail 7@t=10",
        "seed": 2114, "num_drones": 11, "comm_range": 28.0,
        "endpoint_a_pos": (0.0, 0.0), "endpoint_b_pos": (185.0, 0.0),
        "custom_comm_ranges": {5: 20.0}, "failures": [{"step": 10, "agent_id": 7}]
    }
]


def test_param_sweep():
    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load("checkpoints/checkpoint_B_prime_best.pt", map_location="cpu", weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    configs = [
        {"alpha1": 1.0, "alpha2": 1.0, "margin": 0.0},
        {"alpha1": 2.0, "alpha2": 1.0, "margin": 0.0},
        {"alpha1": 3.0, "alpha2": 2.0, "margin": 0.0},
        {"alpha1": 2.0, "alpha2": 1.0, "margin": 0.5},
        {"alpha1": 3.0, "alpha2": 2.0, "margin": 0.5},
        {"alpha1": 4.0, "alpha2": 3.0, "margin": 0.5},
        {"alpha1": 5.0, "alpha2": 4.0, "margin": 0.8},
        {"alpha1": 2.0, "alpha2": 1.0, "margin": 1.0},
    ]

    for cfg in configs:
        a1 = cfg["alpha1"]
        a2 = cfg["alpha2"]
        m = cfg["margin"]

        results = []
        for sc in CAT11_SCENARIOS:
            exp_cfg = ExperimentConfig(
                num_drones=sc["num_drones"],
                comm_range=sc["comm_range"],
                seed=sc["seed"]
            )
            sim = SwarmSimulator(exp_cfg)
            scenario = DisasterRelayScenario(sim)
            scenario.setup_scenario(
                endpoint_a_pos=sc.get("endpoint_a_pos", (0.0, 0.0)),
                endpoint_b_pos=sc.get("endpoint_b_pos", (200.0, 0.0))
            )
            if "custom_comm_ranges" in sc:
                for agent_id, r_val in sc["custom_comm_ranges"].items():
                    sim.comm_ranges[agent_id] = r_val

            policy = CBFFilteredGNNPolicy(
                sim, actor_gnn=actor_bp, alpha1=a1, alpha2=a2, barrier_margin=m, device="cpu"
            )
            recs = run_scenario_with_policy(policy, sc, max_steps=300)
            df = pd.DataFrame(recs)

            snap_150 = df.loc[149, "connected"]
            snap_300 = df.loc[299, "connected"]
            k20_150 = df.loc[130:149, "connected"].all()
            k20_300 = df.loc[280:299, "connected"].all()
            min_slack_300 = df.loc[299, "min_slack"]

            results.append({
                "id": sc["id"],
                "snap_150": snap_150, "snap_300": snap_300,
                "k20_150": k20_150, "k20_300": k20_300,
                "min_slack_300": min_slack_300
            })

        k20_count = sum(1 for r in results if r["k20_300"])
        snap_count = sum(1 for r in results if r["snap_150"])
        print(f"Config a1={a1:.1f}, a2={a2:.1f}, m={m:.1f}m -> K20_300: {k20_count}/4  | Snap150: {snap_count}/4 | Details: {[r['k20_300'] for r in results]} Slacks: {[round(r['min_slack_300'], 2) for r in results]}")


if __name__ == "__main__":
    test_param_sweep()
