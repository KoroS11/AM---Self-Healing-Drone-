"""
FULL VERIFICATION BATTERY FOR LEXICOGRAPHIC CBF-QP FILTERED B-PRIME
===================================================================
Runs:
1. 50-Scenario Benchmark Gate (Seen-35 + Held-Out-15)
2. Category 4: 8-Scenario SO(2) Rotation Invariance
3. Stress Categories 1-3, 5-9 (24+16 scenarios)
4. Category 11: Comm-Range Heterogeneity under both Single-Tick and Sustained K=20 Metrics
Reports full side-by-side comparison tables.
"""
import os, sys, math, pickle, hashlib, subprocess
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
from scratch.evaluate_b_prime_full_battery import build_all_scenarios, find_adversarial_cut_vertex


class LexicographicCBFGNNPolicy(GNNPolicy):
    """GNN Policy with Lexicographic CBF-QP safety filter."""
    def __init__(
        self,
        simulator: SwarmSimulator,
        actor_gnn: ActorGNN,
        alpha1: float = 2.0,
        alpha2: float = 1.0,
        barrier_margin: float = 0.0,
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

    def compute_control_forces(
        self,
        return_pre_clamp: bool = False
    ):
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


def run_single_eval(actor, sc, use_cbf=False, max_steps=150):
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

    if "custom_positions_fn" in sc:
        sim.positions = sc["custom_positions_fn"](sim.positions.copy(), sc["comm_range"])

    if "custom_comm_ranges" in sc:
        for agent_id, r_val in sc["custom_comm_ranges"].items():
            sim.comm_ranges[agent_id] = r_val

    injector = FailureInjector(sim)
    if use_cbf:
        policy = LexicographicCBFGNNPolicy(sim, actor_gnn=actor, alpha1=2.0, alpha2=1.0, device="cpu")
    else:
        policy = GNNPolicy(sim, actor_gnn=actor, device="cpu")

    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()

    failures = list(sc.get("failures", []))
    config_desc = sc.get("desc", f"N={sc['num_drones']}, Seed={sc['seed']}")

    if sc.get("adversarial_cut_vertex", False):
        target_step = sc.get("fail_timestep", 10)
        worst_r, gap_size = find_adversarial_cut_vertex(sim, topo, g_a, g_b)
        failures.append({"step": target_step, "agent_id": worst_r})
        config_desc += f" -> Relay {worst_r} (Gap: {gap_size:.1f}m)"

    if not failures and "fail_timestep" in sc and "fail_agent_id" in sc:
        failures = [{"step": sc["fail_timestep"], "agent_id": sc["fail_agent_id"]}]

    last_fail_step = max([f["step"] for f in failures]) if failures else 0

    ever_reconnected = False
    reconnect_rel = None
    reconnect_abs = None
    drops_after = 0
    prev_reconn = False

    conn_history = []

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
        conn_history.append(is_conn)

        if t >= last_fail_step:
            if is_conn:
                if not ever_reconnected:
                    ever_reconnected = True
                    reconnect_abs = t
                    reconnect_rel = t - last_fail_step
                prev_reconn = True
            else:
                if prev_reconn:
                    drops_after += 1
                prev_reconn = False

    G_final = topo.build_graph()
    final_conn = topo.has_path(g_a, g_b, graph=G_final)
    sum_rate = channel.compute_network_throughput(sim, G_final)
    k20_sustained = all(conn_history[-20:]) if len(conn_history) >= 20 else False

    return {
        "reconnected": final_conn,
        "k20_sustained": k20_sustained,
        "reconnect_rel": reconnect_rel,
        "reconnect_abs": reconnect_abs,
        "sum_rate": sum_rate,
        "transient_drops": drops_after,
        "desc": sc.get("desc", f"N={sc['num_drones']}, Seed={sc['seed']}")
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"

    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load("checkpoints/checkpoint_B_prime_best.pt", map_location=device, weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    # Checkpoint B zero-padded to 9x5
    actor_b = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw_b = torch.load("checkpoints/checkpoint_B_best.pt", map_location=device, weights_only=False)
    sd_old = raw_b["actor_state_dict"] if "actor_state_dict" in raw_b else raw_b
    sd_new = actor_b.state_dict()
    for k, v in sd_old.items():
        if k == "node_encoder.0.weight" and v.shape[1] == 8:
            w = torch.zeros((v.shape[0], 9), dtype=v.dtype)
            w[:, :8] = v
            sd_new[k] = w
        elif k == "edge_encoder.0.weight" and v.shape[1] == 4:
            w = torch.zeros((v.shape[0], 5), dtype=v.dtype)
            w[:, :4] = v
            sd_new[k] = w
        elif k in sd_new:
            sd_new[k] = v
    actor_b.load_state_dict(sd_new, strict=True)
    actor_b.eval()

    print("=" * 100)
    print("  LEXICOGRAPHIC CBF-QP FILTERED B-PRIME: FULL BATTERY & REGRESSION VERIFICATION")
    print("=" * 100)

    # 1. 50-Scenario Benchmark Gate
    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank = pickle.load(f)
    seen_35 = bank["seen_35"]
    held_15 = bank["held_out_15"]
    all_50 = seen_35 + held_15

    print("\n--- 1. 50-SCENARIO BENCHMARK GATE ---")
    res_b_all = [run_single_eval(actor_b, sc, use_cbf=False) for sc in all_50]
    res_bp_raw_all = [run_single_eval(actor_bp, sc, use_cbf=False) for sc in all_50]
    res_bp_cbf_all = [run_single_eval(actor_bp, sc, use_cbf=True) for sc in all_50]

    def summarize_suite(res_list):
        succ = sum(1 for r in res_list if r["reconnected"])
        k20 = sum(1 for r in res_list if r["k20_sustained"])
        times = [r["reconnect_rel"] for r in res_list if r["reconnected"] and r["reconnect_rel"] is not None]
        rates = [r["sum_rate"] for r in res_list if r["reconnected"]]
        return succ, k20, len(res_list), np.mean(times) if times else 0.0, np.mean(rates) if rates else 0.0

    b_summary = summarize_suite(res_b_all)
    bp_raw_summary = summarize_suite(res_bp_raw_all)
    bp_cbf_summary = summarize_suite(res_bp_cbf_all)

    print(f"{'Metric':<30s} | {'Checkpoint B':<18s} | {'B-Prime (Raw)':<18s} | {'B-Prime + CBF-QP':<18s}")
    print(f"{'-'*30}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")
    print(f"{'All 50 Single-Tick Success':<30s} | {f'{b_summary[0]}/{b_summary[2]} ({b_summary[0]/b_summary[2]*100:.1f}%)':<18s} | {f'{bp_raw_summary[0]}/{bp_raw_summary[2]} ({bp_raw_summary[0]/bp_raw_summary[2]*100:.1f}%)':<18s} | {f'{bp_cbf_summary[0]}/{bp_cbf_summary[2]} ({bp_cbf_summary[0]/bp_cbf_summary[2]*100:.1f}%)':<18s}")
    print(f"{'All 50 Sustained K=20 Success':<30s} | {f'{b_summary[1]}/{b_summary[2]} ({b_summary[1]/b_summary[2]*100:.1f}%)':<18s} | {f'{bp_raw_summary[1]}/{bp_raw_summary[2]} ({bp_raw_summary[1]/bp_raw_summary[2]*100:.1f}%)':<18s} | {f'{bp_cbf_summary[1]}/{bp_cbf_summary[2]} ({bp_cbf_summary[1]/bp_cbf_summary[2]*100:.1f}%)':<18s}")
    print(f"{'All 50 Mean Reconnect Time':<30s} | {f'{b_summary[3]:.2f}t':<18s} | {f'{bp_raw_summary[3]:.2f}t':<18s} | {f'{bp_cbf_summary[3]:.2f}t':<18s}")
    print(f"{'All 50 Mean Sum-Rate':<30s} | {f'{b_summary[4]:.2f} Mbps':<18s} | {f'{bp_raw_summary[4]:.2f} Mbps':<18s} | {f'{bp_cbf_summary[4]:.2f} Mbps':<18s}")

    # 2. All Stress Categories
    all_scenarios = build_all_scenarios()
    cat_order = [
        "1. Simultaneous Double Failure", "2. Cascading Failure",
        "3. Scale Test (Large Swarm)", "4. Non-Collinear Corridor",
        "5. Failure Timing Extremes", "6. Tight-Margin Geometry",
        "7. Triple Simultaneous Failure", "8. Adversarial Cut-Vertex",
        "9. Compound Stressors", "11. Comm-Range Heterogeneity"
    ]

    print("\n--- 2. ALL STRESS CATEGORIES SUMMARY ---")
    print(f"{'Category':<35s} | {'Ckpt B (K=20)':<14s} | {'BP Raw (K=20)':<14s} | {'BP+CBF (K=20)':<14s} | {'BP+CBF (Snap)':<14s}")
    print(f"{'-'*35}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

    for cat in cat_order:
        scs = all_scenarios[cat]
        n_sc = len(scs)
        
        # Extended steps for Cat 11 (300 steps)
        steps = 300 if "11." in cat else 150
        
        r_b = [run_single_eval(actor_b, sc, use_cbf=False, max_steps=steps) for sc in scs]
        r_bp = [run_single_eval(actor_bp, sc, use_cbf=False, max_steps=steps) for sc in scs]
        r_cbf = [run_single_eval(actor_bp, sc, use_cbf=True, max_steps=steps) for sc in scs]

        b_k20 = sum(1 for r in r_b if r["k20_sustained"])
        bp_k20 = sum(1 for r in r_bp if r["k20_sustained"])
        cbf_k20 = sum(1 for r in r_cbf if r["k20_sustained"])
        cbf_snap = sum(1 for r in r_cbf if r["reconnected"])

        print(f"{cat:<35s} | {f'{b_k20}/{n_sc}':<14s} | {f'{bp_k20}/{n_sc}':<14s} | {f'{cbf_k20}/{n_sc}':<14s} | {f'{cbf_snap}/{n_sc}':<14s}")

    print("\n" + "=" * 100)
    print("  VERIFICATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
