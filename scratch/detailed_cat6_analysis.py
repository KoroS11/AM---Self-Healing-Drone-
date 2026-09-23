import os
import sys
import math
import numpy as np
import scipy.sparse as sp
import torch
import qpsolvers
import pandas as pd

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


class DiagnosticCBFGNNPolicy(GNNPolicy):
    def __init__(
        self,
        simulator: SwarmSimulator,
        actor_gnn: ActorGNN,
        alpha1: float = 3.0,
        alpha2: float = 1.5,
        barrier_margin: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__(simulator, actor_gnn=actor_gnn, device=device)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.barrier_margin = barrier_margin
        self.cbf_active_count = 0
        self.qp_infeasible_count = 0
        self.last_solver_status = None
        self.last_qp_info = {}

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
        edge_details = []

        for u_node, v_node in edges:
            rc_eff = min(float(sim.comm_ranges[u_node]), float(sim.comm_ranges[v_node]))
            r_safe = max(rc_eff - self.barrier_margin, 1.0)
            
            p_ij = pos[u_node] - pos[v_node]
            v_ij = vel[u_node] - vel[v_node]
            dist_sq = float(np.dot(p_ij, p_ij))
            dist = math.sqrt(dist_sq)
            
            h_ij = r_safe**2 - dist_sq
            h_dot = -2.0 * float(np.dot(p_ij, v_ij))

            psi = h_dot + self.alpha2 * h_ij
            rhs = -2.0 * float(np.dot(v_ij, v_ij)) + self.alpha1 * psi

            row = np.zeros(2 * M, dtype=np.float64)
            if u_node in idx_map:
                k_u = idx_map[u_node]
                row[2 * k_u : 2 * k_u + 2] += 2.0 * p_ij
            if v_node in idx_map:
                k_v = idx_map[v_node]
                row[2 * k_v : 2 * k_v + 2] -= 2.0 * p_ij

            A_rows.append(row)
            b_vals.append(rhs)
            edge_details.append({
                "edge": (u_node, v_node),
                "dist": dist,
                "rc_eff": rc_eff,
                "slack": rc_eff - dist,
                "h_ij": h_ij,
                "h_dot": h_dot,
                "psi": psi,
                "rhs": rhs
            })

        if len(A_rows) > 0:
            G_mat = sp.csc_matrix(np.array(A_rows))
            h_vec = np.array(b_vals)
            try:
                res = qpsolvers.solve_qp(P, q, G_mat, h_vec, solver="osqp", verbose=False)
                sol = res
                self.last_solver_status = "SOLVED" if sol is not None else "INFEASIBLE_NULL"
            except Exception as e:
                sol = None
                self.last_solver_status = f"EXCEPTION_{type(e).__name__}"
        else:
            sol = None
            self.last_solver_status = "NO_CONSTRAINTS"

        self.last_qp_info = {
            "num_constraints": len(A_rows),
            "status": self.last_solver_status,
            "edges": edge_details
        }

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
            self.last_solver_status = "DISCONNECTED_BYPASS"
        else:
            final_acc = self.apply_cbf_qp(raw_world_acc)
            
        if return_pre_clamp:
            mobile_mask_t = (self.sim.statuses == AgentStatus.ACTIVE) & (self.sim.max_speeds > 0.0)
            pulled_this_tick = {i: True for i in np.where(mobile_mask_t)[0]}
            return final_acc, raw_world_acc, pulled_this_tick
        return final_acc


def analyze_all_cat6():
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load(bp_path, map_location="cpu", weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    all_scenarios = build_all_scenarios()
    cat6_scenarios = all_scenarios["6. Tight-Margin Geometry"]

    records = []
    
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
        policy = DiagnosticCBFGNNPolicy(sim, actor_gnn=actor_bp, alpha1=3.0, alpha2=1.5, device="cpu")
        topo = SwarmTopologyManager(sim)

        failures = list(sc.get("failures", []))
        last_fail_step = max([f["step"] for f in failures]) if failures else 0

        for t in range(300):
            for f in failures:
                if t == f["step"]:
                    injector.fail_agent(f["agent_id"])

            acc = policy.compute_control_forces()
            solver_stat = policy.last_solver_status
            
            # Record state before sim step or after? Let's record sim step
            sim.step(accelerations=acc)

            G = topo.build_graph()
            g_a_idx = 0
            g_b_idx = 1
            for a_i, role in enumerate(sim.roles):
                if role == AgentRole.GROUND_A or str(role) == "ground_a":
                    g_a_idx = a_i
                elif role == AgentRole.GROUND_B or str(role) == "ground_b":
                    g_b_idx = a_i

            is_conn = topo.has_path(g_a_idx, g_b_idx, graph=G)

            # Sort active agents along x-axis
            active_mask = (sim.statuses == AgentStatus.ACTIVE)
            active_indices = np.where(active_mask)[0]
            sorted_by_x = sorted(active_indices, key=lambda a: sim.positions[a, 0])
            
            # Find chain hops along corridor
            chain_dists = []
            chain_links = []
            for k in range(len(sorted_by_x) - 1):
                u = sorted_by_x[k]
                v = sorted_by_x[k+1]
                d = np.linalg.norm(sim.positions[u] - sim.positions[v])
                chain_dists.append(d)
                chain_links.append((u, v))

            # Pairwise min slack
            pos = sim.positions
            min_slack = 999.0
            min_slack_pair = None
            for i_idx in range(len(active_indices)):
                for j_idx in range(i_idx + 1, len(active_indices)):
                    u = active_indices[i_idx]
                    v = active_indices[j_idx]
                    d_uv = np.linalg.norm(pos[u] - pos[v])
                    rc_eff = min(sim.comm_ranges[u], sim.comm_ranges[v])
                    slack = rc_eff - d_uv
                    if slack < min_slack:
                        min_slack = slack
                        min_slack_pair = (u, v)

            mobile_mask = active_mask & (sim.max_speeds > 0.0)
            mobile_indices = np.where(mobile_mask)[0]
            speeds = np.linalg.norm(sim.velocities[mobile_indices], axis=1) if len(mobile_indices) > 0 else [0.0]
            max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

            records.append({
                "scenario_idx": sc_idx + 1,
                "desc": sc["desc"],
                "seed": sc["seed"],
                "d_ab": sc.get("endpoint_b_pos", (200,0))[0],
                "fail_id": failures[0]["agent_id"] if failures else None,
                "tick": t,
                "connected": is_conn,
                "solver_status": solver_stat,
                "min_slack": min_slack,
                "min_slack_u": min_slack_pair[0] if min_slack_pair else None,
                "min_slack_v": min_slack_pair[1] if min_slack_pair else None,
                "max_speed": max_speed,
                "chain_order": str(sorted_by_x),
                "max_chain_dist": max(chain_dists) if chain_dists else 0.0,
                "max_chain_link": str(chain_links[np.argmax(chain_dists)]) if chain_dists else "",
                "chain_dists_str": ";".join([f"{u}-{v}:{d:.2f}" for (u,v), d in zip(chain_links, chain_dists)])
            })

    df = pd.DataFrame(records)
    df.to_csv("scratch/cat6_detailed_traces.csv", index=False)
    print("Saved scratch/cat6_detailed_traces.csv")

    # Now print analysis
    for sc_idx in range(1, 5):
        sdf = df[df["scenario_idx"] == sc_idx]
        sc_desc = sdf["desc"].iloc[0]
        reconn_ticks = sdf[sdf["connected"] & (sdf["tick"] >= 10)]["tick"].tolist()
        first_reconn = reconn_ticks[0] if reconn_ticks else None
        
        # drops after first reconn
        drops = 0
        prev_c = False
        drop_ticks = []
        for _, row in sdf[sdf["tick"] >= 10].iterrows():
            c = row["connected"]
            if c:
                prev_c = True
            else:
                if prev_c:
                    drops += 1
                    drop_ticks.append(row["tick"])
        
        k20 = all(sdf[sdf["tick"].between(280, 299)]["connected"])
        c300 = sdf[sdf["tick"] == 299]["connected"].iloc[0]
        
        print(f"\n=======================================================")
        print(f"SCENARIO {sc_idx}: {sc_desc}")
        print(f"First Reconnect: t={first_reconn}, Drops: {drops} (at {drop_ticks}), Conn@300: {c300}, K20 Sustained: {k20}")
        print(f"=======================================================")
        
        # Inspect t=200 to t=300
        print(f"Sampled t=200..300 (every 10 ticks + drops):")
        interesting_ticks = sorted(list(set(list(range(200, 300, 10)) + drop_ticks + [299])))
        interesting_ticks = [t for t in interesting_ticks if 200 <= t <= 299]
        for t in interesting_ticks:
            r = sdf[sdf["tick"] == t].iloc[0]
            print(f"t={t:3d} | Conn={str(r['connected']):<5s} | Stat={r['solver_status']:<18s} | Speed={r['max_speed']:.3f}m/s | MinSlack={r['min_slack']:.2f}m ({r['min_slack_u']}-{r['min_slack_v']}) | MaxChainHop={r['max_chain_link']} ({r['max_chain_dist']:.2f}m)")
            if not r['connected']:
                print(f"      Chain breakdown: {r['chain_dists_str']}")


if __name__ == "__main__":
    analyze_all_cat6()
