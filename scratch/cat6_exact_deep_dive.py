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
from scratch.run_production_verification_battery import LexicographicCBFGNNPolicy


def run_cat6_deep_dive():
    device = "cpu"
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load(bp_path, map_location=device, weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    all_scenarios = build_all_scenarios()
    cat6_scenarios = all_scenarios["6. Tight-Margin Geometry"]

    traces = []
    summary_list = []

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
        policy = LexicographicCBFGNNPolicy(sim, actor_gnn=actor_bp, alpha1=3.0, alpha2=1.5, device="cpu")
        topo = SwarmTopologyManager(sim)

        failures = list(sc.get("failures", []))
        last_fail_step = max([f["step"] for f in failures]) if failures else 0

        first_conn = None
        drops_after = 0
        prev_c = False
        conn_history = []
        drop_ticks = []

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
            conn_history.append(is_conn)

            if t >= last_fail_step:
                if is_conn:
                    if first_conn is None:
                        first_conn = t
                    prev_c = True
                else:
                    if prev_c:
                        drops_after += 1
                        drop_ticks.append(t)
                    prev_c = False

            # Active agent chain ordering along X
            active_mask = (sim.statuses == AgentStatus.ACTIVE)
            active_indices = np.where(active_mask)[0]
            sorted_by_x = sorted(active_indices, key=lambda a: sim.positions[a, 0])

            chain_dists = []
            chain_links = []
            for k in range(len(sorted_by_x) - 1):
                u = sorted_by_x[k]
                v = sorted_by_x[k+1]
                d = float(np.linalg.norm(sim.positions[u] - sim.positions[v]))
                chain_dists.append(d)
                chain_links.append((u, v))

            # Find tightest link in chain
            max_chain_idx = int(np.argmax(chain_dists)) if chain_dists else 0
            max_chain_link = chain_links[max_chain_idx] if chain_links else None
            max_chain_d = chain_dists[max_chain_idx] if chain_dists else 0.0
            min_chain_slack = 28.0 - max_chain_d

            # Max agent speed
            mobile_mask = active_mask & (sim.max_speeds > 0.0)
            mobile_indices = np.where(mobile_mask)[0]
            speeds = np.linalg.norm(sim.velocities[mobile_indices], axis=1) if len(mobile_indices) > 0 else [0.0]
            max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

            # Store per-tick trace
            traces.append({
                "scenario_idx": sc_idx + 1,
                "desc": sc["desc"],
                "seed": sc["seed"],
                "d_ab": sc.get("endpoint_b_pos", (200,0))[0],
                "fail_id": failures[0]["agent_id"] if failures else None,
                "tick": t,
                "connected": is_conn,
                "max_speed": max_speed,
                "max_chain_link": f"{max_chain_link[0]}-{max_chain_link[1]}" if max_chain_link else "",
                "max_chain_d": max_chain_d,
                "min_chain_slack": min_chain_slack,
                "chain_str": "; ".join([f"{u}-{v}:{d:.2f}" for (u,v), d in zip(chain_links, chain_dists)]),
                "cbf_active_total": policy.cbf_active_count,
                "qp_infeasible_total": policy.qp_infeasible_count
            })

        snap_150 = conn_history[149]
        snap_300 = conn_history[299]
        k20_150 = all(conn_history[130:150])
        k20_300 = all(conn_history[280:300])

        summary_list.append({
            "scenario_idx": sc_idx + 1,
            "desc": sc["desc"],
            "seed": sc["seed"],
            "d_ab": sc.get("endpoint_b_pos", (200,0))[0],
            "fail_id": failures[0]["agent_id"] if failures else None,
            "first_reconn": first_conn,
            "drops_count": drops_after,
            "drop_ticks": drop_ticks,
            "snap_150": snap_150,
            "snap_300": snap_300,
            "k20_150": k20_150,
            "k20_300": k20_300,
            "qp_infeasible_final": policy.qp_infeasible_count
        })

    df_traces = pd.DataFrame(traces)
    df_traces.to_csv("scratch/cat6_exact_traces.csv", index=False)
    
    df_summary = pd.DataFrame(summary_list)
    print("\n" + "=" * 100)
    print("CATEGORY 6 EXACT SIMULATION SUMMARY (PRODUCTION CBF-QP)")
    print("=" * 100)
    print(df_summary.to_string(index=False))

    # Detailed inspection of ticks 200..300
    print("\n" + "=" * 100)
    print("DETAILED COMPARISON TICKS 200-300 ACROSS ALL 4 SCENARIOS")
    print("=" * 100)
    
    for t in range(200, 301, 5):
        t_val = min(t, 299)
        row_str = f"Tick {t_val:3d} | "
        for sc_idx in range(1, 5):
            r = df_traces[(df_traces["scenario_idx"] == sc_idx) & (df_traces["tick"] == t_val)].iloc[0]
            row_str += f"Sc{sc_idx}({r['d_ab']}m): c={str(r['connected'])[0]}, max_link={r['max_chain_link']} ({r['max_chain_d']:.2f}m, slk={r['min_chain_slack']:+.2f}m), v={r['max_speed']:.2f} | "
        print(row_str)

    # Detailed inspection around drops for Scenario 3 and Scenario 4
    print("\n" + "=" * 100)
    print("DROP ANALYSIS FOR SCENARIOS 3 & 4")
    print("=" * 100)
    for sc_idx in [3, 4]:
        sc_desc = df_summary[df_summary["scenario_idx"] == sc_idx]["desc"].iloc[0]
        dticks = df_summary[df_summary["scenario_idx"] == sc_idx]["drop_ticks"].iloc[0]
        print(f"\n--- Scenario {sc_idx}: {sc_desc} (Drop Ticks: {dticks}) ---")
        if dticks:
            for drop_t in dticks:
                print(f"  Drop at t={drop_t}:")
                for sub_t in range(max(0, drop_t - 2), min(300, drop_t + 4)):
                    r = df_traces[(df_traces["scenario_idx"] == sc_idx) & (df_traces["tick"] == sub_t)].iloc[0]
                    print(f"    t={sub_t:3d} | Conn={str(r['connected']):<5s} | MaxHop={r['max_chain_link']} ({r['max_chain_d']:.3f}m, slack={r['min_chain_slack']:+.3f}m) | Speed={r['max_speed']:.3f}m/s | Infeasible={r['qp_infeasible_total']}")
                    print(f"           Chain: {r['chain_str']}")


if __name__ == "__main__":
    run_cat6_deep_dive()
