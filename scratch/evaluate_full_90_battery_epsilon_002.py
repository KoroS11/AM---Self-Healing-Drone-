import os, sys, math, pickle, hashlib
import numpy as np
import pandas as pd
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
from scratch.evaluate_b_prime_full_battery import build_all_scenarios, find_adversarial_cut_vertex
from scratch.test_barrier_margin_sweep import ParametricCBFGNNPolicy


def run_300t_eval(actor, sc, alpha=(3.0, 1.5), barrier_margin=0.02):
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
    policy = ParametricCBFGNNPolicy(sim, actor_gnn=actor, alpha1=alpha[0], alpha2=alpha[1], barrier_margin=barrier_margin, device="cpu")
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

    first_conn = None
    drops_after = 0
    prev_c = False
    conn_history = []

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
                prev_c = False

    G_final = topo.build_graph()
    snap_150 = conn_history[149]
    snap_300 = conn_history[299]
    k20_150 = all(conn_history[130:150])
    k20_300 = all(conn_history[280:300])
    reconn_rel = (first_conn - last_fail_step) if first_conn is not None else None
    mean_rate = channel.compute_network_throughput(sim, G_final) if snap_300 else 0.0

    return {
        "desc": config_desc,
        "seed": sc["seed"],
        "num_drones": sc["num_drones"],
        "reconnect_rel": reconn_rel,
        "first_conn": first_conn,
        "snap_150": snap_150,
        "snap_300": snap_300,
        "k20_150": k20_150,
        "k20_300": k20_300,
        "drops_after": drops_after,
        "sum_rate": mean_rate,
        "infeasible_qp": policy.qp_infeasible_count
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"

    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load(bp_path, map_location=device, weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    print("=" * 100)
    print("FULL 90-SCENARIO 300-TICK BATTERY AUDIT: EPSILON = 0.02m")
    print("=" * 100)

    # 1. 50-Scenario Benchmark Gate
    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank = pickle.load(f)
    seen_35 = bank["seen_35"]
    held_15 = bank["held_out_15"]
    all_50 = seen_35 + held_15

    print("\n[PART A: 50-SCENARIO BENCHMARK GATE (300 TICKS)]")
    res_50 = [run_300t_eval(actor_bp, sc, barrier_margin=0.02) for sc in all_50]
    df_50 = pd.DataFrame(res_50)

    snap150_ok = df_50["snap_150"].sum()
    snap300_ok = df_50["snap_300"].sum()
    k20_150_ok = df_50["k20_150"].sum()
    k20_300_ok = df_50["k20_300"].sum()
    reconn_times = df_50["reconnect_rel"].dropna().tolist()
    mean_reconn = np.mean(reconn_times) if reconn_times else 0.0
    mean_rate = df_50["sum_rate"].mean()

    print(f"  Single-Tick @ 150t:     {snap150_ok}/50 ({snap150_ok/50*100:.1f}%)")
    print(f"  Single-Tick @ 300t:     {snap300_ok}/50 ({snap300_ok/50*100:.1f}%)")
    print(f"  Sustained K=20 @ 150t:  {k20_150_ok}/50 ({k20_150_ok/50*100:.1f}%)")
    print(f"  Sustained K=20 @ 300t:  {k20_300_ok}/50 ({k20_300_ok/50*100:.1f}%)")
    print(f"  Mean Reconnect Time:    {mean_reconn:.2f} ticks")
    print(f"  Mean Network Sum-Rate:  {mean_rate:.2f} Mbps")

    # 2. All Stress Categories
    all_scenarios = build_all_scenarios()
    cat_order = [
        "1. Simultaneous Double Failure", "2. Cascading Failure",
        "3. Scale Test (Large Swarm)", "4. Non-Collinear Corridor",
        "5. Failure Timing Extremes", "6. Tight-Margin Geometry",
        "7. Triple Simultaneous Failure", "8. Adversarial Cut-Vertex",
        "9. Compound Stressors", "11. Comm-Range Heterogeneity"
    ]

    print("\n[PART B: ALL STRESS CATEGORIES (300 TICKS, EPSILON=0.02m)]")
    stress_records = []
    
    for cat_name in cat_order:
        sc_list = all_scenarios[cat_name]
        cat_res = [run_300t_eval(actor_bp, sc, barrier_margin=0.02) for sc in sc_list]
        for r in cat_res:
            r["category"] = cat_name
            stress_records.append(r)

        succ_150 = sum(1 for r in cat_res if r["snap_150"])
        succ_300 = sum(1 for r in cat_res if r["snap_300"])
        k20_150 = sum(1 for r in cat_res if r["k20_150"])
        k20_300 = sum(1 for r in cat_res if r["k20_300"])
        total = len(cat_res)
        drops = sum(r["drops_after"] for r in cat_res)
        infeas = sum(r["infeasible_qp"] for r in cat_res)

        print(f"  {cat_name:<35s} | 300t K=20: {k20_300}/{total} ({k20_300/total*100:5.1f}%) | Snap300: {succ_300}/{total} | Drops: {drops} | Infeas: {infeas}")

    df_stress = pd.DataFrame(stress_records)
    df_all = pd.concat([df_50.assign(category="Benchmark Gate"), df_stress], ignore_index=True)
    df_all.to_csv("scratch/full_90_battery_eps002_results.csv", index=False)
    print("\nSaved full results to scratch/full_90_battery_eps002_results.csv")


if __name__ == "__main__":
    main()
