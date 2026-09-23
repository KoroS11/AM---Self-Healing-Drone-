"""
COMPREHENSIVE 300-TICK AUDIT OF ALL BENCHMARK & STRESS SCENARIOS
================================================================
Evaluates all 50 benchmark scenarios + all 40 stress scenarios (90 total)
for 300 full ticks under B-Prime + Lexicographic CBF-QP (alpha1=3.0, alpha2=1.5).
Logs per-scenario:
- snap_150 (t=149) vs snap_300 (t=299)
- k20_150 (t=130..149) vs k20_300 (t=280..299)
- First reconnect time
- Flags any horizon disagreements
"""
import os, sys, math, pickle, hashlib
import numpy as np
import pandas as pd
import scipy.stats as stats
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
from scratch.run_production_verification_battery import LexicographicCBFGNNPolicy


def run_300t_eval(actor, sc, alpha=(3.0, 1.5)):
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
    policy = LexicographicCBFGNNPolicy(sim, actor_gnn=actor, alpha1=alpha[0], alpha2=alpha[1], device="cpu")
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

    snap_150 = conn_history[149]
    snap_300 = conn_history[299]
    k20_150 = all(conn_history[130:150])
    k20_300 = all(conn_history[280:300])
    reconn_rel = (first_conn - last_fail_step) if first_conn is not None else None

    disagreement = (snap_150 != snap_300) or (k20_150 != k20_300)

    return {
        "desc": config_desc,
        "seed": sc["seed"],
        "num_drones": sc["num_drones"],
        "reconnect_rel": reconn_rel,
        "snap_150": snap_150,
        "snap_300": snap_300,
        "k20_150": k20_150,
        "k20_300": k20_300,
        "drops_after": drops_after,
        "disagreement": disagreement
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
    print("      COMPREHENSIVE 300-TICK FULL-HORIZON AUDIT ACROSS ALL SCENARIOS")
    print("=" * 100)

    # 1. 50-Scenario Benchmark Gate
    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank = pickle.load(f)
    seen_35 = bank["seen_35"]
    held_15 = bank["held_out_15"]
    all_50 = seen_35 + held_15

    print("\n[PART A: 50-SCENARIO BENCHMARK GATE AT 300 TICKS]")
    res_50 = []
    for idx, sc in enumerate(all_50):
        r = run_300t_eval(actor_bp, sc, alpha=(3.0, 1.5))
        res_50.append(r)
        if r["disagreement"]:
            print(f"  [DISAGREEMENT] Seed {sc['seed']}: snap150={r['snap_150']} vs snap300={r['snap_300']}, k20_150={r['k20_150']} vs k20_300={r['k20_300']}")

    df_50 = pd.DataFrame(res_50)
    snap150_ok_50 = df_50["snap_150"].sum()
    snap300_ok_50 = df_50["snap_300"].sum()
    k20_150_ok_50 = df_50["k20_150"].sum()
    k20_300_ok_50 = df_50["k20_300"].sum()
    disagree_50 = df_50["disagreement"].sum()

    print(f"\n  50-Scenario Gate Summary:")
    print(f"    Single-Tick @ 150t:     {snap150_ok_50}/50 ({snap150_ok_50/50*100:.1f}%)")
    print(f"    Single-Tick @ 300t:     {snap300_ok_50}/50 ({snap300_ok_50/50*100:.1f}%)")
    print(f"    Sustained K=20 @ 150t:  {k20_150_ok_50}/50 ({k20_150_ok_50/50*100:.1f}%)")
    print(f"    Sustained K=20 @ 300t:  {k20_300_ok_50}/50 ({k20_300_ok_50/50*100:.1f}%)")
    print(f"    Horizon Disagreements:  {disagree_50}/50")

    # 2. All Stress Categories
    all_scenarios = build_all_scenarios()
    cat_order = [
        "1. Simultaneous Double Failure", "2. Cascading Failure",
        "3. Scale Test (Large Swarm)", "4. Non-Collinear Corridor",
        "5. Failure Timing Extremes", "6. Tight-Margin Geometry",
        "7. Triple Simultaneous Failure", "8. Adversarial Cut-Vertex",
        "9. Compound Stressors", "11. Comm-Range Heterogeneity"
    ]

    print("\n" + "=" * 100)
    print("[PART B: ALL STRESS CATEGORIES AT 300 TICKS]")
    print("=" * 100)

    all_stress_records = []
    total_stress_disagreements = 0

    for cat in cat_order:
        scs = all_scenarios[cat]
        n_sc = len(scs)
        print(f"\n--- Category: {cat} ({n_sc} scenarios) ---")
        
        cat_res = []
        for sc in scs:
            sc_with_cat = dict(sc)
            sc_with_cat["category"] = cat
            r = run_300t_eval(actor_bp, sc_with_cat, alpha=(3.0, 1.5))
            r["category"] = cat
            cat_res.append(r)
            all_stress_records.append(r)

            flag = " [HORIZON DISAGREEMENT]" if r["disagreement"] else ""
            print(f"  {r['desc']}")
            print(f"    Snap150={r['snap_150']}, Snap300={r['snap_300']} | K20@150={r['k20_150']}, K20@300={r['k20_300']} | Time={r['reconnect_rel']}t, Drops={r['drops_after']}{flag}")

        df_c = pd.DataFrame(cat_res)
        s150 = df_c["snap_150"].sum()
        s300 = df_c["snap_300"].sum()
        k150 = df_c["k20_150"].sum()
        k300 = df_c["k20_300"].sum()
        dis = df_c["disagreement"].sum()
        total_stress_disagreements += dis
        print(f"  --> Cat Summary: Snap150={s150}/{n_sc}, Snap300={s300}/{n_sc} | K20@150={k150}/{n_sc}, K20@300={k300}/{n_sc} | Disagreements={dis}")

    print("\n" + "=" * 100)
    print("      300-TICK FULL-HORIZON AUDIT SUMMARY")
    print("=" * 100)
    print(f"  Total Scenarios Evaluated:     90 (50 Benchmark + 40 Stress)")
    print(f"  Benchmark Gate Disagreements:  {disagree_50} / 50")
    print(f"  Stress Category Disagreements: {total_stress_disagreements} / 40")
    print("=" * 100)

    # Save to CSV
    df_all_300t = pd.DataFrame(res_50 + all_stress_records)
    df_all_300t.to_csv("scratch/audit_300t_all_scenarios.csv", index=False)
    print("Saved complete 300-tick audit to scratch/audit_300t_all_scenarios.csv")


if __name__ == "__main__":
    main()
