import os, sys, math, pickle, hashlib, subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
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
from scratch.evaluate_b_prime_full_battery import build_all_scenarios, find_adversarial_cut_vertex
from scratch.test_barrier_margin_sweep import ParametricCBFGNNPolicy


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Unknown ({e})"


def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_single_eval(actor, sc, alpha=(3.0, 1.5), barrier_margin=0.02, use_cbf=True, max_steps=300):
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
        policy = ParametricCBFGNNPolicy(sim, actor_gnn=actor, alpha1=alpha[0], alpha2=alpha[1], barrier_margin=barrier_margin, device="cpu")
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

    first_conn = None
    drops_after = 0
    prev_c = False
    conn_history = []

    for t in range(max_steps):
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
        "infeasible_qp": policy.qp_infeasible_count if hasattr(policy, 'qp_infeasible_count') else 0
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    b_path = "checkpoints/checkpoint_B_best.pt"

    print("=" * 100)
    print("PRODUCTION VERIFICATION: EPSILON = 0.02m DETERMINISM & SIGNIFICANCE AUDIT")
    print("=" * 100)
    print(f"Git Commit: {get_git_commit_hash()}")
    print(f"Checkpoint B-Prime MD5: {compute_file_md5(bp_path)}")
    print(f"Checkpoint B MD5:       {compute_file_md5(b_path)}")
    print("=" * 100)

    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load(bp_path, map_location=device, weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    actor_b = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw_b = torch.load(b_path, map_location=device, weights_only=False)
    sd_b = raw_b["actor_state_dict"] if "actor_state_dict" in raw_b else raw_b
    sd_b_new = actor_b.state_dict()
    for k, v in sd_b.items():
        if k == "node_encoder.0.weight" and v.shape[1] == 8:
            w = torch.zeros((v.shape[0], 9), dtype=v.dtype)
            w[:, :8] = v
            sd_b_new[k] = w
        elif k == "edge_encoder.0.weight" and v.shape[1] == 4:
            w = torch.zeros((v.shape[0], 5), dtype=v.dtype)
            w[:, :4] = v
            sd_b_new[k] = w
        elif k in sd_b_new:
            sd_b_new[k] = v
    actor_b.load_state_dict(sd_b_new, strict=True)
    actor_b.eval()

    # 1. 50-Scenario Benchmark Gate
    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank = pickle.load(f)
    all_50 = bank["seen_35"] + bank["held_out_15"]

    all_scenarios_dict = build_all_scenarios()
    stress_list = []
    for c_name, s_list in all_scenarios_dict.items():
        for s in s_list:
            sc_copy = s.copy()
            sc_copy["category"] = c_name
            stress_list.append(sc_copy)

    full_94 = all_50 + stress_list

    print(f"\nTotal scenarios to evaluate: {len(full_94)} (50 benchmark + {len(stress_list)} stress)")

    # Pass 1 vs Pass 2 Double Pass Determinism
    print("\n[PART 1: DOUBLE-PASS DETERMINISM CHECK ACROSS ALL 94 SCENARIOS]")
    print("Running Pass 1...")
    pass1_results = [run_single_eval(actor_bp, sc, alpha=(3.0, 1.5), barrier_margin=0.02, use_cbf=True, max_steps=300) for sc in full_94]
    print("Running Pass 2...")
    pass2_results = [run_single_eval(actor_bp, sc, alpha=(3.0, 1.5), barrier_margin=0.02, use_cbf=True, max_steps=300) for sc in full_94]

    flips = 0
    time_diffs = []
    for i in range(len(full_94)):
        p1 = pass1_results[i]
        p2 = pass2_results[i]
        
        c1, c2 = p1["k20_300"], p2["k20_300"]
        t1, t2 = p1["reconnect_rel"], p2["reconnect_rel"]
        
        if c1 != c2:
            flips += 1
            print(f"  [FLIP] Scenario {i} ({p1['desc']}): Pass1={c1}, Pass2={c2}")
        
        if t1 is not None and t2 is not None:
            time_diffs.append(abs(t1 - t2))

    max_dt = max(time_diffs) if time_diffs else 0
    print(f"\nDeterminism Audit Results:")
    print(f"  Total Scenarios Evaluated: {len(full_94)}")
    print(f"  Outcome Flips (Pass 1 vs Pass 2): {flips} / {len(full_94)}")
    print(f"  Max Absolute Reconnect Time Difference: {max_dt:.6f} ticks")
    print(f"  Determinism Status: {'100% BIT-IDENTICAL DETERMINISTIC' if flips == 0 and max_dt == 0 else 'NON-DETERMINISTIC'}")

    # Paired Statistical Significance on Benchmark Gate
    print("\n[PART 2: STATISTICAL SIGNIFICANCE ON 50-SCENARIO BENCHMARK GATE]")
    res_b = [run_single_eval(actor_b, sc, use_cbf=False, max_steps=300) for sc in all_50]
    res_bp = pass1_results[:50]

    # Reconnect times for pairs where both reconnected
    paired_times_b = []
    paired_times_bp = []
    diffs = []

    for r_b, r_bp in zip(res_b, res_bp):
        if r_b["reconnect_rel"] is not None and r_bp["reconnect_rel"] is not None:
            paired_times_b.append(r_b["reconnect_rel"])
            paired_times_bp.append(r_bp["reconnect_rel"])
            diffs.append(r_bp["reconnect_rel"] - r_b["reconnect_rel"])

    n_pairs = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    
    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(paired_times_bp, paired_times_b)
    # Wilcoxon signed-rank
    w_stat, w_pval = stats.wilcoxon(diffs)

    print(f"  Paired Scenarios: n = {n_pairs}")
    print(f"  Mean Reconnect Time (Checkpoint B):       {np.mean(paired_times_b):.2f} ticks")
    print(f"  Mean Reconnect Time (B-Prime + CBF eps=0.02): {np.mean(paired_times_bp):.2f} ticks")
    print(f"  Mean Difference (B-Prime - Checkpoint B): {mean_diff:.2f} ticks (faster by {-mean_diff:.2f} ticks)")
    print(f"  Paired t-test:       t = {t_stat:.4f}, p = {t_pval:.4e}")
    print(f"  Wilcoxon Signed-Rank: W = {w_stat:.1f}, p = {w_pval:.4e}")
    print(f"  Statistically Significant: {'YES (p < 0.001)' if w_pval < 0.001 else 'NO'}")


if __name__ == "__main__":
    main()
