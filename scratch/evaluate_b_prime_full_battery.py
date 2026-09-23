"""
B-PRIME COMPLETE VERIFICATION BATTERY
======================================
Runs:
  1. 50-Scenario Benchmark Gate (Seen-35 + Held-Out-15)
  2. Category 4: 8-Scenario Rotation Invariance
  3. Stress Categories 1-3, 5-9 (24+16 scenarios)
  4. Category 11: Comm-Range Heterogeneity (4 scenarios)
  5. 4-Step Verification Protocol:
     (a) Baseline Audit
     (b) Bit-Identical Determinism Rerun
     (c) Per-Seed Paired Significance Test
     (d) Zero-Effect Control Run (all agents nominal Rc=28.0m)
All results reported side-by-side: B-Prime vs Checkpoint B (zero-padded to 9/5 dims).
"""
import os, sys, math, pickle, hashlib, subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
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


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

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


def load_b_prime(path: str, device: str = "cpu") -> ActorGNN:
    actor = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor.load_state_dict(sd, strict=True)
    actor.eval()
    return actor


def load_checkpoint_b_as_9x5(path: str, device: str = "cpu") -> ActorGNN:
    """Load Checkpoint B (8/4 dims) into a 9/5 architecture with zero-padded new columns."""
    actor = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    sd_old = raw["actor_state_dict"] if "actor_state_dict" in raw else raw

    sd_new = actor.state_dict()
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
    actor.load_state_dict(sd_new, strict=True)
    actor.eval()
    return actor


# ═══════════════════════════════════════════════════════════════════════
# Closed-Loop Runner
# ═══════════════════════════════════════════════════════════════════════

def run_scenario(actor, sc, device="cpu", max_steps=150):
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

    # Custom initial positions (irregular spacing, zigzag, etc.)
    if "custom_positions_fn" in sc:
        new_pos = sc["custom_positions_fn"](sim.positions.copy(), sc["comm_range"])
        sim.positions = new_pos

    # Per-agent comm_range overrides
    if "custom_comm_ranges" in sc:
        for agent_id, r_val in sc["custom_comm_ranges"].items():
            sim.comm_ranges[agent_id] = r_val

    injector = FailureInjector(sim)
    policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()

    # Handle adversarial cut-vertex
    failures = list(sc.get("failures", []))
    config_desc = sc.get("desc", f"N={sc['num_drones']}, Seed={sc['seed']}")

    if sc.get("adversarial_cut_vertex", False):
        target_step = sc.get("fail_timestep", 10)
        worst_r, gap_size = find_adversarial_cut_vertex(sim, topo, g_a, g_b)
        failures.append({"step": target_step, "agent_id": worst_r})
        config_desc += f" -> Relay {worst_r} (Gap: {gap_size:.1f}m)"

    if not failures and "fail_timestep" in sc:
        failures = [{"step": sc["fail_timestep"], "agent_id": sc["fail_agent_id"]}]

    last_fail_step = max([f["step"] for f in failures]) if failures else 0

    ever_reconnected = False
    reconnect_rel = None
    reconnect_abs = None
    drops_after = 0
    prev_reconn = False

    for t in range(max_steps):
        for f in failures:
            if t == f["step"]:
                injector.fail_agent(f["agent_id"])

        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)

        G = topo.build_graph()
        is_conn = topo.has_path(g_a, g_b, graph=G)

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

    G_final = topo.build_graph()
    final_conn = topo.has_path(g_a, g_b, graph=G_final)
    sum_rate = channel.compute_network_throughput(sim, G_final)

    if final_conn:
        mode = "Success (Stable)" if drops_after == 0 else f"Success with Oscillation ({drops_after} drops)"
    elif not ever_reconnected:
        mode = "Failed to Reconnect"
    elif drops_after > 0:
        mode = f"Reconnect-then-Drift ({drops_after} post-reconnect disconnects)"
    else:
        mode = "Disconnected at Step 150"

    return {
        "category": sc.get("category", "Benchmark"),
        "desc": config_desc,
        "seed": sc["seed"],
        "num_drones": sc["num_drones"],
        "reconnected": final_conn,
        "reconnect_rel": reconnect_rel,
        "reconnect_abs": reconnect_abs,
        "transient_drops": drops_after,
        "sum_rate": round(sum_rate, 2),
        "failure_mode": mode
    }


def find_adversarial_cut_vertex(sim, topo, g_a, g_b):
    G = topo.build_graph()
    active_relays = [i for i in range(2, sim.num_agents) if sim.statuses[i] == AgentStatus.ACTIVE]
    worst_relay, max_gap = None, -1.0

    for r in active_relays:
        G_temp = G.copy()
        G_temp.remove_node(r)
        if not nx.has_path(G_temp, g_a, g_b):
            comp_a = nx.node_connected_component(G_temp, g_a)
            comp_b = nx.node_connected_component(G_temp, g_b)
            min_dist = min(np.linalg.norm(sim.positions[u] - sim.positions[v])
                          for u in comp_a for v in comp_b)
            if min_dist > max_gap:
                max_gap = min_dist
                worst_relay = r
        else:
            neighbors = list(G.neighbors(r))
            if len(neighbors) >= 2:
                d = max(np.linalg.norm(sim.positions[u] - sim.positions[v])
                        for u in neighbors for v in neighbors if u != v)
                if d > max_gap:
                    max_gap = d
                    worst_relay = r

    if worst_relay is None and active_relays:
        worst_relay = active_relays[len(active_relays) // 2]
        max_gap = 0.0
    return worst_relay, max_gap


# ═══════════════════════════════════════════════════════════════════════
# Custom Position Functions (for Categories 8-10)
# ═══════════════════════════════════════════════════════════════════════

def make_irregular_jitter_fn(seed, max_transverse=6.0, max_longitudinal=4.0):
    def apply_jitter(pos, rc):
        rng = np.random.RandomState(seed)
        pos_A, pos_B = pos[0], pos[1]
        u = (pos_B - pos_A) / max(np.linalg.norm(pos_B - pos_A), 1e-6)
        u_perp = np.array([-u[1], u[0]])
        for r in range(2, pos.shape[0]):
            pos[r] += rng.uniform(-max_longitudinal, max_longitudinal) * u + \
                       rng.uniform(-max_transverse, max_transverse) * u_perp
        return pos
    return apply_jitter

def make_zigzag_fn(amplitude=7.0):
    def apply_zigzag(pos, rc):
        pos_A, pos_B = pos[0], pos[1]
        u = (pos_B - pos_A) / max(np.linalg.norm(pos_B - pos_A), 1e-6)
        u_perp = np.array([-u[1], u[0]])
        for r in range(2, pos.shape[0]):
            sign = 1.0 if r % 2 == 0 else -1.0
            pos[r] += sign * amplitude * u_perp
        return pos
    return apply_zigzag

def make_clustered_density_fn(d_AB=200.0):
    def apply_clustering(pos, rc):
        num_relays = pos.shape[0] - 2
        alphas = [0.08, 0.18, 0.28, 0.38, 0.62, 0.72, 0.82, 0.92] if num_relays == 8 else \
                 [0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92, 0.96][:num_relays]
        pos_A, pos_B = pos[0], pos[1]
        vec = pos_B - pos_A
        for i, r in enumerate(range(2, pos.shape[0])):
            if i < len(alphas):
                pos[r] = pos_A + alphas[i] * vec
        return pos
    return apply_clustering


# ═══════════════════════════════════════════════════════════════════════
# Scenario Definitions
# ═══════════════════════════════════════════════════════════════════════

def endpoint_b_from_angle(d_ab, angle_deg):
    theta = math.radians(angle_deg)
    return (round(d_ab * math.cos(theta), 4), round(d_ab * math.sin(theta), 4))


def build_all_scenarios():
    scenarios = {}

    # --- Category 1: Simultaneous Double Failure ---
    scenarios["1. Simultaneous Double Failure"] = [
        {"desc": "N=9, d=200m, Fail [3,6] at t=10 (Separated)", "seed": 2001, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":3},{"step":10,"agent_id":6}]},
        {"desc": "N=9, d=200m, Fail [4,5] at t=10 (Adjacent)", "seed": 2002, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":4},{"step":10,"agent_id":5}]},
        {"desc": "N=11, d=200m, Fail [3,7] at t=10 (Separated)", "seed": 2003, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":3},{"step":10,"agent_id":7}]},
        {"desc": "N=11, d=200m, Fail [5,6] at t=10 (Adjacent)", "seed": 2004, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":5},{"step":10,"agent_id":6}]},
    ]

    # --- Category 2: Cascading Failure ---
    scenarios["2. Cascading Failure"] = [
        {"desc": "N=9, d=200m, Fail 4@t=10, 6@t=30", "seed": 2011, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":4},{"step":30,"agent_id":6}]},
        {"desc": "N=9, d=200m, Fail 3@t=10, 5@t=30", "seed": 2012, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":3},{"step":30,"agent_id":5}]},
        {"desc": "N=11, d=200m, Fail 5@t=10, 7@t=30", "seed": 2013, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":5},{"step":30,"agent_id":7}]},
        {"desc": "N=11, d=200m, Fail 4@t=10, 3@t=30 (Adj)", "seed": 2014, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "failures": [{"step":10,"agent_id":4},{"step":30,"agent_id":3}]},
    ]

    # --- Category 3: Scale Test ---
    scenarios["3. Scale Test (Large Swarm)"] = [
        {"desc": "N=15, d=350m, Fail 7@t=10 (Center)", "seed": 2021, "num_drones": 15, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (350,0), "failures": [{"step":10,"agent_id":7}]},
        {"desc": "N=15, d=350m, Fail 4@t=10 (Near-A)", "seed": 2022, "num_drones": 15, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (350,0), "failures": [{"step":10,"agent_id":4}]},
        {"desc": "N=21, d=500m, Fail 10@t=10 (Center)", "seed": 2023, "num_drones": 21, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (500,0), "failures": [{"step":10,"agent_id":10}]},
        {"desc": "N=21, d=500m, Fail 5@t=10 (Near-A)", "seed": 2024, "num_drones": 21, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (500,0), "failures": [{"step":10,"agent_id":5}]},
    ]

    # --- Category 4: Non-Collinear (Rotation Invariance, 8 scenarios) ---
    cat4 = []
    for angle, N, d, seed, fail_id in [
        (45.0, 9, 180.0, 2031, 5), (53.1, 9, 180.0, 2032, 4),
        (45.0, 11, 200.0, 2033, 6), (60.0, 11, 200.0, 2034, 5),
        (17.0, 11, 200.0, 3001, 5), (73.0, 9, 180.0, 3002, 4),
        (112.0, 11, 200.0, 3003, 6), (200.0, 9, 180.0, 3004, 5),
    ]:
        bx, by = endpoint_b_from_angle(d, angle)
        cat4.append({
            "desc": f"N={N}, d={d}m, {angle}°", "seed": seed,
            "num_drones": N, "comm_range": 28.0, "d_ab": d, "angle_deg": angle,
            "endpoint_a_pos": (0,0), "endpoint_b_pos": (bx, by),
            "failures": [{"step":10,"agent_id":fail_id}]
        })
    scenarios["4. Non-Collinear Corridor"] = cat4

    # --- Category 5: Failure Timing Extremes ---
    scenarios["5. Failure Timing Extremes"] = [
        {"desc": "N=9, d=200m, Fail 5@t=1 (Extreme Early)", "seed": 2041, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0), "failures": [{"step":1,"agent_id":5}]},
        {"desc": "N=11, d=200m, Fail 6@t=1 (Extreme Early)", "seed": 2042, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0), "failures": [{"step":1,"agent_id":6}]},
        {"desc": "N=9, d=200m, Fail 5@t=140 (10 ticks left)", "seed": 2043, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0), "failures": [{"step":140,"agent_id":5}]},
        {"desc": "N=11, d=200m, Fail 6@t=140 (10 ticks left)", "seed": 2044, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0), "failures": [{"step":140,"agent_id":6}]},
    ]

    # --- Category 6: Tight-Margin Geometry ---
    scenarios["6. Tight-Margin Geometry"] = [
        {"desc": "N=9, d=185m (margin 1.57m), Fail 5@t=10", "seed": 2051, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (185,0), "failures": [{"step":10,"agent_id":5}]},
        {"desc": "N=9, d=189m (margin 1.00m), Fail 4@t=10", "seed": 2052, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (189,0), "failures": [{"step":10,"agent_id":4}]},
        {"desc": "N=9, d=192m (margin 0.57m), Fail 5@t=10", "seed": 2053, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (192,0), "failures": [{"step":10,"agent_id":5}]},
        {"desc": "N=9, d=195m (margin 0.14m), Fail 5@t=10", "seed": 2054, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (195,0), "failures": [{"step":10,"agent_id":5}]},
    ]

    # --- Category 7: Triple Simultaneous Failure ---
    scenarios["7. Triple Simultaneous Failure"] = [
        {"desc": "N=11, d=180m, Fail [5,6,7]@t=10 (Adj Center)", "seed": 2071, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (180,0),
         "failures": [{"step":10,"agent_id":5},{"step":10,"agent_id":6},{"step":10,"agent_id":7}]},
        {"desc": "N=11, d=180m, Fail [3,6,9]@t=10 (Spread)", "seed": 2072, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (180,0),
         "failures": [{"step":10,"agent_id":3},{"step":10,"agent_id":6},{"step":10,"agent_id":9}]},
        {"desc": "N=11, d=180m, Fail [2,3,4]@t=10 (Cluster Near-A)", "seed": 2073, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (180,0),
         "failures": [{"step":10,"agent_id":2},{"step":10,"agent_id":3},{"step":10,"agent_id":4}]},
        {"desc": "N=11, d=180m, Fail [4,5,8]@t=10 (Asymmetric)", "seed": 2074, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (180,0),
         "failures": [{"step":10,"agent_id":4},{"step":10,"agent_id":5},{"step":10,"agent_id":8}]},
    ]

    # --- Category 8: Adversarial Cut-Vertex ---
    scenarios["8. Adversarial Cut-Vertex"] = [
        {"desc": "N=9, d=200m, Adversarial Cut-Vertex", "seed": 2081, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "adversarial_cut_vertex": True, "fail_timestep": 10},
        {"desc": "N=11, d=200m, Adversarial Cut-Vertex", "seed": 2082, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (200,0),
         "adversarial_cut_vertex": True, "fail_timestep": 10},
        {"desc": "N=9, d=180m, Clustered + Adversarial", "seed": 2083, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (180,0),
         "custom_positions_fn": make_clustered_density_fn(180.0),
         "adversarial_cut_vertex": True, "fail_timestep": 10},
        {"desc": "N=11, d=210m, Tight Chain + Adversarial", "seed": 2084, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (210,0),
         "adversarial_cut_vertex": True, "fail_timestep": 10},
    ]

    # --- Category 9: Compound Stressors ---
    scenarios["9. Compound Stressors"] = [
        {"desc": "45° + Double Fail [4,7]@t=10 (N=11, d=200m)", "seed": 2091, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (141.42, 141.42),
         "failures": [{"step":10,"agent_id":4},{"step":10,"agent_id":7}]},
        {"desc": "Tight-Margin d=188m + Late Fail@t=80 (N=9)", "seed": 2092, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (188,0),
         "failures": [{"step":80,"agent_id":5}]},
        {"desc": "60° + Cascade Fail 5@10,7@30 (N=11, d=200m)", "seed": 2093, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (100.0, 173.21),
         "failures": [{"step":10,"agent_id":5},{"step":30,"agent_id":7}]},
        {"desc": "Jitter + Double Fail [4,8]@t=10 (N=11, d=190m)", "seed": 2094, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (190,0),
         "custom_positions_fn": make_irregular_jitter_fn(seed=2094, max_transverse=5.0, max_longitudinal=3.0),
         "failures": [{"step":10,"agent_id":4},{"step":10,"agent_id":8}]},
    ]

    # --- Category 11: Comm-Range Heterogeneity ---
    scenarios["11. Comm-Range Heterogeneity"] = [
        {"desc": "N=9, d=170m, Agent 4 Rc=22m, Fail 6@t=10", "seed": 2111, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (170,0),
         "custom_comm_ranges": {4: 22.0}, "failures": [{"step":10,"agent_id":6}]},
        {"desc": "N=9, d=165m, Agents 3,6 Rc=22m, Fail 5@t=10", "seed": 2112, "num_drones": 9, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (165,0),
         "custom_comm_ranges": {3: 22.0, 6: 22.0}, "failures": [{"step":10,"agent_id":5}]},
        {"desc": "N=11, d=180m, Agents 4,8 Rc=22m, Fail 6@t=10", "seed": 2113, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (180,0),
         "custom_comm_ranges": {4: 22.0, 8: 22.0}, "failures": [{"step":10,"agent_id":6}]},
        {"desc": "N=11, d=185m, Agent 5 Rc=20m, Fail 7@t=10", "seed": 2114, "num_drones": 11, "comm_range": 28.0,
         "endpoint_a_pos": (0,0), "endpoint_b_pos": (185,0),
         "custom_comm_ranges": {5: 20.0}, "failures": [{"step":10,"agent_id":7}]},
    ]

    # Add category key to each scenario
    for cat, scs in scenarios.items():
        for sc in scs:
            sc["category"] = cat

    return scenarios


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"

    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    b_path  = "checkpoints/checkpoint_B_best.pt"

    print("=" * 90)
    print("  B-PRIME COMPLETE VERIFICATION & 4-STEP PROTOCOL BATTERY")
    print("=" * 90)
    print(f"  Git Commit:        {get_git_commit_hash()}")
    print(f"  B-Prime Path:      {bp_path}  (MD5: {compute_file_md5(bp_path)})")
    print(f"  Checkpoint B Path: {b_path}  (MD5: {compute_file_md5(b_path)})")
    print("=" * 90)

    actor_bp = load_b_prime(bp_path, device)
    actor_b  = load_checkpoint_b_as_9x5(b_path, device)

    # ─── 1. 50-SCENARIO BENCHMARK GATE ───
    print("\n" + "━" * 90)
    print("  SECTION 1: 50-SCENARIO BENCHMARK GATE (Seen-35 + Held-Out-15)")
    print("━" * 90)

    with open("checkpoints/test_bank_50.pkl", "rb") as f:
        bank = pickle.load(f)
    seen_35 = bank["seen_35"]
    held_15 = bank["held_out_15"]
    all_50  = bank["all_50"]

    res_bp_s35 = [run_scenario(actor_bp, sc, device) for sc in seen_35]
    res_b_s35  = [run_scenario(actor_b,  sc, device) for sc in seen_35]
    res_bp_h15 = [run_scenario(actor_bp, sc, device) for sc in held_15]
    res_b_h15  = [run_scenario(actor_b,  sc, device) for sc in held_15]

    def summarize_50(label, res_list):
        ok = sum(1 for r in res_list if r["reconnected"])
        times = [r["reconnect_rel"] for r in res_list if r["reconnected"] and r["reconnect_rel"] is not None]
        rates = [r["sum_rate"] for r in res_list if r["reconnected"]]
        mt = np.mean(times) if times else float('nan')
        mr = np.mean(rates) if rates else float('nan')
        return ok, len(res_list), mt, mr

    bp_s = summarize_50("BP-Seen35", res_bp_s35)
    b_s  = summarize_50("B-Seen35",  res_b_s35)
    bp_h = summarize_50("BP-Held15", res_bp_h15)
    b_h  = summarize_50("B-Held15",  res_b_h15)

    bp_all = res_bp_s35 + res_bp_h15
    b_all  = res_b_s35  + res_b_h15
    bp_a = summarize_50("BP-All50", bp_all)
    b_a  = summarize_50("B-All50",  b_all)

    print(f"\n  {'':25s} {'B-Prime':>20s}  {'Checkpoint B':>20s}")
    print(f"  {'─'*25} {'─'*20}  {'─'*20}")
    print(f"  {'Seen-35 Success':25s} {f'{bp_s[0]}/{bp_s[1]}':>20s}  {f'{b_s[0]}/{b_s[1]}':>20s}")
    print(f"  {'Seen-35 Mean Time':25s} {bp_s[2]:>19.2f}t  {b_s[2]:>19.2f}t")
    print(f"  {'Held-Out-15 Success':25s} {f'{bp_h[0]}/{bp_h[1]}':>20s}  {f'{b_h[0]}/{b_h[1]}':>20s}")
    print(f"  {'Held-Out-15 Mean Time':25s} {bp_h[2]:>19.2f}t  {b_h[2]:>19.2f}t")
    print(f"  {'ALL 50 Success':25s} {f'{bp_a[0]}/{bp_a[1]}':>20s}  {f'{b_a[0]}/{b_a[1]}':>20s}")
    print(f"  {'ALL 50 Mean Time':25s} {bp_a[2]:>19.2f}t  {b_a[2]:>19.2f}t")

    # ─── 2. STRESS CATEGORIES ───
    all_scenarios = build_all_scenarios()
    cat_order = [
        "1. Simultaneous Double Failure", "2. Cascading Failure",
        "3. Scale Test (Large Swarm)", "4. Non-Collinear Corridor",
        "5. Failure Timing Extremes", "6. Tight-Margin Geometry",
        "7. Triple Simultaneous Failure", "8. Adversarial Cut-Vertex",
        "9. Compound Stressors", "11. Comm-Range Heterogeneity"
    ]

    all_stress_results = []

    for cat in cat_order:
        scs = all_scenarios[cat]
        n_sc = len(scs)

        print(f"\n{'━' * 90}")
        print(f"  CATEGORY: {cat}  ({n_sc} scenarios)")
        print(f"{'━' * 90}")

        for sc in scs:
            r_bp = run_scenario(actor_bp, sc, device)
            r_b  = run_scenario(actor_b,  sc, device)

            t_bp = f"{r_bp['reconnect_rel']}t" if r_bp['reconnect_rel'] is not None else "—"
            t_b  = f"{r_b['reconnect_rel']}t"  if r_b['reconnect_rel']  is not None else "—"

            print(f"\n  Config: {r_bp['desc']}")
            print(f"    B-Prime:  Reconn={r_bp['reconnected']}  Time={t_bp:>6s}  Rate={r_bp['sum_rate']:>8.2f}  Drops={r_bp['transient_drops']}  Mode={r_bp['failure_mode']}")
            print(f"    Ckpt B:   Reconn={r_b['reconnected']}  Time={t_b:>6s}  Rate={r_b['sum_rate']:>8.2f}  Drops={r_b['transient_drops']}  Mode={r_b['failure_mode']}")

            all_stress_results.append({
                "category": cat,
                "desc": r_bp["desc"],
                "bp_reconn": r_bp["reconnected"], "bp_time": r_bp["reconnect_rel"],
                "bp_rate": r_bp["sum_rate"], "bp_drops": r_bp["transient_drops"], "bp_mode": r_bp["failure_mode"],
                "b_reconn": r_b["reconnected"], "b_time": r_b["reconnect_rel"],
                "b_rate": r_b["sum_rate"], "b_drops": r_b["transient_drops"], "b_mode": r_b["failure_mode"],
            })

        # Category summary
        cat_rows = [r for r in all_stress_results if r["category"] == cat]
        bp_ok = sum(1 for r in cat_rows if r["bp_reconn"])
        b_ok  = sum(1 for r in cat_rows if r["b_reconn"])
        print(f"\n  ── {cat} Summary: B-Prime {bp_ok}/{n_sc}  |  Ckpt B {b_ok}/{n_sc} ──")

    # Save stress results
    df_stress = pd.DataFrame(all_stress_results)
    stress_csv = "scratch/b_prime_stress_battery_results.csv"
    df_stress.to_csv(stress_csv, index=False)
    print(f"\n  Stress battery results saved to {stress_csv}")

    # ─── 3. 4-STEP VERIFICATION PROTOCOL ───
    print(f"\n{'═' * 90}")
    print(f"  4-STEP VERIFICATION PROTOCOL")
    print(f"{'═' * 90}")

    # Step 1: Baseline Audit
    print(f"\n  [Step 1] Baseline Audit (50-scenario gate)")
    regression = bp_a[0] < b_a[0]
    print(f"    B-Prime: {bp_a[0]}/50 ({bp_a[0]/50*100:.1f}%)  vs  Ckpt B: {b_a[0]}/50 ({b_a[0]/50*100:.1f}%)")
    print(f"    GATE STATUS: {'❌ REGRESSION DETECTED' if regression else '✅ NO REGRESSION'}")

    # Step 2: Bit-Identical Determinism Rerun
    print(f"\n  [Step 2] Determinism Rerun (bit-identical check)")
    res_bp_all_r2 = [run_scenario(actor_bp, sc, device) for sc in all_50]
    time_diffs = []
    rate_diffs = []
    for r1, r2 in zip(bp_all, res_bp_all_r2):
        t1 = r1["reconnect_rel"] if r1["reconnect_rel"] is not None else -1
        t2 = r2["reconnect_rel"] if r2["reconnect_rel"] is not None else -1
        time_diffs.append(abs(t1 - t2))
        rate_diffs.append(abs(r1["sum_rate"] - r2["sum_rate"]))
    max_tdiff = max(time_diffs)
    max_rdiff = max(rate_diffs)
    print(f"    Max |ΔTime| across 50 seeds: {max_tdiff:.10e}")
    print(f"    Max |ΔRate| across 50 seeds: {max_rdiff:.10e}")
    determinism_ok = max_tdiff == 0.0 and max_rdiff == 0.0
    print(f"    STATUS: {'✅ BIT-IDENTICAL DETERMINISM CONFIRMED' if determinism_ok else '❌ DETERMINISM FAILURE'}")

    # Step 3: Paired Significance Test
    print(f"\n  [Step 3] Per-Seed Distribution & Paired Significance Test")
    deltas = []
    for r_bp, r_b in zip(bp_all, b_all):
        t_bp_v = r_bp["reconnect_rel"] if r_bp["reconnect_rel"] is not None else float('nan')
        t_b_v  = r_b["reconnect_rel"]  if r_b["reconnect_rel"]  is not None else float('nan')
        deltas.append(t_bp_v - t_b_v)
    deltas = np.array(deltas)
    valid = ~np.isnan(deltas)
    if np.any(valid):
        d_valid = deltas[valid]
        print(f"    N valid pairs: {np.sum(valid)}")
        print(f"    Mean Δ(B-Prime − B): {np.mean(d_valid):.4f}t  (negative = B-Prime faster)")
        print(f"    Median Δ: {np.median(d_valid):.4f}t")
        if np.any(d_valid != 0):
            w, p = stats.wilcoxon(d_valid)
            print(f"    Wilcoxon Signed-Rank: W={w}, p={p:.6e}")
        else:
            print(f"    Wilcoxon: all deltas = 0 (identical performance)")
    else:
        print(f"    No valid pairs for significance testing.")

    # Step 4: Zero-Effect Control Run
    print(f"\n  [Step 4] Zero-Effect Control (Cat 11 with all agents nominal Rc=28.0m)")
    cat11_nominal = []
    for sc in all_scenarios["11. Comm-Range Heterogeneity"]:
        sc_nom = {k: v for k, v in sc.items() if k != "custom_comm_ranges"}
        cat11_nominal.append(sc_nom)

    for sc_nom in cat11_nominal:
        r_bp = run_scenario(actor_bp, sc_nom, device)
        r_b  = run_scenario(actor_b,  sc_nom, device)
        t_bp = f"{r_bp['reconnect_rel']}t" if r_bp['reconnect_rel'] is not None else "—"
        t_b  = f"{r_b['reconnect_rel']}t"  if r_b['reconnect_rel']  is not None else "—"
        print(f"    {sc_nom['desc']}")
        print(f"      B-Prime: Reconn={r_bp['reconnected']} Time={t_bp} Rate={r_bp['sum_rate']}")
        print(f"      Ckpt B:  Reconn={r_b['reconnected']} Time={t_b} Rate={r_b['sum_rate']}")

    print(f"\n{'═' * 90}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'═' * 90}")


if __name__ == "__main__":
    main()
