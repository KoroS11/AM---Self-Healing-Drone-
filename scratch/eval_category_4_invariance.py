"""Evaluate rotation-invariant checkpoint on 8 Category 4 non-collinear corridor scenarios.

4 original angles + 4 new arbitrary non-round angles (17°, 73°, 112°, 200°).
Uses feasible corridor distances: N=9 → d=180m, N=11 → d=200m.
"""
import os
import sys
import math
import torch
import numpy as np
import pandas as pd

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel


def run_scenario(actor, sc, device="cpu", max_steps=150):
    config = ExperimentConfig(
        num_drones=sc["num_drones"],
        comm_range=sc["comm_range"],
        seed=sc["seed"]
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario(
        endpoint_a_pos=sc["endpoint_a_pos"],
        endpoint_b_pos=sc["endpoint_b_pos"]
    )
    injector = FailureInjector(sim)
    policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()

    failures = sc.get("failures", [])
    if not failures and "fail_timestep" in sc:
        failures = [{"step": sc["fail_timestep"], "agent_id": sc["fail_agent_id"]}]

    last_fail_step = max(f["step"] for f in failures) if failures else 0
    ever_reconnected = False
    reconnect_time = None

    for t in range(max_steps):
        for f in failures:
            if t == f["step"]:
                injector.fail_agent(f["agent_id"])

        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)

        G = topo.build_graph()
        is_conn = topo.has_path(g_a, g_b, graph=G)

        if t >= last_fail_step and is_conn and not ever_reconnected:
            ever_reconnected = True
            reconnect_time = t - last_fail_step

    G_final = topo.build_graph()
    final_conn = topo.has_path(g_a, g_b, graph=G_final)
    sum_rate = channel.compute_network_throughput(sim, G_final)

    if final_conn:
        failure_mode = "Success (Stable)"
    elif not ever_reconnected:
        failure_mode = "Failed to Reconnect"
    else:
        failure_mode = "Reconnected then Lost"

    return {
        "desc": sc["desc"],
        "angle_deg": sc.get("angle_deg", "N/A"),
        "N": sc["num_drones"],
        "d_AB": sc["d_ab"],
        "reconnected": final_conn,
        "reconnect_time": reconnect_time,
        "final_sum_rate_mbps": round(sum_rate, 2),
        "failure_mode": failure_mode
    }


def endpoint_b_from_angle(d_ab, angle_deg):
    theta = math.radians(angle_deg)
    return (round(d_ab * math.cos(theta), 4), round(d_ab * math.sin(theta), 4))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"

    actor = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    ckpt_path = "checkpoints/checkpoint_B_best.pt"
    actor.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    actor.eval()
    print(f"Loaded retrained rotation-invariant weights from {ckpt_path}")

    # ---------- 8 Category 4 Scenarios ----------
    # Feasible distances: N=9 → d=180m  (7 relays survive, 7×28=196>180 ✓)
    #                     N=11 → d=200m  (9 relays survive, 9×28=252>200 ✓)
    scenarios = []

    # --- 4 Original Category 4 Angles ---
    # 1) N=9, 45°, d=180m
    scenarios.append({
        "desc": "N=9, d=180m, 45° diagonal",
        "angle_deg": 45.0,
        "seed": 2031, "num_drones": 9, "comm_range": 28.0,
        "d_ab": 180.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(180.0, 45.0),
        "failures": [{"step": 10, "agent_id": 5}]
    })
    # 2) N=9, 53.1°, d=180m
    scenarios.append({
        "desc": "N=9, d=180m, 53.1° offset",
        "angle_deg": 53.1,
        "seed": 2032, "num_drones": 9, "comm_range": 28.0,
        "d_ab": 180.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(180.0, 53.1),
        "failures": [{"step": 10, "agent_id": 4}]
    })
    # 3) N=11, 45°, d=200m
    scenarios.append({
        "desc": "N=11, d=200m, 45° diagonal",
        "angle_deg": 45.0,
        "seed": 2033, "num_drones": 11, "comm_range": 28.0,
        "d_ab": 200.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(200.0, 45.0),
        "failures": [{"step": 10, "agent_id": 6}]
    })
    # 4) N=11, 60°, d=200m
    scenarios.append({
        "desc": "N=11, d=200m, 60° offset",
        "angle_deg": 60.0,
        "seed": 2034, "num_drones": 11, "comm_range": 28.0,
        "d_ab": 200.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(200.0, 60.0),
        "failures": [{"step": 10, "agent_id": 5}]
    })

    # --- 4 New Arbitrary Non-Round Angles (never seen in training or prior tests) ---
    # 5) N=11, 17°, d=200m
    scenarios.append({
        "desc": "N=11, d=200m, 17° (new)",
        "angle_deg": 17.0,
        "seed": 3001, "num_drones": 11, "comm_range": 28.0,
        "d_ab": 200.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(200.0, 17.0),
        "failures": [{"step": 10, "agent_id": 5}]
    })
    # 6) N=9, 73°, d=180m
    scenarios.append({
        "desc": "N=9, d=180m, 73° (new)",
        "angle_deg": 73.0,
        "seed": 3002, "num_drones": 9, "comm_range": 28.0,
        "d_ab": 180.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(180.0, 73.0),
        "failures": [{"step": 10, "agent_id": 4}]
    })
    # 7) N=11, 112°, d=200m
    scenarios.append({
        "desc": "N=11, d=200m, 112° (new)",
        "angle_deg": 112.0,
        "seed": 3003, "num_drones": 11, "comm_range": 28.0,
        "d_ab": 200.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(200.0, 112.0),
        "failures": [{"step": 10, "agent_id": 6}]
    })
    # 8) N=9, 200°, d=180m
    scenarios.append({
        "desc": "N=9, d=180m, 200° (new)",
        "angle_deg": 200.0,
        "seed": 3004, "num_drones": 9, "comm_range": 28.0,
        "d_ab": 180.0,
        "endpoint_a_pos": (0.0, 0.0),
        "endpoint_b_pos": endpoint_b_from_angle(180.0, 200.0),
        "failures": [{"step": 10, "agent_id": 5}]
    })

    print(f"\n{'='*90}")
    print(f"  Category 4 Non-Collinear Corridor Invariance Test — 8 Scenarios")
    print(f"{'='*90}")

    results = []
    for i, sc in enumerate(scenarios, 1):
        print(f"\n  [{i}/8] {sc['desc']}  |  B={sc['endpoint_b_pos']}")
        res = run_scenario(actor, sc, device=device)
        results.append(res)
        status = "✓ PASS" if res["reconnected"] else "✗ FAIL"
        rt = f"{res['reconnect_time']}t" if res["reconnect_time"] is not None else "N/A"
        print(f"         {status}  |  Reconnect Time: {rt}  |  Sum Rate: {res['final_sum_rate_mbps']} Mbps  |  {res['failure_mode']}")

    # Summary table
    df = pd.DataFrame(results)
    csv_path = "scratch/category4_invariance_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    n_pass = sum(1 for r in results if r["reconnected"])
    print(f"  Total: {n_pass}/8 passed")
    print(f"\n  Original 4 angles: {sum(1 for r in results[:4] if r['reconnected'])}/4")
    print(f"  New 4 angles:      {sum(1 for r in results[4:] if r['reconnected'])}/4")

    if n_pass == 8:
        print(f"\n  ✓ ROTATION INVARIANCE VERIFIED: All 8 scenarios passed.")
    else:
        failed = [r for r in results if not r["reconnected"]]
        print(f"\n  ✗ ROTATION INVARIANCE INCOMPLETE: {len(failed)} scenarios failed.")
        for f in failed:
            print(f"    - {f['desc']}: {f['failure_mode']}")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
