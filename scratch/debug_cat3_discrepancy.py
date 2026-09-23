"""
DEBUG CATEGORY 3 DISCREPANCY
============================
Checks exact per-scenario behavior for Category 3 under:
1. 150 steps vs 300 steps
2. (alpha1=2.0, alpha2=1.0) vs (alpha1=3.0, alpha2=1.5)
3. Checkpoint B vs Raw B-Prime vs B-Prime+CBF
"""
import os, sys, math, pickle, hashlib
import numpy as np
import pandas as pd
import torch

from swarm_sim.policies.gnn_actor import ActorGNN
from scratch.evaluate_filtered_b_prime_full_battery import LexicographicCBFGNNPolicy, run_single_eval
from scratch.evaluate_b_prime_full_battery import build_all_scenarios


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    b_path  = "checkpoints/checkpoint_B_best.pt"

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

    all_scs = build_all_scenarios()
    cat3_scs = all_scs["3. Scale Test (Large Swarm)"]

    print("=" * 90)
    print("  CATEGORY 3 DETAILED PER-SCENARIO AUDIT")
    print("=" * 90)

    for sc in cat3_scs:
        print(f"\nScenario: {sc['desc']}")
        for steps in [150, 300]:
            print(f"  --- Horizon = {steps} ticks ---")
            r_b = run_single_eval(actor_b, sc, use_cbf=False, max_steps=steps)
            r_bp_raw = run_single_eval(actor_bp, sc, use_cbf=False, max_steps=steps)
            r_bp_cbf_2 = run_single_eval(actor_bp, sc, use_cbf=True, max_steps=steps)

            # Also run with (3.0, 1.5)
            from swarm_sim.core.simulator import SwarmSimulator
            from swarm_sim.utils.config import ExperimentConfig
            from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
            from swarm_sim.graph.topology import SwarmTopologyManager
            from swarm_sim.graph.channel import ChannelModel
            from swarm_sim.core.failure import FailureInjector
            from swarm_sim.utils.enums import AgentRole

            cfg = ExperimentConfig(num_drones=sc["num_drones"], comm_range=sc["comm_range"], seed=sc["seed"])
            sim = SwarmSimulator(cfg)
            scenario = DisasterRelayScenario(sim)
            scenario.setup_scenario(endpoint_a_pos=sc.get("endpoint_a_pos", (0.0, 0.0)), endpoint_b_pos=sc.get("endpoint_b_pos", (200.0, 0.0)))
            pol3 = LexicographicCBFGNNPolicy(sim, actor_gnn=actor_bp, alpha1=3.0, alpha2=1.5, device=device)
            topo = SwarmTopologyManager(sim)
            channel = ChannelModel()
            inj = FailureInjector(sim)
            failures = list(sc.get("failures", []))
            if not failures and "fail_timestep" in sc and "fail_agent_id" in sc:
                failures = [{"step": sc["fail_timestep"], "agent_id": sc["fail_agent_id"]}]

            conn_h = []
            first_c = None
            drops = 0
            prev_c = False
            for t in range(steps):
                for f in failures:
                    if t == f["step"]:
                        inj.fail_agent(f["agent_id"])
                acc = pol3.compute_control_forces()
                sim.step(accelerations=acc)
                G = topo.build_graph()
                is_c = topo.has_path(0, 1, graph=G)
                conn_h.append(is_c)
                if t >= 10:
                    if is_c:
                        if first_c is None:
                            first_c = t
                        prev_c = True
                    else:
                        if prev_c:
                            drops += 1
                        prev_c = False
            k20_3 = all(conn_h[-20:]) if len(conn_h) >= 20 else False
            snap_3 = conn_h[-1]
            t_rel_3 = (first_c - 10) if first_c is not None else None

            print(f"    Ckpt B:          Snap={r_b['reconnected']}, K20={r_b['k20_sustained']}, Time={r_b['reconnect_rel']}t, Drops={r_b['transient_drops']}")
            print(f"    Raw B-Prime:     Snap={r_bp_raw['reconnected']}, K20={r_bp_raw['k20_sustained']}, Time={r_bp_raw['reconnect_rel']}t, Drops={r_bp_raw['transient_drops']}")
            print(f"    BP+CBF (2.0,1.0): Snap={r_bp_cbf_2['reconnected']}, K20={r_bp_cbf_2['k20_sustained']}, Time={r_bp_cbf_2['reconnect_rel']}t, Drops={r_bp_cbf_2['transient_drops']}")
            print(f"    BP+CBF (3.0,1.5): Snap={snap_3}, K20={k20_3}, Time={t_rel_3}t, Drops={drops}")


if __name__ == "__main__":
    main()
