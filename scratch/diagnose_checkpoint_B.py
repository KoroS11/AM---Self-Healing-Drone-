import sys
import os
import pickle
import torch
import numpy as np
import pandas as pd

from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.scenarios.test_bank import TestBankGenerator

def run_diagnostics():
    sys.stdout.reconfigure(line_buffering=True)
    pipeline = TrainPipeline(output_dir="checkpoints")
    device = "cpu"
    
    # Load spare bank
    spare_path = "checkpoints/test_bank_spare.pkl"
    with open(spare_path, "rb") as f:
        spare_bank = pickle.load(f)
    spare_10 = spare_bank["spare_10"]
    
    # Load test bank
    bank_path = "checkpoints/test_bank_50.pkl"
    with open(bank_path, "rb") as f:
        test_bank = pickle.load(f)
    held_out_15 = test_bank["held_out_15"]
    
    # Load models
    actor_A = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_A.load_state_dict(torch.load("checkpoints/checkpoint_A_best_k2.pt", map_location=device), strict=False)
    actor_A.eval()
    
    actor_B = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor_B.load_state_dict(torch.load("checkpoints/checkpoint_B_best.pt", map_location=device), strict=False)
    actor_B.eval()
    
    print("=================================================================")
    print("PART 2: 5-PASS RE-EVALUATION OF CHECKPOINT B ON 10 SPARE SCENARIOS")
    print("=================================================================")
    pass_results = []
    for p in range(1, 6):
        res = pipeline.evaluate_policy_on_scenarios(
            lambda s: GNNPolicy(s, actor_gnn=actor_B, device=device),
            spare_10,
            max_steps=150
        )
        pass_results.append({
            "pass": p,
            "success_rate": res["success_rate"],
            "success_count": int(res["success_rate"] * len(spare_10)),
            "mean_time": res["mean_time"],
            "mean_sum_rate": res["mean_sum_rate"]
        })
        print(f"Pass {p}: Success = {res['success_rate']*100:.1f}% ({int(res['success_rate']*len(spare_10))}/10), Mean Time = {res['mean_time']:.2f} t, Mean SR = {res['mean_sum_rate']:.2f} Mbps")
        
    print("\n=================================================================")
    print("PART 3: HELD-OUT 15 SCENARIOS COMPARISON (CHECKPOINT A vs B)")
    print("=================================================================")
    held_out_table = []
    failed_seeds_B = []
    for sc in held_out_15:
        seed = sc["seed"]
        # Eval A
        res_A = pipeline.evaluate_policy_on_scenarios(
            lambda s: GNNPolicy(s, actor_gnn=actor_A, device=device),
            [sc],
            max_steps=150
        )
        # Eval B
        res_B = pipeline.evaluate_policy_on_scenarios(
            lambda s: GNNPolicy(s, actor_gnn=actor_B, device=device),
            [sc],
            max_steps=150
        )
        
        succ_A = res_A["success_rate"] == 1.0
        time_A = res_A["mean_time"]
        sr_A = res_A["mean_sum_rate"]
        
        succ_B = res_B["success_rate"] == 1.0
        time_B = res_B["mean_time"]
        sr_B = res_B["mean_sum_rate"]
        
        held_out_table.append({
            "seed": seed,
            "drones": sc.get("num_drones", len(sc.get("initial_positions", []))),
            "fail_id": sc["fail_agent_id"],
            "A_succ": succ_A,
            "A_time": time_A,
            "A_sr": sr_A,
            "B_succ": succ_B,
            "B_time": time_B,
            "B_sr": sr_B
        })
        n_drones = sc.get("num_drones", len(sc.get("initial_positions", [])))
        print(f"Seed {seed:4d} (N={n_drones:2d}, fail={sc['fail_agent_id']}): Checkpoint A = {'PASS' if succ_A else 'FAIL'} ({time_A:5.1f}t, {sr_A:7.2f}M) | Checkpoint B = {'PASS' if succ_B else 'FAIL'} ({time_B:5.1f}t, {sr_B:7.2f}M)")
        if not succ_B:
            failed_seeds_B.append(seed)
            
    print(f"\nFailed held-out seeds under Checkpoint B: {failed_seeds_B}")
    
    # Extended 300-tick diagnosis on failed seeds
    for fail_seed in failed_seeds_B:
        print(f"\n=================================================================")
        print(f"EXTENDED 300-TICK TRACE DIAGNOSIS ON FAILED HELD-OUT SEED {fail_seed}")
        print(f"=================================================================")
        sc_config = next(s for s in held_out_15 if s["seed"] == fail_seed)
        
        from swarm_sim.utils.config import ExperimentConfig
        
        config = ExperimentConfig(
            num_drones=sc_config["num_drones"],
            comm_range=sc_config["comm_range"],
            seed=sc_config["seed"]
        )
        sim = SwarmSimulator(config)
        scenario = DisasterRelayScenario(sim)
        g_a, g_b = scenario.setup_scenario(
            endpoint_a_pos=sc_config.get("endpoint_a_pos", (0.0, 0.0)),
            endpoint_b_pos=sc_config.get("endpoint_b_pos", (200.0, 0.0))
        )
        injector = FailureInjector(sim)
        topo = SwarmTopologyManager(sim)
        
        policy = GNNPolicy(sim, actor_gnn=actor_B, device=device)
        
        timeline = []
        first_reconnect_tick = None
        
        for t in range(300):
            if t == sc_config["fail_timestep"]:
                injector.fail_agent(sc_config["fail_agent_id"])
                
            sim.step(policy=policy)
            
            G = topo.build_graph()
            is_connected = topo.has_path(g_a, g_b, graph=G)
            
            if is_connected and first_reconnect_tick is None:
                first_reconnect_tick = t
                
            timeline.append({
                "tick": t,
                "connected": is_connected,
                "num_components": topo.get_num_connected_components(),
                "min_d_ab": float(np.linalg.norm(sim.positions[g_a] - sim.positions[g_b]))
            })
            
        final_connected = timeline[-1]["connected"]
        connected_ticks = [row["tick"] for row in timeline if row["connected"]]
        
        print(f"Seed {fail_seed} Diagnostic Summary:")
        print(f"  First Reconnection Event Tick: {first_reconnect_tick}")
        print(f"  Connected at 150 ticks: {timeline[149]['connected']}")
        print(f"  Final Connected at 300 ticks: {final_connected}")
        print(f"  Total Connected Ticks (out of 300): {len(connected_ticks)} / 300 ({len(connected_ticks)/300*100:.1f}%)")
        
        if len(connected_ticks) > 0:
            print(f"  Connected Tick Ranges: min={min(connected_ticks)}, max={max(connected_ticks)}")
            # Show transitions
            transitions = []
            prev = False
            for row in timeline:
                if row["connected"] != prev:
                    transitions.append((row["tick"], row["connected"]))
                    prev = row["connected"]
            print(f"  State Transitions (tick, new_state): {transitions}")

if __name__ == "__main__":
    run_diagnostics()
