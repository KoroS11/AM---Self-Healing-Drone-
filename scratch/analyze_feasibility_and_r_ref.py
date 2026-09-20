import os
import pickle
import numpy as np
import pandas as pd
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.scenarios.test_bank import TestBankGenerator

bank_file = "checkpoints/test_bank_50.pkl"
spare_file = "checkpoints/test_bank_spare.pkl"

if not os.path.exists(bank_file):
    tb_gen = TestBankGenerator()
    tb_gen.save_bank(bank_file, spare_file)

with open(bank_file, "rb") as f:
    bank_data = pickle.load(f)

scenarios = bank_data["all_50"]

print(f"=== Feasibility Margin Analysis for All 50 Scenarios ===")

negative_margin_scenarios = []

for sc in scenarios:
    seed = sc["seed"]
    num_drones = sc["num_drones"]
    num_relays = num_drones - 2
    comm_range = sc["comm_range"]
    
    pos_a = sc["endpoint_a_pos"]
    pos_b = sc["endpoint_b_pos"]
    dist_ab = float(np.linalg.norm(np.array(pos_b) - np.array(pos_a)))
    
    # Formula 1: user-specified (num_relays - 1) * comm_range - dist_ab
    margin_user = (num_relays - 1) * comm_range - dist_ab
    
    # Formula 2: physical edge max reach (active_relays + 1) * comm_range - dist_ab = num_relays * comm_range - dist_ab
    max_reach = num_relays * comm_range
    margin_physical = max_reach - dist_ab
    
    is_neg = (margin_user < 0) or (margin_physical < 0)
    
    sc_info = {
        "seed": seed,
        "num_drones": num_drones,
        "num_relays": num_relays,
        "active_relays_after_fail": num_relays - 1,
        "comm_range": comm_range,
        "dist_ab": dist_ab,
        "user_margin": margin_user,
        "physical_max_reach": max_reach,
        "physical_margin": margin_physical,
        "is_unrecoverable": margin_physical < 0
    }
    
    if margin_physical < 0 or margin_user < 0:
        negative_margin_scenarios.append(sc_info)

df_all = pd.DataFrame([
    {
        "seed": sc["seed"],
        "num_drones": sc["num_drones"],
        "num_relays": sc["num_drones"] - 2,
        "comm_range": sc["comm_range"],
        "dist_ab": float(np.linalg.norm(np.array(sc["endpoint_b_pos"]) - np.array(sc["endpoint_a_pos"]))),
        "user_margin": (sc["num_drones"] - 3) * sc["comm_range"] - float(np.linalg.norm(np.array(sc["endpoint_b_pos"]) - np.array(sc["endpoint_a_pos"]))),
        "physical_max_reach": (sc["num_drones"] - 2) * sc["comm_range"],
        "physical_margin": (sc["num_drones"] - 2) * sc["comm_range"] - float(np.linalg.norm(np.array(sc["endpoint_b_pos"]) - np.array(sc["endpoint_a_pos"])))
    } for sc in scenarios
])

print(f"\nTotal scenarios analyzed: {len(scenarios)}")
print(f"Scenarios with negative physical margin (num_relays * Rc < dist_AB): {sum(df_all['physical_margin'] < 0)}")
print(f"Scenarios with negative user margin ((num_relays - 1) * Rc < dist_AB): {sum(df_all['user_margin'] < 0)}")

if len(negative_margin_scenarios) > 0:
    print("\n--- Negative Margin Scenarios ---")
    df_neg = pd.DataFrame(negative_margin_scenarios)
    print(df_neg.to_string(index=False))
else:
    print("\nZero scenarios have negative physical margin.")

print("\n=== Re-running precompute_r_ref() for Seed 42 Actual Config ===")
channel = ChannelModel()
# Seed 42 config: num_drones = 9 (7 relays), dist_ab = 200.0m
r_ref_seed42 = channel.precompute_r_ref(num_relays=7, dist_ab=200.0)
print(f"Seed 42 Config: num_relays = 7, dist_AB = 200.0m")
print(f"precompute_r_ref(num_relays=7, dist_ab=200.0) = {r_ref_seed42:.4f} Mbps")
