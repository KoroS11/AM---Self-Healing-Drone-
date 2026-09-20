import os
import pickle
import pandas as pd
import numpy as np
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.scenarios.test_bank import TestBankGenerator

print("=== 1. Summary CSV Logs for Seed 42 (150 steps vs 250 steps) ===")
df_dara_150 = pd.read_csv("logs/summary_heuristic_dara_42.csv")
df_greedy_150 = pd.read_csv("logs/summary_greedy_dara_42.csv")

if os.path.exists("logs/summary_heuristic_dara_42_250.csv"):
    df_dara_250 = pd.read_csv("logs/summary_heuristic_dara_42_250.csv")
else:
    df_dara_250 = df_dara_150

if os.path.exists("logs/summary_greedy_dara_42_250.csv"):
    df_greedy_250 = pd.read_csv("logs/summary_greedy_dara_42_250.csv")
else:
    df_greedy_250 = df_greedy_150

print("\n--- DARA Summary (150 steps) ---")
print(df_dara_150[["run_id", "time_to_reconnect", "final_connectivity", "min_connectivity_during_failure", "final_sum_rate_mbps"]].to_string())

print("\n--- Greedy DARA Summary (150 steps) ---")
print(df_greedy_150[["run_id", "time_to_reconnect", "final_connectivity", "min_connectivity_during_failure", "final_sum_rate_mbps"]].to_string())

if os.path.exists("logs/summary_heuristic_dara_42_250.csv"):
    print("\n--- DARA Summary (250 steps) ---")
    print(df_dara_250[["run_id", "time_to_reconnect", "final_connectivity", "min_connectivity_during_failure", "final_sum_rate_mbps"]].to_string())

if os.path.exists("logs/summary_greedy_dara_42_250.csv"):
    print("\n--- Greedy DARA Summary (250 steps) ---")
    print(df_greedy_250[["run_id", "time_to_reconnect", "final_connectivity", "min_connectivity_during_failure", "final_sum_rate_mbps"]].to_string())
