import os
import sys
import hashlib
import pathlib
import subprocess
import numpy as np
import pandas as pd
import torch

from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.policies.gnn_actor import ActorGNN

def main():
    print("=== EXECUTING CHECKPOINT A PIPELINE WITH FULL INSTRUMENTATION ===")
    
    # 0. Print device and hashes
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    
    files = [
        "swarm_sim/policies/gnn_actor.py",
        "swarm_sim/policies/gnn_critic.py",
        "swarm_sim/policies/gnn_policy.py",
        "train_pipeline.py"
    ]
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit_hash = "N/A"
        
    print(f"Git Commit Hash: {commit_hash}")
    for f in files:
        md5 = hashlib.md5(pathlib.Path(f).read_bytes()).hexdigest()
        print(f"  {f:35s}: {md5}")
        
    pipeline = TrainPipeline(output_dir="checkpoints")
    
    # Run Checkpoint A
    res = pipeline.run_checkpoint_A(max_dagger_rounds=5, bc_epochs=30, dagger_epochs=15, lr=1e-3, seed=42)
    
    print("\n=================================================================")
    print("                 CHECKPOINT A RESULTS SUMMARY")
    print("=================================================================")
    print(f"Final Passing k:             k = {res['k_hop']}")
    print(f"Final Passing Round:         Round {res['round']}")
    print(f"Open-Loop Action RMSE:       {res['open_loop_rmse']:.4f} m/s^2 (Threshold: {res['rmse_threshold']:.2f} m/s^2)")
    
    # Epoch loss curves
    print("\n--- BC Round 0 Training & Validation Loss Curves ---")
    loss_df = pd.DataFrame({
        "Epoch": list(range(1, len(res["train_loss_curve"]) + 1)),
        "Train_MSE_Loss": res["train_loss_curve"],
        "Val_MSE_Loss": res["val_loss_curve"]
    })
    print(loss_df.to_string(index=False))
    
    # Per-scenario tables
    actor = res["actor"]
    actor.eval()
    
    seen_details = res["seen_eval"]["details"]
    held_out_details = res["held_out_eval"]["details"]
    dara_seen_details = res["dara_seen"]["details"]
    dara_held_details = res["dara_held_out"]["details"]
    
    # Match DARA vs GNN per scenario
    seen_table = []
    for i, g_sc in enumerate(seen_details):
        d_sc = dara_seen_details[i]
        seen_table.append({
            "seed": g_sc["seed"],
            "num_drones": g_sc["num_drones"],
            "fail_agent_id": g_sc["fail_agent_id"],
            "DARA_Success": d_sc["success"],
            "DARA_Time": d_sc["time_to_reconnect"],
            "DARA_SumRate": d_sc["final_sum_rate_mbps"],
            "GNN_Success": g_sc["success"],
            "GNN_Time": g_sc["time_to_reconnect"],
            "GNN_SumRate": g_sc["final_sum_rate_mbps"]
        })
        
    held_table = []
    for i, g_sc in enumerate(held_out_details):
        d_sc = dara_held_details[i]
        held_table.append({
            "seed": g_sc["seed"],
            "num_drones": g_sc["num_drones"],
            "fail_agent_id": g_sc["fail_agent_id"],
            "DARA_Success": d_sc["success"],
            "DARA_Time": d_sc["time_to_reconnect"],
            "DARA_SumRate": d_sc["final_sum_rate_mbps"],
            "GNN_Success": g_sc["success"],
            "GNN_Time": g_sc["time_to_reconnect"],
            "GNN_SumRate": g_sc["final_sum_rate_mbps"]
        })
        
    df_seen = pd.DataFrame(seen_table)
    df_held = pd.DataFrame(held_table)
    
    print("\n--- PER-SCENARIO EVALUATION: SEEN 35 SCENARIOS (Seeds 1000..1034) ---")
    print(df_seen.to_string(index=False))
    
    print("\n--- PER-SCENARIO EVALUATION: HELD-OUT 15 SCENARIOS (Seeds 1035..1049) ---")
    print(df_held.to_string(index=False))
    
    # 7. Reproducibility Check
    print("\n=== STEP 7: REPRODUCIBILITY RERUN (50 SCENARIOS) ===")
    rerun_eval = pipeline.evaluate_policy_on_scenarios(
        lambda s: GNNPolicy(s, actor_gnn=actor, device=str(pipeline.device)),
        pipeline.bank_data["all_50"]
    )
    
    pass1_success = [r["success"] for r in res["all_eval"]["details"]]
    pass2_success = [r["success"] for r in rerun_eval["details"]]
    pass1_times = [r["time_to_reconnect"] for r in res["all_eval"]["details"]]
    pass2_times = [r["time_to_reconnect"] for r in rerun_eval["details"]]
    pass1_rates = [r["final_sum_rate_mbps"] for r in res["all_eval"]["details"]]
    pass2_rates = [r["final_sum_rate_mbps"] for r in rerun_eval["details"]]
    
    success_eq = (pass1_success == pass2_success)
    times_eq = np.array_equal(np.nan_to_num(pass1_times), np.nan_to_num(pass2_times))
    rates_eq = np.array_equal(pass1_rates, pass2_rates)
    max_rate_diff = np.max(np.abs(np.array(pass1_rates) - np.array(pass2_rates)))
    
    print(f"Reproducibility Results (Pass 1 vs Pass 2 on 50 Scenarios):")
    print(f"  - Reconnection Success Identical: {success_eq}")
    print(f"  - Reconnection Times Identical:   {times_eq}")
    print(f"  - Network Sum-Rates Identical:    {rates_eq} (Max abs diff: {max_rate_diff:.8f} Mbps)")
    print(f"  - Deterministic Reproducibility:  {'PASSED (100% BIT-IDENTICAL)' if (success_eq and times_eq and rates_eq) else 'STOCHASTIC/FAILED'}")
    
    # Save results to CSV
    os.makedirs("scratch", exist_ok=True)
    df_seen.to_csv("scratch/checkpoint_A_seen_35_results.csv", index=False)
    df_held.to_csv("scratch/checkpoint_A_held_out_15_results.csv", index=False)
    loss_df.to_csv("scratch/checkpoint_A_loss_curves.csv", index=False)
    print("\nSaved result CSVs to scratch/ directory.")

if __name__ == "__main__":
    main()
