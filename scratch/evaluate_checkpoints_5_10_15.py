import os
import torch
import numpy as np
import pandas as pd
from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy

def main():
    pipeline = TrainPipeline(output_dir="checkpoints")
    spare_bank = pipeline.bank_data
    seen_35 = pipeline.bank_data["seen_35"]
    held_out_15 = pipeline.bank_data["held_out_15"]
    all_50 = pipeline.bank_data["all_50"]
    
    with open("checkpoints/test_bank_spare.pkl", "rb") as f:
        import pickle
        spare_10 = pickle.load(f)["spare_10"]
    val_25 = spare_10 + held_out_15
    
    checkpoints = {
        "Iter 1": "checkpoints/checkpoint_B_iter_1.pt",
        "Iter 5": "checkpoints/checkpoint_B_iter_5.pt",
        "Iter 10": "checkpoints/checkpoint_B_iter_10.pt",
        "Iter 15": "checkpoints/checkpoint_B_iter_15.pt"
    }
    
    rows = []
    for label, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {label} (not found: {ckpt_path})")
            continue
            
        actor = ActorGNN(k_hops=2, max_accel=5.0).to(pipeline.device)
        actor.load_state_dict(torch.load(ckpt_path, map_location=pipeline.device), strict=False)
        actor.eval()
        
        eval_val25 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor, device=str(pipeline.device)), val_25)
        eval_seen35 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor, device=str(pipeline.device)), seen_35)
        eval_held15 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor, device=str(pipeline.device)), held_out_15)
        eval_all50 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor, device=str(pipeline.device)), all_50)
        
        row = {
            "Checkpoint": label,
            "Val 25 Success": eval_val25["success_rate"],
            "Val 25 Time": eval_val25["mean_time"],
            "Seen 35 Success": eval_seen35["success_rate"],
            "Seen 35 Time": eval_seen35["mean_time"],
            "Held-Out 15 Success": eval_held15["success_rate"],
            "Held-Out 15 Time": eval_held15["mean_time"],
            "All 50 Success": eval_all50["success_rate"],
            "All 50 Time": eval_all50["mean_time"],
            "All 50 Sum-Rate": eval_all50["mean_sum_rate"]
        }
        rows.append(row)
        print(f"\n--- {label} ---")
        print(f"Val 25:      Success = {row['Val 25 Success']*100:.1f}%, Mean Time = {row['Val 25 Time']:.2f} t")
        print(f"Seen 35:     Success = {row['Seen 35 Success']*100:.1f}%, Mean Time = {row['Seen 35 Time']:.2f} t")
        print(f"Held-Out 15: Success = {row['Held-Out 15 Success']*100:.1f}%, Mean Time = {row['Held-Out 15 Time']:.2f} t")
        print(f"All 50:      Success = {row['All 50 Success']*100:.1f}%, Mean Time = {row['All 50 Time']:.2f} t, SR = {row['All 50 Sum-Rate']:.2f} Mbps")
        
    df = pd.DataFrame(rows)
    df.to_csv("scratch/checkpoint_B_plateau_neighbor_audit.csv", index=False)
    print("\nSaved neighbor comparison audit to scratch/checkpoint_B_plateau_neighbor_audit.csv")

if __name__ == "__main__":
    main()
