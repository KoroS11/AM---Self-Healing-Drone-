import sys
import os
import torch
import numpy as np
import pandas as pd
from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.policies.gnn_actor import ActorGNN

def main():
    sys.stdout.reconfigure(line_buffering=True)
    pipeline = TrainPipeline(output_dir="checkpoints")
    
    print("=== Launching Checkpoint B Training with Sustained 3-Consecutive Gate on 25-Scenario Suite ===")
    results = pipeline.run_checkpoint_B(
        checkpoint_A_path="checkpoints/checkpoint_A_best_k2.pt",
        max_iterations=100,
        rollout_batch_size=8,
        ppo_epochs=4,
        lr_actor=5e-5,
        lr_critic=2.5e-4,
        clip_param=0.2,
        entropy_coef=0.0005,
        gamma=0.99,
        gae_lambda=0.95,
        seed=1042,
        seeds=[1042],
        val_suite_mode="expanded_25",
        required_consecutive_passes=3
    )
    
    # Save full trajectory to CSV
    df = pd.DataFrame(results["history"])
    df.to_csv("scratch/checkpoint_B_sustained_trajectory.csv", index=False)
    print("\n--- Checkpoint B Sustained Training Run Complete ---")
    print(f"Gate Passed: {results['gate_passed']}")
    print(f"Best Val Success: {results['best_val_success']*100:.1f}%, Best Val Time: {results['best_val_time']:.2f} t")
    print(f"Saved full trajectory to scratch/checkpoint_B_sustained_trajectory.csv")

if __name__ == "__main__":
    main()
