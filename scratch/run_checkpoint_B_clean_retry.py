import sys
import os
import torch
import numpy as np
import pandas as pd
from train_pipeline import TrainPipeline

def main():
    sys.stdout.reconfigure(line_buffering=True)
    pipeline = TrainPipeline(output_dir="checkpoints")
    
    print("=== Launching Clean Checkpoint B Retry (seed=1042, lr_actor=5e-5, lr_critic=2.5e-4) ===")
    results = pipeline.run_checkpoint_B(
        checkpoint_A_path="checkpoints/checkpoint_A_best_k2.pt",
        max_iterations=300,
        rollout_batch_size=8,
        ppo_epochs=4,
        lr_actor=5e-5,
        lr_critic=2.5e-4,
        clip_param=0.2,
        entropy_coef=0.0005,
        gamma=0.99,
        gae_lambda=0.95,
        seed=1042,
        seeds=[1042]
    )
    
    # Save trajectory to CSV
    df = pd.DataFrame(results["history"])
    df.to_csv("scratch/checkpoint_B_retry_clean_trajectory.csv", index=False)
    print("\n--- CHECKPOINT B CLEAN RETRY RUN COMPLETE ---")
    print(f"Gate Passed: {results['gate_passed']}")
    print(f"Best Val Success: {results['best_val_success']*100:.1f}%, Best Val Time: {results['best_val_time']:.2f} t")
    print(f"Saved trajectory to scratch/checkpoint_B_retry_clean_trajectory.csv")

if __name__ == "__main__":
    main()
