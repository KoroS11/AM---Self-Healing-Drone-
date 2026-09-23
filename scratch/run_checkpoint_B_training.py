import sys
import os
import torch
import numpy as np
from train_pipeline import TrainPipeline

def main():
    sys.stdout.reconfigure(line_buffering=True)
    pipeline = TrainPipeline(output_dir="checkpoints")
    results = pipeline.run_checkpoint_B(
        checkpoint_A_path="checkpoints/checkpoint_A_best_k2.pt",
        max_iterations=300,
        rollout_batch_size=8,
        ppo_epochs=4,
        lr_actor=1e-4,
        lr_critic=5e-4,
        clip_param=0.2,
        entropy_coef=0.0005,
        gamma=0.99,
        gae_lambda=0.95,
        seed=42
    )
    print("\n--- CHECKPOINT B RUN COMPLETE ---")
    print(f"Gate Passed: {results['gate_passed']}")
    print(f"Total Validation Steps Tracked: {len(results['history'])}")

if __name__ == "__main__":
    main()
