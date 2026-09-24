"""
Generate Ground-Truth Baseline for Checkpoint B-Prime (N=50 Scenarios).

Explicit, standalone baseline generator.
Evaluates Checkpoint B-Prime weights on test_bank_50.pkl with alpha=0 (pure connectivity/margin reward)
and saves the per-seed outcome metrics to results/checkpoint_b_prime_baseline.json.
This script must be executed independently from verify_zero_effect_control.py.
"""
import os
import sys
import json
import argparse
from train_pipeline import TrainPipeline, AlphaSchedule

DEFAULT_CHECKPOINT_PATH = "checkpoints/checkpoint_B_prime_best.pt"
DEFAULT_TEST_BANK_PATH = "checkpoints/test_bank_50.pkl"
DEFAULT_OUTPUT_PATH = "results/checkpoint_b_prime_baseline.json"

def generate_b_prime_baseline(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    test_bank_path: str = DEFAULT_TEST_BANK_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_steps: int = 150
) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(test_bank_path):
        raise FileNotFoundError(f"Test bank file not found: {test_bank_path}")

    print(f"[Generate Baseline] Loading checkpoint from {checkpoint_path}...")
    pipeline = TrainPipeline.load_checkpoint(checkpoint_path)
    test_bank = pipeline.load_test_bank(test_bank_path)
    
    alpha_sched = AlphaSchedule(alpha_max=0.0, k_anneal=1)  # alpha strictly 0.0
    baseline_data = {}

    print(f"[Generate Baseline] Evaluating {len(test_bank)} scenarios with alpha=0.0...")
    for seed_id, scenario in sorted(test_bank.items(), key=lambda x: int(x[0])):
        outcome = pipeline.run_episode(
            scenario,
            alpha_schedule=alpha_sched,
            iteration=0,
            max_steps=max_steps
        )
        baseline_data[str(seed_id)] = {
            "reconnect_ticks": outcome.reconnect_ticks,
            "success": outcome.success,
            "final_reward": outcome.total_reward,
            "sum_rate": outcome.sum_rate
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"[Generate Baseline] Successfully generated and saved ground truth to {output_path} ({len(baseline_data)} seeds).")
    return baseline_data

def main():
    parser = argparse.ArgumentParser(description="Generate Checkpoint B-Prime Ground Truth Baseline")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH, help="Path to B-Prime checkpoint")
    parser.add_argument("--test-bank", type=str, default=DEFAULT_TEST_BANK_PATH, help="Path to 50-scenario test bank")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    generate_b_prime_baseline(
        checkpoint_path=args.checkpoint,
        test_bank_path=args.test_bank,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
