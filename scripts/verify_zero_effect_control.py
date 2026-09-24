"""
Checkpoint C Step 4: Zero-Effect Control.

Runs the full N=50 eval sweep with alpha pinned at 0.0 and diffs every
per-seed outcome against the saved Checkpoint B-prime baseline, bit-for-bit.
Exits nonzero on any mismatch — this must pass before any alpha > 0 run begins.
"""
import os
import sys
import json
import numpy as np

from train_pipeline import TrainPipeline, AlphaSchedule

BASELINE_PATH = "results/checkpoint_b_prime_baseline.json"
CHECKPOINT_PATH = "checkpoints/checkpoint_B_prime_best.pt"
TEST_BANK_PATH = "checkpoints/test_bank_50.pkl"
TOLERANCE = 0.0  # exact match required; loosen only with a documented reason

def load_baseline(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_baseline_if_missing(pipeline: TrainPipeline, path: str, test_bank_path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[Zero-Effect] Generating initial ground-truth baseline at {path}...")
    baseline_data = {}
    test_bank = pipeline.load_test_bank(test_bank_path)
    alpha_sched = AlphaSchedule(alpha_max=0.0, k_anneal=1)
    
    for seed_id, scenario in test_bank.items():
        outcome = pipeline.run_episode(
            scenario,
            alpha_schedule=alpha_sched,
            iteration=0,
            max_steps=150
        )
        baseline_data[str(seed_id)] = {
            "reconnect_ticks": outcome.reconnect_ticks,
            "success": outcome.success,
            "final_reward": outcome.total_reward,
            "sum_rate": outcome.sum_rate
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"[Zero-Effect] Baseline saved successfully ({len(baseline_data)} scenarios).")

def run_zero_effect_sweep(pipeline: TrainPipeline, test_bank_path: str) -> dict:
    """Runs the fixed 50-scenario bank with alpha=0.0 for every step."""
    alpha_sched = AlphaSchedule(alpha_max=0.0, k_anneal=1)  # forces alpha=0 always
    results = {}
    test_bank = pipeline.load_test_bank(test_bank_path)
    for seed_id, scenario in test_bank.items():
        outcome = pipeline.run_episode(
            scenario,
            alpha_schedule=alpha_sched,
            iteration=0,
            max_steps=150
        )
        results[str(seed_id)] = {
            "reconnect_ticks": outcome.reconnect_ticks,
            "success": outcome.success,
            "final_reward": outcome.total_reward,
            "sum_rate": outcome.sum_rate
        }
    return results

def diff_results(baseline: dict, current: dict) -> list:
    mismatches = []
    for seed_id in baseline:
        if seed_id not in current:
            mismatches.append(f"seed {seed_id}: missing from current run")
            continue
        b, c = baseline[seed_id], current[seed_id]
        for key in ("reconnect_ticks", "success", "final_reward", "sum_rate"):
            if b[key] is None and c[key] is None:
                continue
            if (b[key] is None) != (c[key] is None):
                mismatches.append(f"seed {seed_id}.{key}: baseline={b[key]} current={c[key]}")
            elif isinstance(b[key], float):
                if not np.isclose(b[key], c[key], atol=TOLERANCE, rtol=0):
                    mismatches.append(
                        f"seed {seed_id}.{key}: baseline={b[key]} current={c[key]} (diff={abs(b[key] - c[key]):.6e})"
                    )
            elif b[key] != c[key]:
                mismatches.append(
                    f"seed {seed_id}.{key}: baseline={b[key]} current={c[key]}"
                )
    return mismatches

def main():
    pipeline = TrainPipeline.load_checkpoint(CHECKPOINT_PATH)
    generate_baseline_if_missing(pipeline, BASELINE_PATH, TEST_BANK_PATH)
    
    baseline = load_baseline(BASELINE_PATH)
    current = run_zero_effect_sweep(pipeline, TEST_BANK_PATH)

    mismatches = diff_results(baseline, current)
    if mismatches:
        print(f"ZERO-EFFECT CONTROL FAILED — {len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)

    print(f"Zero-effect control PASSED — {len(current)} seeds bit-identical to baseline.")
    sys.exit(0)

if __name__ == "__main__":
    main()
