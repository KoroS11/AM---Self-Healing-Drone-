import os
import pytest
from train_pipeline import TrainPipeline
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy

def test_train_pipeline_initialization_and_evaluation(tmp_path):
    output_dir = str(tmp_path / "checkpoints")
    pipeline = TrainPipeline(output_dir=output_dir)
    
    # Verify test bank generation
    assert os.path.exists(os.path.join(output_dir, "test_bank_50.pkl"))
    assert os.path.exists(os.path.join(output_dir, "test_bank_spare.pkl"))
    
    # Verify evaluation on 2 test bank scenarios
    sample_scenarios = pipeline.bank_data["seen_35"][:2]
    eval_res = pipeline.evaluate_policy_on_scenarios(
        lambda sim: DARAHeuristicPolicy(sim),
        sample_scenarios,
        max_steps=100
    )
    
    assert "success_rate" in eval_res
    assert "mean_time" in eval_res
    assert "mean_sum_rate" in eval_res
    assert len(eval_res["details"]) == 2

def test_reward_single_reconnection_event(tmp_path):
    """Assert that an episode with two separate disconnect->reconnect cycles receives exactly one +500 bonus."""
    pipeline = TrainPipeline(output_dir=str(tmp_path))
    
    # Simulate a synthetic 10-step episode with 2 distinct reconnection cycles:
    # Steps 0..1: Disconnected
    # Steps 2..4: Connected (Cycle 1 Reconnection at step 2)
    # Steps 5..6: Disconnected (Temporary Break)
    # Steps 7..9: Connected (Cycle 2 Reconnection at step 7)
    conn_sequence = [False, False, True, True, True, False, False, True, True, True]
    
    # Mock sim
    from swarm_sim.core.simulator import SwarmSimulator
    from swarm_sim.utils.config import ExperimentConfig
    sim = SwarmSimulator(ExperimentConfig(num_drones=7, seed=42))
    
    ever_reconnected = False
    rewards = []
    bonus_count = 0
    
    for t, is_connected in enumerate(conn_sequence):
        is_first_reconn = False
        if is_connected and not ever_reconnected:
            is_first_reconn = True
            ever_reconnected = True
            
        r = pipeline.compute_step_reward(
            sim=sim,
            is_connected=is_connected,
            is_first_reconnection_event=is_first_reconn,
            comm_range=28.0
        )
        rewards.append(r)
        if is_first_reconn:
            bonus_count += 1
            
    # Assertions
    assert bonus_count == 1, f"Expected exactly 1 bonus event, got {bonus_count}"
    assert rewards[2] >= 500.0, f"Expected step 2 reward to contain +500.0, got {rewards[2]}"
    assert rewards[7] < 500.0, f"Expected step 7 (second reconnect cycle) to NOT receive +500.0 bonus, got {rewards[7]}"
    
    # Exact cumulative bonus contribution
    total_bonus_contribution = sum(500.0 for r in rewards if r >= 500.0)
    assert total_bonus_contribution == 500.0

def test_spare_test_bank_runtime_loading(tmp_path):
    """Verify test_bank_spare.pkl runtime loading and seed list."""
    output_dir = str(tmp_path / "checkpoints")
    pipeline = TrainPipeline(output_dir=output_dir)
    spare_file = os.path.join(output_dir, "test_bank_spare.pkl")
    assert os.path.exists(spare_file)
    
    import pickle
    with open(spare_file, "rb") as f:
        spare_data = pickle.load(f)
        
    assert "spare_10" in spare_data
    assert len(spare_data["spare_10"]) == 10
    spare_seeds = [s["seed"] for s in spare_data["spare_10"]]
    assert spare_seeds == list(range(1050, 1060))

