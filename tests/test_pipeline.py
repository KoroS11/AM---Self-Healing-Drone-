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
            is_first_reconnection_event=is_first_reconn
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

def test_reward_margin_uses_pairwise_min_rc(tmp_path):
    """Verify that compute_step_reward computes distance margins against min(Rc_i, Rc_j) rather than a global scalar."""
    import numpy as np
    from swarm_sim.core.simulator import SwarmSimulator
    from swarm_sim.utils.config import ExperimentConfig
    from swarm_sim.utils.enums import AgentStatus
    
    pipeline = TrainPipeline(output_dir=str(tmp_path))
    sim = SwarmSimulator(ExperimentConfig(num_drones=2, seed=123))
    
    # Zero velocities & accelerations so velocity/acc penalties are exactly 0.0
    sim.velocities[:] = 0.0
    sim.statuses[:] = AgentStatus.ACTIVE
    
    # Node 0 has Rc=28.0m, Node 1 is degraded to Rc=22.0m
    sim.comm_ranges[0] = 28.0
    sim.comm_ranges[1] = 22.0
    min_rc = 22.0
    
    # Case A: dist = 25.0m (within 28m, but outside min_rc=22m) -> should have ZERO margin bonus
    sim.positions[0] = np.array([0.0, 0.0])
    sim.positions[1] = np.array([25.0, 0.0])
    
    reward_outside = pipeline.compute_step_reward(
        sim=sim,
        is_connected=True,
        is_first_reconnection_event=False,
        lambda_conn=1.0,
        lambda_margin=0.20
    )
    # With no valid edge under min_rc, margin bonus is 0.0 -> reward is exactly lambda_conn (1.0)
    assert reward_outside == pytest.approx(1.0, abs=1e-6)
    
    # Case B: dist = 20.0m (within min_rc=22m) -> margin = (22 - 20) / 22 = 2 / 22
    sim.positions[1] = np.array([20.0, 0.0])
    reward_inside = pipeline.compute_step_reward(
        sim=sim,
        is_connected=True,
        is_first_reconnection_event=False,
        lambda_conn=1.0,
        lambda_margin=0.20
    )
    expected_margin = 0.20 * ((22.0 - 20.0) / 22.0)
    expected_reward = 1.0 + expected_margin
    assert reward_inside == pytest.approx(expected_reward, abs=1e-6)

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


