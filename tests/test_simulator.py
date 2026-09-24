import time
import pytest
import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator

def test_simulator_initialization_and_kinematics():
    config = ExperimentConfig(num_drones=3, dt=0.1, max_speed=5.0)
    sim = SwarmSimulator(config)
    
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0]
    ])
    sim.initialize_positions(positions)
    
    # Apply initial velocities (one exceeding max_speed)
    sim.velocities = np.array([
        [2.0, 0.0],
        [0.0, 10.0],  # Should be clamped to max_speed (5.0)
        [0.0, 0.0]
    ])
    
    sim.step()
    
    # Check agent 0 position: [0, 0] + [2, 0]*0.1 = [0.2, 0.0]
    assert np.allclose(sim.positions[0], [0.2, 0.0])
    
    # Check agent 1 velocity clamping: magnitude was 10.0 -> clamped to 5.0
    assert np.allclose(sim.velocities[1], [0.0, 5.0])
    # Check agent 1 position: [10, 0] + [0, 5]*0.1 = [10.0, 0.5]
    assert np.allclose(sim.positions[1], [10.0, 0.5])

def test_simulator_bounds_clamping():
    config = ExperimentConfig(num_drones=1, dt=1.0, max_speed=100.0, bounds=(0.0, 10.0, 0.0, 10.0))
    sim = SwarmSimulator(config)
    sim.positions = np.array([[9.0, 9.0]])
    sim.velocities = np.array([[5.0, 5.0]])
    
    sim.step()
    
    # Clamped to bounds max (10.0, 10.0)
    assert np.allclose(sim.positions[0], [10.0, 10.0])

def test_spatial_tree_reindexing_and_neighbor_pairs():
    config = ExperimentConfig(num_drones=5, comm_range=15.0)
    sim = SwarmSimulator(config)
    
    # Linear layout: 0 --(10)-- 1 --(10)-- 2 --(10)-- 3 --(10)-- 4
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
        [40.0, 0.0]
    ])
    sim.initialize_positions(positions)
    
    # Initially all active: pairs within 15.0 should be (0,1), (1,2), (2,3), (3,4)
    pairs = sim.get_neighbor_pairs()
    expected_pairs = {(0, 1), (1, 2), (2, 3), (3, 4)}
    assert set(pairs) == expected_pairs
    
    # Fail agent 2!
    sim.statuses[2] = AgentStatus.FAILED
    
    # Remaining active: 0, 1, 3, 4
    # Distance (1, 3) is 20.0 > 15.0, so valid active pairs are (0,1) and (3,4)
    pairs_after_failure = sim.get_neighbor_pairs()
    expected_after_failure = {(0, 1), (3, 4)}
    assert set(pairs_after_failure) == expected_after_failure

def test_single_active_agent_edge_case():
    """Verify velocity clamping and step execution when exactly 1 agent is active."""
    config = ExperimentConfig(num_drones=3, dt=0.1, max_speed=5.0)
    sim = SwarmSimulator(config)
    sim.statuses = np.array([AgentStatus.FAILED, AgentStatus.ACTIVE, AgentStatus.FAILED], dtype=object)
    sim.positions[1] = np.array([5.0, 5.0])
    sim.velocities[1] = np.array([0.0, 20.0])  # Exceeds max_speed
    
    sim.step()
    
    # Single active agent velocity should be clamped cleanly without 0-d scalar array errors
    assert np.allclose(sim.velocities[1], [0.0, 5.0])
    assert np.allclose(sim.positions[1], [5.0, 5.5])

def test_performance_benchmark_n100():
    """Verify simulation step performance exceeds 20 Hz for N=100 agents (NFR-1)."""
    config = ExperimentConfig(num_drones=100, dt=0.1, comm_range=20.0, seed=42)
    sim = SwarmSimulator(config)
    
    # Random initial positions and velocities
    sim.positions = np.random.uniform(0.0, 200.0, size=(100, 2))
    sim.velocities = np.random.uniform(-5.0, 5.0, size=(100, 2))
    
    num_steps = 500
    start_time = time.perf_counter()
    for _ in range(num_steps):
        sim.step()
        _ = sim.get_neighbor_pairs()
    elapsed = time.perf_counter() - start_time
    
    hz = num_steps / elapsed
    assert hz >= 20.0, f"Performance target failed: achieved {hz:.2f} Hz < 20 Hz"
