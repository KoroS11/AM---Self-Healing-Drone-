import numpy as np
import pytest
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.utils.enums import AgentRole, AgentStatus

def test_greedy_dara_policy_init_and_step():
    config = ExperimentConfig(num_drones=9, comm_range=28.0)
    sim = SwarmSimulator(config)
    
    positions = np.array([
        [0.0, 0.0],    # GROUND_A
        [25.0, 0.0],   # RELAY 1
        [50.0, 0.0],   # RELAY 2
        [75.0, 0.0],   # RELAY 3 (Failed)
        [100.0, 0.0],  # RELAY 4
        [125.0, 0.0],  # RELAY 5
        [150.0, 0.0],  # RELAY 6
        [175.0, 0.0],  # RELAY 7
        [200.0, 0.0],  # GROUND_B
    ])
    roles = [
        AgentRole.GROUND_A,
        AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY,
        AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY,
        AgentRole.GROUND_B
    ]
    sim.initialize_positions(positions, roles)
    sim.max_speeds[0] = 0.0
    sim.max_speeds[8] = 0.0
    
    injector = FailureInjector(sim)
    injector.fail_agent(3)  # Fail relay 3
    
    policy = GreedyDARAHeuristicPolicy(sim)
    accelerations = policy.compute_control_forces()
    
    assert accelerations.shape == (9, 2)
    # Ground nodes stationary
    assert np.allclose(accelerations[0], [0.0, 0.0])
    assert np.allclose(accelerations[8], [0.0, 0.0])
    # Failed node acceleration zeroed
    assert np.allclose(accelerations[3], [0.0, 0.0])
    
    # Active relays should have forces computed
    active_forces = np.linalg.norm(accelerations[[1, 2, 4, 5, 6, 7]], axis=1)
    assert np.any(active_forces > 0.0)

def test_greedy_dara_simulation_loop():
    config = ExperimentConfig(num_drones=9, comm_range=28.0)
    sim = SwarmSimulator(config)
    
    positions = np.array([
        [0.0, 0.0],    # GROUND_A
        [25.0, 0.0],
        [50.0, 0.0],
        [75.0, 0.0],   # RELAY 3 (Failed)
        [100.0, 0.0],
        [125.0, 0.0],
        [150.0, 0.0],
        [175.0, 0.0],
        [200.0, 0.0],  # GROUND_B
    ])
    roles = [
        AgentRole.GROUND_A,
        AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY,
        AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY,
        AgentRole.GROUND_B
    ]
    sim.initialize_positions(positions, roles)
    sim.max_speeds[0] = 0.0
    sim.max_speeds[8] = 0.0
    
    injector = FailureInjector(sim)
    injector.fail_agent(3)
    
    policy = GreedyDARAHeuristicPolicy(sim)
    
    # Step simulation 10 steps while disconnected (lexicographic Mode 1: Pure DARA)
    for _ in range(10):
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
    assert sim.positions.shape == (9, 2)
    assert policy.ticks_pure_dara == 10
    assert policy.ticks_throughput_polishing == 0
    assert policy.total_solve_count == 0

def test_cbf_static_tight_margin_scenario():
    """Unit test (a): Static tight-margin scenario asserting no edge ever exceeds R_c after integration."""
    rc = 28.0
    config = ExperimentConfig(num_drones=3, comm_range=rc)
    sim = SwarmSimulator(config)
    
    # Tight initial margin: distance is 27.5m (close to 28.0m limit)
    positions = np.array([
        [0.0, 0.0],
        [27.5, 0.0],
        [55.0, 0.0]
    ])
    roles = [AgentRole.GROUND_A, AgentRole.RELAY, AgentRole.GROUND_B]
    sim.initialize_positions(positions, roles)
    sim.max_speeds[0] = 0.0
    sim.max_speeds[2] = 0.0
    
    policy = GreedyDARAHeuristicPolicy(sim, k_tp=10.0, alpha1=2.0, alpha2=2.0)
    
    # Step for 20 ticks (network is connected, so enters Mode 2: CBF-QP polishing)
    for _ in range(20):
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        d01 = np.linalg.norm(sim.positions[1] - sim.positions[0])
        d12 = np.linalg.norm(sim.positions[1] - sim.positions[2])
        # Assert no edge exceeds communication range
        assert d01 <= rc + 1e-3, f"Edge (0, 1) violated R_c: {d01} > {rc}"
        assert d12 <= rc + 1e-3, f"Edge (1, 2) violated R_c: {d12} > {rc}"

    assert policy.ticks_throughput_polishing == 20
    assert policy.total_solve_count == 20

def test_cbf_joint_coupled_two_relays_moving_apart():
    """Unit test (b): Scenario with two mobile relays moving simultaneously away from each other toward constraint boundary."""
    rc = 28.0
    config = ExperimentConfig(num_drones=4, comm_range=rc)
    sim = SwarmSimulator(config)
    
    # Two mobile relays at 20.0 and 46.0 (separation 26.0m, tight margin to 28.0m)
    positions = np.array([
        [0.0, 0.0],     # Ground A
        [20.0, 0.0],    # Mobile Relay 1
        [46.0, 0.0],    # Mobile Relay 2
        [66.0, 0.0],    # Ground B
    ])
    roles = [AgentRole.GROUND_A, AgentRole.RELAY, AgentRole.RELAY, AgentRole.GROUND_B]
    sim.initialize_positions(positions, roles)
    sim.max_speeds[0] = 0.0
    sim.max_speeds[3] = 0.0
    
    # Give both relays initial diverging velocities toward the constraint boundary
    sim.velocities[1] = np.array([-2.0, 0.0])  # Relay 1 moving left
    sim.velocities[2] = np.array([2.0, 0.0])   # Relay 2 moving right (relative speed 4.0 m/s apart!)
    
    policy = GreedyDARAHeuristicPolicy(sim, k_tp=5.0, alpha1=2.0, alpha2=2.0)
    
    for _ in range(15):
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        # Check coupled inter-relay distance
        d12 = np.linalg.norm(sim.positions[1] - sim.positions[2])
        d01 = np.linalg.norm(sim.positions[0] - sim.positions[1])
        d23 = np.linalg.norm(sim.positions[2] - sim.positions[3])
        
        assert d12 <= rc + 1e-3, f"Coupled edge (1, 2) violated R_c: {d12} > {rc}"
        assert d01 <= rc + 1e-3, f"Edge (0, 1) violated R_c: {d01} > {rc}"
        assert d23 <= rc + 1e-3, f"Edge (2, 3) violated R_c: {d23} > {rc}"

    assert policy.ticks_throughput_polishing == 15
    assert policy.total_solve_count == 15
