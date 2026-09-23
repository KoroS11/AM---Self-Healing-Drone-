import pytest
import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario

def test_single_agent_failure():
    config = ExperimentConfig(num_drones=3, comm_range=15.0)
    sim = SwarmSimulator(config)
    injector = FailureInjector(sim)
    
    sim.velocities[1] = np.array([5.0, 5.0])
    injector.fail_agent(1)
    
    assert sim.statuses[1] == AgentStatus.FAILED
    assert np.array_equal(sim.velocities[1], [0.0, 0.0])

def test_tick_ordering_zero_kinematic_motion_on_failure_tick():
    """Verify that when an agent is failed before simulator.step(), its position is strictly unchanged."""
    config = ExperimentConfig(num_drones=3, dt=0.1, comm_range=15.0)
    sim = SwarmSimulator(config)
    injector = FailureInjector(sim)
    
    sim.positions[1] = np.array([10.0, 10.0])
    sim.velocities[1] = np.array([5.0, 5.0])
    
    # 1. Failure injection BEFORE simulator.step()
    injector.fail_agent(1)
    
    # 2. Simulator step
    sim.step()
    
    # Position MUST remain strictly at [10.0, 10.0]
    assert np.array_equal(sim.positions[1], [10.0, 10.0])

def test_ground_endpoint_protection():
    config = ExperimentConfig(num_drones=3, comm_range=15.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    scenario.setup_scenario((0.0, 0.0), (30.0, 0.0))
    
    injector = FailureInjector(sim)
    
    # Singular fail_agent on ground node without force MUST raise ValueError
    with pytest.raises(ValueError, match="Cannot fail ground endpoint"):
        injector.fail_agent(0, force=False)
        
    # Singular fail_agent on ground node WITH force=True MUST succeed
    injector.fail_agent(0, force=True)
    assert sim.statuses[0] == AgentStatus.FAILED

def test_bulk_fail_agents_skips_ground_nodes():
    config = ExperimentConfig(num_drones=4, comm_range=15.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    scenario.setup_scenario((0.0, 0.0), (45.0, 0.0))
    
    injector = FailureInjector(sim)
    
    # Try bulk-failing agents [0, 2, 3] where 0 is GROUND_A
    failed_ids = injector.fail_agents([0, 2, 3], force=False)
    
    # Should skip 0 gracefully and return [2, 3]
    assert failed_ids == [2, 3]
    assert sim.statuses[0] == AgentStatus.ACTIVE
    assert sim.statuses[2] == AgentStatus.FAILED
    assert sim.statuses[3] == AgentStatus.FAILED

def test_fail_spatial_zone():
    config = ExperimentConfig(num_drones=5, comm_range=15.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    scenario.setup_scenario((0.0, 0.0), (60.0, 0.0))
    
    injector = FailureInjector(sim)
    
    # Fail spatial zone around relay 2 position
    pos_2 = sim.positions[2]
    failed_ids = injector.fail_spatial_zone(tuple(pos_2), radius=5.0)
    
    assert 2 in failed_ids
    assert sim.statuses[2] == AgentStatus.FAILED

def test_one_step_fragmentation_detection():
    """Verify that injected failures are correctly detected as fragmentation events within 1 simulation step."""
    config = ExperimentConfig(num_drones=5, comm_range=15.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario((0.0, 0.0), (60.0, 0.0), topology="line_relay")
    
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    
    # Initial healthy check
    frag_before = injector.check_fragmentation(top_mgr, g_a, g_b)
    assert frag_before["is_fragmented"] is False
    assert frag_before["path_exists"] is True
    assert frag_before["num_components"] == 1
    
    # Inject failure in middle relay node 2
    injector.fail_agent(2)
    sim.step()  # 1 simulation step
    
    # Immediate 1-step fragmentation detection check
    frag_after = injector.check_fragmentation(top_mgr, g_a, g_b)
    assert frag_after["is_fragmented"] is True
    assert frag_after["path_exists"] is False
    assert frag_after["num_components"] > 1
    assert frag_after["algebraic_connectivity"] == 0.0
