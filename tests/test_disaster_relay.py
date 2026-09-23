import pytest
import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario

def test_disaster_relay_line_relay_setup():
    config = ExperimentConfig(num_drones=6, comm_range=50.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    
    endpoint_a = (0.0, 0.0)
    endpoint_b = (200.0, 0.0)
    
    g_a_id, g_b_id = scenario.setup_scenario(endpoint_a, endpoint_b, topology="line_relay")
    
    assert g_a_id == 0
    assert g_b_id == 1
    assert np.allclose(sim.positions[0], [0.0, 0.0])
    assert np.allclose(sim.positions[1], [200.0, 0.0])
    assert sim.roles[0] == AgentRole.GROUND_A
    assert sim.roles[1] == AgentRole.GROUND_B
    assert sim.roles[2] == AgentRole.RELAY
    
    top_mgr = SwarmTopologyManager(sim)
    assert top_mgr.has_path(0, 1) is True
    assert top_mgr.is_fully_connected() is True

def test_disaster_relay_grid_lattice_diagonal():
    """Verify grid_lattice works cleanly on non-axis-aligned diagonal paths."""
    config = ExperimentConfig(num_drones=8, comm_range=50.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    
    endpoint_a = (0.0, 0.0)
    endpoint_b = (100.0, 100.0)
    
    g_a_id, g_b_id = scenario.setup_scenario(endpoint_a, endpoint_b, topology="grid_lattice")
    top_mgr = SwarmTopologyManager(sim)
    assert top_mgr.has_path(g_a_id, g_b_id) is True
    assert top_mgr.is_fully_connected() is True

def test_disaster_relay_grid_lattice_capacity_validation():
    """Verify grid_lattice raises clear ValueError when num_relays is insufficient."""
    config = ExperimentConfig(num_drones=26, comm_range=50.0)  # N=26 -> 2 ground + 24 relays
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    
    # ab_dist = 1000.0, comm_range = 50.0 -> spacing = 40.0 -> num_cols = ceil(1000/40) = 25
    # Needs at least 25 relays, but we only have 24!
    endpoint_a = (0.0, 0.0)
    endpoint_b = (1000.0, 0.0)
    
    with pytest.raises(ValueError, match="grid_lattice needs at least 25 relay agents"):
        scenario.setup_scenario(endpoint_a, endpoint_b, topology="grid_lattice")
        
    # Increasing to N=27 (2 ground + 25 relays) must succeed
    config_27 = ExperimentConfig(num_drones=27, comm_range=50.0)
    sim_27 = SwarmSimulator(config_27)
    scenario_27 = DisasterRelayScenario(sim_27)
    g_a, g_b = scenario_27.setup_scenario(endpoint_a, endpoint_b, topology="grid_lattice")
    top_mgr_27 = SwarmTopologyManager(sim_27)
    assert top_mgr_27.has_path(g_a, g_b) is True

def test_disaster_relay_grid_lattice_direct_range_bypass():
    """Verify direct-range bypass (ab_dist <= rc) succeeds with 0 relays and with extra relays."""
    # Case A: 0 relays
    config_0 = ExperimentConfig(num_drones=2, comm_range=50.0)
    sim_0 = SwarmSimulator(config_0)
    scenario_0 = DisasterRelayScenario(sim_0)
    g_a, g_b = scenario_0.setup_scenario((0.0, 0.0), (30.0, 0.0), topology="grid_lattice")
    assert SwarmTopologyManager(sim_0).has_path(g_a, g_b) is True

    # Case B: 30 extra relays (ab_dist <= rc) -> centered bounded grid, stays fully connected
    config_30 = ExperimentConfig(num_drones=32, comm_range=100.0)  # 2 ground + 30 relays
    sim_30 = SwarmSimulator(config_30)
    scenario_30 = DisasterRelayScenario(sim_30)
    g_a30, g_b30 = scenario_30.setup_scenario((0.0, 0.0), (20.0, 0.0), topology="grid_lattice")
    top_30 = SwarmTopologyManager(sim_30)
    assert top_30.has_path(g_a30, g_b30) is True
    assert top_30.is_fully_connected() is True

def test_disaster_relay_random_scatter_guaranteed_connectivity():
    config = ExperimentConfig(num_drones=10, comm_range=50.0, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    
    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (120.0, 0.0), topology="random_scatter")
    top_mgr = SwarmTopologyManager(sim)
    assert top_mgr.has_path(g_a_id, g_b_id) is True
    assert top_mgr.is_fully_connected() is True

def test_ground_endpoints_stationarity():
    """Verify Ground Endpoint A and B positions remain strictly unchanged after step simulation with accelerations."""
    config = ExperimentConfig(num_drones=5, dt=0.1, comm_range=50.0)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    
    pos_a = (0.0, 0.0)
    pos_b = (100.0, 0.0)
    scenario.setup_scenario(pos_a, pos_b, topology="line_relay")
    
    # Try applying large non-zero accelerations to all agents
    accelerations = np.full((5, 2), 50.0, dtype=np.float64)
    
    for _ in range(50):
        sim.step(accelerations)
        
    # Ground Endpoint positions MUST be strictly unchanged
    assert np.allclose(sim.positions[0], [0.0, 0.0])
    assert np.allclose(sim.positions[1], [100.0, 0.0])
    assert sim.velocities[0].tolist() == [0.0, 0.0]
    assert sim.velocities[1].tolist() == [0.0, 0.0]
