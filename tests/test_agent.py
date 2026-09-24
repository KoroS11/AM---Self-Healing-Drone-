import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator

def test_uav_agent_read_only_proxy():
    config = ExperimentConfig(num_drones=3, comm_range=40.0, max_speed=8.0)
    sim = SwarmSimulator(config)
    
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0]
    ])
    roles = [AgentRole.GROUND_A, AgentRole.RELAY, AgentRole.GROUND_B]
    sim.initialize_positions(positions, roles)
    
    agent_1 = sim.get_agent(1)
    assert agent_1.agent_id == 1
    assert np.array_equal(agent_1.position, np.array([10.0, 0.0]))
    assert agent_1.role == AgentRole.RELAY
    assert agent_1.status == AgentStatus.ACTIVE
    assert agent_1.comm_range == 40.0
    assert agent_1.max_speed == 8.0
    
    # Verify modifying agent position matrix updates proxy view dynamically
    sim.positions[1] = np.array([12.5, 3.0])
    assert np.array_equal(agent_1.position, np.array([12.5, 3.0]))
