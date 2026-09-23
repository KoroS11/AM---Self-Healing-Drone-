import pytest
import numpy as np
import networkx as nx
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager

def test_topology_graph_building():
    config = ExperimentConfig(num_drones=3, comm_range=15.0)
    sim = SwarmSimulator(config)
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0]
    ])
    sim.initialize_positions(positions)
    
    top_mgr = SwarmTopologyManager(sim)
    G = top_mgr.build_graph()
    
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert set(G.edges()) == {(0, 1), (1, 2)}

def test_has_path_and_shortest_path():
    config = ExperimentConfig(num_drones=4, comm_range=12.0)
    sim = SwarmSimulator(config)
    positions = np.array([
        [0.0, 0.0],    # Node 0
        [10.0, 0.0],   # Node 1
        [20.0, 0.0],   # Node 2
        [100.0, 0.0]   # Node 3 (Disconnected)
    ])
    sim.initialize_positions(positions)
    top_mgr = SwarmTopologyManager(sim)
    
    # Path 0 -> 2 exists (0-1-2, 2 hops)
    assert top_mgr.has_path(0, 2) is True
    assert top_mgr.get_shortest_path_length(0, 2) == 2
    
    # Path 0 -> 3 does not exist
    assert top_mgr.has_path(0, 3) is False
    assert top_mgr.get_shortest_path_length(0, 3) is None

def test_algebraic_connectivity_connected_vs_disconnected():
    config = ExperimentConfig(num_drones=3, comm_range=15.0)
    sim = SwarmSimulator(config)
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0]
    ])
    sim.initialize_positions(positions)
    top_mgr = SwarmTopologyManager(sim)
    
    # Connected line graph: lambda_2 > 0
    lambda_2_connected = top_mgr.get_algebraic_connectivity()
    assert lambda_2_connected > 0.0
    
    # Disconnect graph by failing middle agent (node 1)
    sim.statuses[1] = AgentStatus.FAILED
    
    # Disconnected graph must safely return 0.0 without raising NetworkXError
    lambda_2_disconnected = top_mgr.get_algebraic_connectivity()
    assert lambda_2_disconnected == 0.0
    assert top_mgr.is_fully_connected() is False
    assert top_mgr.get_num_connected_components() == 2

def test_get_network_metrics_single_pass():
    config = ExperimentConfig(num_drones=3, comm_range=15.0)
    sim = SwarmSimulator(config)
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0]
    ])
    sim.initialize_positions(positions)
    top_mgr = SwarmTopologyManager(sim)
    
    metrics = top_mgr.get_network_metrics(0, 2)
    assert metrics["num_active_nodes"] == 3
    assert metrics["num_edges"] == 2
    assert metrics["num_components"] == 1
    assert metrics["path_exists"] is True
    assert metrics["hop_count"] == 2
    assert metrics["algebraic_connectivity"] > 0.0
