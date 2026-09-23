import numpy as np
import pytest
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager

def test_channel_model_path_loss():
    channel = ChannelModel()
    
    # Path loss should increase with distance
    pl_near = channel.compute_a2g_path_loss(10.0)
    pl_far = channel.compute_a2g_path_loss(100.0)
    assert pl_far > pl_near
    
    # Cellular backhaul path loss
    pl_cell = channel.compute_cellular_backhaul_path_loss(50.0)
    assert pl_cell > 0.0

def test_channel_model_throughput():
    channel = ChannelModel()
    
    snr, rate = channel.compute_achievable_rate(path_loss_db=70.0)
    assert snr > 0.0
    assert rate > 0.0
    
    # Higher path loss -> lower rate
    _, rate_high_pl = channel.compute_achievable_rate(path_loss_db=100.0)
    assert rate > rate_high_pl

def test_link_throughput_and_network_throughput():
    channel = ChannelModel()
    config = ExperimentConfig(num_drones=5, comm_range=28.0)
    sim = SwarmSimulator(config)
    
    pos = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [40.0, 0.0],
        [60.0, 0.0],
        [80.0, 0.0]
    ])
    sim.initialize_positions(pos)
    
    topo = SwarmTopologyManager(sim)
    G = topo.build_graph()
    
    net_throughput = channel.compute_network_throughput(sim, G)
    assert net_throughput > 0.0

def test_precompute_r_ref():
    channel = ChannelModel()
    r_ref = channel.precompute_r_ref(num_relays=7, dist_ab=223.6)
    assert r_ref > 0.0
