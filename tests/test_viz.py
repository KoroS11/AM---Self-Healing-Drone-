import os
import numpy as np
import pytest
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.viz import SwarmAnimator, GIFExporter

def test_swarm_animator_and_gif_exporter(tmp_path):
    config = ExperimentConfig(num_drones=5, comm_range=28.0)
    sim = SwarmSimulator(config)
    topo = SwarmTopologyManager(sim)
    
    pos = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [40.0, 0.0],
        [60.0, 0.0],
        [80.0, 0.0]
    ])
    sim.initialize_positions(pos)
    
    animator = SwarmAnimator()
    exporter = GIFExporter()
    
    G = topo.build_graph()
    
    for t in range(3):
        frame = animator.render_frame(sim, graph=G, timestep=t)
        assert frame is not None
        exporter.add_frame(frame)
        
    gif_path = os.path.join(tmp_path, "test_output.gif")
    exporter.save_gif(gif_path, fps=5)
    
    assert os.path.exists(gif_path)
    assert os.path.getsize(gif_path) > 0
