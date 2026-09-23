import os
import numpy as np
import pytest
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.utils.logger import TelemetryLogger

def test_telemetry_logger_logging_and_csv(tmp_path):
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
    
    logger = TelemetryLogger(run_id="test_run_1")
    G = topo.build_graph()
    
    logger.log_timestep(sim, G, timestep=0)
    logger.log_timestep(sim, G, timestep=1)
    
    logger.log_run_summary(
        config=config,
        seed=42,
        failure_timestep=10,
        failure_count=1,
        strategy="heuristic_dara",
        time_to_reconnect=12.5,
        final_connectivity=True,
        min_connectivity_during_failure=0.0,
        final_sum_rate_mbps=150.0
    )
    
    df_ts = logger.get_timestep_df()
    df_sum = logger.get_summary_df()
    
    assert len(df_ts) == 10  # 5 agents * 2 timesteps
    assert len(df_sum) == 1
    
    ts_csv = os.path.join(tmp_path, "timestep_log.csv")
    sum_csv = os.path.join(tmp_path, "summary_log.csv")
    
    logger.save_to_csv(ts_csv, sum_csv)
    
    assert os.path.exists(ts_csv)
    assert os.path.exists(sum_csv)
