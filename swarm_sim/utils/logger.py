from typing import Optional, Dict, Any, List
import os
import pandas as pd
import numpy as np
import networkx as nx
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

class TelemetryLogger:
    """Two-level Pandas telemetry logger for simulation runs."""
    def __init__(self, run_id: str = "run_0", channel_model: Optional[ChannelModel] = None):
        self.run_id = run_id
        self.channel_model = channel_model if channel_model is not None else ChannelModel()
        
        # Per-timestep records
        self.timestep_records: List[Dict[str, Any]] = []
        
        # Per-run summary record
        self.summary_record: Optional[Dict[str, Any]] = None

    def log_timestep(self, sim: SwarmSimulator, graph: nx.Graph, timestep: int) -> None:
        """Record per-agent state row at given simulation timestep."""
        pos = sim.positions
        statuses = sim.statuses
        roles = sim.roles
        
        for i in range(sim.num_agents):
            deg = graph.degree(i) if i in graph else 0
            
            # Compute average link SNR/rate with neighbors
            snr_avg = 0.0
            rate_avg = 0.0
            neighbors = list(graph.neighbors(i)) if i in graph else []
            if neighbors:
                snrs_rates = [
                    self.channel_model.get_link_throughput(
                        pos[i], pos[n], is_backhaul=(roles[n].value.startswith("ground"))
                    ) for n in neighbors
                ]
                snr_avg = float(np.mean([s[0] for s in snrs_rates]))
                rate_avg = float(np.mean([s[1] for s in snrs_rates]))
                
            self.timestep_records.append({
                "run_id": self.run_id,
                "timestep": timestep,
                "agent_id": i,
                "role": roles[i].value,
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "status": statuses[i].value,
                "degree": deg,
                "avg_snr_db": snr_avg,
                "avg_rate_mbps": rate_avg
            })

    def log_run_summary(
        self,
        config: Any,
        seed: Optional[int],
        failure_timestep: int,
        failure_count: int,
        strategy: str,
        time_to_reconnect: Optional[float],
        final_connectivity: bool,
        min_connectivity_during_failure: float,
        final_sum_rate_mbps: float
    ) -> None:
        """Record run summary metrics."""
        self.summary_record = {
            "run_id": self.run_id,
            "seed": seed,
            "num_drones": config.num_drones,
            "comm_range": config.comm_range,
            "failure_timestep": failure_timestep,
            "failure_count": failure_count,
            "strategy": strategy,
            "time_to_reconnect": time_to_reconnect if time_to_reconnect is not None else np.nan,
            "final_connectivity": final_connectivity,
            "min_connectivity_during_failure": min_connectivity_during_failure,
            "final_sum_rate_mbps": final_sum_rate_mbps
        }

    def get_timestep_df(self) -> pd.DataFrame:
        """Return per-timestep Pandas DataFrame."""
        return pd.DataFrame(self.timestep_records)

    def get_summary_df(self) -> pd.DataFrame:
        """Return per-run summary Pandas DataFrame."""
        if self.summary_record is None:
            return pd.DataFrame()
        return pd.DataFrame([self.summary_record])

    def save_to_csv(self, timestep_csv_path: str, summary_csv_path: str) -> None:
        """Export telemetry dataframes to CSV files."""
        os.makedirs(os.path.dirname(os.path.abspath(timestep_csv_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        
        self.get_timestep_df().to_csv(timestep_csv_path, index=False)
        self.get_summary_df().to_csv(summary_csv_path, index=False)
