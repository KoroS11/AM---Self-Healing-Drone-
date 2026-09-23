import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario

# Artifact target directory for embedding media
ARTIFACT_DIR = r"C:\Users\harsh\.gemini\antigravity-ide\brain\3008cc19-832e-42b2-b2d0-33c5c21b5d5f"

def run_demo():
    endpoint_a = (0.0, 0.0)
    endpoint_b = (200.0, 100.0)
    topologies = ["line_relay", "grid_lattice", "random_scatter"]
    
    print("==================================================================")
    print("               SWARM Phase 2 Topology Demo Output                 ")
    print("==================================================================")
    
    for topo in topologies:
        config = ExperimentConfig(
            num_drones=15,
            comm_range=40.0,
            dt=0.1,
            seed=42
        )
        sim = SwarmSimulator(config)
        scenario = DisasterRelayScenario(sim)
        top_mgr = SwarmTopologyManager(sim)
        
        g_a_id, g_b_id = scenario.setup_scenario(endpoint_a, endpoint_b, topology=topo)
        metrics = top_mgr.get_network_metrics(g_a_id, g_b_id)
        
        print(f"\n--- Topology Mode: {topo.upper()} ---")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:<24}: {val:.4f}")
            else:
                print(f"  {key:<24}: {val}")
                
        # Matplotlib visualization
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Draw communication edges
        pairs = sim.get_neighbor_pairs()
        for u, v in pairs:
            pos_u = sim.positions[u]
            pos_v = sim.positions[v]
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
            
        # Draw agents color-coded by role
        roles = sim.roles
        relays = [i for i in range(sim.num_agents) if roles[i] == AgentRole.RELAY]
        
        ax.scatter(sim.positions[relays, 0], sim.positions[relays, 1], color="#1f77b4", label="Relay UAV", s=80, zorder=3)
        ax.scatter(sim.positions[g_a_id, 0], sim.positions[g_a_id, 1], color="#d62728", marker="s", label="Ground Endpoint A", s=140, zorder=4)
        ax.scatter(sim.positions[g_b_id, 0], sim.positions[g_b_id, 1], color="#2ca02c", marker="s", label="Ground Endpoint B", s=140, zorder=4)
        
        ax.set_title(f"SWARM Topology: {topo} (N=15, Rc=40.0)", fontsize=13, fontweight="bold")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")
        
        plt.tight_layout()
        
        png_name = f"demo_{topo}.png"
        local_png_path = os.path.abspath(png_name)
        plt.savefig(local_png_path, dpi=150)
        plt.close(fig)
        
        # Copy PNG to artifact directory for display
        if os.path.exists(ARTIFACT_DIR):
            artifact_png_path = os.path.join(ARTIFACT_DIR, png_name)
            shutil.copy(local_png_path, artifact_png_path)
            
        print(f"  Saved plot to: {local_png_path}")

if __name__ == "__main__":
    run_demo()
