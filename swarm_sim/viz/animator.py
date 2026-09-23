from typing import Optional, Dict, Any, List
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from PIL import Image
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.utils.enums import AgentRole, AgentStatus

class SwarmAnimator:
    """2D Matplotlib Swarm Visualizer rendering simulation frames to Pillow Images."""
    def __init__(self, figsize: tuple = (8, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def render_frame(
        self,
        sim: SwarmSimulator,
        graph: Optional[nx.Graph] = None,
        timestep: int = 0,
        title: Optional[str] = None
    ) -> Image.Image:
        """Render a single 2D simulation frame as a Pillow Image."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        pos = sim.positions
        statuses = sim.statuses
        roles = sim.roles
        N = sim.num_agents
        
        # Plot edges
        if graph is not None:
            for u, v in graph.edges():
                ax.plot(
                    [pos[u, 0], pos[v, 0]],
                    [pos[u, 1], pos[v, 1]],
                    color='#38bdf8', alpha=0.6, linewidth=1.5, zorder=1
                )
                
        # Plot active mobile relays
        active_relays = np.where((statuses == AgentStatus.ACTIVE) & (roles == AgentRole.RELAY))[0]
        if active_relays.size > 0:
            ax.scatter(
                pos[active_relays, 0], pos[active_relays, 1],
                c='#3b82f6', s=80, marker='o', label='Active Relay', zorder=3
            )
            for idx in active_relays:
                ax.annotate(str(idx), (pos[idx, 0], pos[idx, 1]), color='white', fontsize=8, ha='center', va='center', zorder=4)

        # Plot ground endpoints
        ground_a = np.where(roles == AgentRole.GROUND_A)[0]
        if ground_a.size > 0:
            ax.scatter(pos[ground_a, 0], pos[ground_a, 1], c='#22c55e', s=150, marker='s', label='Ground A', zorder=3)
            
        ground_b = np.where(roles == AgentRole.GROUND_B)[0]
        if ground_b.size > 0:
            ax.scatter(pos[ground_b, 0], pos[ground_b, 1], c='#10b981', s=150, marker='^', label='Ground B', zorder=3)

        # Plot failed agents
        failed = np.where(statuses == AgentStatus.FAILED)[0]
        if failed.size > 0:
            ax.scatter(pos[failed, 0], pos[failed, 1], c='#ef4444', s=120, marker='x', label='Failed UAV', zorder=5)

        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        frame_title = title if title is not None else f"Swarm Simulator - Timestep {timestep}"
        ax.set_title(frame_title, fontsize=12, fontweight='bold')
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend(loc='upper right', fontsize=8)
        
        # Save figure to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf)
        return img.convert('RGB')
