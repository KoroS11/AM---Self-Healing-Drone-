from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import networkx as nx
from swarm_sim.utils.enums import AgentStatus

if TYPE_CHECKING:
    from swarm_sim.core.simulator import SwarmSimulator

class SwarmTopologyManager:
    """Manages NetworkX communication graph construction and topology metric evaluation."""
    def __init__(self, simulator: "SwarmSimulator"):
        self.sim = simulator

    def build_graph(self) -> nx.Graph:
        """Construct NetworkX graph from active simulator nodes and neighbor pairs."""
        G = nx.Graph()
        active_indices = np.where(self.sim.statuses == AgentStatus.ACTIVE)[0]
        
        for idx in active_indices:
            agent = self.sim.get_agent(idx)
            G.add_node(
                idx,
                role=agent.role,
                status=agent.status,
                position=agent.position
            )
            
        pairs = self.sim.get_neighbor_pairs()
        for u, v in pairs:
            G.add_edge(u, v)
            
        return G

    def has_path(self, source_id: int, target_id: int, graph: Optional[nx.Graph] = None) -> bool:
        """Check if a connected path exists between source and target nodes."""
        G = graph if graph is not None else self.build_graph()
        if source_id not in G or target_id not in G:
            return False
        return nx.has_path(G, source_id, target_id)

    def get_shortest_path_length(self, source_id: int, target_id: int, graph: Optional[nx.Graph] = None) -> Optional[int]:
        """Return hop count between source and target nodes, or None if disconnected."""
        G = graph if graph is not None else self.build_graph()
        if source_id not in G or target_id not in G or not nx.has_path(G, source_id, target_id):
            return None
        return int(nx.shortest_path_length(G, source_id, target_id))

    def get_algebraic_connectivity(self, graph: Optional[nx.Graph] = None) -> float:
        """Compute algebraic connectivity (lambda_2 of Laplacian matrix).
        
        Explicitly catches disconnected graphs and graphs with <2 nodes, returning 0.0
        instead of raising NetworkXError.
        """
        G = graph if graph is not None else self.build_graph()
        if G.number_of_nodes() < 2 or not nx.is_connected(G):
            return 0.0
        try:
            return float(nx.algebraic_connectivity(G))
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            return 0.0

    def get_num_connected_components(self, graph: Optional[nx.Graph] = None) -> int:
        """Return number of connected components in current topology graph."""
        G = graph if graph is not None else self.build_graph()
        return int(nx.number_connected_components(G))

    def is_fully_connected(self, graph: Optional[nx.Graph] = None) -> bool:
        """Return True if active graph forms a single connected component."""
        G = graph if graph is not None else self.build_graph()
        return G.number_of_nodes() > 0 and nx.is_connected(G)

    def get_network_metrics(self, source_id: int, target_id: int) -> Dict[str, Any]:
        """Return summary metric dictionary by building NetworkX graph ONCE for high performance (NFR-1)."""
        G = self.build_graph()
        path_exists = self.has_path(source_id, target_id, graph=G)
        hop_count = self.get_shortest_path_length(source_id, target_id, graph=G) if path_exists else None
        
        return {
            "num_active_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_components": self.get_num_connected_components(graph=G),
            "algebraic_connectivity": self.get_algebraic_connectivity(graph=G),
            "path_exists": path_exists,
            "hop_count": hop_count
        }
