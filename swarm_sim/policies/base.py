from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import networkx as nx
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.utils.enums import RecoveryStrategy

class BasePolicy(ABC):
    """Abstract base class for all swarm recovery policies.
    
    Subclasses must implement compute_control_forces() to generate an (N, 2) array of
    acceleration vectors for active agents.
    """
    def __init__(self, simulator: SwarmSimulator, strategy: RecoveryStrategy = RecoveryStrategy.NONE):
        self.sim = simulator
        self.strategy = strategy

    @abstractmethod
    def compute_control_forces(self, graph: Optional[nx.Graph] = None) -> np.ndarray:
        """Compute acceleration vectors for active agents.
        
        Args:
            graph: Optional pre-built NetworkX graph to preserve single-pass performance.
            
        Returns:
            np.ndarray of shape (N, 2) containing acceleration vectors.
        """
        pass
