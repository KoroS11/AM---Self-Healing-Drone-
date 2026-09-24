from enum import Enum

class AgentRole(Enum):
    """Role assigned to a UAV or ground node in the swarm network."""
    RELAY = "relay"
    GROUND_A = "ground_a"
    GROUND_B = "ground_b"

class AgentStatus(Enum):
    """Operational status of a swarm agent."""
    ACTIVE = "active"
    FAILED = "failed"

class RecoveryStrategy(Enum):
    """Self-healing network reorganization policy strategy."""
    NONE = "none"
    HEURISTIC_DARA = "heuristic_dara"
    HEURISTIC_CHAIN = "heuristic_chain"
    GREEDY_DARA = "greedy_dara"
    LEARNED_GNN = "learned_gnn"
