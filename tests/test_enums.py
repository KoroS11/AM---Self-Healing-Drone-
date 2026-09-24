import pytest
from swarm_sim.utils.enums import AgentRole, AgentStatus, RecoveryStrategy

def test_agent_role_enum_values():
    assert AgentRole.RELAY.value == "relay"
    assert AgentRole.GROUND_A.value == "ground_a"
    assert AgentRole.GROUND_B.value == "ground_b"

def test_agent_status_enum_values():
    assert AgentStatus.ACTIVE.value == "active"
    assert AgentStatus.FAILED.value == "failed"

def test_recovery_strategy_enum_values():
    assert RecoveryStrategy.NONE.value == "none"
    assert RecoveryStrategy.HEURISTIC_DARA.value == "heuristic_dara"
    assert RecoveryStrategy.HEURISTIC_CHAIN.value == "heuristic_chain"
    assert RecoveryStrategy.LEARNED_GNN.value == "learned_gnn"
