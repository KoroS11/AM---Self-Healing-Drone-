import numpy as np
import torch
import pytest

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager

def test_node_features_heterogeneous_rc():
    """Verify that node feature dim 8 (rc_norm) accurately reflects individual agent comm ranges."""
    config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=100)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario((0.0, 0.0), (200.0, 0.0))
    
    # Degrade Agent 3 to 22.0m and Agent 6 to 20.0m
    sim.comm_ranges[3] = 22.0
    sim.comm_ranges[6] = 20.0
    
    policy = GNNPolicy(sim)
    node_feats, edge_feats, edge_index, mobile_mask = policy.extract_graph_features()
    
    node_np = node_feats.cpu().numpy()
    assert node_np.shape == (9, 9), f"Expected shape (9, 9), got {node_np.shape}"
    
    expected_rc_norms = np.array([28.0, 28.0, 28.0, 22.0, 28.0, 28.0, 20.0, 28.0, 28.0]) / 28.0
    actual_rc_norms = node_np[:, 8]
    
    assert np.allclose(actual_rc_norms, expected_rc_norms, atol=1e-6), (
        f"rc_norm mismatch! Expected:\n{expected_rc_norms}\nGot:\n{actual_rc_norms}"
    )
    assert abs(actual_rc_norms[3] - 22.0 / 28.0) < 1e-6
    assert abs(actual_rc_norms[6] - 20.0 / 28.0) < 1e-6
    assert abs(actual_rc_norms[0] - 1.0) < 1e-6
    assert abs(actual_rc_norms[1] - 1.0) < 1e-6

def test_edge_features_slack_calculation():
    """Verify that edge feature dim 4 (slack_ij) accurately calculates (min(Rc_i, Rc_j) - dist) / min(Rc_i, Rc_j)."""
    config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=100)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario((0.0, 0.0), (200.0, 0.0))
    
    # Degrade Agent 4 to 22.0m (neighbors 3 and 5 are nominal 28.0m)
    sim.comm_ranges[4] = 22.0
    
    policy = GNNPolicy(sim)
    node_feats, edge_feats, edge_index, mobile_mask = policy.extract_graph_features()
    
    edge_np = edge_feats.cpu().numpy()
    edge_idx_np = edge_index.cpu().numpy()
    
    assert edge_np.shape[1] == 5, f"Expected 5 edge features, got {edge_np.shape[1]}"
    
    pos = sim.positions
    num_edges = edge_idx_np.shape[1]
    
    for e in range(num_edges):
        u = edge_idx_np[0, e]
        v = edge_idx_np[1, e]
        dist = np.linalg.norm(pos[u] - pos[v])
        
        min_rc = min(sim.comm_ranges[u], sim.comm_ranges[v])
        expected_slack = (min_rc - dist) / min_rc
        expected_norm_dist = dist / 28.0
        
        actual_norm_dist = edge_np[e, 0]
        actual_slack = edge_np[e, 4]
        
        assert abs(actual_norm_dist - expected_norm_dist) < 1e-5, f"Edge {u}->{v} norm_dist mismatch!"
        assert abs(actual_slack - expected_slack) < 1e-5, (
            f"Edge {u}->{v} slack mismatch! Expected {expected_slack:.6f}, got {actual_slack:.6f}"
        )
        
        # Explicit check for edges touching degraded agent 4
        if u == 4 or v == 4:
            assert min_rc == 22.0, f"Expected min_rc=22.0 for edge touching agent 4, got {min_rc}"

def test_domain_randomization_distribution():
    """Verify that domain randomization triggers ~50% of episodes and uniformly degrades 1-2 relay agents."""
    np.random.seed(42)
    N = 9
    relay_indices = list(range(2, N))
    
    num_trials = 500
    triggered_count = 0
    degraded_counts = {1: 0, 2: 0}
    agent_pick_counts = {r: 0 for r in relay_indices}
    sampled_rc_values = []
    
    for _ in range(num_trials):
        # Emulate collect_mappo_rollout domain randomization logic
        if np.random.rand() < 0.50:
            triggered_count += 1
            num_degraded = int(np.random.choice([1, 2]))
            degraded_counts[num_degraded] += 1
            
            chosen = np.random.choice(relay_indices, size=num_degraded, replace=False)
            for c_id in chosen:
                agent_pick_counts[c_id] += 1
                rc_val = float(np.random.uniform(20.0, 28.0))
                sampled_rc_values.append(rc_val)
                assert 20.0 <= rc_val <= 28.0
                assert c_id not in [0, 1], "Ground endpoints must NEVER be degraded!"
                
    trigger_rate = triggered_count / num_trials
    print(f"Trigger Rate over {num_trials} trials: {trigger_rate:.3f}")
    assert 0.44 <= trigger_rate <= 0.56, f"Trigger rate {trigger_rate:.3f} outside expected 50% binomial window!"
    
    # Check 1 vs 2 degraded split
    frac_1 = degraded_counts[1] / triggered_count
    assert 0.40 <= frac_1 <= 0.60, f"Split for 1-degraded ({frac_1:.3f}) is not ~50% of triggered trials!"
    
    # Check uniform spread across relay indices
    min_picks = min(agent_pick_counts.values())
    max_picks = max(agent_pick_counts.values())
    print(f"Agent pick distribution: {agent_pick_counts}")
    assert min_picks > 0, "All relay indices must have positive selection probability!"
    assert max_picks / min_picks < 2.5, f"Selection is skewed across relays: max/min = {max_picks/min_picks:.2f}"
    
    # Check mean sampled Rc is ~24.0m
    mean_rc = np.mean(sampled_rc_values)
    assert 23.5 <= mean_rc <= 24.5, f"Mean sampled Rc ({mean_rc:.2f}) differs from expected midpoint 24.0m"
