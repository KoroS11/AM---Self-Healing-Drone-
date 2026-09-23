import numpy as np
import torch
import pytest
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.utils.enums import AgentRole

def test_actor_gnn_forward_pass():
    actor = ActorGNN(node_in_dim=8, edge_in_dim=4, hidden_dim=32, k_hops=1, max_accel=5.0)
    
    N = 9
    E = 16
    node_feats = torch.randn(N, 8)
    edge_feats = torch.randn(E, 4)
    edge_index = torch.randint(0, N, (2, E))
    mobile_mask = torch.tensor([False, True, True, True, True, True, True, True, False])
    
    actions = actor(node_feats, edge_feats, edge_index, mobile_mask)
    
    assert actions.shape == (N, 2)
    # Ground nodes zeroed out
    assert torch.allclose(actions[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(actions[8], torch.tensor([0.0, 0.0]))
    # Clamped to max_accel
    assert torch.all(torch.abs(actions) <= 5.0)

def test_critic_gnn_forward_pass():
    critic = CriticGNN(node_in_dim=8, edge_in_dim=4, hidden_dim=32, num_layers=2)
    
    N = 9
    E = 16
    node_feats = torch.randn(N, 8)
    edge_feats = torch.randn(E, 4)
    edge_index = torch.randint(0, N, (2, E))
    
    value = critic(node_feats, edge_feats, edge_index)
    
    assert value.shape == (1,)
    assert not torch.isnan(value).any()

def test_gnn_policy_integration():
    config = ExperimentConfig(num_drones=9, comm_range=28.0)
    sim = SwarmSimulator(config)
    
    positions = np.array([
        [0.0, 0.0],    # GROUND_A
        [25.0, 0.0],
        [50.0, 0.0],
        [75.0, 0.0],   # RELAY 3 (Failed)
        [100.0, 0.0],
        [125.0, 0.0],
        [150.0, 0.0],
        [175.0, 0.0],
        [200.0, 0.0],  # GROUND_B
    ])
    roles = [
        AgentRole.GROUND_A,
        AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY,
        AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY, AgentRole.RELAY,
        AgentRole.GROUND_B
    ]
    sim.initialize_positions(positions, roles)
    sim.max_speeds[0] = 0.0
    sim.max_speeds[8] = 0.0
    
    injector = FailureInjector(sim)
    injector.fail_agent(3)
    
    policy = GNNPolicy(sim)
    accelerations = policy.compute_control_forces()
    
    assert accelerations.shape == (9, 2)
    assert np.allclose(accelerations[0], [0.0, 0.0])
    assert np.allclose(accelerations[8], [0.0, 0.0])

def test_actor_gnn_stochastic_sampling():
    actor = ActorGNN(node_in_dim=8, edge_in_dim=4, hidden_dim=32, k_hops=1, max_accel=5.0)
    
    N = 9
    E = 16
    node_feats = torch.randn(N, 8)
    edge_feats = torch.randn(E, 4)
    edge_index = torch.randint(0, N, (2, E))
    mobile_mask = torch.tensor([False, True, True, True, True, True, True, True, False])
    
    # Deterministic pass
    det_actions, det_raw, det_lp, det_ent = actor.get_action(
        node_feats, edge_feats, edge_index, mobile_mask, deterministic=True
    )
    assert det_actions.shape == (N, 2)
    assert det_raw.shape == (N, 2)
    assert det_lp.shape == ()
    assert det_ent.shape == ()
    assert torch.allclose(det_actions[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(det_actions[8], torch.tensor([0.0, 0.0]))
    
    # Stochastic pass
    stoch_actions, stoch_raw, stoch_lp, stoch_ent = actor.get_action(
        node_feats, edge_feats, edge_index, mobile_mask, deterministic=False
    )
    assert stoch_actions.shape == (N, 2)
    assert stoch_raw.shape == (N, 2)
    assert stoch_lp.shape == ()
    assert not torch.isnan(stoch_lp)
    assert not torch.isnan(stoch_ent)
    assert torch.all(torch.abs(stoch_actions) <= 5.0)
    assert torch.allclose(stoch_actions[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(stoch_actions[8], torch.tensor([0.0, 0.0]))
    
    # Evaluate actions on pre-clamp raw actions
    eval_lp, eval_ent = actor.evaluate_actions(
        node_feats, edge_feats, edge_index, mobile_mask, stoch_raw
    )
    assert eval_lp.shape == ()
    assert torch.allclose(eval_lp, stoch_lp, atol=1e-4)
    assert torch.allclose(eval_ent, stoch_ent, atol=1e-4)
    
    # Gradient flow test
    eval_lp.backward()
    assert actor.action_head[0].weight.grad is not None
    assert actor.log_std.grad is not None

def test_actor_gnn_pre_clamp_log_prob_out_of_bounds():
    """Assert that when raw_sampled falls outside [-a_max, a_max], log_prob is computed on raw_sampled."""
    # Set high log_std to guarantee samples outside [-5.0, 5.0]
    actor = ActorGNN(node_in_dim=8, edge_in_dim=4, hidden_dim=32, k_hops=1, max_accel=5.0, init_log_std=2.0)
    
    N = 5
    E = 4
    node_feats = torch.randn(N, 8)
    edge_feats = torch.randn(E, 4)
    edge_index = torch.randint(0, N, (2, E))
    mobile_mask = torch.tensor([False, True, True, True, False])
    
    torch.manual_seed(123)
    actions, raw_sampled, recorded_log_prob, _ = actor.get_action(
        node_feats, edge_feats, edge_index, mobile_mask, deterministic=False
    )
    
    # 1. Assert raw sample exceeded [-5.0, 5.0] on active mobile nodes
    mobile_raw = raw_sampled[mobile_mask]
    has_out_of_bounds = torch.any(torch.abs(mobile_raw) > 5.0)
    assert has_out_of_bounds, "Expected sample outside [-5.0, 5.0] with log_std=2.0"
    
    # 2. Assert executed actions are strictly bounded to [-5.0, 5.0]
    assert torch.all(torch.abs(actions) <= 5.0)
    
    # 3. Compute ground-truth Gaussian log-prob at raw_sampled vs at clamped actions
    mu = actor._compute_mean(node_feats, edge_feats, edge_index, mobile_mask)
    std = torch.exp(actor.log_std)
    dist = torch.distributions.Normal(mu, std)
    
    expected_raw_log_prob = dist.log_prob(raw_sampled)[mobile_mask].sum()
    clamped_log_prob = dist.log_prob(actions)[mobile_mask].sum()
    
    # Assert recorded log-prob matches distribution evaluated at actual pre-clamp sample
    assert torch.allclose(recorded_log_prob, expected_raw_log_prob, atol=1e-5)
    # And differs from the incorrect post-clamp evaluation
    assert not torch.allclose(recorded_log_prob, clamped_log_prob, atol=1e-3)
