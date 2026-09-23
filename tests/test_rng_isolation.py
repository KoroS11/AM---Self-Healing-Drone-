import os
import copy
import torch
import numpy as np
import pytest

from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy

def test_evaluate_policy_rng_isolation():
    """Regression test: assert that interleaving evaluate_policy_on_scenarios does NOT alter the training RNG stream or model weights."""
    pipeline = TrainPipeline(output_dir="checkpoints")
    device = "cpu"
    val_scenarios = pipeline.bank_data["seen_35"][:3]
    
    # -------------------------------------------------------------
    # RUN 1: 3-step loop WITHOUT evaluation calls
    # -------------------------------------------------------------
    np.random.seed(42)
    torch.manual_seed(42)
    
    actor_1 = ActorGNN(k_hops=1, max_accel=5.0).to(device)
    optimizer_1 = torch.optim.Adam(actor_1.parameters(), lr=1e-3)
    
    drawn_seeds_1 = []
    
    for it in range(3):
        # Draw batch of scenarios
        choice = np.random.choice([s["seed"] for s in pipeline.bank_data["seen_35"]], size=2, replace=False)
        drawn_seeds_1.append(list(choice))
        
        # Synthetic step
        optimizer_1.zero_grad()
        dummy_loss = (actor_1.log_std ** 2).sum() + torch.randn(1).sum()
        dummy_loss.backward()
        optimizer_1.step()
        
    state_dict_1 = copy.deepcopy(actor_1.state_dict())
    
    # -------------------------------------------------------------
    # RUN 2: 3-step loop WITH interleaving evaluate_policy_on_scenarios calls
    # -------------------------------------------------------------
    np.random.seed(42)
    torch.manual_seed(42)
    
    actor_2 = ActorGNN(k_hops=1, max_accel=5.0).to(device)
    optimizer_2 = torch.optim.Adam(actor_2.parameters(), lr=1e-3)
    
    drawn_seeds_2 = []
    
    for it in range(3):
        # Interleaved evaluation call that simulates rollouts
        _ = pipeline.evaluate_policy_on_scenarios(
            lambda s: GNNPolicy(s, actor_gnn=actor_2, device=device),
            val_scenarios,
            max_steps=10
        )
        
        # Draw batch of scenarios
        choice = np.random.choice([s["seed"] for s in pipeline.bank_data["seen_35"]], size=2, replace=False)
        drawn_seeds_2.append(list(choice))
        
        # Synthetic step
        optimizer_2.zero_grad()
        dummy_loss = (actor_2.log_std ** 2).sum() + torch.randn(1).sum()
        dummy_loss.backward()
        optimizer_2.step()
        
    state_dict_2 = copy.deepcopy(actor_2.state_dict())
    
    # Assert identical sequence of drawn seeds
    assert drawn_seeds_1 == drawn_seeds_2, f"RNG state diverged! Run 1: {drawn_seeds_1} vs Run 2: {drawn_seeds_2}"
    
    # Assert bit-identical model weights
    for k in state_dict_1:
        diff = torch.max(torch.abs(state_dict_1[k] - state_dict_2[k])).item()
        assert diff == 0.0, f"Model parameter {k} diverged by {diff} due to unisolated evaluation RNG!"
