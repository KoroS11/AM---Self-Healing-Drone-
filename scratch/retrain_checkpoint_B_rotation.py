import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.optim as optim

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from train_pipeline import TrainPipeline

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pipeline = TrainPipeline(device=device)
    
    # Load warm-start weights from existing checkpoint_B_best.pt (iteration 10)
    warm_start_path = "checkpoints/checkpoint_B_best.pt"
    assert os.path.exists(warm_start_path), f"Missing {warm_start_path}"
    
    actor = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor.load_state_dict(torch.load(warm_start_path, map_location=device), strict=False)
    print(f"Loaded warm-start weights from {warm_start_path}")
    
    critic = CriticGNN(num_layers=2).to(device)
    
    seen_scenarios = pipeline.bank_data["seen_35"]
    held_out_scenarios = pipeline.bank_data["held_out_15"]
    
    # Validation suite (25 scenarios)
    spare_path = os.path.join(pipeline.output_dir, "test_bank_spare.pkl")
    with open(spare_path, "rb") as f:
        spare_bank = pickle.load(f)
    val_suite = spare_bank["spare_10"] + held_out_scenarios
    
    # Initial validation evaluation
    init_eval = pipeline.evaluate_policy_on_scenarios(
        lambda s: GNNPolicy(s, actor_gnn=actor, device=str(device)),
        val_suite
    )
    print(f"Initial Validation Success: {init_eval['success_rate']*100:.1f}%, Mean Time: {init_eval['mean_time']:.2f}t")
    
    # Retrain with randomized corridor headings
    lr_actor = 5e-5
    lr_critic = 2e-4
    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)
    
    clip_param = 0.2
    entropy_coef = 0.0005
    gamma = 0.99
    gae_lambda = 0.95
    rollout_batch_size = 8
    ppo_epochs = 4
    num_iterations = 12
    
    best_success = init_eval['success_rate']
    best_time = init_eval['mean_time']
    best_weights = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
    
    print("\n--- Starting Checkpoint B Retraining with Randomized Corridor Headings (0-360 deg) ---")
    
    for it in range(1, num_iterations + 1):
        t_start = time.perf_counter()
        actor.eval()
        critic.eval()
        
        batch_scenarios = np.random.choice(seen_scenarios, size=min(rollout_batch_size, len(seen_scenarios)), replace=False)
        
        batch_states = []
        batch_raw_actions = []
        batch_old_log_probs = []
        batch_advantages = []
        batch_returns = []
        batch_total_returns = []
        
        for sc in batch_scenarios:
            rollout = pipeline.collect_mappo_rollout(
                sc, actor, critic, max_steps=150, gamma=gamma, gae_lambda=gae_lambda,
                randomize_corridor_heading=True
            )
            batch_states.extend(rollout["states"])
            batch_raw_actions.extend(rollout["raw_actions"])
            batch_old_log_probs.extend(rollout["old_log_probs"])
            batch_advantages.extend(rollout["advantages"])
            batch_returns.extend(rollout["returns"])
            batch_total_returns.append(rollout["total_return"])
            
        adv_tensor = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        old_log_probs_tensor = torch.stack(batch_old_log_probs).to(device)
        
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std() + 1e-8
        norm_advantages = (adv_tensor - adv_mean) / adv_std
        
        actor.train()
        critic.train()
        
        num_samples = len(batch_states)
        indices = np.arange(num_samples)
        minibatch_size = 32
        
        for ppo_ep in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                b_node, b_edge, b_edge_idx, b_mask, b_raw_act, b_idx = pipeline._batch_graph_transitions(
                    batch_states, batch_raw_actions, mb_idx
                )
                mb_adv = norm_advantages[mb_idx]
                mb_ret = ret_tensor[mb_idx]
                mb_old_lp = old_log_probs_tensor[mb_idx]
                
                new_lp, entropy = actor.evaluate_actions(
                    b_node, b_edge, b_edge_idx, b_mask, b_raw_act, batch_idx=b_idx
                )
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_adv
                actor_clip_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = -entropy_coef * entropy.mean()
                actor_loss = actor_clip_loss + entropy_bonus
                
                val_pred = critic(b_node, b_edge, b_edge_idx, batch_idx=b_idx)
                critic_loss = 0.5 * ((val_pred - mb_ret) ** 2).mean()
                
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                actor_opt.step()
                
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                critic_opt.step()
                
        t_elapsed = time.perf_counter() - t_start
        mean_ret = float(np.mean(batch_total_returns))
        curr_log_std = float(actor.log_std.mean().item())
        
        # Periodic evaluation
        val_eval = pipeline.evaluate_policy_on_scenarios(
            lambda s: GNNPolicy(s, actor_gnn=actor, device=str(device)),
            val_suite
        )
        val_success = val_eval["success_rate"]
        val_mean_time = val_eval["mean_time"]
        
        print(f"Retrain Iter {it:2d}/{num_iterations} | Return: {mean_ret:7.2f} | log_std: {curr_log_std:.4f} | Val Success: {val_success*100:5.1f}% | Val Time: {val_mean_time:5.2f}t | Time: {t_elapsed:.2f}s")
        
        if val_success >= best_success:
            best_success = val_success
            best_time = val_mean_time
            best_weights = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
            
    # Save best retrained weights
    retrained_save_path = "checkpoints/checkpoint_B_rotation_best.pt"
    torch.save(best_weights, retrained_save_path)
    print(f"\nSaved best retrained rotation-invariant weights to {retrained_save_path}")
    print(f"Best Validation Success: {best_success*100:.1f}%, Mean Time: {best_time:.2f}t")
    
    # Overwrite checkpoint_B_best.pt with updated invariant model
    torch.save(best_weights, "checkpoints/checkpoint_B_best.pt")
    print("Updated checkpoints/checkpoint_B_best.pt with retrained rotation-invariant weights.")

if __name__ == "__main__":
    main()
