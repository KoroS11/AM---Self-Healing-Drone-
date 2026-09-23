import time
import os
import torch
import numpy as np
from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
import torch.optim as optim

def profile_iteration():
    pipeline = TrainPipeline(output_dir="checkpoints")
    device = pipeline.device
    
    # Model setup
    checkpoint_A_path = "checkpoints/checkpoint_A_best_k2.pt"
    assert os.path.exists(checkpoint_A_path), f"Checkpoint A weights not found at {checkpoint_A_path}"
    
    actor = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor.load_state_dict(torch.load(checkpoint_A_path, map_location=device), strict=False)
    critic = CriticGNN(num_layers=2).to(device)
    
    lr_actor = 3e-4
    lr_critic = 1e-3
    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)
    
    clip_param = 0.2
    entropy_coef = 0.01
    gamma = 0.99
    gae_lambda = 0.95
    rollout_batch_size = 8
    ppo_epochs = 4
    
    seen_scenarios = pipeline.bank_data["seen_35"]
    
    print("=================================================================")
    print("PROFILING 1 PPO ITERATION (Rollout Collection + Mini-batch Updates)")
    print(f"Device: {device}")
    print(f"Rollout Batch Size: {rollout_batch_size} scenarios (150 steps each = {rollout_batch_size*150} transitions)")
    print(f"PPO Epochs: {ppo_epochs}, Minibatch Size: 32")
    print("=================================================================")
    
    # 1. Profile Rollout Collection
    t0_rollout = time.perf_counter()
    actor.eval()
    critic.eval()
    
    batch_scenarios = np.random.choice(seen_scenarios, size=rollout_batch_size, replace=False)
    
    batch_states = []
    batch_actions = []
    batch_raw_actions = []
    batch_old_log_probs = []
    batch_advantages = []
    batch_returns = []
    batch_total_returns = []
    
    for sc in batch_scenarios:
        rollout = pipeline.collect_mappo_rollout(
            sc, actor, critic, max_steps=150, gamma=gamma, gae_lambda=gae_lambda
        )
        batch_states.extend(rollout["states"])
        batch_actions.extend(rollout["actions"])
        batch_raw_actions.extend(rollout["raw_actions"])
        batch_old_log_probs.extend(rollout["old_log_probs"])
        batch_advantages.extend(rollout["advantages"])
        batch_returns.extend(rollout["returns"])
        batch_total_returns.append(rollout["total_return"])
        
    t1_rollout = time.perf_counter()
    rollout_time = t1_rollout - t0_rollout
    
    # 2. Advantage Normalization
    t0_prep = time.perf_counter()
    adv_tensor = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
    ret_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    old_log_probs_tensor = torch.stack(batch_old_log_probs).to(device)
    
    adv_mean = adv_tensor.mean()
    adv_std = adv_tensor.std() + 1e-8
    norm_advantages = (adv_tensor - adv_mean) / adv_std
    t1_prep = time.perf_counter()
    prep_time = t1_prep - t0_prep
    
    # 3. Profile PPO Updates
    t0_update = time.perf_counter()
    actor.train()
    critic.train()
    
    num_samples = len(batch_states)
    indices = np.arange(num_samples)
    
    epoch_actor_losses = []
    epoch_critic_losses = []
    
    for ppo_ep in range(ppo_epochs):
        np.random.shuffle(indices)
        mini_batch_size = 32
        
        for start_idx in range(0, num_samples, mini_batch_size):
            mb_idx = indices[start_idx:start_idx + mini_batch_size]
            
            actor_loss_sum = torch.tensor(0.0, device=device)
            critic_loss_sum = torch.tensor(0.0, device=device)
            
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            
            for i in mb_idx:
                node_t, edge_t, edge_idx_t, mobile_mask_t = batch_states[i]
                raw_act_t = batch_raw_actions[i]
                old_lp = old_log_probs_tensor[i]
                adv = norm_advantages[i]
                ret = ret_tensor[i]
                
                new_lp, entropy = actor.evaluate_actions(
                    node_t, edge_t, edge_idx_t, mobile_mask_t, raw_act_t
                )
                
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
                actor_clip_loss = -torch.min(surr1, surr2)
                entropy_bonus = -entropy_coef * entropy
                actor_step_loss = actor_clip_loss + entropy_bonus
                
                val_pred = critic(node_t, edge_t, edge_idx_t)
                critic_step_loss = 0.5 * (val_pred - ret) ** 2
                
                actor_loss_sum = actor_loss_sum + actor_step_loss
                critic_loss_sum = critic_loss_sum + critic_step_loss
                
            actor_loss_mb = actor_loss_sum / len(mb_idx)
            critic_loss_mb = critic_loss_sum / len(mb_idx)
            
            actor_loss_mb.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_opt.step()
            
            critic_loss_mb.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            critic_opt.step()
            
            epoch_actor_losses.append(actor_loss_mb.item())
            epoch_critic_losses.append(critic_loss_mb.item())
            
    t1_update = time.perf_counter()
    update_time = t1_update - t0_update
    
    total_iter_time = rollout_time + prep_time + update_time
    
    print("\n--- TIMING BREAKDOWN (1 PPO ITERATION) ---")
    print(f"Rollout Collection: {rollout_time:.3f} s ({rollout_time/total_iter_time*100:.1f}%)")
    print(f"Tensor Prep / GAE:  {prep_time:.3f} s ({prep_time/total_iter_time*100:.1f}%)")
    print(f"PPO Update Loop:    {update_time:.3f} s ({update_time/total_iter_time*100:.1f}%)")
    print(f"Total Iteration:    {total_iter_time:.3f} s")
    
    for n_iter in [50, 100, 200]:
        est_sec = total_iter_time * n_iter
        est_min = est_sec / 60.0
        est_hr = est_min / 60.0
        print(f"Extrapolated for {n_iter:3d} iterations: {est_sec:.1f} s = {est_min:.2f} min = {est_hr:.3f} hours")

if __name__ == "__main__":
    profile_iteration()
