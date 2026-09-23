import sys
import os
import pickle
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
from swarm_sim.policies.gnn_policy import GNNPolicy

def run_window_audit():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    pipeline = TrainPipeline(output_dir="checkpoints")
    
    # Load spare validation scenarios
    spare_path = "checkpoints/test_bank_spare.pkl"
    with open(spare_path, "rb") as f:
        spare_bank = pickle.load(f)
    val_suite = spare_bank["spare_10"]
    
    # Load seen scenarios for rollouts
    seen_scenarios = pipeline.bank_data["seen_35"]
    
    # Retry hyperparameters
    run_seed = 1042
    lr_actor = 5.0e-05
    lr_critic = 2.5e-04
    clip_param = 0.2
    entropy_coef = 0.0005
    gamma = 0.99
    gae_lambda = 0.95
    rollout_batch_size = 8
    ppo_epochs = 4
    minibatch_size = 32
    max_iterations = 30
    
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    
    actor = ActorGNN(k_hops=2, max_accel=5.0, init_log_std=-1.0).to(device)
    actor.load_state_dict(torch.load("checkpoints/checkpoint_A_best_k2.pt", map_location=device), strict=False)
    critic = CriticGNN(num_layers=2).to(device)
    
    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)
    
    print("=================================================================")
    print("  CHECKPOINT B STABILITY WINDOW AUDIT (ITERATIONS 20-30)")
    print("  Seed: 1042, lr_actor: 5e-5, lr_critic: 2.5e-4")
    print("=================================================================")
    
    # Baseline eval at iteration 0
    init_eval = pipeline.evaluate_policy_on_scenarios(
        lambda s: GNNPolicy(s, actor_gnn=actor, device=device),
        val_suite
    )
    print(f"Iter  0 (Baseline) | Val Success: {init_eval['success_rate']*100:5.1f}% | Val Time: {init_eval['mean_time']:5.2f}t | Val SR: {init_eval['mean_sum_rate']:7.2f}M")
    
    audit_results = []
    
    for it in range(1, max_iterations + 1):
        t_iter_start = time.perf_counter()
        actor.eval()
        critic.eval()
        
        # 1. Rollout batch
        batch_scenarios = np.random.choice(seen_scenarios, size=min(rollout_batch_size, len(seen_scenarios)), replace=False)
        batch_states = []
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
        
        # 2. PPO updates
        actor.train()
        critic.train()
        num_samples = len(batch_states)
        indices = np.arange(num_samples)
        
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
                
        t_iter_total = time.perf_counter() - t_iter_start
        mean_ret = float(np.mean(batch_total_returns))
        curr_log_std = float(actor.log_std.mean().item())
        
        # Evaluate on the requested window: every iteration for it in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] and at it % 5 == 0
        if it >= 20 or it % 5 == 0 or it == 1:
            val_eval = pipeline.evaluate_policy_on_scenarios(
                lambda s: GNNPolicy(s, actor_gnn=actor, device=device),
                val_suite
            )
            succ = val_eval["success_rate"]
            m_time = val_eval["mean_time"]
            m_sr = val_eval["mean_sum_rate"]
            
            audit_results.append({
                "iteration": it,
                "val_success": succ,
                "val_mean_time": m_time,
                "val_mean_sr": m_sr,
                "mean_return": mean_ret,
                "log_std": curr_log_std,
                "wall_clock_s": t_iter_total
            })
            
            print(f"Iter {it:2d} | Return: {mean_ret:7.2f} | log_std: {curr_log_std:.4f} | Val Success: {succ*100:5.1f}% ({int(succ*10)}/10) | Val Time: {m_time:5.2f}t | Val SR: {m_sr:7.2f}M | Iter Time: {t_iter_total:.2f}s")
        else:
            print(f"Iter {it:2d} | Return: {mean_ret:7.2f} | log_std: {curr_log_std:.4f} | [Training Step] | Iter Time: {t_iter_total:.2f}s")

    df = pd.DataFrame(audit_results)
    df.to_csv("scratch/checkpoint_B_window_audit.csv", index=False)
    print("\nSaved window audit results to scratch/checkpoint_B_window_audit.csv")

if __name__ == "__main__":
    run_window_audit()
