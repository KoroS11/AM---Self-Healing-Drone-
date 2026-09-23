import os
import sys
import pickle
import hashlib
import subprocess
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.scenarios.test_bank import TestBankGenerator
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from train_pipeline import TrainPipeline

def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Unknown ({e})"

def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def warm_start_b_prime(checkpoint_b_path: str, device="cpu") -> Tuple[ActorGNN, CriticGNN]:
    """Instantiate Actor (9, 5) and Critic (9, 5) with zero-initialized new feature slices from Checkpoint B."""
    b_state = torch.load(checkpoint_b_path, map_location=device)
    actor_state_b = b_state["actor_state_dict"] if "actor_state_dict" in b_state else b_state
    
    # 1. Warm-start Actor
    actor = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0).to(device)
    actor_dict = actor.state_dict()
    
    for k, v in actor_state_b.items():
        if k == "node_encoder.0.weight":
            w_new = torch.zeros((64, 9), dtype=v.dtype, device=device)
            w_new[:, :8] = v.to(device)
            w_new[:, 8] = 0.0
            actor_dict[k] = w_new
        elif k == "edge_encoder.0.weight":
            w_new = torch.zeros((32, 5), dtype=v.dtype, device=device)
            w_new[:, :4] = v.to(device)
            w_new[:, 4] = 0.0
            actor_dict[k] = w_new
        elif k in actor_dict:
            actor_dict[k] = v.to(device)
            
    actor.load_state_dict(actor_dict, strict=True)
    
    # 2. Warm-start Critic
    critic = CriticGNN(node_in_dim=9, edge_in_dim=5, hidden_dim=64, num_layers=2).to(device)
    critic_dict = critic.state_dict()
    
    if "critic_state_dict" in b_state:
        critic_state_b = b_state["critic_state_dict"]
        for k, v in critic_state_b.items():
            if k == "node_encoder.0.weight":
                w_new = torch.zeros((64, 9), dtype=v.dtype, device=device)
                w_new[:, :8] = v.to(device)
                w_new[:, 8] = 0.0
                critic_dict[k] = w_new
            elif k == "edge_encoder.0.weight":
                w_new = torch.zeros((32, 5), dtype=v.dtype, device=device)
                w_new[:, :4] = v.to(device)
                w_new[:, 4] = 0.0
                critic_dict[k] = w_new
            elif k in critic_dict:
                critic_dict[k] = v.to(device)
        critic.load_state_dict(critic_dict, strict=True)
    else:
        # If no critic in checkpoint, initialize critic and train critic warmup
        print("[WarmStart] Initializing fresh Critic with 9/5 dimensions.")
        
    return actor, critic

def evaluate_suite(actor, scenarios, device="cpu", max_steps=150):
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    
    actor.eval()
    channel = ChannelModel()
    results = []
    
    for sc in scenarios:
        config = ExperimentConfig(
            num_drones=sc["num_drones"],
            comm_range=sc["comm_range"],
            seed=sc["seed"]
        )
        sim = SwarmSimulator(config)
        scenario = DisasterRelayScenario(sim)
        g_a, g_b = scenario.setup_scenario(
            endpoint_a_pos=sc.get("endpoint_a_pos", (0.0, 0.0)),
            endpoint_b_pos=sc.get("endpoint_b_pos", (200.0, 0.0))
        )
        injector = FailureInjector(sim)
        policy = GNNPolicy(sim, actor_gnn=actor, device=str(device))
        topo = SwarmTopologyManager(sim)
        
        ever_reconnected = False
        reconnected_step = None
        
        for t in range(max_steps):
            if t == sc["fail_timestep"]:
                injector.fail_agent(sc["fail_agent_id"])
                
            acc = policy.compute_control_forces()
            sim.step(accelerations=acc)
            
            G = topo.build_graph()
            is_conn = topo.has_path(g_a, g_b, graph=G)
            if t >= sc["fail_timestep"]:
                if is_conn:
                    if not ever_reconnected:
                        ever_reconnected = True
                        reconnected_step = t - sc["fail_timestep"]
                    
        G_final = topo.build_graph()
        final_conn = topo.has_path(g_a, g_b, graph=G_final)
        sum_rate = channel.compute_network_throughput(sim, G_final)
        
        results.append({
            "seed": sc["seed"],
            "reconnected": final_conn,
            "reconnect_time": reconnected_step,
            "sum_rate": sum_rate
        })
        
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    
    df = pd.DataFrame(results)
    success_rate = (df["reconnected"].sum() / len(df)) * 100.0
    mean_time = df.loc[df["reconnected"], "reconnect_time"].mean() if df["reconnected"].any() else float('nan')
    mean_rate = df["sum_rate"].mean()
    
    return success_rate, mean_time, mean_rate, df

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    checkpoint_b_path = "checkpoints/checkpoint_B_best.pt"
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=================================================================")
    print("     TRAINING B-PRIME: PPO WITH DOMAIN-RANDOMIZED HETEROGENEITY")
    print("=================================================================")
    print(f"Git Commit: {get_git_commit_hash()}")
    print(f"Checkpoint B Best MD5: {compute_file_md5(checkpoint_b_path)}")
    
    pipeline = TrainPipeline(output_dir=output_dir, device=device)
    
    # 1. Warm start B-prime
    actor, critic = warm_start_b_prime(checkpoint_b_path, device=device)
    
    # 2. Validation suites
    seen_35 = pipeline.bank_data["seen_35"]
    held_out_15 = pipeline.bank_data["held_out_15"]
    all_50 = pipeline.bank_data["all_50"]
    
    spare_path = os.path.join(output_dir, "test_bank_spare.pkl")
    with open(spare_path, "rb") as f:
        spare_bank = pickle.load(f)
    spare_10 = spare_bank["spare_10"]
    val_25 = spare_10 + held_out_15
    
    # 3. Baseline pre-training evaluation (Iteration 0)
    succ_val_0, time_val_0, rate_val_0, _ = evaluate_suite(actor, val_25, device=device)
    print(f"\n[Iter 0 Pre-Train Val (25-scenarios)]: Success={succ_val_0:.1f}%, Time={time_val_0:.2f}t, Rate={rate_val_0:.2f} Mbps")
    
    # Optimizer setup
    lr_actor = 1e-4
    lr_critic = 5e-4
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor, eps=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic, eps=1e-5)
    
    rollout_batch_size = 8
    max_iterations = 15
    ppo_epochs = 4
    clip_param = 0.2
    entropy_coef = 0.0005
    
    best_val_success = 0.0
    best_val_time = float('inf')
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_B_prime_best.pt")
    
    history = []
    
    print("\nStarting PPO Fine-Tuning across 15 Iterations...")
    print("------------------------------------------------------------------------------------------------------")
    print(f"{'Iter':>4} | {'Mean Ret':>9} | {'log_std':>8} | {'Val25 Succ':>10} | {'Val25 Time':>10} | {'Seen35':>10} | {'Status':>12}")
    print("------------------------------------------------------------------------------------------------------")
    
    for iteration in range(1, max_iterations + 1):
        actor.train()
        critic.train()
        
        # Sample training batch with replacement from seen 35 scenarios
        batch_configs = [seen_35[idx] for idx in np.random.choice(len(seen_35), size=rollout_batch_size, replace=True)]
        
        batch_states = []
        batch_actions = []
        batch_raw_actions = []
        batch_old_log_probs = []
        batch_advantages = []
        batch_returns = []
        episode_returns = []
        
        for sc in batch_configs:
            rollout = pipeline.collect_mappo_rollout(
                scenario_config=sc,
                actor=actor,
                critic=critic,
                max_steps=150,
                randomize_corridor_heading=True,
                domain_randomize_rc=True
            )
            
            episode_returns.append(sum(rollout["rewards"]))
            batch_states.extend(rollout["states"])
            batch_actions.extend(rollout["actions"])
            batch_raw_actions.extend(rollout["raw_actions"])
            batch_old_log_probs.extend(rollout["old_log_probs"])
            batch_advantages.extend(rollout["advantages"])
            batch_returns.extend(rollout["returns"])
            
        mean_ret = float(np.mean(episode_returns))
        
        # Advantage normalization
        adv_tensor = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        old_lp_tensor = torch.stack(batch_old_log_probs).to(device)
        
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
            
        # PPO Update
        T_total = len(batch_states)
        minibatch_size = 64
        
        for epoch in range(ppo_epochs):
            perm = np.random.permutation(T_total)
            for start in range(0, T_total, minibatch_size):
                mb_idx = perm[start:start + minibatch_size]
                
                b_node, b_edge, b_edge_idx, b_mask, b_raw_act, b_graph_idx = pipeline._batch_graph_transitions(
                    batch_states, batch_raw_actions, mb_idx
                )
                
                # Evaluate new log probs and entropy
                new_lp, entropy = actor.evaluate_actions(b_node, b_edge, b_edge_idx, b_mask, b_raw_act, b_graph_idx)
                
                # PPO Clipped Loss
                mb_old_lp = old_lp_tensor[mb_idx]
                mb_adv = adv_tensor[mb_idx]
                
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
                
                optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                optimizer_actor.step()
                
                # Critic update
                mb_ret = ret_tensor[mb_idx]
                val_pred = critic(b_node, b_edge, b_edge_idx, b_graph_idx)
                critic_loss = F.mse_loss(val_pred, mb_ret)
                
                optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                optimizer_critic.step()
                
        # Evaluate on Validation Suite
        val_succ, val_time, val_rate, _ = evaluate_suite(actor, val_25, device=device)
        seen_succ, seen_time, _, _ = evaluate_suite(actor, seen_35, device=device)
        
        log_std_val = float(actor.log_std.mean().item()) if hasattr(actor.log_std, "mean") else -1.0
        
        is_best = False
        if val_succ > best_val_success or (val_succ == best_val_success and val_time < best_val_time):
            best_val_success = val_succ
            best_val_time = val_time
            is_best = True
            
            # Save best checkpoint
            torch.save({
                "iteration": iteration,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "val_success": val_succ,
                "val_time": val_time
            }, best_checkpoint_path)
            
        status = "BEST (Saved)" if is_best else ""
        print(f"{iteration:>4} | {mean_ret:>9.2f} | {log_std_val:>8.4f} | {val_succ:>9.1f}% | {val_time:>9.2f}t | {seen_succ:>9.1f}% | {status:>12}")
        
        # Save per-iteration checkpoint
        torch.save(actor.state_dict(), os.path.join(output_dir, f"checkpoint_B_prime_iter_{iteration}.pt"))
        
        history.append({
            "iteration": iteration,
            "mean_return": mean_ret,
            "log_std": log_std_val,
            "val_25_success": val_succ,
            "val_25_time": val_time,
            "seen_35_success": seen_succ,
            "seen_35_time": seen_time
        })
        
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("scratch/b_prime_training_history.csv", index=False)
    print(f"\nTraining Complete. Best Checkpoint Saved to {best_checkpoint_path}")
    print(f"Best Validation Performance: Success={best_val_success:.1f}%, Time={best_val_time:.2f}t")

if __name__ == "__main__":
    import torch.nn.functional as F
    main()
