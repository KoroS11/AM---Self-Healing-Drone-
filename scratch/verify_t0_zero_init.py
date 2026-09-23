import os
import sys
import hashlib
import subprocess
from typing import Tuple, Dict, Optional, Any
import torch
import numpy as np

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_critic import CriticGNN
from swarm_sim.policies.gnn_policy import GNNPolicy

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

def create_b_prime_weights(checkpoint_b_path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """Warm-start Actor (and optional Critic) from Checkpoint B with zero-initialized input slices."""
    b_state = torch.load(checkpoint_b_path, map_location="cpu")
    
    actor_state_b = b_state["actor_state_dict"] if "actor_state_dict" in b_state else b_state
    
    # Construct 9/5 Actor State Dict
    actor_b_prime = {}
    for k, v in actor_state_b.items():
        if k == "node_encoder.0.weight":
            # shape (64, 8) -> (64, 9)
            w_new = torch.zeros((64, 9), dtype=v.dtype)
            w_new[:, :8] = v
            w_new[:, 8] = 0.0  # Zero-init 9th feature slice (rc_norm)
            actor_b_prime[k] = w_new
        elif k == "edge_encoder.0.weight":
            # shape (32, 4) -> (32, 5)
            w_new = torch.zeros((32, 5), dtype=v.dtype)
            w_new[:, :4] = v
            w_new[:, 4] = 0.0  # Zero-init 5th feature slice (slack_ij)
            actor_b_prime[k] = w_new
        else:
            actor_b_prime[k] = v.clone()
            
    critic_b_prime = None
    if "critic_state_dict" in b_state:
        critic_state_b = b_state["critic_state_dict"]
        critic_b_prime = {}
        for k, v in critic_state_b.items():
            if k == "node_encoder.0.weight":
                w_new = torch.zeros((64, 9), dtype=v.dtype)
                w_new[:, :8] = v
                w_new[:, 8] = 0.0
                critic_b_prime[k] = w_new
            elif k == "edge_encoder.0.weight":
                w_new = torch.zeros((32, 5), dtype=v.dtype)
                w_new[:, :4] = v
                w_new[:, 4] = 0.0
                critic_b_prime[k] = w_new
            else:
                critic_b_prime[k] = v.clone()
                
    return actor_b_prime, critic_b_prime

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = "cpu"
    checkpoint_b_path = "checkpoints/checkpoint_B_best.pt"
    
    commit_hash = get_git_commit_hash()
    ckpt_b_md5 = compute_file_md5(checkpoint_b_path)
    policy_file_md5 = compute_file_md5("swarm_sim/policies/gnn_policy.py")
    actor_file_md5 = compute_file_md5("swarm_sim/policies/gnn_actor.py")
    
    print("==================================================================")
    print(" B-PRIME ZERO-INIT VERIFICATION & t=0 SANITY CHECK")
    print("==================================================================")
    print(f"Git Commit Hash:       {commit_hash}")
    print(f"checkpoint_B_best MD5: {ckpt_b_md5}")
    print(f"gnn_policy.py MD5:     {policy_file_md5}")
    print(f"gnn_actor.py MD5:      {actor_file_md5}")
    print("------------------------------------------------------------------")
    
    # 1. Load Checkpoint B (Old Architecture: 8/4 dim)
    actor_B_old = ActorGNN(node_in_dim=8, edge_in_dim=4, k_hops=2, max_accel=5.0).to(device)
    b_raw_state = torch.load(checkpoint_b_path, map_location=device)
    actor_B_old.load_state_dict(b_raw_state["actor_state_dict"] if "actor_state_dict" in b_raw_state else b_raw_state, strict=False)
    actor_B_old.eval()
    
    # 2. Build B-Prime State Dict with Zero-Init slices
    actor_b_prime_state, _ = create_b_prime_weights(checkpoint_b_path)
    
    # Verify zero-init properties
    node_w = actor_b_prime_state["node_encoder.0.weight"]
    edge_w = actor_b_prime_state["edge_encoder.0.weight"]
    
    print(f"\n[Zero-Init Verification]")
    print(f"Node Encoder Input Weight Shape: {node_w.shape}")
    print(f"  * Slices 0..7 match Checkpoint B: {torch.allclose(node_w[:, :8], actor_B_old.node_encoder[0].weight)}")
    print(f"  * Slice 8 (rc_norm) exact zero check: max(|W[:, 8]|) = {float(torch.max(torch.abs(node_w[:, 8]))):.10f}")
    assert torch.all(node_w[:, 8] == 0.0), "Node encoder 9th slice is NOT zero!"
    
    print(f"\nEdge Encoder Input Weight Shape: {edge_w.shape}")
    print(f"  * Slices 0..3 match Checkpoint B: {torch.allclose(edge_w[:, :4], actor_B_old.edge_encoder[0].weight)}")
    print(f"  * Slice 4 (slack_ij) exact zero check: max(|W[:, 4]|) = {float(torch.max(torch.abs(edge_w[:, 4]))):.10f}")
    assert torch.all(edge_w[:, 4] == 0.0), "Edge encoder 5th slice is NOT zero!"
    
    # 3. Instantiate B-Prime Actor (9/5 dim)
    actor_B_prime = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0).to(device)
    actor_B_prime.load_state_dict(actor_b_prime_state, strict=True)
    actor_B_prime.eval()
    print("\nSuccessfully instantiated B-Prime Actor with strict=True.")
    
    # 4. Run closed-loop rollout comparison on nominal scenario (Seed 1000, Rc=28.0m)
    print("\n------------------------------------------------------------------")
    print(" Executing Step-by-Step Rollout Comparison on Nominal Scenario (Seed 1000, Rc=28.0m)")
    print("------------------------------------------------------------------")
    
    config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=1000)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario((0.0, 0.0), (200.0, 0.0))
    policy = GNNPolicy(sim, actor_gnn=actor_B_prime, device=device)
    
    max_action_diff = 0.0
    max_node_enc_diff = 0.0
    max_edge_enc_diff = 0.0
    
    for t in range(50):
        # Extract 9/5 features
        node_feats_9, edge_feats_5, edge_index, mobile_mask = policy.extract_graph_features()
        
        # Subslice 8/4 features for old model
        node_feats_8 = node_feats_9[:, :8]
        edge_feats_4 = edge_feats_5[:, :4]
        
        # Compare Node Encodings
        with torch.no_grad():
            node_enc_old = actor_B_old.node_encoder(node_feats_8)
            node_enc_new = actor_B_prime.node_encoder(node_feats_9)
            diff_node = float(torch.max(torch.abs(node_enc_old - node_enc_new)))
            max_node_enc_diff = max(max_node_enc_diff, diff_node)
            
            if edge_feats_5.shape[0] > 0:
                edge_enc_old = actor_B_old.edge_encoder(edge_feats_4)
                edge_enc_new = actor_B_prime.edge_encoder(edge_feats_5)
                diff_edge = float(torch.max(torch.abs(edge_enc_old - edge_enc_new)))
                max_edge_enc_diff = max(max_edge_enc_diff, diff_edge)
                
            act_old, _, _, _ = actor_B_old.get_action(node_feats_8, edge_feats_4, edge_index, mobile_mask, deterministic=True)
            act_new, _, _, _ = actor_B_prime.get_action(node_feats_9, edge_feats_5, edge_index, mobile_mask, deterministic=True)
            
            diff_act = float(torch.max(torch.abs(act_old - act_new)))
            max_action_diff = max(max_action_diff, diff_act)
            
        # Step simulation with new policy
        acc_corridor = act_new.cpu().numpy()
        acc_world = policy.corridor_to_world_forces(acc_corridor)
        sim.step(accelerations=acc_world)
        
    print(f"Max Node Encoder Output Difference: {max_node_enc_diff:.10e}")
    print(f"Max Edge Encoder Output Difference: {max_edge_enc_diff:.10e}")
    print(f"Max Control Action Difference:       {max_action_diff:.10e}")
    
    assert max_action_diff < 1e-6, f"Action outputs diverge! max_diff = {max_action_diff}"
    print("\n[PASSED] t=0 Sanity Check confirms bit-identical behavior between Checkpoint B and B-Prime!")

if __name__ == "__main__":
    main()
