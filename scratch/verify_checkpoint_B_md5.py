import os
import hashlib
import time
import pickle
import torch
import numpy as np

from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy

def main():
    file_path = "checkpoints/checkpoint_B_best.pt"
    assert os.path.exists(file_path), f"{file_path} not found"
    
    # 1. Compute MD5
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        md5_hash = hashlib.md5(file_bytes).hexdigest()
        
    mtime = os.path.getmtime(file_path)
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    file_size = len(file_bytes)
    
    print("=================================================================")
    print("  CHECKPOINT_B_BEST.PT INTEGRITY & ORIGIN AUDIT")
    print("=================================================================")
    print(f"File Path:          {file_path}")
    print(f"File Size:          {file_size} bytes")
    print(f"Last Modified:      {mtime_str}")
    print(f"MD5 Hash:           {md5_hash}")
    
    # 2. Re-evaluate on 10 spare scenarios
    device = "cpu"
    pipeline = TrainPipeline(output_dir="checkpoints")
    spare_path = "checkpoints/test_bank_spare.pkl"
    with open(spare_path, "rb") as f:
        spare_bank = pickle.load(f)
    val_suite = spare_bank["spare_10"]
    
    actor = ActorGNN(k_hops=2, max_accel=5.0).to(device)
    actor.load_state_dict(torch.load(file_path, map_location=device), strict=False)
    actor.eval()
    
    eval_res = pipeline.evaluate_policy_on_scenarios(
        lambda s: GNNPolicy(s, actor_gnn=actor, device=device),
        val_suite,
        max_steps=150
    )
    
    print("\nEvaluation of checkpoint_B_best.pt on 10 Spare Scenarios:")
    print(f"  Success Rate:     {eval_res['success_rate']*100:.1f}% ({int(eval_res['success_rate']*len(val_suite))}/10)")
    print(f"  Mean Time:        {eval_res['mean_time']:.2f} ticks")
    print(f"  Mean Sum-Rate:    {eval_res['mean_sum_rate']:.2f} Mbps")

if __name__ == "__main__":
    main()
