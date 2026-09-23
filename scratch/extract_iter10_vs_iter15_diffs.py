import sys
import torch
import pandas as pd
from train_pipeline import TrainPipeline
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy

def main():
    sys.stdout.reconfigure(line_buffering=True)
    pipeline = TrainPipeline()
    all_50 = pipeline.bank_data['all_50']
    
    actor_10 = ActorGNN(k_hops=2, max_accel=5.0).to(pipeline.device)
    actor_10.load_state_dict(torch.load('checkpoints/checkpoint_B_iter_10.pt', map_location=pipeline.device), strict=False)
    
    actor_15 = ActorGNN(k_hops=2, max_accel=5.0).to(pipeline.device)
    actor_15.load_state_dict(torch.load('checkpoints/checkpoint_B_iter_15.pt', map_location=pipeline.device), strict=False)

    res10 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor_10), all_50)
    res15 = pipeline.evaluate_policy_on_scenarios(lambda s: GNNPolicy(s, actor_gnn=actor_15), all_50)

    diffs = []
    for r10, r15 in zip(res10['details'], res15['details']):
        if r10['success'] != r15['success']:
            diffs.append({
                'seed': r10['seed'],
                'num_drones': r10['num_drones'],
                'fail_agent_id': r10['fail_agent_id'],
                'iter10_success': r10['success'],
                'iter10_time': r10['time_to_reconnect'],
                'iter10_sum_rate': r10['final_sum_rate_mbps'],
                'iter15_success': r15['success'],
                'iter15_time': r15['time_to_reconnect'],
                'iter15_sum_rate': r15['final_sum_rate_mbps']
            })
    df = pd.DataFrame(diffs)
    print("\n--- Per-Scenario Differences (Final Success: Iter 10 vs Iter 15) ---")
    print(df.to_string(index=False))
    df.to_csv('scratch/iter10_vs_iter15_final_conn_diffs.csv', index=False)
    print(f"\nTotal differences found: {len(diffs)}")

if __name__ == "__main__":
    main()
