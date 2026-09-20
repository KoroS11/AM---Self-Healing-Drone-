import os
import pickle
import numpy as np
import pandas as pd

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.utils.logger import TelemetryLogger
from swarm_sim.viz.animator import SwarmAnimator
from swarm_sim.viz.exporter import GIFExporter

# Load seed 1000 config from test_bank_50.pkl
bank_file = "checkpoints/test_bank_50.pkl"
with open(bank_file, "rb") as f:
    bank_data = pickle.load(f)

sc1000 = bank_data["seen_35"][0]  # First scenario is seed 1000
print(f"=== Seed 1000 Scenario Configuration ===")
for k, v in sc1000.items():
    print(f"  {k}: {v}")

def run_seed_1000(strategy_name, policy_cls, gif_filename):
    config = ExperimentConfig(
        num_drones=sc1000["num_drones"],
        comm_range=sc1000["comm_range"],
        seed=sc1000["seed"]
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    ground_a_id, ground_b_id = scenario.setup_scenario(
        endpoint_a_pos=sc1000["endpoint_a_pos"],
        endpoint_b_pos=sc1000["endpoint_b_pos"]
    )
    
    injector = FailureInjector(sim)
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()
    logger = TelemetryLogger(run_id=f"run_{strategy_name}_1000", channel_model=channel)
    
    animator = SwarmAnimator()
    exporter = GIFExporter()
    
    policy = policy_cls(sim)
    failed = False
    reconnected_step = None
    min_connectivity = 1.0
    fail_agent_id = sc1000["fail_agent_id"]
    fail_timestep = sc1000["fail_timestep"]
    max_steps = 150
    
    for t in range(max_steps):
        if t == fail_timestep and not failed:
            injector.fail_agent(fail_agent_id)
            failed = True
            
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        logger.log_timestep(sim, G, timestep=t)
        
        if t % 2 == 0 or t == max_steps - 1:
            frame = animator.render_frame(sim, graph=G, timestep=t, title=f"{strategy_name.upper()} Seed 1000 - Step {t}")
            exporter.add_frame(frame)
            
        if failed:
            has_conn = topo.has_path(ground_a_id, ground_b_id, graph=G)
            alg_conn = topo.get_algebraic_connectivity(graph=G)
            if alg_conn < min_connectivity:
                min_connectivity = alg_conn
                
            if has_conn and reconnected_step is None:
                reconnected_step = t - fail_timestep
                
    G_final = topo.build_graph()
    final_conn = topo.has_path(ground_a_id, ground_b_id, graph=G_final)
    sum_rate = channel.compute_network_throughput(sim, G_final)
    
    logger.log_run_summary(
        config=config,
        seed=sc1000["seed"],
        failure_timestep=fail_timestep,
        failure_count=1,
        strategy=strategy_name,
        time_to_reconnect=reconnected_step,
        final_connectivity=final_conn,
        min_connectivity_during_failure=min_connectivity,
        final_sum_rate_mbps=sum_rate
    )
    
    os.makedirs("logs", exist_ok=True)
    ts_csv = f"logs/timestep_{strategy_name}_1000.csv"
    sum_csv = f"logs/summary_{strategy_name}_1000.csv"
    logger.save_to_csv(ts_csv, sum_csv)
    exporter.save_gif(gif_filename, fps=10)
    
    print(f"[{strategy_name.upper()} Seed 1000] Final Connected: {final_conn}, Time to Reconnect: {reconnected_step} ticks, Sum-rate: {sum_rate:.2f} Mbps")

run_seed_1000("heuristic_dara", DARAHeuristicPolicy, "experiment_dara_1000.gif")
run_seed_1000("greedy_dara", GreedyDARAHeuristicPolicy, "experiment_greedy_1000.gif")

# Load and print summary CSV rows side-by-side
df_dara = pd.read_csv("logs/summary_heuristic_dara_1000.csv")
df_greedy = pd.read_csv("logs/summary_greedy_dara_1000.csv")

print("\n=== Side-by-Side Summary Results for Seed 1000 ===")
combined = pd.concat([df_dara, df_greedy], ignore_index=True)
print(combined[["strategy", "time_to_reconnect", "final_connectivity", "min_connectivity_during_failure", "final_sum_rate_mbps"]].to_string())
