import argparse
import os
import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from swarm_sim.utils.logger import TelemetryLogger
from swarm_sim.viz.animator import SwarmAnimator
from swarm_sim.viz.exporter import GIFExporter
from swarm_sim.utils.enums import RecoveryStrategy

def run_single_experiment(
    num_drones: int = 9,
    comm_range: float = 28.0,
    strategy: str = "heuristic_dara",
    fail_agent_id: int = 3,
    fail_timestep: int = 20,
    max_steps: int = 150,
    seed: int = 42,
    export_gif: bool = False,
    gif_output: str = "experiment.gif",
    log_dir: str = "logs"
) -> None:
    print(f"=== Running Experiment: Strategy={strategy}, Drones={num_drones}, Seed={seed} ===")
    
    config = ExperimentConfig(num_drones=num_drones, comm_range=comm_range, seed=seed)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    ground_a_id, ground_b_id = scenario.setup_scenario(endpoint_a_pos=(0.0, 0.0), endpoint_b_pos=(200.0, 0.0))
    
    injector = FailureInjector(sim)
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()
    logger = TelemetryLogger(run_id=f"run_{strategy}_{seed}", channel_model=channel)
    
    if export_gif:
        animator = SwarmAnimator()
        exporter = GIFExporter()
        
    # Select Policy
    if strategy == "greedy_dara":
        policy = GreedyDARAHeuristicPolicy(sim, channel_model=channel)
    elif strategy == "learned_gnn":
        policy = GNNPolicy(sim, channel_model=channel)
    else:
        policy = DARAHeuristicPolicy(sim)
        
    failed = False
    reconnected_step = None
    min_connectivity = 1.0
    
    for t in range(max_steps):
        if t == fail_timestep and not failed:
            injector.fail_agent(fail_agent_id)
            failed = True
            
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        logger.log_timestep(sim, G, timestep=t)
        
        if export_gif and (t % 2 == 0 or t == max_steps - 1):
            frame = animator.render_frame(sim, graph=G, timestep=t, title=f"{strategy.upper()} - Step {t}")
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
        seed=seed,
        failure_timestep=fail_timestep,
        failure_count=1,
        strategy=strategy,
        time_to_reconnect=reconnected_step,
        final_connectivity=final_conn,
        min_connectivity_during_failure=min_connectivity,
        final_sum_rate_mbps=sum_rate
    )
    
    ts_csv = os.path.join(log_dir, f"timestep_{strategy}_{seed}.csv")
    sum_csv = os.path.join(log_dir, f"summary_{strategy}_{seed}.csv")
    logger.save_to_csv(ts_csv, sum_csv)
    print(f"Logged run to CSV: {sum_csv}")
    
    if export_gif:
        exporter.save_gif(gif_output, fps=10)
        print(f"Exported animation GIF to: {gif_output}")
        
    print(f"Result -> Final Connected: {final_conn}, Time to Reconnect: {reconnected_step} ticks, Sum-rate: {sum_rate:.2f} Mbps")

def main():
    parser = argparse.ArgumentParser(description="UAV Swarm Experiment CLI Runner")
    parser.add_argument("--num-drones", type=int, default=9)
    parser.add_argument("--comm-range", type=float, default=28.0)
    parser.add_argument("--strategy", type=str, default="heuristic_dara", choices=["heuristic_dara", "greedy_dara", "learned_gnn"])
    parser.add_argument("--fail-agent-id", type=int, default=3)
    parser.add_argument("--fail-timestep", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-gif", action="store_true")
    parser.add_argument("--gif-output", type=str, default="experiment.gif")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()
    
    run_single_experiment(
        num_drones=args.num_drones,
        comm_range=args.comm_range,
        strategy=args.strategy,
        fail_agent_id=args.fail_agent_id,
        fail_timestep=args.fail_timestep,
        max_steps=args.max_steps,
        seed=args.seed,
        export_gif=args.export_gif,
        gif_output=args.gif_output,
        log_dir=args.log_dir
    )

if __name__ == "__main__":
    main()
