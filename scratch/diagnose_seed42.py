import numpy as np
import networkx as nx
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.policies.greedy_dara import GreedyDARAHeuristicPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel

def diagnose(policy_cls, name):
    config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    ground_a_id, ground_b_id = scenario.setup_scenario(endpoint_a_pos=(0.0, 0.0), endpoint_b_pos=(200.0, 0.0))
    
    print(f"\n================ {name} (N=9, AB=200m) ================")
    print(f"ground_a_id={ground_a_id}, ground_b_id={ground_b_id}")
    print(f"Ground A pos: {sim.positions[ground_a_id]}")
    print(f"Ground B pos: {sim.positions[ground_b_id]}")
    
    injector = FailureInjector(sim)
    topo = SwarmTopologyManager(sim)
    channel = ChannelModel()
    policy = policy_cls(sim)
    
    failed = False
    reconnected_step = None
    
    for t in range(250):
        if t == 20:
            print(f"[Step 20] Injecting failure into agent 3 (position: {sim.positions[3]})")
            injector.fail_agent(3)
            failed = True
            
        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)
        
        G = topo.build_graph()
        has_path = topo.has_path(ground_a_id, ground_b_id, graph=G)
        
        if failed and reconnected_step is None and has_path:
            reconnected_step = t - 20
            print(f"===> RECONNECTED at tick t={t} (time_to_reconnect = {reconnected_step} ticks) <===")
            
        if t in [0, 20, 50, 100, 150, 200, 249]:
            rate = channel.compute_network_throughput(sim, G)
            print(f"Step {t:3d} | Path A->B: {str(has_path):5s} | Sum-Rate: {rate:7.2f} Mbps | Components: {topo.get_num_connected_components(G)}")

diagnose(DARAHeuristicPolicy, "DARA")
diagnose(GreedyDARAHeuristicPolicy, "Greedy DARA")
