import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy

config = ExperimentConfig(num_drones=10, comm_range=35.0, dt=0.1, max_speed=10.0, max_accel=5.0, seed=42)
sim = SwarmSimulator(config)
scenario = DisasterRelayScenario(sim)
injector = FailureInjector(sim)
top_mgr = SwarmTopologyManager(sim)
policy = DARAHeuristicPolicy(sim, r_safe=5.0, k_repair=5.0, k_rep=4.0)

g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")

for t in range(10):
    sim.step()

injector.fail_agent(5)
sim.step()

for t in range(30):
    G = top_mgr.build_graph()
    frag = injector.check_fragmentation(top_mgr, g_a_id, g_b_id, graph=G)
    print(f"t={t}: frag={frag['is_fragmented']}, path={frag['path_exists']}, comps={frag['num_components']}")
    if not frag['is_fragmented']:
        print(f"RECONNECTED AT RECOVERY STEP {t}")
        break
    forces = policy.compute_control_forces(graph=G)
    sim.step(accelerations=forces)
