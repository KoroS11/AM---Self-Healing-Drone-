import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy
from swarm_sim.graph.topology import SwarmTopologyManager

config = ExperimentConfig(num_drones=9, comm_range=28.0, seed=42)
sim = SwarmSimulator(config)
scenario = DisasterRelayScenario(sim)
g_a, g_b = scenario.setup_scenario((0.0, 0.0), (200.0, 0.0))

injector = FailureInjector(sim)
policy = DARAHeuristicPolicy(sim)

print("Agent roles:")
for i in range(9):
    print(f"  Agent {i}: role={sim.roles[i].value}, max_speed={sim.max_speeds[i]}, pos={sim.positions[i]}")

for t in range(50):
    if t == 20:
        injector.fail_agent(3)
        print(f"\n--- [Step 20] Agent 3 failed at pos {sim.positions[3]} ---")
        
    acc = policy.compute_control_forces()
    sim.step(accelerations=acc)
    
    if t >= 20:
        print(f"Step {t:2d}: positions = {[round(p[0], 1) for p in sim.positions]}")
