import os, sys, math, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import qpsolvers
import torch

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.gnn_actor import ActorGNN
from swarm_sim.policies.gnn_policy import GNNPolicy
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.graph.channel import ChannelModel
from scratch.evaluate_b_prime_full_battery import build_all_scenarios
from scratch.run_production_verification_battery import LexicographicCBFGNNPolicy


def analyze_initial_conditions_and_tension():
    all_scenarios = build_all_scenarios()
    cat6_scenarios = all_scenarios["6. Tight-Margin Geometry"]

    print("=" * 100)
    print("TASK 3: INITIAL CONDITIONS & SEED-SPECIFIC GEOMETRY")
    print("=" * 100)

    for idx, sc in enumerate(cat6_scenarios):
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
        
        failures = sc.get("failures", [])
        fail_id = failures[0]["agent_id"] if failures else sc.get("fail_agent_id")
        fail_t = failures[0]["step"] if failures else sc.get("fail_timestep")

        print(f"\nScenario {idx+1}: {sc['desc']}")
        print(f"  Seed: {sc['seed']}, N: {sc['num_drones']}, d_AB: {sc.get('endpoint_b_pos')[0]}m")
        print(f"  Fault injection: Agent {fail_id} at t={fail_t}")
        print(f"  Initial Agent Positions (x, y):")
        for a in range(sim.num_agents):
            role_name = sim.roles[a].name if hasattr(sim.roles[a], 'name') else str(sim.roles[a])
            print(f"    Agent {a:2d} ({role_name:<10s}): ({sim.positions[a,0]:6.2f}, {sim.positions[a,1]:6.2f})")

        # Initial chain spacing along corridor
        active_relays = [a for a in range(sim.num_agents) if a not in [0, 1, fail_id]]
        chain_agents = [0] + sorted(active_relays, key=lambda a: sim.positions[a,0]) + [1]
        print(f"  Post-failure surviving chain order: {chain_agents}")
        init_dists = [np.linalg.norm(sim.positions[chain_agents[k+1]] - sim.positions[chain_agents[k]]) for k in range(len(chain_agents)-1)]
        print(f"  Initial distances along post-failure chain:")
        for k in range(len(chain_agents)-1):
            u, v = chain_agents[k], chain_agents[k+1]
            print(f"    {u}-{v}: {init_dists[k]:.2f}m (Gap to Rc=28: {28.0-init_dists[k]:+.2f}m)")


def analyze_osqp_solver_and_tolerances():
    bp_path = "checkpoints/checkpoint_B_prime_best.pt"
    actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
    raw = torch.load(bp_path, map_location="cpu", weights_only=False)
    sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
    actor_bp.load_state_dict(sd, strict=True)
    actor_bp.eval()

    all_scenarios = build_all_scenarios()
    sc3 = all_scenarios["6. Tight-Margin Geometry"][2] # d=192m

    print("\n" + "=" * 100)
    print("TASK 4: OSQP NUMERICAL SOLVER BEHAVIOR ON SCENARIO 3 (d=192m) AROUND t=289")
    print("=" * 100)

    config = ExperimentConfig(
        num_drones=sc3["num_drones"],
        comm_range=sc3["comm_range"],
        seed=sc3["seed"]
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario(
        endpoint_a_pos=sc3.get("endpoint_a_pos", (0.0, 0.0)),
        endpoint_b_pos=sc3.get("endpoint_b_pos", (200.0, 0.0))
    )
    injector = FailureInjector(sim)
    policy = LexicographicCBFGNNPolicy(sim, actor_gnn=actor_bp, alpha1=3.0, alpha2=1.5, device="cpu")
    topo = SwarmTopologyManager(sim)

    failures = list(sc3.get("failures", []))

    for t in range(300):
        for f in failures:
            if t == f["step"]:
                injector.fail_agent(f["agent_id"])

        acc = policy.compute_control_forces()
        sim.step(accelerations=acc)

        G = topo.build_graph()
        g_a_id = 0
        g_b_id = 1
        for idx, role in enumerate(sim.roles):
            if role == AgentRole.GROUND_A or str(role) == "ground_a":
                g_a_id = idx
            elif role == AgentRole.GROUND_B or str(role) == "ground_b":
                g_b_id = idx

        is_conn = topo.has_path(g_a_id, g_b_id, graph=G)

        if 285 <= t <= 295:
            pos = sim.positions
            vel = sim.velocities
            d34 = np.linalg.norm(pos[3] - pos[4])
            v34 = vel[3] - vel[4]
            p34 = pos[3] - pos[4]
            h34 = 28.0**2 - d34**2
            h_dot34 = -2.0 * np.dot(p34, v34)
            rhs34 = -2.0 * np.dot(v34, v34) + 3.0 * h_dot34 + 1.5 * h34

            print(f"t={t:3d} | Conn={str(is_conn):<5s} | d(3,4)={d34:10.6f}m | slack={28.0-d34:+10.6f}m | h={h34:+10.6f} | h_dot={h_dot34:+10.6f} | rhs={rhs34:+10.6f}")
            print(f"      acc[3]=({acc[3,0]:+.3f},{acc[3,1]:+.3f}), acc[4]=({acc[4,0]:+.3f},{acc[4,1]:+.3f}), v[3]=({vel[3,0]:+.3f},{vel[3,1]:+.3f}), v[4]=({vel[4,0]:+.3f},{vel[4,1]:+.3f})")


if __name__ == "__main__":
    analyze_initial_conditions_and_tension()
    analyze_osqp_solver_and_tolerances()
