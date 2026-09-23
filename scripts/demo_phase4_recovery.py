import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy

def run_phase4_demo():
    print("=" * 66)
    print("       SWARM Phase 4 DARA Self-Healing Recovery Demo       ")
    print("=" * 66)

    # 1. Initialize simulator with N=10, Rc=28.0m
    config = ExperimentConfig(
        num_drones=10,
        comm_range=28.0,
        dt=0.1,
        max_speed=10.0,
        max_accel=5.0,
        seed=42
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    policy = DARAHeuristicPolicy(sim, r_safe=5.0, k_repair=5.0, k_rep=4.0, k_damp=3.0, max_cascade_depth=5)

    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    print(f"\n[Step 0] Initialized Line Relay Scenario (N={sim.num_agents} total agents, Rc={config.comm_range}m)")

    # 2. Run healthy steps
    for t in range(10):
        sim.step()

    # 3. Inject single node failure (Node 5)
    target_failed_id = 5
    failed_pos = sim.positions[target_failed_id].copy()
    injector.fail_agent(target_failed_id)
    sim.step()
    print(f"\n[Step 10] Injected Failure into Relay Node {target_failed_id}")

    # Compute physical bounds
    active_relays_at_failure = [
        i for i in range(sim.num_agents)
        if sim.statuses[i] == AgentStatus.ACTIVE and sim.roles[i] == AgentRole.RELAY
    ]
    actual_gap_dist = min(
        np.linalg.norm(sim.positions[i] - failed_pos)
        for i in active_relays_at_failure
    )
    print(f"  Actual Gap Distance: {actual_gap_dist:.2f} m")

    v_max = config.max_speed
    a_max = config.max_accel
    t_accel = v_max / a_max
    d_accel = 0.5 * a_max * (t_accel ** 2)

    if actual_gap_dist <= d_accel:
        t_primary = np.sqrt(2.0 * actual_gap_dist / a_max)
    else:
        t_primary = t_accel + (actual_gap_dist - d_accel) / v_max

    a_cascade = 0.5 * a_max
    t_cascade = np.sqrt(2.0 * actual_gap_dist / a_cascade)
    total_base_steps = (t_primary + t_cascade) / config.dt
    max_recovery_steps = int(np.ceil(total_base_steps * 2.5))
    print(f"  Max Recovery Steps : {max_recovery_steps} steps (Multi-hop physics bound)")

    # 4. Execute DARA recovery loop
    reconnection_step = None
    for t in range(250):
        G = top_mgr.build_graph()
        frag = injector.check_fragmentation(top_mgr, g_a_id, g_b_id, graph=G)
        if (not frag["is_fragmented"] or frag["path_exists"]) and reconnection_step is None:
            reconnection_step = 10 + t
        forces = policy.compute_control_forces(graph=G)
        sim.step(accelerations=forces)

    final_G = top_mgr.build_graph()
    final_frag = injector.check_fragmentation(top_mgr, g_a_id, g_b_id, graph=final_G)
    metrics = top_mgr.get_network_metrics(g_a_id, g_b_id)

    print(f"\n[Final Status] Step {10 + 250}")
    print(f"  Reconnection Step: {reconnection_step if reconnection_step else 'N/A'}")
    print(f"  Path A->B Exists : {final_frag['path_exists']}")
    print(f"  Components       : {metrics['num_components']}")
    print(f"  Alg Connectivity : {metrics['algebraic_connectivity']:.4f}")

    print(f"\n  [Numeric Edge Verification at Final Step (comm_range = {config.comm_range}m)]:")
    edges = list(final_G.edges())
    for u, v in sorted(edges):
        dist = np.linalg.norm(sim.positions[u] - sim.positions[v])
        print(f"    Edge ({u:2d} <-> {v:2d}): distance = {dist:6.2f} m  [<= {config.comm_range}m: {dist <= config.comm_range}]")

    # 5. Plotting results
    plt.figure(figsize=(10, 5), dpi=150)
    
    # Draw real graph edges ONLY
    for u, v in final_G.edges():
        p1, p2 = sim.positions[u], sim.positions[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#2ecc71', lw=2, zorder=2, label='Restored Comm Edge' if 'Restored Comm Edge' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Draw active relays
    active_mask = (sim.statuses == AgentStatus.ACTIVE) & (sim.roles == AgentRole.RELAY)
    plt.scatter(sim.positions[active_mask, 0], sim.positions[active_mask, 1], color='#3498db', s=100, zorder=3, label='Active Relay')

    # Draw ground stations
    ground_mask = (sim.roles == AgentRole.GROUND_A) | (sim.roles == AgentRole.GROUND_B)
    plt.scatter(sim.positions[ground_mask, 0], sim.positions[ground_mask, 1], color='#e74c3c', s=150, marker='s', zorder=4, label='Ground Endpoint')

    # Draw failed node
    plt.scatter(failed_pos[0], failed_pos[1], color='#7f8c8d', s=120, marker='x', zorder=4, label='Failed Node (No Edge)')

    # Labels for active nodes
    for i in range(sim.num_agents):
        if sim.statuses[i] == AgentStatus.ACTIVE:
            plt.annotate(f"r{i}" if sim.roles[i] == AgentRole.RELAY else f"G", 
                         (sim.positions[i, 0], sim.positions[i, 1] + 3.0),
                         ha='center', fontsize=9, fontweight='bold', color='#2c3e50')

    plt.title(f"Phase 4: DARA Recovery Reconnection (Reconnected at Step {reconnection_step if reconnection_step else 'N/A'})")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()

    out_file = "demo_phase4_recovery.png"
    plt.savefig(out_file)
    plt.close()
    print(f"\nSaved Phase 4 recovery plot to: {out_file}")

if __name__ == "__main__":
    run_phase4_demo()
