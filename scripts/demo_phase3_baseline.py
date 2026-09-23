import os
import shutil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario

ARTIFACT_DIR = r"C:\Users\harsh\.gemini\antigravity-ide\brain\3008cc19-832e-42b2-b2d0-33c5c21b5d5f"

def run_baseline_demo():
    print("==================================================================")
    print("       SWARM Phase 3 Unmitigated Fragmentation Baseline Demo       ")
    print("==================================================================")
    
    # 10-node scenario (2 ground endpoints + 8 relays) with Rc=28.0 creating a 1-hop relay chain
    config = ExperimentConfig(
        num_drones=10,
        comm_range=28.0,
        dt=0.1,
        seed=42,
        failure_timestep=10,
        failure_count=1
    )
    
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    
    endpoint_a = (0.0, 0.0)
    endpoint_b = (200.0, 100.0)
    g_a_id, g_b_id = scenario.setup_scenario(endpoint_a, endpoint_b, topology="line_relay")
    
    print("\n[Step 0] Initialized Line Relay Scenario (N=10 total agents, Rc=28.0)")
    initial_metrics = top_mgr.get_network_metrics(g_a_id, g_b_id)
    print(f"  Path A->B Exists : {initial_metrics['path_exists']}")
    print(f"  Components       : {initial_metrics['num_components']}")
    print(f"  Alg Connectivity : {initial_metrics['algebraic_connectivity']:.4f}")
    
    # Run healthy steps t = 0..9
    for t in range(10):
        injector.step(t)
        sim.step()
        
    print("\n[Step 10] Pre-Failure Check (t=10)")
    healthy_metrics = top_mgr.get_network_metrics(g_a_id, g_b_id)
    print(f"  Path A->B Exists : {healthy_metrics['path_exists']}")
    
    # Inject failure into middle relay node 5 at t = 10 BEFORE sim.step()
    target_failed_id = 5
    injector.fail_agent(target_failed_id)
    sim.step()  # step on tick 10
    
    print(f"\n[Step 10] Injected Failure into Relay Node {target_failed_id}")
    frag_metrics = injector.check_fragmentation(top_mgr, g_a_id, g_b_id)
    print(f"  Fragmented Status: {frag_metrics['is_fragmented']}")
    print(f"  Path A->B Exists : {frag_metrics['path_exists']}")
    print(f"  Components       : {frag_metrics['num_components']}")
    print(f"  Alg Connectivity : {frag_metrics['algebraic_connectivity']:.4f}")
    
    # Run unmitigated steps t = 11..30
    for t in range(11, 30):
        sim.step()
        
    print("\n[Step 30] Final Unmitigated Status (t=30)")
    final_metrics = injector.check_fragmentation(top_mgr, g_a_id, g_b_id)
    print(f"  Path A->B Exists : {final_metrics['path_exists']}")
    print(f"  Components       : {final_metrics['num_components']}")
    
    # Plot baseline visualization using exact NetworkX graph edges
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # Build NetworkX graph once
    G = top_mgr.build_graph()
    
    # Draw ONLY active valid communication edges from G.edges()
    edge_drawn = False
    for u, v in G.edges():
        pos_u = sim.positions[u]
        pos_v = sim.positions[v]
        label = "Communication Edge" if not edge_drawn else ""
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color="#1f77b4", linestyle="-", alpha=0.7, linewidth=2.0, label=label)
        edge_drawn = True

    # Identify nodes in Ground A component vs Ground B component
    comp_a = nx.node_connected_component(G, g_a_id) if g_a_id in G else set()
    comp_b = nx.node_connected_component(G, g_b_id) if g_b_id in G else set()
    
    relays_a = [i for i in comp_a if sim.roles[i] == AgentRole.RELAY]
    relays_b = [i for i in comp_b if sim.roles[i] == AgentRole.RELAY]
    failed_nodes = [i for i in range(sim.num_agents) if sim.statuses[i] == AgentStatus.FAILED]
    
    # Scatter active relay nodes by component
    if relays_a:
        ax.scatter(sim.positions[relays_a, 0], sim.positions[relays_a, 1], color="#1f77b4", label="Component A Relays", s=90, zorder=3)
    if relays_b:
        ax.scatter(sim.positions[relays_b, 0], sim.positions[relays_b, 1], color="#ff7f0e", label="Component B Relays", s=90, zorder=3)
        
    # Scatter ground endpoints
    ax.scatter(sim.positions[g_a_id, 0], sim.positions[g_a_id, 1], color="#d62728", marker="s", label="Ground Endpoint A", s=150, zorder=4)
    ax.scatter(sim.positions[g_b_id, 0], sim.positions[g_b_id, 1], color="#2ca02c", marker="s", label="Ground Endpoint B", s=150, zorder=4)
    
    # Scatter failed node with large red 'X'
    for f_id in failed_nodes:
        pos_f = sim.positions[f_id]
        ax.scatter(pos_f[0], pos_f[1], color="darkred", marker="x", label=f"Failed Node (ID {f_id})", s=200, linewidths=3.5, zorder=5)
        # Draw red outage circle representing broken link gap
        circle = plt.Circle(pos_f, 28.0, color="red", fill=True, alpha=0.12, linestyle="--", linewidth=1.5, zorder=1)
        ax.add_patch(circle)
        # Annotate broken gap text
        ax.annotate(
            "BROKEN LINK GAP\n(Path A->B Lost)",
            xy=(pos_f[0], pos_f[1]),
            xytext=(pos_f[0], pos_f[1] + 35),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="darkred", lw=1.5),
            zorder=6
        )

    ax.set_title("Phase 3 Baseline: Network Graph Partitioning Following Node Failure (N=10)", fontsize=12, fontweight="bold", color="darkred")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    
    plt.tight_layout()
    
    png_name = "demo_phase3_baseline.png"
    local_png = os.path.abspath(png_name)
    plt.savefig(local_png, dpi=150)
    plt.close(fig)
    
    if os.path.exists(ARTIFACT_DIR):
        artifact_png = os.path.join(ARTIFACT_DIR, png_name)
        shutil.copy(local_png, artifact_png)
        
    print(f"\nSaved Phase 3 baseline plot to: {local_png}")

if __name__ == "__main__":
    run_baseline_demo()
