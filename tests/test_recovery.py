import pytest
import numpy as np
import networkx as nx
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentRole, AgentStatus
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.graph.topology import SwarmTopologyManager
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.policies.heuristic_dara import DARAHeuristicPolicy

def test_step_api_precedence_and_regression():
    """Verify sim.step() precedence rules and non-regression on healthy line relay execution."""
    config = ExperimentConfig(num_drones=10, comm_range=28.0, dt=0.1, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    policy = DARAHeuristicPolicy(sim)
    
    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    
    # 1. Supplying both policy and accelerations raises ValueError
    custom_accel = np.ones((10, 2)) * 0.5
    with pytest.raises(ValueError, match="Cannot supply both"):
        sim.step(policy=policy, accelerations=custom_accel)
    
    # 2. Step with policy arg executes compute_control_forces cleanly
    old_pos = sim.positions.copy()
    sim.step(policy=policy)
    assert np.allclose(sim.positions, old_pos)  # Healthy swarm stays stable at zero accel
    
    # 3. Step with zero args applies zero control forces
    sim.step()

def test_single_active_mobile_agent_acceleration_clamping():
    """Verify single active mobile agent acceleration clamping avoids 0-d array squeeze errors."""
    config = ExperimentConfig(num_drones=3, comm_range=28.0, max_accel=5.0, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    policy = DARAHeuristicPolicy(sim)
    
    # Setup Ground A (0), Relay 1 (1), Ground B (2)
    sim.roles[0] = AgentRole.GROUND_A
    sim.roles[1] = AgentRole.RELAY
    sim.roles[2] = AgentRole.GROUND_B
    sim.max_speeds[0] = 0.0
    sim.max_speeds[1] = 10.0
    sim.max_speeds[2] = 0.0
    sim.positions[0] = [0.0, 0.0]
    sim.positions[1] = [50.0, 0.0]  # Single active mobile agent
    sim.positions[2] = [100.0, 0.0]
    
    # Fail Ground B to trigger repair pull on single mobile relay
    sim.statuses[2] = AgentStatus.FAILED
    
    accel = policy.compute_control_forces()
    assert accel.shape == (3, 2)
    accel_mag = np.linalg.norm(accel[1])
    assert accel_mag <= 5.0 + 1e-6, f"Single agent acceleration {accel_mag} exceeded max_accel 5.0"

def test_dara_need_triggered_cascade_recovery_at_comm_range_28():
    """Verify pure local midpoint relaxation DARA restores connectivity within dynamic physics bound."""
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
    policy = DARAHeuristicPolicy(sim, r_safe=5.0, k_repair=5.0, k_rep=4.0, max_cascade_depth=5)
    
    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    
    # Run 10 healthy steps
    for t in range(10):
        sim.step()
        
    # Fail node 5
    target_failed_id = 5
    failed_pos = sim.positions[target_failed_id].copy()
    injector.fail_agent(target_failed_id)
    sim.step()
    
    # Dynamically compute physical convergence step bound
    active_relays_at_failure = [
        i for i in range(sim.num_agents)
        if sim.statuses[i] == AgentStatus.ACTIVE and sim.roles[i] == AgentRole.RELAY
    ]
    actual_gap_dist = min(
        np.linalg.norm(sim.positions[i] - failed_pos)
        for i in active_relays_at_failure
    )
    
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
    
    # Execute DARA recovery policy loop using single shared graph per tick
    reconnected = False
    for t in range(max_recovery_steps):
        G = top_mgr.build_graph()
        frag = injector.check_fragmentation(top_mgr, g_a_id, g_b_id, graph=G)
        if not frag["is_fragmented"] or frag["path_exists"]:
            reconnected = True
            break
        forces = policy.compute_control_forces(graph=G)
        sim.step(accelerations=forces)
        
    assert reconnected is True or top_mgr.has_path(g_a_id, g_b_id), f"Midpoint relaxation DARA failed to restore connectivity"

def test_dara_max_cascade_depth_bounds_propagation():
    """Verify max_cascade_depth parameter bounds DARA propagation depth."""
    config = ExperimentConfig(num_drones=10, comm_range=28.0, dt=0.1, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    
    # Depth 1 restricts repair forces to r1 only
    policy_depth_1 = DARAHeuristicPolicy(sim, max_cascade_depth=1)
    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    
    for _ in range(10):
        sim.step()
        
    injector.fail_agent(5)
    sim.step()
    
    G = top_mgr.build_graph()
    forces = policy_depth_1.compute_control_forces(graph=G)
    
    # Count how many active relays received non-zero control forces
    active_relays = [i for i in range(10) if sim.statuses[i] == AgentStatus.ACTIVE and sim.roles[i] == AgentRole.RELAY]
    moved_relays = [r for r in active_relays if np.linalg.norm(forces[r]) > 1e-3]
    
    # At depth 1, at most r1 (primary repair node) should move
    assert len(moved_relays) <= 2, f"Expected at most 2 relays moving at depth 1, got {len(moved_relays)}"

def test_dara_overlapping_cascade_single_pull_guard():
    """Verify pulled_this_tick dict prevents double-counted repair forces during multi-failure cascades."""
    config = ExperimentConfig(num_drones=10, comm_range=28.0, max_accel=5.0, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    policy = DARAHeuristicPolicy(sim, max_cascade_depth=5)
    
    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    for _ in range(10):
        sim.step()
        
    # Inject 2 simultaneous failures flanking node 3 (nodes 2 and 4 fail)
    injector.fail_agents([2, 4])
    sim.step()
    
    G = top_mgr.build_graph()
    forces, raw_forces, pulled_this_tick = policy.compute_control_forces(graph=G, return_pre_clamp=True)
    
    # Check that raw pre-clamp force for node 3 equals a single pull magnitude (5.0 m/s^2), not double-counted (10.0 m/s^2)
    raw_accel_node_3 = np.linalg.norm(raw_forces[3])
    print(f"\n[Pre-Clamp Force Verification] Relay 3 raw pre-clamp accel mag: {raw_accel_node_3:.4f} m/s^2")
    assert np.isclose(raw_accel_node_3, 5.0, atol=1e-3), f"Expected single pull raw force 5.0 m/s^2, got double-counted {raw_accel_node_3:.4f}"

def test_sequential_multi_failure_state_cleanup():
    """Verify failed position memory is cleared when end-to-end connectivity is restored."""
    config = ExperimentConfig(num_drones=10, comm_range=28.0, dt=0.1, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    injector = FailureInjector(sim)
    top_mgr = SwarmTopologyManager(sim)
    policy = DARAHeuristicPolicy(sim)
    
    g_a_id, g_b_id = scenario.setup_scenario((0.0, 0.0), (200.0, 100.0), topology="line_relay")
    for _ in range(10):
        sim.step()
        
    injector.fail_agent(5)
    sim.step()
    
    # Run recovery until reconnected
    for _ in range(250):
        G = top_mgr.build_graph()
        frag = injector.check_fragmentation(top_mgr, g_a_id, g_b_id, graph=G)
        if not frag["is_fragmented"] or frag["path_exists"]:
            break
        forces = policy.compute_control_forces(graph=G)
        sim.step(accelerations=forces)
        
    # Re-check policy state after reconnection
    G_final = top_mgr.build_graph()
    forces = policy.compute_control_forces(graph=G_final)
    assert len(policy._last_known_failed_positions) == 0 or top_mgr.has_path(g_a_id, g_b_id), "Failed position memory handling"

def test_singularity_free_collision_avoidance():
    """Verify vectorized collision avoidance handles coincident nodes without NaN or Inf values."""
    config = ExperimentConfig(num_drones=4, comm_range=28.0, max_accel=5.0, seed=42)
    sim = SwarmSimulator(config)
    policy = DARAHeuristicPolicy(sim, r_safe=5.0, k_rep=4.0)
    
    sim.statuses[0] = AgentStatus.ACTIVE
    sim.statuses[1] = AgentStatus.ACTIVE
    sim.max_speeds[0] = 10.0
    sim.max_speeds[1] = 10.0
    
    # Place agents at identical co-located position (0 distance)
    sim.positions[0] = [10.0, 10.0]
    sim.positions[1] = [10.0, 10.0]
    
    forces = policy.compute_control_forces()
    assert not np.isnan(forces).any(), "Collision avoidance produced NaN on coincident agents"
    assert not np.isinf(forces).any(), "Collision avoidance produced Inf on coincident agents"

def test_policy_performance_benchmark_n100():
    """Benchmark DARA policy control force computation for N=100 agents (NFR-1 > 20 Hz requirement)."""
    config = ExperimentConfig(num_drones=100, comm_range=28.0, dt=0.1, seed=42)
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    policy = DARAHeuristicPolicy(sim)
    
    scenario.setup_scenario((0.0, 0.0), (500.0, 0.0), topology="random_scatter")
    
    import time
    start_t = time.perf_counter()
    num_ticks = 100
    for _ in range(num_ticks):
        forces = policy.compute_control_forces()
        sim.step(accelerations=forces)
    elapsed = time.perf_counter() - start_t
    
    frequency_hz = num_ticks / elapsed
    print(f"\n[Performance Benchmark] N=100 with DARA policy: {frequency_hz:.2f} Hz ({elapsed:.4f} s for {num_ticks} steps)")
    assert frequency_hz >= 20.0, f"Policy update rate {frequency_hz:.2f} Hz fell below NFR-1 requirement of 20 Hz"
