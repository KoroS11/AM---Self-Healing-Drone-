import torch
from scratch.evaluate_b_prime_full_battery import build_all_scenarios
from swarm_sim.policies.gnn_actor import ActorGNN
from scratch.test_barrier_margin_sweep import ParametricCBFGNNPolicy
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import AgentStatus, AgentRole
from swarm_sim.core.simulator import SwarmSimulator
from swarm_sim.core.failure import FailureInjector
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario
from swarm_sim.graph.topology import SwarmTopologyManager

bp_path = "checkpoints/checkpoint_B_prime_best.pt"
actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
raw = torch.load(bp_path, map_location="cpu", weights_only=False)
sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
actor_bp.load_state_dict(sd, strict=True)
actor_bp.eval()

all_scenarios = build_all_scenarios()
sc21_center = all_scenarios["3. Scale Test (Large Swarm)"][2]

print("=== N=21 d=500m Center Failure: Margin Sweep ===")
for eps in [0.00, 0.01, 0.02, 0.03, 0.05]:
    config = ExperimentConfig(
        num_drones=sc21_center["num_drones"],
        comm_range=sc21_center["comm_range"],
        seed=sc21_center["seed"]
    )
    sim = SwarmSimulator(config)
    scenario = DisasterRelayScenario(sim)
    g_a, g_b = scenario.setup_scenario(
        endpoint_a_pos=sc21_center.get("endpoint_a_pos", (0.0, 0.0)),
        endpoint_b_pos=sc21_center.get("endpoint_b_pos", (500.0, 0.0))
    )
    injector = FailureInjector(sim)
    policy = ParametricCBFGNNPolicy(sim, actor_gnn=actor_bp, alpha1=3.0, alpha2=1.5, barrier_margin=eps, device="cpu")
    topo = SwarmTopologyManager(sim)

    failures = list(sc21_center.get("failures", []))
    last_fail_step = max([f["step"] for f in failures]) if failures else 0

    first_conn = None
    drops = 0
    prev_c = False
    conn_hist = []

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
        conn_hist.append(is_conn)

        if t >= last_fail_step:
            if is_conn:
                if first_conn is None:
                    first_conn = t
                prev_c = True
            else:
                if prev_c:
                    drops += 1
                prev_c = False

    k20 = all(conn_hist[280:300])
    c300 = conn_hist[299]
    c150 = conn_hist[149]
    print(f"eps={eps:.2f}m: FirstReconn={first_conn}, Drops={drops}, Conn@150={c150}, Conn@300={c300}, K20={k20}")
