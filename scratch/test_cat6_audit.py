import torch
from scratch.evaluate_b_prime_full_battery import build_all_scenarios
from swarm_sim.policies.gnn_actor import ActorGNN
from scratch.audit_300_ticks_full_battery import run_300t_eval

bp_path = "checkpoints/checkpoint_B_prime_best.pt"
actor_bp = ActorGNN(node_in_dim=9, edge_in_dim=5, k_hops=2, max_accel=5.0)
raw = torch.load(bp_path, map_location="cpu", weights_only=False)
sd = raw["actor_state_dict"] if "actor_state_dict" in raw else raw
actor_bp.load_state_dict(sd, strict=True)
actor_bp.eval()

all_scenarios = build_all_scenarios()
cat6 = all_scenarios["6. Tight-Margin Geometry"]

for sc in cat6:
    res = run_300t_eval(actor_bp, sc, alpha=(3.0, 1.5))
    print(sc["desc"])
    print(f"  snap_150: {res['snap_150']}, snap_300: {res['snap_300']}, k20_150: {res['k20_150']}, k20_300: {res['k20_300']}, drops: {res['drops_after']}")
