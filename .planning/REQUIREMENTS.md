# Project Requirements: SWARM

## Functional Requirements

### FR-1: UAV Agent Kinematics & State Representation
- **FR-1.1**: Maintain state matrices for $N$ UAV agents including position $\mathbf{p}_i \in \mathbb{R}^2$ (or $\mathbb{R}^3$), velocity $\mathbf{v}_i$, max speed $v_{\max}$, max acceleration $a_{\max}$, and active/failed status.
- **FR-1.2**: Update positions using kinematic integration $\mathbf{p}_i(t+\Delta t) = \mathbf{p}_i(t) + \mathbf{v}_i(t)\Delta t$ with velocity clamping.
- **FR-1.3**: Support configurable communication radius $R_c$ and sensing radius $R_s$.
- **FR-1.4**: Standardized Enums (`swarm_sim/utils/enums.py`):
  All policy, simulation, and telemetry/logging code MUST reference these central Python `Enum` classes instead of raw string literals:
  ```python
  class AgentRole(Enum):
      RELAY = "relay"
      GROUND_A = "ground_a"
      GROUND_B = "ground_b"

  class AgentStatus(Enum):
      ACTIVE = "active"
      FAILED = "failed"

  class RecoveryStrategy(Enum):
      NONE = "none"
      HEURISTIC_DARA = "heuristic_dara"
      GREEDY_DARA = "greedy_dara"
      LEARNED_GNN = "learned_gnn"
  ```

### FR-2: Spatial Indexing & Dynamic Channel Topology Engine
- **FR-2.1**: Use `scipy.spatial.cKDTree` for fast $O(N \log N)$ neighbor detection within radius $R_c$.
- **FR-2.2**: Build dynamic `NetworkX.Graph` representing current communication topology.
- **FR-2.3**: Shakhatreh et al. (2021) Path-Loss & Throughput Model (`swarm_sim/graph/channel.py`):
  - Air-to-Ground (A2G) / Relay-Relay Link: Elevation-dependent $P_{\text{LOS}}(\theta)$ path loss and achievable rate $R_{ij}$.
  - Cellular Backhaul / Relay-Base-Station Link: Cellular-to-UAV path loss formula.
  - Edge feature `edge_type` flag discriminating A2G/Relay-Relay (0) vs. Relay-Cellular-Backhaul (1).
- **FR-2.4**: Compute network health metrics:
  - End-to-end path existence between Endpoint A and Endpoint B.
  - Algebraic connectivity ($\lambda_2$ of graph Laplacian).
  - Shortest path length (hop count) between A and B.
  - Number of connected components (fragmentation state).
  - Aggregate network sum-rate (achievable throughput).

### FR-3: Disaster-Zone Scenario & Relay Chain Setup
- **FR-3.1**: Define Ground Endpoint A (Base) and Ground Endpoint B (Target Field Unit).
- **FR-3.2**: Initialize UAV relay chain bridging A and B across specified obstacle or distance configuration.
- **FR-3.3**: Support customizable initial swarm topologies (line relay, grid lattice, random scatter with initial convergence).

### FR-4: Failure & Perturbation Injector
- **FR-4.1**: Scheduled or stochastic node failures (single node death, simultaneous multi-node failure, localized spatial EMP/outage zone).
- **FR-4.2**: Real-time fragmentation detection triggering recovery alarms when path A-to-B breaks or graph partitions into $>1$ component.

### FR-5: Recovery & Self-Organization Policies
- **FR-5.1**: Baseline DARA Heuristic (`swarm_sim/policies/heuristic_dara.py`): Varadharajan et al. / DARA self-healing chain reorganization using 100% graph neighbor perception.
- **FR-5.2**: Baseline Greedy DARA (`swarm_sim/policies/greedy_dara.py`): DARA midpoint relaxation with greedy throughput-aware tiebreak toward higher SNR.
- **FR-5.3**: CTDE GNN RL Policy (`swarm_sim/policies/gnn_actor.py`, `swarm_sim/policies/gnn_critic.py`):
  - Actor: $k$-hop subgraph message-passing GNN ($k=1$ default/primary, ablations $k \in \{1, 2, 3\}$).
  - Critic: Full-graph encoder GNN evaluating team value $V(s)$.
  - Features: Node features (pos, vel, role, status, degree, `has_lost_neighbor`, `offset_to_f_pos`, `is_ground_anchor`), Edge features ($d_{ij}$, SNR/rate, `edge_type`, relative direction vector).

### FR-6: 3-Checkpoint Training & Evaluation Pipeline
- **FR-6.1**: Checkpoint A — Imitation / Behavior Cloning (BC) off DARA rollouts + DAgger fallback (up to 5 rounds). Gates: Open-loop RMSE $< 0.1 R_c$, Closed-loop $N=50$ test bank success $\ge 95\%$, mean time $\le 1.2\times$ DARA.
- **FR-6.2**: Checkpoint B — RL Fine-Tuning connectivity-only ($\alpha=0$). Gate: Success rate drop $\le 5\text{pp}$, time increase $\le 10\%$ vs A.
- **FR-6.3**: Checkpoint C — Throughput Annealing ($\alpha > 0$). Gates: Success rate drop $\le 5\text{pp}$, time $\le 1.3\times$ DARA Phase 4 baseline, aggregate sum-rate $\ge 10\%$ higher than B.
- **FR-6.4**: $N=50$ Fixed Test Bank evaluation methodology with binomial CI awareness.

### FR-7: Telemetry, Logging & Visualization
- **FR-7.1**: Two-Level Telemetry Logging (`swarm_sim/utils/logger.py`):
  - Per-Timestep Log: `run_id`, `timestep`, `agent_id`, `agent_role`, `x`, `y`, `status`, `degree`.
  - Per-Run Summary Log: `run_id`, `config_hash`, `seed`, `num_drones`, `comm_range`, `failure_timestep`, `failure_count`, `recovery_strategy`, `time_to_reconnect`, `final_connectivity`, `min_connectivity_during_failure`.
- **FR-7.2**: Matplotlib 2D real-time visualizer (`swarm_sim/viz/animator.py`).
- **FR-7.3**: Pillow GIF exporter (`swarm_sim/viz/exporter.py`).
- **FR-7.4**: Executable CLI script (`run_experiment.py`) with reproducible seed logging.

---

## Non-Functional Requirements

### NFR-1: Performance & Scalability
- Simulation update loop must achieve $>20$ Hz execution speed for $N \ge 100$ agents using NumPy vectorization and cKDTree.

### NFR-2: Software Engineering & Modularity
- Clean modular package structure (`swarm_sim/` with submodules `core`, `graph`, `policies`, `scenarios`, `viz`, `utils`).
- Package management configured via `uv` (`pyproject.toml`).

### NFR-3: Reproducibility & Testing
- Deterministic random seed initialization for reproducible simulation runs.
- Every experiment run MUST log its random seed in the output CSV so results are strictly reproducible from the CSV alone.
- Unit and integration test coverage for channel math, GNN models, training pipeline, telemetry, and CLI runner.
