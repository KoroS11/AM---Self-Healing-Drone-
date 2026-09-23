# SWARM Roadmap

## Milestones & Phases

### Phase 1: Environment Setup & Core Kinematic Engine
- **Goal**: Establish project environment with `uv`, create core package structure, implement UAV kinematic state representation and simulation loop.
- **Deliverables**:
  - `pyproject.toml` with dependencies (`numpy`, `scipy`, `networkx`, `matplotlib`, `pillow`, `pandas`, `pytest`).
  - `swarm_sim/utils/enums.py`: Standardized `AgentRole`, `AgentStatus`, and `RecoveryStrategy` enums.
  - `swarm_sim/core/agent.py` & `swarm_sim/core/simulator.py`: Kinematic updates and spatial neighbor queries via `cKDTree`.
  - Basic unit tests verifying kinematic motion and neighbor search accuracy.

### Phase 2: Communication Graph & Disaster Scenario Environment
- **Goal**: Build network graph engine and disaster-zone scenario featuring Endpoints A & B.
- **Deliverables**:
  - `swarm_sim/graph/topology.py`: NetworkX integration, algebraic connectivity ($\lambda_2$), pathfinding, and graph fragmentation metrics.
  - `swarm_sim/scenarios/disaster_relay.py`: Setup for Ground Endpoints A & B and initial relay chain configuration.
  - Verification test for end-to-end path detection.

### Phase 3: Failure Injector & Fragmentation Detection
- **Goal**: Implement node failure simulation and real-time graph fragmentation monitoring without recovery logic.
- **Deliverables**:
  - `swarm_sim/core/failure.py`: Failure modes (instant death, battery drain, spatial outage zone).
  - Unit/integration test verifying that injected failures are correctly detected as fragmentation events (graph partition count > 1, or A-to-B path loss) within one simulation step of the failure.
  - Demonstration plot showing connectivity/fragmentation visibly degrading after failure with no response (independently demonstrable baseline).

### Phase 4: Heuristic Recovery Policy
- **Goal**: Implement the DARA / chain-healing algorithm (Varadharajan et al. 2020 / Wang et al. 2016) for self-healing gap closure.
- **Deliverables**:
  - `swarm_sim/policies/base.py`: Abstract policy interface referencing standardized enums.
  - `swarm_sim/policies/heuristic_dara.py`: DARA / Varadharajan et al. chain repair heuristic policy.
  - Integration test verifying network repair after node failure: assert connectivity is restored (single connected component, A-to-B path exists) within a bounded number of steps after failure injection.
  - Demonstration plot matching Phase 3 baseline but showing successful connectivity recovery.

### Phase 5: Throughput-Aware Recovery Policy & Telemetry Engine
- **Goal**: Build Shakhatreh et al. (2021) path-loss/throughput channel model, CTDE GNN RL policy engine (Actor/Critic), DARA + greedy throughput baseline, 3-checkpoint training pipeline (BC+DAgger, RL connectivity, RL throughput annealing), 2-level Pandas telemetry logger, Matplotlib animator, Pillow GIF exporter, and CLI experiment runner.
- **Deliverables**:
  - `swarm_sim/graph/channel.py`: A2G and cellular backhaul path-loss/SNR/data-rate model.
  - `swarm_sim/policies/greedy_dara.py`: DARA + greedy throughput-aware tiebreak baseline.
  - `swarm_sim/policies/gnn_actor.py` & `swarm_sim/policies/gnn_critic.py`: $k$-hop subgraph Actor GNN & full-graph Critic GNN (CTDE MAPPO).
  - `swarm_sim/utils/logger.py`: Two-level Pandas telemetry logger (per-step and per-run summary with seed logging).
  - `swarm_sim/viz/animator.py` & `swarm_sim/viz/exporter.py`: 2D Matplotlib animation renderer & Pillow GIF exporter.
  - `train_pipeline.py` & `run_experiment.py`: 3-checkpoint training pipeline and CLI experiment runner.
  - Integration test suite validating channel math, GNN message passing, training checkpoints, telemetry CSV export, and GIF generation.
