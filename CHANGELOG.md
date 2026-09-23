# Changelog

All notable changes to the SWARM simulator project will be documented in this file.

## [0.4.0] - Phase 4: Heuristic Recovery Policy - 2026-09-03

### Added
- **Core Recovery Policy Architecture**:
  - `BasePolicy` abstract class in `swarm_sim/policies/base.py` referencing standardized `RecoveryStrategy` enum.
  - `DARAHeuristicPolicy` in `swarm_sim/policies/heuristic_dara.py` implementing multi-hop localized DARA (Wang et al. 2016) and Varadharajan et al. (2020) self-healing chain repair with configurable `max_cascade_depth` (default 5).
- **Localized Gap Repair & Safety**:
  - Primary repair node selection ($r_1 = \arg\min_{i} \|\mathbf{p}_i - \mathbf{p}_k^{\text{last}}\|$).
  - Multi-hop cascade propagation bounded by `max_cascade_depth` (default 5), enabling full gap closure under original $R_c = 28.0$ m constraint.
  - 100% NumPy-Vectorized Collision Avoidance ($R_{\text{safe}}$): pairwise displacement and repulsive force calculation eliminating Python loops over agent pairs for NFR-1 performance.
  - Branchless Safe-Division Acceleration Clamping: `np.maximum(acc_mags, 1e-9)` preventing 0-d scalar array errors when $M=1$.
  - Explicit State Cleanup & Velocity Zeroing: `.clear()` removes tracked failed node positions once global reconnection is confirmed and locks equilibrium.
- **Simulator Policy Integration**:
  - Refactored `SwarmSimulator.step(accelerations, policy)` with explicit precedence rules (raises `ValueError` if both are supplied).
- **Demonstration & Tests**:
  - `scripts/demo_phase4_recovery.py` generating `demo_phase4_recovery.png` demonstrating self-healing network repair under original $R_c = 28.0$ m constraint with explicit numeric inter-relay edge verification ($27.89 \text{ m} \dots 27.99 \text{ m} \le 28.0 \text{ m}$).
  - `tests/test_recovery.py` testing policy precedence, $M=1$ acceleration clamping, multi-hop DARA recovery at $R_c = 28.0$ m, `max_cascade_depth` propagation bounding, multi-failure cleanup, topological cascade scoping, singularity-free repulsion, and $N=100$ performance benchmark ($>20$ Hz).

---

## [0.3.0] - Phase 3: Failure Injector & Fragmentation Detection - 2026-09-03

### Added
- **Core Failure Injector Engine**:
  - `FailureInjector` in `swarm_sim/core/failure.py` supporting single agent failure, bulk failure (`fail_agents`), spatial outage zones (`fail_spatial_zone`), and scheduled timestep failures (`step`).
  - Velocity Zeroing: Zeroes velocity vector immediately upon failure so failed agents stop motion.
  - Ground Endpoint Protection: Guards `GROUND_A` and `GROUND_B` nodes from failure unless explicitly passed `force=True`. Single `fail_agent(0, force=False)` raises `ValueError`, while bulk methods skip protected nodes gracefully.
  - Tick-Ordering Guarantee: Enforces failure injection execution *before* `SwarmSimulator.step()`, guaranteeing zero kinematic motion on the tick of failure.
  - Single-Pass Fragmentation Detection: `check_fragmentation()` evaluates `is_fragmented`, `path_exists`, `num_components`, and `algebraic_connectivity` by constructing the `NetworkX` graph once to preserve NFR-1 performance.
- **Demonstration & Baseline**:
  - `scripts/demo_phase3_baseline.py` generating `demo_phase3_baseline.png` to visually demonstrate unmitigated graph fragmentation following node failure.
- **Test Suite**:
  - `tests/test_failure.py` verifying single node failure, tick-ordering zero motion, spatial outage zones, bulk failure ground node skipping, force override API, and 1-step fragmentation detection.

---

## [0.2.0] - Phase 2: Communication Graph & Disaster Scenarios - 2026-09-03

### Added
- **Graph Topology Engine**:
  - `SwarmTopologyManager` in `swarm_sim/graph/topology.py` supporting NetworkX graph construction, KD-tree neighbor pair filtering, shortest path, and single-pass metrics.
- **Disaster Relay Scenarios**:
  - `DisasterRelayScenario` in `swarm_sim/scenarios/disaster_relay.py` supporting `line_relay`, `grid_lattice`, and `random_scatter` topologies.
