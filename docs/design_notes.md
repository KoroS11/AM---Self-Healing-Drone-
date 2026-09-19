# SWARM Design Notes & Architectural Decisions Log

This document records key technical decisions, architectural rationale, mathematical formulations, and empirical verification benchmarks established across project phases.

---

## Phase 1: Environment Setup & Core Kinematic Engine

### 1. Vectorized NumPy State Storage vs. Object-Oriented Agent Instances
- **Decision**: Store agent positions, velocities, roles, and statuses in contiguous 2D/1D NumPy arrays inside `SwarmSimulator`, exposed via a read-only `UAVAgent` lightweight proxy.
- **Rationale**: 
  - **Memory & Cache Locality**: Vectorized NumPy operations achieve $>1000$ Hz update loops for $N=100$ agents, whereas iterating Python object lists drops below 50 Hz.
  - **Encapsulation Protection**: `UAVAgent` properties return scalar or array copies to prevent external code from mutating internal simulation state without going through `SwarmSimulator.step()`.

### 2. KD-Tree Spatial Indexing & Re-indexing Policy
- **Decision**: Rebuild SciPy `cKDTree` dynamically inside `get_neighbor_pairs()` whenever spatial queries are executed.
- **Rationale**:
  - Spatial tree rebuild time for $N=100$ 2D points is sub-millisecond ($<0.1$ ms). Storing KD-tree instances avoids stale spatial queries after state updates while guaranteeing $O(N \log N)$ neighbor search bounds.

---

## Phase 2: Communication Graph & Disaster Scenario Environment

### 1. Topology Metric Single-Pass Performance Design
- **Decision**: Redesigned `get_network_metrics()` to construct `G = build_graph()` once and pass it directly into sub-metric calculations (`has_path`, `get_num_connected_components`, `get_algebraic_connectivity`).
- **Rationale**:
  - Eliminates 3x redundant NetworkX graph reconstructions per metric check, preserving NFR-1 performance ($>20$ Hz) during simulation ticks.

### 2. Disconnected Graph Algebraic Connectivity ($\lambda_2$)
- **Decision**: `get_algebraic_connectivity()` returns `0.0` explicitly when NetworkX graph is disconnected (`nx.is_connected(G) == False`).
- **Rationale**:
  - `nx.algebraic_connectivity()` raises `NetworkXError` on disconnected graphs. Catching this and returning `0.0` matches spectral graph theory semantics ($\lambda_2 = 0 \iff$ graph is disconnected) and avoids try/except bloat in caller code.

---

## Phase 3: Failure Injector & Fragmentation Detection

### 1. Pre-Kinematics Tick-Ordering Guarantee
- **Decision**: Scheduled failure injection in `FailureInjector.step()` executes *before* `SwarmSimulator.step()`.
- **Rationale**:
  - Guarantees zero kinematic motion on the exact tick of failure injection, matching physical failure events.

### 2. Differentiated Ground Endpoint Protection API
- **Decision**: Singular `fail_agent(id, force=False)` raises a `ValueError` if `id` is a ground endpoint and `force=False`. Bulk methods (`fail_agents`, `fail_spatial_zone`) skip protected ground endpoints gracefully when `force=False` and return `list[int]` of agent IDs actually failed.
- **Rationale**:
  - Explicit single calls (`fail_agent`) fail fast on invalid arguments to catch hardcoded mistakes, while batch operations skip static endpoints cleanly without throwing mid-batch partial failures.

---

## Phase 4: Heuristic Recovery Policy & Self-Healing Chain Repair

### 1. 100% Live Graph Perception Architecture (`G.neighbors`)
- **Decision**: Implemented 100% graph-based local perception recovery using `_get_chain_neighbors(r_idx, G, active_set)` driven strictly by `G.neighbors(r_idx)`.
- **Scope of Dependency & Zero Global State**:
  - **Primary Bridging Nodes**: Relays flanking a failure gap target $\mathbf{p}_f$ directly (local memory of a formerly connected neighbor's last position) only while degree $< 2$ in $G$.
  - **Surviving Relays**: All degree $\ge 2$ relays target the midpoint of their two graph neighbors ($\frac{1}{2}(\mathbf{p}_{\text{left}} + \mathbf{p}_{\text{right}})$).
  - **Zero Global State**: $0\%$ of active relays reference global corridor projections ($\mathbf{p}_A + \frac{\text{rank}+1}{M+1} \mathbf{u}_{AB}$), 1D projection sorts (`projections`, `sorted_order`), or $x$-coordinates (`pos[:, 0]`).

### 2. Overlapping Cascade Single-Pull Guard & Velocity Damping
- **Decision**: `pulled_this_tick: Dict[int, bool] = {}` prevents double-counted repair forces when multiple failure cascades reach the same relay node. Active mobile velocities are zeroed (`self.sim.velocities[active_mask] = 0.0`) as soon as global end-to-end connectivity is restored (`repair_needed == False`).
- **Rationale**:
  - **Single Pull Guard**: Prevents acceleration magnitude spikes ($\le a_{\max}$) on shared relays during concurrent multi-node failures. Verified via pre-clamp raw acceleration inspection ($5.0000$ m/s$^2$).
  - **Equilibrium Locking**: Zeroing velocity upon repair completion locks the swarm in its reconnected equilibrium, preventing micro-drift past $R_c$.

### 3. 100% Vectorized $O(N^2)$ Collision Avoidance & Branchless Clamping
- **Decision**: Implemented pairwise displacement tensor $\mathbf{D}_{\text{diff}} = \mathbf{p}[:, \text{None}, :] - \mathbf{p}[\text{None}, :, :]$ and masked matrix vector sums for collision avoidance, coupled with `np.maximum(acc_mags, 1e-9)` acceleration clamping.
- **Rationale**:
  - **NFR-1 Performance Protection**: Nested Python loops over agent pairs for $R_{\text{safe}}$ repulsion drop execution speeds below 20 Hz for $N \ge 100$. Vectorizing pairwise distance calculations in NumPy maintains SIMD execution speeds ($>20$ Hz).
  - **0-D Array Squeeze Prevention**: Using `np.maximum` for acceleration clamping prevents 0-d scalar array indexing crashes when exactly $M=1$ active mobile node requires clamping.

---

## Phase 5: Baseline Heuristic, Control Barrier Functions & Lexicographic Controller

### 1. Architectural Evolution of Greedy DARA & Audit Trail

| Evolution Step | Architecture | Reconnection (35 Seeds) | Throughput vs DARA | Status & Post-Mortem |
| :--- | :--- | :---: | :---: | :--- |
| **v1: Additive Blend** | $\mathbf{u}_{\text{DARA}} + k_{\text{tp}} \nabla \text{SNR}$ | 7 / 35 (20.0%) | N/A | **Defective**: SNR gradient force directly opposed and canceled DARA gap-bridging pull. |
| **v2: Null-Space Projection** | $\mathbf{u}_{\text{DARA}} + k_{\text{tp}} \mathbf{s}_{\text{orth}}$ | 7 / 35 (20.0%) | N/A | **Defective**: Lateral force rotation $\theta = \arctan(\|\mathbf{F}_{\text{snr}}\|/\|\mathbf{F}_{\text{DARA}}\|)$ deflected drones sideways, breaking chain laterally. |
| **v3: Geometric Slack Clamp** | Clamp $\|\mathbf{F}_{\text{snr}}\| \le 0.8 \cdot \text{slack}$ | 7 / 35 (20.0%) | N/A | **Defective**: Scale mismatch ($\|\mathbf{F}_{\text{snr}}\| \le 1.0 \le 10.08$) resulted in 0 triggers (0.00% binding), leaving lateral drift unmitigated. |
| **v4: Pure CBF-QP** | $\min \frac{1}{2}\|\mathbf{u} - \mathbf{u}_{\text{des}}\|^2$ s.t. 2nd-Order CBF | 5 / 35 (14.3%) | *+12.04% (5-seed subset)* | **Defective (Partition Trap)**: CBF enforces $h_{ij} \ge 0$ only on active edges; disconnected clusters optimized local SNR peaks instead of crossing the gap. |
| **v5: Lexicographic CBF-QP (FINAL)** | Disconnected $\to$ Pure DARA; Connected $\to$ CBF-QP Polishing | **35 / 35 (100.0%)** | **+0.06% ($p=1.2\times 10^{-7}$)** | **VERIFIED & ADOPTED**: 100% reconnection + guaranteed safety + statistically significant throughput gain. |

### 2. Explicit Retirement of Intermediate Figures
- **`+12.04%` Figure RETIRED**: Erroneously calculated over a cherry-picked 5-seed subset where non-lexicographic CBF reconnected by chance, rather than the full 35-seed test bank.
- **`+0.03%` Figure RETIRED**: Erroneously calculated from unconfirmed clamp engagement where the geometric slack clamp never actually triggered due to scale/unit mismatch.

### 3. Final Verified Performance Benchmarks (`seeds 1000–1034`, $N=35$)
- **Reconnection Success**: **35 / 35 (100.0%)** (identical to DARA baseline).
- **Mean Time-to-Reconnect**: **75.89 ticks** (vs DARA baseline 78.66 ticks).
- **Mean Network Sum-Rate**: **1214.35 Mbps** (vs DARA baseline **1213.66 Mbps**).
- **Net Throughput Delta ($\Delta$)**: **`+0.688092 Mbps` (`+0.0563%`)**, strictly positive on all 35 seeds ($35/35 > 0$).
- **Statistical Significance**:
  - Paired Wilcoxon Signed-Rank Test: $W = 630.0$ (max possible rank sum for $N=35$), **$p = 1.2305 \times 10^{-7} \ll 0.001$**.
  - Paired $t$-test: $t = 12.3114, p = 4.4135 \times 10^{-14}$.
- **Zero-Effect Control ($k_{\text{tp}} = 0.0$)**:
  - Trajectory is bit-identical to DARA baseline on all 35 seeds ($\max |\Delta| = 0.00000000$ Mbps).

---

## Standard Four-Check Verification Protocol (Mandatory for Checkpoints A/B/C)

To prevent unconfirmed heuristic artifacts, non-deterministic drift, or subset cherry-picking during RL training, all evaluation runs must execute this 4-step protocol:

1. **Baseline Invariant Audit**:
   - Verify that the reference baseline (e.g. Heuristic DARA) produces exact, invariant numbers across the full evaluation suite, independent of run length ($T=150$ vs $T=300$).
2. **Determinism Rerun (Bit-Identical Check)**:
   - Run two consecutive identical evaluation passes. Assert that all state variables, reconnection times, and sum-rate metrics are 100% bit-identical ($\max |\Delta| = 0.0$).
3. **Raw Per-Seed Distribution & Non-Parametric Significance Testing**:
   - Report raw per-seed tables for all test seeds.
   - Run paired Wilcoxon signed-rank tests ($p < 0.001$) to prove that claimed gains are statistically distinguishable from zero.
4. **Zero-Effect / Ablation Control**:
   - Run a zero-effect control (e.g., $k_{\text{tp}} = 0$ or frozen baseline weights) and assert that the wrapped pipeline reproduces the unperturbed baseline down to machine precision.
