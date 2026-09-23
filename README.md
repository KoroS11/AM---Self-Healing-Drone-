# Autonomous Self-Healing Drone Swarm (AM-SHDS)
### *Decentralized Multi-Agent Graph Neural Networks with Second-Order Control Barrier Function (CBF-QP) Safety Filters for Resilient Multi-Hop UAV Communications*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OSQP Solver](https://img.shields.io/badge/Solver-OSQP-00599C.svg)](https://osqp.org/)
[![Tests](https://img.shields.io/badge/Tests-53%20Passed-brightgreen.svg)]()
[![Benchmark Gate](https://img.shields.io/badge/50--Scenario%20Gate-100%25%20Sustained%20Lock-success.svg)]()
[![Full Battery](https://img.shields.io/badge/94--Scenario%20Battery-91.5%25%20Pass%20Rate-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Visual Study Overview & Trajectory Replay

![Autonomous Self-Healing Swarm Dynamic Recovery](demo_phase4_recovery.png)

*Figure 1: Autonomous dynamic gap closure and relay reorganization following catastrophic in-flight drone failure between Ground Station A and Ground Station B.*

---

## Executive Overview

**Autonomous Self-Healing Drone Swarm (AM-SHDS)** is a high-performance simulation and control framework designed to study and resolve catastrophic network fragmentation in aerial multi-hop drone relay chains operating in contested, disaster, or infrastructure-deprived environments.

When intermediate relays experience in-flight dropouts (kinetic destruction, hardware failure, battery depletion, or signal jamming), the end-to-end communication link between ground stations $A$ and $B$ breaks. AM-SHDS deploys a **Centralized Training with Decentralized Execution (CTDE)** multi-agent pipeline combining:

1. **$SO(2)$-Invariant Graph Neural Network (GNN)** local message-passing actor for gap-closing mobilization.
2. **Heterogeneity-Aware $9 \times 5$ Graph Feature Schema** resolving communication range differences ($R_c$).
3. **Lexicographic Second-Order Control Barrier Function Quadratic Program (CBF-QP)** safety filter enforcing forward-invariant inter-relay connectivity constraints while eliminating elastic post-reconnection drift.

---

## Key Technical Innovations

### 1. Second-Order Lexicographic CBF-QP Safety Filter

To resolve the fundamental tension between rapid gap closure (high acceleration) and steady-state edge stability (zero overshoot), AM-SHDS introduces a **two-phase lexicographic controller**:

- **Phase 1: Disconnected Search ($A \nleftrightarrow B$)**: The policy executes raw continuous RL acceleration outputs with zero CBF interference, preserving 100% mobilization force to close voids rapidly.
- **Phase 2: Connected Safe Operation ($A \leftrightarrow B$)**: Once end-to-end connectivity is restored, raw RL control forces $\mathbf{u}_{\mathrm{desired}}$ are filtered through a per-tick Quadratic Program (OSQP) enforcing relative acceleration barrier constraints across all live edges $(i, j) \in E(G)$:

$$
2(\mathbf{p}_i - \mathbf{p}_j)^\top (\mathbf{a}_i - \mathbf{a}_j) \le -2\Vert\mathbf{v}_i - \mathbf{v}_j\Vert^2 + \alpha_1 \dot{h}_{ij} + \alpha_2 h_{ij}
$$

where:

$$
h_{ij} = r_{\mathrm{safe}}^2 - \Vert\mathbf{p}_i - \mathbf{p}_j\Vert^2 \ge 0, \qquad r_{\mathrm{safe}} = \max\left(\min(R_{c,i}, R_{c,j}) - \epsilon,\ 1.0\text{ m}\right)
$$

![Inter-Relay Distance Dynamics and Throughput Stability](docs/images/recovery_dynamics.png)

*Figure 2: Distance invariance and Shannon sum-rate throughput stability under B-Prime + Second-Order CBF-QP (α₁ = 3.0, α₂ = 1.5, ε = 0.02 m) versus unfiltered baseline drift.*

### 2. $9 \times 5$ Heterogeneity-Aware Graph Schema

Standard spatial GNNs assume homogeneous communication radii, leading to severe edge chattering when individual drones suffer degraded transceivers ($R_c < 28\,\text{m}$). AM-SHDS incorporates exact local hardware degradation into graph message-passing:

- **Node Features (9-dim)**:

$$
p_x',\ p_y',\ v_x',\ v_y',\ \mathrm{role}_{\text{one-hot}}\ (4),\ R_{c,i} / R_{c,\mathrm{nominal}}
$$

- **Edge Features (5-dim)**:

$$
\Delta p_x',\ \Delta p_y',\ \Vert\Delta p'\Vert,\ \mathrm{LOS}_{\mathrm{flag}},\ \mathrm{slack}_{ij}
$$

$$
\mathrm{slack}_{ij} = \frac{\min(R_{c,i}, R_{c,j}) - \Vert\mathbf{p}_i - \mathbf{p}_j\Vert}{\max\left(\min(R_{c,i}, R_{c,j}), 10^{-3}\right)}
$$

### 3. $SO(2)$ Rotation-Invariant Coordinate Projection

All local observations and relative vectors are dynamically projected onto the unit corridor frame $\hat{\mathbf{u}}_{\parallel} = \frac{\mathbf{p}_B - \mathbf{p}_A}{\Vert\mathbf{p}_B - \mathbf{p}_A\Vert}$, guaranteeing 100% rotational invariance across all $360^\circ$ spatial deployment geometries.

---

## Benchmark & Verification Results

### A. 50-Scenario Benchmark Gate (300 Ticks, Seen 35 + Held-Out 15)

| Metric | Checkpoint B (Warm-Start) | B-Prime (Raw GNN) | B-Prime + CBF-QP [Production] | Delta vs Checkpoint B |
| :--- | :---: | :---: | :---: | :--- |
| **Single-Tick Success (150t)** | 48 / 50 (96.0%) | 48 / 50 (96.0%) | **50 / 50 (100.0%)** | **+4.0 pp (100% Reconnection)** |
| **Single-Tick Success (300t)** | 48 / 50 (96.0%) | 48 / 50 (96.0%) | **50 / 50 (100.0%)** | **+4.0 pp (100% Reconnection)** |
| **Sustained $K=20$ (150t)** | 41 / 50 (82.0%) | 42 / 50 (84.0%) | **50 / 50 (100.0%)** | **+18.0 pp (Zero Drift)** |
| **Sustained $K=20$ (300t)** | 41 / 50 (82.0%) | 42 / 50 (84.0%) | **50 / 50 (100.0%)** | **+18.0 pp (Zero Drift)** |
| **Mean Time-to-Reconnect** | 66.96 ticks | 61.69 ticks | **61.32 ticks** | **-5.64 ticks faster ($p = 1.25 \times 10^{-7}$)** |
| **Mean Network Sum-Rate** | 1231.02 Mbps | 1231.11 Mbps | **1216.97 Mbps** | High-throughput Shannon capacity |
| **Double-Pass Determinism** | -- | -- | **0 flips / 50 ($\max \Vert\Delta t\Vert = 0.0$)** | 100% bit-identical |

---

### B. Out-of-Distribution (OOD) Stress Test Battery (300 Ticks, 94 Scenarios Total)

![Stress Battery Performance Comparison](docs/images/stress_battery_performance.png)

*Figure 3: Sustained K = 20 pass rates across all 10 stress categories comparing baseline Checkpoint B versus B-Prime + Second-Order CBF-QP (α₁ = 3.0, α₂ = 1.5, ε = 0.02 m).*

| Stress Category | Total $N$ | Baseline Checkpoint B | Production B-Prime + CBF-QP | Performance & Mechanism Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Benchmark Gate** | **50** | 41 / 50 (82.0%) | **50 / 50 (100.0%)** | 100% sustained lock; zero drift over 300 ticks. |
| **1. Simultaneous Double Failure** | 4 | 2 / 4 (50.0%) | **2 / 4 (50.0%)** | 100% on geometrically feasible configs (21t, 51t). |
| **2. Cascading Sequential Failure** | 4 | 2 / 4 (50.0%) | **2 / 4 (50.0%)** | 100% on geometrically feasible configs (47t, 50t). |
| **3. Large Swarm Scale ($N=15, 21$)** | 4 | 1 / 4 (25.0%) | **3 / 4 (75.0%)** | Both $N=15$ solved (66t, 101t); $N=21$ center solved (75t). |
| **4. Non-Collinear Corridor ($SO(2)$)** | **8** | 8 / 8 (100.0%) | **8 / 8 (100.0%)** | **100% solved across all 8 angles** ($17^\circ \dots 200^\circ$). |
| **5. Failure Timing Extremes** | 4 | 1 / 4 (25.0%) | **2 / 4 (50.0%)** | Early failure healed in 21t; late failure @ $t=140$ healed in 24t. |
| **6. Tight-Margin Geometry** | 4 | 2 / 4 (50.0%) | **4 / 4 (100.0%)** | Solves $1.57\,\text{m}, 1.00\,\text{m}, 0.57\,\text{m}, 0.14\,\text{m}$ margins with zero drops. |
| **7. Triple Simultaneous Failure** | 4 | 3 / 4 (75.0%) | **4 / 4 (100.0%)** | **100% sustained lock** (18t, 53t, 71t, 128t). |
| **8. Adversarial Cut-Vertex** | 4 | 2 / 4 (50.0%) | **3 / 4 (75.0%)** | 100% on feasible cut-vertices with zero transient drops. |
| **9. Compound Multi-Axis Stressors**| 4 | 3 / 4 (75.0%) | **4 / 4 (100.0%)** | **100% sustained lock** across all multi-stressor scenarios. |
| **11. Comm-Range Heterogeneity** | 4 | 1 / 4 (25.0%) | **4 / 4 (100.0%)** | **100% solved** with exact $R_c$-aware local slack buffers. |
| **Total Full-Battery Passed** | **94** | **66 / 94 (70.2%)** | **86 / 94 (91.5%)** | **+20 additional stress scenarios solved** |

---

## Topology Formations & Kinematic States

| Linear Multi-Hop Relay | 2D Grid Lattice Topology | Random Geographic Scatter |
| :---: | :---: | :---: |
| ![Line Relay](demo_line_relay.png) | ![Grid Lattice](demo_grid_lattice.png) | ![Random Scatter](demo_random_scatter.png) |
| *Collinear Ground $A \leftrightarrow B$ Chain* | *2D Meshed Relay Lattice* | *Scattered Initial Node Placements* |

---

## Repository Structure

```tree
SWARM/
├── checkpoints/                        # Model weights & test bank datasets
│   ├── checkpoint_A_best_k2.pt         # Checkpoint A (Imitation Learning DAgger model)
│   ├── checkpoint_B_best.pt            # Checkpoint B (PPO RL model, iteration 15)
│   ├── checkpoint_B_prime_best.pt      # Production Checkpoint B-Prime (9x5 schema)
│   ├── test_bank_50.pkl                # Standard 50-scenario benchmark dataset
│   └── test_bank_spare.pkl             # 10-seed spare evaluation dataset
├── docs/images/                        # High-resolution research study figures
│   ├── recovery_dynamics.png           # Inter-relay distance & Shannon rate stability
│   └── stress_battery_performance.png  # OOD stress testing performance breakdown
├── swarm_sim/                          # Core simulation package
│   ├── core/                           # Kinematics, agent state, and failure injection
│   │   ├── agent.py                    # UAV kinematic state representation
│   │   ├── failure.py                  # Stochastic & deterministic failure injector
│   │   └── simulator.py                # 2D swarm physics simulator
│   ├── graph/                          # Topology & radio channel propagation
│   │   ├── channel.py                  # Shakhatreh et al. (2021) A2G path loss & Shannon rate
│   │   └── topology.py                 # Spatial cKDTree & NetworkX graph analyzer
│   ├── policies/                       # Decentralized control policies
│   │   ├── base.py                     # BasePolicy abstract interface
│   │   ├── gnn_actor.py                # 2-hop spatial message-passing actor
│   │   ├── gnn_critic.py               # Centralized team-value critic V(s)
│   │   ├── gnn_policy.py               # GNNPolicy wrapper with corridor coordinate framing
│   │   ├── greedy_dara.py              # Second-Order CBF-QP safety filter
│   │   └── heuristic_dara.py           # Baseline Heuristic DARA spring-damper policy
│   ├── scenarios/                      # Scenario generators
│   │   ├── disaster_relay.py           # Ground-to-ground disaster relay environment
│   │   └── test_bank.py                # Benchmark test bank loader & generator
│   ├── utils/                          # Configuration, enums, and logging
│   │   ├── config.py                   # ExperimentConfig dataclass
│   │   ├── enums.py                    # AgentRole, AgentStatus, FailureType
│   │   └── logger.py                   # Two-tier Pandas telemetry recorder
│   └── viz/                            # Visualization & animation
│       ├── animator.py                 # 2D Matplotlib frame renderer
│       └── exporter.py                 # Pillow animated GIF exporter
├── tests/                              # Automated test suite (53 tests)
│   ├── test_agent.py                   # Agent kinematics & limits
│   ├── test_channel.py                 # Channel propagation & SNR
│   ├── test_config.py                  # Configuration parameter bounds
│   ├── test_disaster_relay.py          # Disaster scenario mechanics
│   ├── test_enums.py                   # Enum invariants
│   ├── test_failure.py                 # Dropout mechanics & cut-vertices
│   ├── test_gnn_policy.py              # GNN message passing & tensor dimensions
│   ├── test_greedy_dara.py             # CBF-QP forward invariance tests
│   ├── test_heterogeneous_features.py  # 9x5 feature schema & t=0 mathematical check
│   ├── test_logger.py                  # Telemetry DataFrame schema validation
│   ├── test_pipeline.py                # Training pipeline & DAgger tests
│   ├── test_recovery.py                # End-to-end recovery verification
│   ├── test_rng_isolation.py           # Seed isolation & reproducibility
│   ├── test_simulator.py               # Physics integration & clamping
│   ├── test_topology.py                # Graph connectivity & shortest paths
│   └── test_viz.py                     # Matplotlib animation & GIF export tests
├── scripts/                            # Visual demo scripts & figure generators
│   ├── demo_phase2.py                  # Phase 2 kinematic lattice demo
│   ├── demo_phase3_baseline.py         # Phase 3 DARA heuristic baseline demo
│   ├── demo_phase4_recovery.py         # Phase 4 recovery replay demo
│   └── generate_study_figures.py       # High-resolution figure renderer
├── train_pipeline.py                   # Checkpoints A/B/B-Prime training pipeline
├── run_experiment.py                   # CLI experiment runner
├── CHANGELOG.md                        # Complete chronological release history
├── walkthrough.md                      # Comprehensive verification audit documentation
└── pyproject.toml / uv.lock            # Python dependencies & packaging
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python `3.10` or higher
- Recommended: [`uv`](https://github.com/astral-sh/uv) package manager (for ultra-fast reproducible environments)

### 2. Clone & Install
```bash
git clone https://github.com/KoroS11/AM---Self-Healing-Drone-.git
cd AM---Self-Healing-Drone-

# Using uv (recommended)
uv sync

# Or using standard pip
pip install -e .
```

### 3. Run Automated Tests
```bash
# Execute all 53 unit & integration tests
pytest tests/ -v
```

---

## CLI Usage

### Run an Interactive Experiment
```bash
# Run a single evaluation seed with GIF export
python run_experiment.py --seed 1000 --strategy gnn --export-gif

# Run with baseline Heuristic DARA
python run_experiment.py --seed 1000 --strategy heuristic --export-gif
```

### Run Full Benchmark Gate Evaluation
```bash
# Evaluate Production B-Prime + CBF-QP across 50 benchmark scenarios
python -m scratch.run_production_verification_battery
```

### Run Full 94-Scenario Stress Battery
```bash
# Evaluate all 94 benchmark & stress scenarios over 300 full ticks
python -m scratch.verify_eps002_determinism_significance
```

---

## Mathematical Formulation

### 1. Kinematics & Physics Model

Each drone $i \in \{1, \dots, N\}$ obeys discrete double-integrator kinematics:

$$
\begin{aligned}
\mathbf{p}_i(t + \Delta t) &= \mathbf{p}_i(t) + \mathbf{v}_i(t) \Delta t + \frac{1}{2} \mathbf{a}_i(t) \Delta t^2 \\
\mathbf{v}_i(t + \Delta t) &= \text{clamp}\left(\mathbf{v}_i(t) + \mathbf{a}_i(t) \Delta t, -v_{\max}, v_{\max}\right)
\end{aligned}
$$

$$
\Vert\mathbf{a}_i(t)\Vert \le a_{\max} = 5.0\,\text{m/s}^2, \quad \Vert\mathbf{v}_i(t)\Vert \le v_{\max} = 10.0\,\text{m/s}
$$

### 2. Second-Order Control Barrier Function Formulation

For relative distance $d_{ij} = \Vert\mathbf{p}_i - \mathbf{p}_j\Vert$, the safety barrier function $h_{ij}(\mathbf{p})$ and its time derivatives are defined as:

$$
\begin{aligned}
h_{ij}(\mathbf{p}) &= r_{\mathrm{safe}}^2 - \Vert\mathbf{p}_i - \mathbf{p}_j\Vert^2 \\
\dot{h}_{ij} &= -2 (\mathbf{p}_i - \mathbf{p}_j)^\top (\mathbf{v}_i - \mathbf{v}_j) \\
\ddot{h}_{ij} &= -2 \Vert\mathbf{v}_i - \mathbf{v}_j\Vert^2 - 2 (\mathbf{p}_i - \mathbf{p}_j)^\top (\mathbf{a}_i - \mathbf{a}_j)
\end{aligned}
$$

The second-order CBF forward invariance condition $\ddot{h}_{ij} + \alpha_1 \dot{h}_{ij} + \alpha_2 h_{ij} \ge 0$ yields the affine linear inequality constraint on control accelerations:

$$
2(\mathbf{p}_i - \mathbf{p}_j)^\top (\mathbf{a}_i - \mathbf{a}_j) \le -2\Vert\mathbf{v}_i - \mathbf{v}_j\Vert^2 + \alpha_1 \dot{h}_{ij} + \alpha_2 h_{ij}
$$

**Production Parameterization**:
- Velocity damping gain: $\alpha_1 = 3.0$
- Position stiffness gain: $\alpha_2 = 1.5$
- Barrier margin buffer: $\epsilon = 0.02\,\text{m}$ ($2\,\text{cm}$ discrete Euler integration guard)

### 3. Air-to-Ground (A2G) Radio Channel Model

Following *Shakhatreh et al. (2021)*, the Line-of-Sight probability $P_{\mathrm{LOS}}$ as a function of elevation angle $\theta_{ij}$ is:

$$
P_{\mathrm{LOS}}(\theta_{ij}) = \frac{1}{1 + a \exp\left(-b (\theta_{ij} - a)\right)}
$$

Path loss $\mathrm{PL}_{ij}$ and Shannon capacity achievable rate $C_{ij}$ are given by:

$$
\mathrm{PL}_{ij} = 20 \log_{10}(d_{ij}) + 20 \log_{10}(f_c) + 20 \log_{10}\left(\frac{4\pi}{c}\right) + P_{\mathrm{LOS}} \eta_{\mathrm{LOS}} + (1 - P_{\mathrm{LOS}}) \eta_{\mathrm{NLOS}}
$$

$$
C_{ij} = B \log_2 \left(1 + \frac{P_t \cdot 10^{-\mathrm{PL}_{ij}/10}}{N_0 B}\right) \quad \text{[Mbps]}
$$

---

## Authors & Citation

Developed by **Harsh Jain** ([@KoroS11](https://github.com/KoroS11)).

If you use this simulator or benchmark in your research, please cite:

```bibtex
@software{am_self_healing_drone_2026,
  author = {Harsh Jain},
  title = {AM-SHDS: Decentralized Multi-Agent GNN with Second-Order CBF-QP for Resilient UAV Swarm Communications},
  year = {2026},
  url = {https://github.com/KoroS11/AM---Self-Healing-Drone-}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
