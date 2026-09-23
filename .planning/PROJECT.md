# SWARM: Decentralized UAV Swarm Communication Resilience Simulator

## Overview
A lightweight Python simulator designed to study communication-network resilience in UAV swarms operating in disaster zones. The system simulates a relay chain between two ground endpoints (Endpoints A and B), detects network fragmentation upon node failure, and executes a heuristic reorganization policy to restore end-to-end connectivity. The architecture is designed with modular hooks to support learned (GNN/RL) policies in later phases.

## Objectives
1. **Kinematic Swarm Simulation**: Simulate 2D/3D UAV dynamics using lightweight kinematic equations without heavy physics engines.
2. **Dynamic Graph Topology**: Continuously maintain communication graphs using spatial neighbor queries (`cKDTree`) and analyze topology via `NetworkX` (algebraic connectivity $\lambda_2$, shortest paths, graph partitioning).
3. **Disaster-Zone Relay Scenario**: Model a relay chain spanning between two fixed/mobile ground endpoints (e.g., base station to search-and-rescue unit).
4. **Failure & Fragmentation Detection**: Inject simulated node failures (battery depletion, destruction, signal loss) and detect network disconnection in real time.
5. **Heuristic Reorganization Policy**: Implement distributed potential-field / spring-damper / relay repositioning algorithms to repair graph gaps autonomously.
6. **Telemetry & Visual Export**: Log per-step metrics (tidy Pandas DataFrames) and render Matplotlib animations with GIF export capabilities via Pillow.

## Tech Stack
- **Language**: Python 3.10+
- **Package Management**: `uv`
- **State & Kinematics**: `NumPy`
- **Spatial Indexing & Queries**: `SciPy` (`scipy.spatial.cKDTree`)
- **Network & Topology Analysis**: `NetworkX`
- **Visualization & Export**: `Matplotlib` + `Pillow` (GIF export)
- **Data & Telemetry**: `Pandas`

## Core Architecture Principles
- **Decentralized Perception**: Each UAV acts on local neighbor information bounded by communication range $R_c$.
- **Kinematic Focus**: Simple, fast 2D/3D position and velocity updates allowing fast simulation runs ($>20$ Hz for 100+ agents).
- **Extensible Policy Engine**: Abstract `BasePolicy` interface allowing seamless swapping between Heuristic (Phase 3) and GNN/RL (Phase 5).
- **Reproducible Experiments**: Random seeds for scenario generation, agent initialization, and failure injection.
