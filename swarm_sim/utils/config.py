from dataclasses import dataclass
from typing import Optional, Tuple
from swarm_sim.utils.enums import RecoveryStrategy

@dataclass
class ExperimentConfig:
    """Central configuration object for simulation runs and parameter sweeps."""
    num_drones: int = 20
    dt: float = 0.1
    comm_range: float = 50.0
    sensing_range: float = 50.0
    max_speed: float = 10.0
    max_accel: float = 5.0
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, max_x, min_y, max_y)
    seed: Optional[int] = 42
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE
    failure_timestep: Optional[int] = None
    failure_count: int = 0
