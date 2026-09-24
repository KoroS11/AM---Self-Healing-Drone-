from typing import TYPE_CHECKING
import numpy as np
from swarm_sim.utils.enums import AgentRole, AgentStatus

if TYPE_CHECKING:
    from swarm_sim.core.simulator import SwarmSimulator

class UAVAgent:
    """Read-only proxy view over a single row of SwarmSimulator's state matrices.
    
    Provided for inspection, assertions in tests, and debugging.
    SwarmSimulator.step() operates directly on NumPy state matrices, NOT UAVAgent instances.
    """
    def __init__(self, simulator: "SwarmSimulator", agent_id: int):
        self._sim = simulator
        self.agent_id = agent_id

    @property
    def position(self) -> np.ndarray:
        return self._sim.positions[self.agent_id].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._sim.velocities[self.agent_id].copy()

    @property
    def role(self) -> AgentRole:
        return self._sim.roles[self.agent_id]

    @property
    def status(self) -> AgentStatus:
        return self._sim.statuses[self.agent_id]

    @property
    def comm_range(self) -> float:
        return float(self._sim.comm_ranges[self.agent_id])

    @property
    def max_speed(self) -> float:
        return float(self._sim.max_speeds[self.agent_id])

    @property
    def max_accel(self) -> float:
        return float(self._sim.max_accels[self.agent_id])

    def __repr__(self) -> str:
        return (
            f"<UAVAgent id={self.agent_id} "
            f"role={self.role.value} "
            f"status={self.status.value} "
            f"pos={self.position.tolist()}>"
        )
