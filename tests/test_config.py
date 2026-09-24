from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.utils.enums import RecoveryStrategy

def test_experiment_config_defaults():
    config = ExperimentConfig()
    assert config.num_drones == 20
    assert config.dt == 0.1
    assert config.comm_range == 50.0
    assert config.sensing_range == 50.0
    assert config.max_speed == 10.0
    assert config.max_accel == 5.0
    assert config.bounds is None
    assert config.seed == 42
    assert config.recovery_strategy == RecoveryStrategy.NONE
    assert config.failure_timestep is None
    assert config.failure_count == 0

def test_experiment_config_custom():
    config = ExperimentConfig(
        num_drones=50,
        dt=0.05,
        comm_range=30.0,
        bounds=(0.0, 100.0, 0.0, 100.0),
        seed=123,
        recovery_strategy=RecoveryStrategy.HEURISTIC_DARA
    )
    assert config.num_drones == 50
    assert config.dt == 0.05
    assert config.comm_range == 30.0
    assert config.bounds == (0.0, 100.0, 0.0, 100.0)
    assert config.seed == 123
    assert config.recovery_strategy == RecoveryStrategy.HEURISTIC_DARA
