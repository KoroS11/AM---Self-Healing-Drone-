from typing import List, Dict, Any, Tuple
import os
import pickle
import numpy as np
from swarm_sim.utils.config import ExperimentConfig
from swarm_sim.scenarios.disaster_relay import DisasterRelayScenario

class TestBankGenerator:
    """Generates and manages fixed N=50 scenario test bank (35 seen, 15 held-out + 10 spare seeds)."""
    def __init__(
        self,
        seen_seeds: List[int] = list(range(1000, 1035)),
        held_out_seeds: List[int] = list(range(1035, 1050)),
        spare_seeds: List[int] = list(range(1050, 1060))
    ):
        self.seen_seeds = seen_seeds
        self.held_out_seeds = held_out_seeds
        self.spare_seeds = spare_seeds

    def generate_scenario_config(self, seed: int) -> Dict[str, Any]:
        """Generate a reproducible scenario parameter dictionary given a random seed."""
        np.random.seed(seed)
        
        # Vary num_drones in range [7, 11]
        num_drones = int(np.random.choice([7, 9, 11]))
        comm_range = 28.0
        
        # Random failure agent choice among relays (relay IDs are 2..num_drones-1)
        fail_agent_id = int(np.random.choice(list(range(2, num_drones))))
        fail_timestep = int(np.random.choice([10, 15, 20]))
        
        # Scale endpoint B position to guarantee initial connectivity based on num_drones
        endpoint_b_pos = ((num_drones - 1) * 22.4, 0.0)
        
        return {
            "seed": seed,
            "num_drones": num_drones,
            "comm_range": comm_range,
            "endpoint_a_pos": (0.0, 0.0),
            "endpoint_b_pos": endpoint_b_pos,
            "fail_agent_id": fail_agent_id,
            "fail_timestep": fail_timestep
        }

    def generate_bank(self) -> Dict[str, Any]:
        """Generate complete 50-scenario dataset + 10 spare dataset."""
        seen_configs = [self.generate_scenario_config(s) for s in self.seen_seeds]
        held_out_configs = [self.generate_scenario_config(s) for s in self.held_out_seeds]
        spare_configs = [self.generate_scenario_config(s) for s in self.spare_seeds]
        
        return {
            "seen_35": seen_configs,
            "held_out_15": held_out_configs,
            "all_50": seen_configs + held_out_configs,
            "spare_10": spare_configs
        }

    def save_bank(self, bank_filepath: str, spare_filepath: str) -> None:
        """Save test bank datasets to pickle files."""
        bank_data = self.generate_bank()
        
        os.makedirs(os.path.dirname(os.path.abspath(bank_filepath)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(spare_filepath)), exist_ok=True)
        
        with open(bank_filepath, "wb") as f:
            pickle.dump({"seen_35": bank_data["seen_35"], "held_out_15": bank_data["held_out_15"], "all_50": bank_data["all_50"]}, f)
            
        with open(spare_filepath, "wb") as f:
            pickle.dump({"spare_10": bank_data["spare_10"]}, f)

    @staticmethod
    def load_bank(bank_filepath: str) -> Dict[str, Any]:
        """Load test bank dataset from pickle file."""
        with open(bank_filepath, "rb") as f:
            return pickle.load(f)
