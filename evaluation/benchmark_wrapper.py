"""
Benchmark Wrapper for unified evaluation across algorithms.

This wrapper ensures consistent logging and metrics regardless of the agent type.
"""

import numpy as np
from typing import Tuple, Any, Dict, Optional
import gymnasium as gym
from gymnasium import spaces

class BenchmarkWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env):
        """
        env: Base DroneDeliveryEnv enironment
        """
        super().__init__(env)
        self.episode_stats = {
            "total_reward": 0.0,
            "episode_length": 0,
            "deliveries_completed": 0,
            "total_deliveries": 0,
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "energy_consumed": 0.0,
            "energy_distance": 0.0,
            "total_distance": 0.0,
            "success": False,
            "failure_reason": None
        }
        self.step_count = 0
        self.drone_prev_pos = None
        self.client_positions = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment with deterministic seed.
        Args:
            options: Addiitonal reset options
        REturns:
            Observation and info dict
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.episode_stats = {
            "total_reward": 0.0,
            "episode_length": 0,
            "deliveries_completed": 0,
            "total_deliveries": info.get("total_deliveries", 0),
            "initial_energy": 100.0,
            "final_energy": 0.0,
            "energy_consumed": 0.0,
            "total_distance": 0.0,
            "success": False,
            "failure_reason": None
        }
        self.step_count = 0
        self.drone_prev_pos = None

        # store client positions for distance calculation
        self.client_positions = info.get("client_positions", np.array([]))

        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment and track metrics
        Return:
            Observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track staticits
        self.episode_stats["total_reward"] +=reward
        self.episode_stats["episode_length"] +=1
        self.step_count += 1

        # track deliveris
        self.episode_stats["deliveries_completed"] = info.get("deliveries_completed", 0)

        # track energy
        current_energy = info.get("energy", 0.0)
        self.episode_stats["final_energy"] = current_energy
        self.episode_stats["energy_consumed"] = 100.0 - current_energy

        # track fistance (from observation: position is first 3 elemnets)
        if len(obs) >= 3:
            current_pos = obs[:3]
            if self.drone_prev_pos is not None:
                distance = np.linalg.norm(current_pos - self.drone_prev_pos)
                self.episode_stats["total_distance"] += distance
            self.drone_prev_pos = current_pos.copy()

        # check termination conditiions
        if terminated or truncated:
            if self.episode_stats["deliveries_completed"] == self.episode_stats["total_deliveries"]:
                self.episode_stats["success"] = True
                self.episode_stats["failure_reason"] = "success"
            elif current_energy <= 0.0:
                self.episode_stats["success"] = False
                self.episode_stats["failure_reason"] = "energy_depleted"
            else:
                self.episode_stats["success"] = False
                self.episode_stats["failure_reason"] = "out_of_bounds_or_timeout"
        
        # add benchmark starts to info
        info["benchmark_stats"] = self.episode_stats.copy()

        return obs, reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        get the accumulated episode statisitcs.
        return:
            Dictionary of episode statistics
        """
        return self.episode_stats.copy()
    
    def compute_benchmark_metrics(self) -> Dict[str, float]:
        """
        Compute unified benchmark metrics from episode statistics
        Returns:
            Dicitonary of computed metrics
        """
        stats = self.episode_stats

        # success rate is binary (success or not)
        success_rate = 1.0 if stats["success"] else 0.0

        # energy efficiency: distance per unit energy
        energy_consumed = max(stats["energy_consumed"], 0.001) 
        energy_efficiency = stats["total_distance"] / energy_consumed if energy_consumed > 0 else 0.0

        # delivery efficiency: packages delivered per unit energy
        deliveries_per_energy = stats["deliveries_completed"] / energy_consumed if energy_consumed > 0 else 0.0

        # time efficiency: deliveres per timestep
        time_per_delivery = stats["episode_length"] / max(stats["deliveries_completed"], 1)

        return {
            "success": success_rate,
            "energy_efficiency": energy_efficiency,
            "deliveries_per_energy": deliveries_per_energy,
            "time_per_delivery": time_per_delivery,
            "total_reward": stats["total_reward"],
            "episode_length": stats["episode_length"],
            "energy_consumed": stats["energy_consumed"],
            "deliveries_completed": stats["deliveries_completed"],
            "total_deliveries": stats["total_deliveries"],
            "failure_reason": stats["failure_reason"]
        }
    
