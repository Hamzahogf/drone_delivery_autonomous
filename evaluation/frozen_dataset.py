"""
Frozen evaluation dataset generator for consistent benchmarking.

This module creates and manages a fixed set of evaluation scenarios
to ensure all algorithms are tested on identical delivery problems.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class EvaluationScenario:
    """Represents a single evaluation scenario with fixed client positions."""
    
    def __init__(
        self,
        scenario_id: int,
        num_clients: int,
        client_positions: np.ndarray,
        initial_energy: float = 100.0,
        seed: int = 42,
        wind_profile: Optional[Dict[str, Any]] = None,
        wind_on: bool = False,
        obstacle_positions: Optional[np.ndarray] = None,
        obstacle_radii: Optional[np.ndarray] = None,
    ):
        """
        Initialize an evaluation scenario.
        
        Args:
            scenario_id: Unique identifier for this scenario
            num_clients: Number of delivery clients
            client_positions: Fixed client positions (num_clients x 3)
            initial_energy: Starting energy budget
            seed: Random seed used to generate this scenario
            wind_profile: Optional wind parameters (volatility/mean_reversion/max_speed).
            wind_on: Whether wind is active for this scenario.
            obstacle_positions: Fixed sphere-obstacle centres (K x 3), K may be 0.
            obstacle_radii: Fixed sphere-obstacle radii (K,).
        """
        self.scenario_id = scenario_id
        self.num_clients = num_clients
        self.client_positions = client_positions.astype(np.float32)
        self.initial_energy = float(initial_energy)
        self.seed = int(seed)
        self.wind_profile = dict(wind_profile) if wind_profile is not None else None
        self.wind_on = bool(wind_on)
        self.obstacle_positions = (
            np.zeros((0, 3), dtype=np.float32) if obstacle_positions is None
            else np.array(obstacle_positions, dtype=np.float32)
        )
        self.obstacle_radii = (
            np.zeros((0,), dtype=np.float32) if obstacle_radii is None
            else np.array(obstacle_radii, dtype=np.float32)
        )

    @property
    def obstacles_on(self) -> bool:
        return len(self.obstacle_positions) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for JSON serialization."""
        data = {
            "scenario_id": self.scenario_id,
            "num_clients": self.num_clients,
            "client_positions": self.client_positions.tolist(),
            "initial_energy": self.initial_energy,
            "seed": self.seed,
            "wind_on": self.wind_on,
            "obstacle_positions": self.obstacle_positions.tolist(),
            "obstacle_radii": self.obstacle_radii.tolist(),
        }
        if self.wind_profile is not None:
            data["wind_profile"] = self.wind_profile
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationScenario":
        """Create scenario from dictionary."""
        return cls(
            scenario_id=data["scenario_id"],
            num_clients=data["num_clients"],
            client_positions=np.array(data["client_positions"]),
            initial_energy=data["initial_energy"],
            seed=data["seed"],
            wind_profile=data.get("wind_profile"),
            wind_on=data.get("wind_on", False),
            obstacle_positions=data.get("obstacle_positions"),
            obstacle_radii=data.get("obstacle_radii"),
        )


class EvaluationDataset:
    """Manages a collection of frozen evaluation scenarios."""
    
    def __init__(self, scenarios: Optional[List[EvaluationScenario]] = None):
        """
        Initialize the evaluation dataset.
        
        Args:
            scenarios: List of evaluation scenarios
        """
        self.scenarios = scenarios or []
    
    @staticmethod
    def _sample_obstacles(
        rng: np.random.RandomState,
        k: int,
        arena_size: float,
        altitude: float,
        exclude_positions: np.ndarray,
        radius_min: float = 0.3,
        radius_max: float = 0.8,
        min_clearance: float = 1.0,
        max_attempts: int = 200,
    ):
        """Self-contained rejection sampler for K sphere obstacles (mirrors
        drone_dynamics.sample_obstacles; duplicated here so frozen_dataset.py
        has no dependency on the env package)."""
        half = arena_size / 2.0
        positions, radii = [], []
        for _ in range(k):
            candidate, r = None, radius_min
            for _attempt in range(max_attempts):
                candidate = np.array([
                    rng.uniform(-half, half),
                    rng.uniform(-half, half),
                    altitude,
                ], dtype=np.float32)
                r = float(rng.uniform(radius_min, radius_max))
                ok = all(
                    np.linalg.norm(candidate - ex) >= (min_clearance + r)
                    for ex in exclude_positions
                ) and all(
                    np.linalg.norm(candidate - p) >= (r + pr + 0.3)
                    for p, pr in zip(positions, radii)
                )
                if ok:
                    break
            positions.append(candidate)
            radii.append(r)
        if k == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.array(positions, dtype=np.float32), np.array(radii, dtype=np.float32)

    @staticmethod
    def generate(
        num_scenarios: int = 500,
        num_clients_range: tuple = (3, 8),
        arena_size: float = 10.0,
        delivery_altitude: float = 1.0,
        base_seed: int = 42,
        obstacles_k_range: tuple = (2, 5),
        wind_profile: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationDataset":
        """
        Generate a new frozen evaluation dataset.

        Scenarios cycle evenly through all four domain-randomization
        combinations — neither / wind-only / obstacles-only / both — so
        each is equally represented across the dataset.

        Args:
            num_scenarios: Number of scenarios to generate (default 500)
            num_clients_range: (min, max) number of clients per scenario
            arena_size: Size of the arena
            delivery_altitude: Altitude of delivery positions
            base_seed: Base random seed
            obstacles_k_range: (min_k, max_k) sphere obstacles when obstacles are on
            wind_profile: Wind params (volatility/mean_reversion/max_speed) to
                          attach when wind is on for a scenario

        Returns:
            EvaluationDataset with generated scenarios
        """
        scenarios = []
        half_arena = arena_size / 2.0
        default_wind_profile = wind_profile or {
            "volatility": 0.5, "mean_reversion": 0.1, "max_speed": 3.0,
        }
        # The 4 combinations, cycled evenly across the dataset.
        combos = [
            (False, False), (True, False), (False, True), (True, True),
        ]

        for i in range(num_scenarios):
            scenario_seed = base_seed + i
            rng = np.random.RandomState(scenario_seed)

            num_clients = rng.randint(num_clients_range[0], num_clients_range[1] + 1)
            client_positions = rng.uniform(
                low=-half_arena, high=half_arena, size=(num_clients, 3)
            ).astype(np.float32)
            client_positions[:, 2] = delivery_altitude

            wind_on, obstacles_on = combos[i % len(combos)]

            if obstacles_on:
                k = int(rng.randint(obstacles_k_range[0], obstacles_k_range[1] + 1))
                exclude = np.vstack([
                    np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                    client_positions,
                ])
                obstacle_positions, obstacle_radii = EvaluationDataset._sample_obstacles(
                    rng, k, arena_size, delivery_altitude, exclude
                )
            else:
                obstacle_positions = np.zeros((0, 3), dtype=np.float32)
                obstacle_radii = np.zeros((0,), dtype=np.float32)

            scenario = EvaluationScenario(
                scenario_id=i,
                num_clients=num_clients,
                client_positions=client_positions,
                initial_energy=100.0,
                seed=scenario_seed,
                wind_profile=default_wind_profile if wind_on else None,
                wind_on=wind_on,
                obstacle_positions=obstacle_positions,
                obstacle_radii=obstacle_radii,
            )
            scenarios.append(scenario)

        return EvaluationDataset(scenarios)
    
    def save_json(self, filepath: Path):
        """
        Save dataset to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            "num_scenarios": len(self.scenarios),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved evaluation dataset to {filepath}")
    
    def save_pickle(self, filepath: Path):
        """
        Save dataset to pickle file.
        
        Args:
            filepath: Path to save pickle file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved evaluation dataset to {filepath}")
    
    @staticmethod
    def load_json(filepath: Path) -> "EvaluationDataset":
        """
        Load dataset from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            EvaluationDataset loaded from file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        scenarios = [EvaluationScenario.from_dict(s) for s in data["scenarios"]]
        print(f"Loaded {len(scenarios)} evaluation scenarios from {filepath}")
        return EvaluationDataset(scenarios)
    
    @staticmethod
    def load_pickle(filepath: Path) -> "EvaluationDataset":
        """
        Load dataset from pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            EvaluationDataset loaded from file
        """
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset.scenarios)} evaluation scenarios from {filepath}")
        return dataset
    
    def __len__(self) -> int:
        """Return number of scenarios in dataset."""
        return len(self.scenarios)
    
    def __getitem__(self, idx: int) -> EvaluationScenario:
        """Get scenario by index."""
        return self.scenarios[idx]
    
    def __iter__(self):
        """Iterate over scenarios."""
        return iter(self.scenarios)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        num_clients_list = [s.num_clients for s in self.scenarios]
        return {
            "total_scenarios": len(self.scenarios),
            "avg_clients": np.mean(num_clients_list),
            "min_clients": int(np.min(num_clients_list)),
            "max_clients": int(np.max(num_clients_list)),
            "scenarios_by_difficulty": {
                "easy (3 clients)": sum(1 for s in self.scenarios if s.num_clients == 3),
                "medium (4-5 clients)": sum(1 for s in self.scenarios if 4 <= s.num_clients <= 5),
                "hard (6-8 clients)": sum(1 for s in self.scenarios if s.num_clients >= 6),
            },
            "scenarios_by_domain_randomization": {
                "neither":          sum(1 for s in self.scenarios if not s.wind_on and not s.obstacles_on),
                "wind_only":        sum(1 for s in self.scenarios if s.wind_on and not s.obstacles_on),
                "obstacles_only":   sum(1 for s in self.scenarios if not s.wind_on and s.obstacles_on),
                "both":             sum(1 for s in self.scenarios if s.wind_on and s.obstacles_on),
            },
        }