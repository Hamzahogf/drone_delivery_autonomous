from typing import Tuple
import numpy as np

class DroneEnergyModel:
    def __init__(
            self,
            initial_energy: float = 100.0,
            base_drain: float = 0.01,
            speed_coefficient: float = 0.05
    ):
        """
            iniitial_energy= Max energy budget
            base_drain: Energy lost per timestep at zero velocity
            speed_coefficient: Energy loss scaling factor per m/s of velocity
        """
    
        self.initial_energy = initial_energy
        self.base_drain = base_drain
        self.speed_coefficient = speed_coefficient
    
    def compute_energy_drain(self, velocity: float)-> float:
        speed = np.linalg.norm(velocity)
        return self.base_drain + self.speed_coefficient * speed
    
    def compute_remaining_energy(self, current_energy: float, velocity: np.ndarray) -> float:
        """
            update energy after consuming power for one timestep
            Args:
                current_energy: Current energy level
                velocity: 3D velocity vector of the drone [vx, vy, vz] in m/s

            returns:
                remaining energy after one timestep
        """
        drain = self.compute_energy_drain(velocity)
        remaining_energy = max(0.0, current_energy - drain)
        return remaining_energy
    
def compute_distance_to_target(
        drone_pos: np.ndarray,
        target_pos: np.ndarray
) -> float :
    """
       compute euclidean distance between drone and target
         Args:
                drone_pos: 3D position of the drone [x, y, z]
                target_pos: 3D position of the target [x, y, z]
    
          returns:
                euclidean distance between drone and target
    """
    return float(np.linalg.norm(drone_pos - target_pos))

def compute_relative_vector(
        drone_pos: np.ndarray,
        target_pos: np.ndarray
) -> np.ndarray:
    """
        compute relative position vector from drone to target
         Args:
                drone_pos: 3D position of the drone [x, y, z]
                target_pos: 3D position of the target [x, y, z]
        returns: 
                relative position vector from drone to target
    """
    return target_pos - drone_pos

def is_withing_delivery_radius(
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        radius: float 
) -> bool:
    return compute_distance_to_target(drone_pos, target_pos) <= radius  

class WindModel:
    """Ornstein-Uhlenbeck wind process. Call reset() at episode start, step() once per control tick."""

    def __init__(
            self,
            volatility: float = 0.5,
            mean_reversion: float = 0.1,
            max_speed: float = 3.0,
    ):
        """
            volatility: OU noise scale (sigma)
            mean_reversion: OU pull-back-to-zero rate (theta)
            max_speed: clamp on wind velocity magnitude (m/s)
        """
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.max_speed = max_speed
        self.wind_velocity = np.zeros(3)

    def reset(self) -> None:
        self.wind_velocity = np.zeros(3)

    def step(self, dt: float = 1.0) -> np.ndarray:
        """Advance the OU process one tick and return the new wind velocity."""
        noise = np.random.normal(0.0, 1.0, size=3) * self.volatility
        self.wind_velocity += -self.mean_reversion * self.wind_velocity * dt + noise * np.sqrt(dt)
        speed = np.linalg.norm(self.wind_velocity)
        if speed > self.max_speed:
            self.wind_velocity *= self.max_speed / speed
        return self.wind_velocity.copy()


def sample_obstacles(
        num_obstacles: int,
        arena_size: float,
        altitude: float,
        exclude_positions: np.ndarray,
        radius_min: float = 0.3,
        radius_max: float = 0.8,
        min_clearance: float = 1.0,
        max_attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Sample non-overlapping sphere obstacles inside the arena, keeping
        clearance from drone-start / client positions and from each other.

        Args:
            num_obstacles: number of spheres (K) to place
            arena_size: side length of the square arena
            altitude: z-height to place obstacle centres at
            exclude_positions: positions (e.g. drone start + client positions)
                                to keep clear of
            radius_min / radius_max: sphere radius range (m)
            min_clearance: minimum gap between an obstacle surface and any
                            excluded position
            max_attempts: rejection-sampling attempts per obstacle before
                            giving up and placing it anyway

        returns:
            (positions [K,3], radii [K])
    """
    half = arena_size / 2.0
    positions: list = []
    radii: list = []

    for _ in range(num_obstacles):
        candidate = None
        r = radius_min
        for _attempt in range(max_attempts):
            candidate = np.array([
                np.random.uniform(-half, half),
                np.random.uniform(-half, half),
                altitude,
            ], dtype=np.float32)
            r = float(np.random.uniform(radius_min, radius_max))

            ok = True
            for ex in exclude_positions:
                if np.linalg.norm(candidate - ex) < (min_clearance + r):
                    ok = False
                    break
            if ok:
                for p, pr in zip(positions, radii):
                    if np.linalg.norm(candidate - p) < (r + pr + 0.3):
                        ok = False
                        break
            if ok:
                break
        positions.append(candidate)
        radii.append(r)

    if num_obstacles == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.array(positions, dtype=np.float32), np.array(radii, dtype=np.float32)


def is_colliding_with_obstacle(
        drone_pos: np.ndarray,
        obstacle_positions: np.ndarray,
        obstacle_radii: np.ndarray,
        drone_radius: float = 0.15,
) -> bool:
    """
        check whether the drone body intersects any sphere obstacle.
        Args:
            drone_pos: 3D position of the drone
            obstacle_positions: [K,3] obstacle centres (K may be 0)
            obstacle_radii: [K] obstacle radii
            drone_radius: physical radius of the drone body (m)
        returns:
            True if colliding with any obstacle, False otherwise (or if K == 0)
    """
    if obstacle_positions is None or len(obstacle_positions) == 0:
        return False
    dists = np.linalg.norm(obstacle_positions - drone_pos, axis=1)
    return bool(np.any(dists <= (obstacle_radii + drone_radius)))


def is_out_of_bounds(
        drone_pos: np.ndarray,
        arena_size: float,
        z_max: float = 5.0
) -> bool:
    """
        check if drone is out of bounds of the arena
         Args:
                drone_pos: 3D position of the drone [x, y, z]
                arena_size: size of the square arena (length of one side)
                z_max: maximum allowed altitude for the drone
        returns:
                True if drone is out of bounds, False otherwise
    """
    x, y, z = drone_pos
    half_arena = arena_size / 2.0
    return (
            x < -half_arena or x > half_arena
            or y < -half_arena or y > half_arena
            or z > z_max
            or z < 0.0
        )
