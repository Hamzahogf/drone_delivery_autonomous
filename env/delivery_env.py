"""
Drone delivery environment for reinforcement learning — RaiSim backend.
 
This module implements a custom Gymnasium environment for drone-based package
delivery.  The physics simulation is driven by RaiSim (via ``raisimpy``)
instead of PyBullet / gym-pybullet-drones.
 
Key design decisions vs. the original PyBullet version
-------------------------------------------------------
* ``DroneDeliveryEnv`` now inherits directly from ``gymnasium.Env`` (no
  ``HoverAviary`` base class) and owns the RaiSim ``World`` and drone
  ``ArticulatedSystem`` objects.
* Velocity control is implemented as a simple kinematic integrator:
  the action [vx, vy, vz] is scaled and added to the drone position every
  step (after calling ``world.integrate()`` for gravity / collision).
  For a more realistic rotor model, swap ``_apply_velocity_command()`` for
  a full PD-torque scheme once you have a drone URDF.
* Visualisation uses ``RaisimServer`` + raisimUnity (the standard RaiSim
  visualiser) instead of PyBullet's built-in GUI.
* The observation space and reward logic are **identical** to the original.
"""

from __future__ import annotations
import os
from typing import Any, Optional, List, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import raisimpy as raisim
    _RAISIM_AVAILABLE  = True
except ImportError:
    raisim = None
    _RAISIM_AVAILABLE  = False

from .drone_dynamics import(
    DroneEnergyModel,
    WindModel,
    compute_distance_to_target,
    compute_relative_vector,
    is_withing_delivery_radius,
    is_out_of_bounds,
    sample_obstacles,
    is_colliding_with_obstacle,
)

from .visual_markers import VisualMarkerManager

# ---------------------------------------------------------------------------
# Path to the drone URDF shipped with the package (or override via env-var).
# Set DRONE_URDF_PATH in your shell to point to your own model file.
# ---------------------------------------------------------------------------
_DEFAULT_URDF = os.path.join(
    os.path.dirname(__file__), "assets", "cf2x.urdf"
)
DRONE_URDF_PATH: str = os.environ.get("DRONE_URDF_PATH", _DEFAULT_URDF)

# raisim activation key path  (required by raisimLib ≤ 1.x; ignored on 2.x+).
RAISIM_ACTIVATION_KEY: str = os.environ.get(
    "RAISIM_ACTIVATION_KEY",
    os.path.expanduser("~/.raisim/activation.raisim"),
)

# physics time-step and control decimation
_PHYSICS_DT: float = 1.0/ 240.0 # 240 Hz physics
_CTRL_EVERY: int = 8

class DroneDeliveryEnv(gym.Env):
    """
    Gymnasium environment for drone-based package delivery using RaiSim.
 
    The drone must visit every client position in the arena before energy
    runs out or the episode time-limit is reached.
 
    Observation
    -----------
    [drone_pos(3), drone_vel(3), energy(1), remaining_frac(1),
     rel_vec_to_next(3), distances(max_clients), delivered_mask(max_clients)]
 
    Action
    ------
    Normalised 3-D velocity command in [−1, 1]³ → scaled to max_speed.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    # construction
    def __init__(
          self,
          gui: bool = False,
          num_clients_min: int = 3,
          num_clients_max: int = 8,
          max_clients: int = 8,
          arena_size: float = 10.0,
          delivery_altitude: float = 1.0,
          delivery_radius: float = 0.5,
          max_speed: float = 2.0,
          initial_energy: float = 100.0,    
          base_drain: float = 0.01,
          speed_coefficient: float = 0.05,
          max_episode_steps: int = 2000,
          delivery_reward: float = 50.0,
          completion_bonus: float = 100.0,
          energy_bonus_coeff: float = 0.5,
          failure_penalty: float = -100.0,
          out_of_bounds_penalty: float = -20.0,
          seed: Optional[int] = None,
          dense_shaping_enabled: bool = False,
          shaping_coeff: float = 0.5,
          # ── Domain randomization (single-stage: sampled fresh every reset) ──
          domain_randomization_enabled: bool = True,
          wind_prob: float = 0.5,
          obstacles_prob: float = 0.5,
          wind_volatility: float = 0.5,
          wind_mean_reversion: float = 0.1,
          wind_max_speed: float = 3.0,
          obstacles_min_k: int = 2,
          obstacles_max_k: int = 5,
          obstacle_radius_min: float = 0.3,
          obstacle_radius_max: float = 0.8,
          obstacle_collision_penalty: float = -50.0,
          evaluation_mode: bool = False,
          urdf_path: str = DRONE_URDF_PATH,
          raisim_activation_key: str = RAISIM_ACTIVATION_KEY,
    ) -> None:
        super().__init__()
 
        if not _RAISIM_AVAILABLE:
            raise ImportError(
                "raisimpy is not installed.  "
                "Follow the instructions at https://raisim.com/sections/Installation.html "
                "to build and install raisimLib + raisimpy."
            )
        # ── store configuration ──────────────────────────────────────────
        self.gui                  = gui
        self.num_clients_min      = num_clients_min
        self.num_clients_max      = num_clients_max
        self.max_clients          = max_clients
        self.arena_size           = arena_size
        self.delivery_altitude    = delivery_altitude
        self.max_speed            = max_speed
        self.max_episode_steps    = max_episode_steps
        self.urdf_path            = urdf_path
 
        self.evaluation_mode      = evaluation_mode
        self.dense_shaping_enabled = dense_shaping_enabled
        self.shaping_coeff        = shaping_coeff
 
        # Single fixed delivery radius — no curriculum annealing.
        self.delivery_radius = float(delivery_radius)

        # ── Domain randomization config ──────────────────────────────────
        # Every reset() independently coin-flips wind-on / obstacles-on so
        # all four combinations (neither / wind-only / obstacles-only / both)
        # occur during training. Disabled entirely in evaluation_mode so the
        # frozen eval dataset controls wind/obstacles explicitly instead.
        self.domain_randomization_enabled = domain_randomization_enabled and not evaluation_mode
        self.wind_prob      = float(wind_prob)
        self.obstacles_prob = float(obstacles_prob)
        self.obstacles_min_k = int(obstacles_min_k)
        self.obstacles_max_k = int(obstacles_max_k)
        self.obstacle_radius_min = float(obstacle_radius_min)
        self.obstacle_radius_max = float(obstacle_radius_max)
        self.obstacle_collision_penalty = float(obstacle_collision_penalty)
        self.max_obstacles = int(obstacles_max_k)

        self.wind_model = WindModel(
            volatility=wind_volatility,
            mean_reversion=wind_mean_reversion,
            max_speed=wind_max_speed,
        )
        self._wind_on: bool = False
        self._obstacles_on: bool = False
        self.obstacle_positions: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.obstacle_radii:     np.ndarray = np.zeros((0,), dtype=np.float32)

        # Reward weights
        self.delivery_reward     = delivery_reward
        self.completion_bonus    = completion_bonus
        self.energy_bonus_coeff  = energy_bonus_coeff
        self.failure_penalty     = failure_penalty
        self.out_of_bounds_penalty = out_of_bounds_penalty
 
        # Energy model
        self.energy_model   = DroneEnergyModel(
            initial_energy=initial_energy,
            base_drain=base_drain,
            speed_coefficient=speed_coefficient,
        )
        self.current_energy = float(initial_energy)
 
        # Episode state (initialised properly in reset())
        self.client_positions: np.ndarray = np.zeros((max_clients, 3), np.float32)
        self.delivered_mask:   np.ndarray = np.ones(max_clients, bool)
        self.num_clients:      int        = 0
        self.step_count:       int        = 0
        self.episode_reward:   float      = 0.0
        self._prev_nearest_dist: float    = 0.0
 
        # Drone state (maintained by our kinematic integrator)
        self._drone_pos: np.ndarray = np.array([0.0, 0.0, 0.5])
        self._drone_vel: np.ndarray = np.zeros(3)
 
        # ── Gymnasium spaces ─────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # + obstacle relative vectors (max_obstacles*3, nearest-first, zero-padded)
        # + obstacle active mask (max_obstacles, 1 = real obstacle this episode)
        obs_size = (
            3 + 3 + 1 + 1 + 3 + max_clients + max_clients
            + self.max_obstacles * 3 + self.max_obstacles
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
 
        # ── RaiSim world ─────────────────────────────────────────────────
        # Activate licence for raisimLib ≤ 1.x (silently skipped on 2.x).
        if os.path.isfile(raisim_activation_key):
            try:
                raisim.World.setLicenseFile(raisim_activation_key)
            except AttributeError:
                pass  # raisimLib 2.x dropped licence files
 
        self._world = raisim.World()
        self._world.setTimeStep(_PHYSICS_DT)
        self._world.addGround()
 
        # Load drone URDF (floating-base: root link must not be named "world")
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(
                f"Drone URDF not found at '{self.urdf_path}'.  "
                "Set the DRONE_URDF_PATH environment variable or place "
                "assets/cf2x.urdf next to delivery_env.py."
            )
        self._drone_body = self._world.addArticulatedSystem(self.urdf_path)
        self._gc_dim  = self._drone_body.getGeneralizedCoordinateDim()
        self._gv_dim  = self._drone_body.getDOF()
 
        # ── RaiSim visualisation server ──────────────────────────────────
        if gui:
            self._server = raisim.RaisimServer(self._world)
            self._server.launchServer(8080)
            self._server.focusOn(self._drone_body)
            self.marker_manager: Optional[VisualMarkerManager] = (
                VisualMarkerManager(self._server, gui_mode=True)
            )
        else:
            self._server = None
            self.marker_manager = None
 
        # Seed the RNG
        if seed is not None:
            np.random.seed(seed)

        
    def set_energy_drain(self, base_drain: float, speed_coefficient: float) -> None:
        """Update the energy-drain coefficients in place (e.g. for manual tuning/eval sweeps)."""
        self.energy_model.base_drain = float(base_drain)
        self.energy_model.speed_coefficient = float(speed_coefficient)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _nearest_undelivered_distance(self, drone_pos: np.ndarray) -> float:
        active = np.where(~self.delivered_mask[: self.num_clients])[0]
        if len(active) == 0:
            return 0.0
        return float(min(
            compute_distance_to_target(drone_pos, self.client_positions[i])
            for i in active
        ))
 
    def _sample_clients_in_disk(self, spawn_radius: float) -> np.ndarray:
        """Uniform random positions within a disk of *spawn_radius* around origin."""
        positions = np.zeros((self.num_clients, 3), dtype=np.float32)
        for i in range(self.num_clients):
            theta       = np.random.uniform(0.0, 2.0 * np.pi)
            r           = spawn_radius * np.sqrt(np.random.uniform(0.0, 1.0))
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = r * np.sin(theta)
            positions[i, 2] = self.delivery_altitude
        return positions
 
    def _energy_for_observation(self) -> float:
        """Observed energy — single-stage training always uses the real value."""
        return float(self.current_energy)
 
    # ------------------------------------------------------------------
    # Drone state helpers (RaiSim-based)
    # ------------------------------------------------------------------
 
    def _reset_drone_state(self) -> None:
        """Place the drone at the origin (0, 0, 0.5) with zero velocity."""
        # Generalised coordinate for a floating-base drone URDF:
        #   [x, y, z, qw, qx, qy, qz, rotor_angles...]
        gc = np.zeros(self._gc_dim)
        gc[0], gc[1], gc[2] = 0.0, 0.0, 0.5   # position
        gc[3] = 1.0                              # quaternion w = 1 (identity)
        gv = np.zeros(self._gv_dim)
 
        self._drone_body.setState(gc, gv)
        self._drone_pos = np.array([0.0, 0.0, 0.5])
        self._drone_vel = np.zeros(3)

    def _apply_velocity_command(self, velocity_cmd: np.ndarray) -> None:
        """
        Apply a 3D velocity command by updating the generalised velocity.
        This is a kinematic velocity controller: we directly set the linear
        velocity of the drone base, then let RaiSim integrate one physics
        step so gravity and collisions are still respected.
        For a full rotor-torque model, replace this method with a PD
        controller that drives the four rotor speeds.
        """
        # Read current generalised coordinate (position + orientation + rotor angles)
        gc, gv = self._drone_body.getState()
        gc = np.array(gc)
        gv = np.array(gv)

        # Overwrite linear velocity part of gv (first 3 DoFs of a floating base)
        gv[0] = velocity_cmd[0]
        gv[1] = velocity_cmd[1]
        gv[2] = velocity_cmd[2]
        self._drone_body.setState(gc, gv)

        # Integrate one physics step
        self._world.integrate()

        # Read back updated state
        gc, gv = self._drone_body.getState()
        self._drone_pos = np.array(gc)[0:3].copy()
        self._drone_vel = np.array(gv)[0:3].copy()
 
    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
 
    def _compute_obs(self) -> np.ndarray:
        drone_pos = self._drone_pos
        drone_vel = self._drone_vel
 
        # Relative vector to the nearest undelivered client
        undelivered = np.where(~self.delivered_mask[: self.num_clients])[0]
        if len(undelivered) > 0:
            next_pos       = self.client_positions[int(undelivered[0])]
            relative_vector = compute_relative_vector(drone_pos, next_pos)
        else:
            relative_vector = np.zeros(3)
 
        # Per-client distances (padded)
        distances = np.zeros(self.max_clients, dtype=np.float32)
        for i in range(self.num_clients):
            distances[i] = compute_distance_to_target(drone_pos, self.client_positions[i])
 
        remaining_frac = float(
            np.sum(~self.delivered_mask[: self.num_clients]) / max(self.num_clients, 1)
        )
        energy_obs = self._energy_for_observation() / 100.0

        # Obstacle relative vectors, nearest-first, zero-padded to max_obstacles.
        obstacle_rel = np.zeros((self.max_obstacles, 3), dtype=np.float32)
        obstacle_mask = np.zeros(self.max_obstacles, dtype=np.float32)
        n_obs = len(self.obstacle_positions)
        if n_obs > 0:
            dists = np.linalg.norm(self.obstacle_positions - drone_pos, axis=1)
            order = np.argsort(dists)[: self.max_obstacles]
            for slot, idx in enumerate(order):
                obstacle_rel[slot] = compute_relative_vector(
                    drone_pos, self.obstacle_positions[idx]
                )
                obstacle_mask[slot] = 1.0

        obs = np.concatenate([
            drone_pos.astype(np.float32),                        # 3
            drone_vel.astype(np.float32),                        # 3
            np.array([energy_obs],  dtype=np.float32),           # 1
            np.array([remaining_frac], dtype=np.float32),        # 1
            relative_vector.astype(np.float32),                  # 3
            distances,                                            # max_clients
            self.delivered_mask.astype(np.float32),              # max_clients
            obstacle_rel.flatten(),                              # max_obstacles*3
            obstacle_mask,                                        # max_obstacles
        ])
        return obs.astype(np.float32)
 
    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
 
    def _compute_step_reward(self, shaping_reward: float) -> float:
        reward = 0.0
        energy_cost = self.energy_model.compute_energy_drain(self._drone_vel)
        reward -= energy_cost * 0.5
        reward += shaping_reward
        return float(reward)
 
    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------
 
    def _is_terminated(self) -> bool:
        if np.all(self.delivered_mask[: self.num_clients]):
            return True
        if self.current_energy <= 0.0:
            return True
        if is_out_of_bounds(self._drone_pos, self.arena_size, z_max=5.0):
            return True
        if is_colliding_with_obstacle(
            self._drone_pos, self.obstacle_positions, self.obstacle_radii
        ):
            return True
        return False
 
    def _is_truncated(self) -> bool:
        return self.step_count >= self.max_episode_steps
 
    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
 
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
 
        # ── Client positions ─────────────────────────────────────────────
        if options is not None and "client_positions" in options:
            self.client_positions = np.array(
                options["client_positions"], dtype=np.float32
            )
            self.num_clients = len(self.client_positions)
        else:
            self.num_clients = int(
                np.random.randint(self.num_clients_min, self.num_clients_max + 1)
            )
            half = self.arena_size / 2.0
            self.client_positions = np.random.uniform(
                low=-half, high=half, size=(self.num_clients, 3)
            ).astype(np.float32)
            self.client_positions[:, 2] = self.delivery_altitude
 
        # Pad client_positions to max_clients rows
        padded = np.zeros((self.max_clients, 3), dtype=np.float32)
        padded[: self.num_clients] = self.client_positions[: self.num_clients]
        self.client_positions = padded
 
        # ── Delivery mask ────────────────────────────────────────────────
        self.delivered_mask = np.ones(self.max_clients, bool)
        self.delivered_mask[: self.num_clients] = False
 
        # ── Energy ──────────────────────────────────────────────────────
        self.current_energy = float(self.energy_model.initial_energy)
 
        # ── Episode counters ─────────────────────────────────────────────
        self.step_count    = 0
        self.episode_reward = 0.0
 
        # ── Physics reset ─────────────────────────────────────────────────
        self._reset_drone_state()

        # ── Domain randomization: wind & obstacles ───────────────────────
        # Independent coin-flips so neither / wind-only / obstacles-only /
        # both all occur across training.
        if self.domain_randomization_enabled:
            self._wind_on = bool(np.random.random() < self.wind_prob)
            self._obstacles_on = bool(np.random.random() < self.obstacles_prob)
        else:
            self._wind_on = False
            self._obstacles_on = False

        self.wind_model.reset()

        if self._obstacles_on:
            k = int(np.random.randint(self.obstacles_min_k, self.obstacles_max_k + 1))
            exclude = np.vstack([
                self._drone_pos.reshape(1, 3),
                self.client_positions[: self.num_clients],
            ])
            self.obstacle_positions, self.obstacle_radii = sample_obstacles(
                num_obstacles=k,
                arena_size=self.arena_size,
                altitude=self.delivery_altitude,
                exclude_positions=exclude,
                radius_min=self.obstacle_radius_min,
                radius_max=self.obstacle_radius_max,
            )
        else:
            self.obstacle_positions = np.zeros((0, 3), dtype=np.float32)
            self.obstacle_radii = np.zeros((0,), dtype=np.float32)

        # Explicit overrides (used by frozen eval scenarios for reproducibility)
        if options is not None:
            if "obstacle_positions" in options:
                self.obstacle_positions = np.array(
                    options["obstacle_positions"], dtype=np.float32
                ).reshape(-1, 3)
                default_radii = np.full(
                    len(self.obstacle_positions), self.obstacle_radius_min, dtype=np.float32
                )
                self.obstacle_radii = np.array(
                    options.get("obstacle_radii", default_radii), dtype=np.float32
                )
                self._obstacles_on = len(self.obstacle_positions) > 0
            if "wind_on" in options:
                self._wind_on = bool(options["wind_on"])
 
        # ── Visual markers ───────────────────────────────────────────────
        if self.marker_manager is not None:
            self.marker_manager.reset(
                self.client_positions[: self.num_clients],
                delivery_radius=self.delivery_radius,
            )
            self.marker_manager.sync_obstacles(
                self.obstacle_positions, self.obstacle_radii
            )
            self.marker_manager.update_hud(
                self._energy_for_observation(),
                self.num_clients,
                self.num_clients,
                target_index=0,
            )
 
        obs = self._compute_obs()
        self._prev_nearest_dist = self._nearest_undelivered_distance(self._drone_pos)
 
        info: Dict[str, Any] = {
            "num_clients":          self.num_clients,
            "client_positions":     self.client_positions[: self.num_clients].copy(),
            "energy":               self._energy_for_observation(),
            "deliveries_completed": int(np.sum(self.delivered_mask[: self.num_clients])),
            "total_deliveries":     self.num_clients,
            "shaping_reward":       0.0,
            "delivery_radius":      float(self.delivery_radius),
            "wind_on":              self._wind_on,
            "obstacles_on":         self._obstacles_on,
            "num_obstacles":        int(len(self.obstacle_positions)),
        }
        return obs, info
 
    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
 
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # ── Scale action → velocity command ──────────────────────────────
        velocity_cmd = np.clip(action, -1.0, 1.0) * self.max_speed

        # ── Wind disturbance (only when this episode drew wind-on) ───────
        if self._wind_on:
            wind_vel = self.wind_model.step()
            velocity_cmd = velocity_cmd + wind_vel

        # ── Physics step (sub-steps for accuracy) ────────────────────────
        for _ in range(_CTRL_EVERY):
            self._apply_velocity_command(velocity_cmd)

        drone_pos = self._drone_pos
        drone_vel = self._drone_vel

        # ── Energy update (single-stage: always strict drain) ────────────
        drain = self.energy_model.compute_energy_drain(drone_vel)
        self.current_energy = float(max(0.0, self.current_energy - drain))
 
        # ── Shaping reward ────────────────────────────────────────────────
        active_before = np.where(~self.delivered_mask[: self.num_clients])[0]
        nearest_after = (
            self._nearest_undelivered_distance(drone_pos)
            if len(active_before) > 0
            else 0.0
        )
        shaping_reward = 0.0
        if self.dense_shaping_enabled and len(active_before) > 0:
            shaping_reward = float(
                self.shaping_coeff * (self._prev_nearest_dist - nearest_after)
            )
 
        # ── Delivery check ────────────────────────────────────────────────
        delivery_reward_this_step = 0.0
        for i in range(self.num_clients):
            if not self.delivered_mask[i]:
                if is_withing_delivery_radius(
                    drone_pos, self.client_positions[i], self.delivery_radius
                ):
                    self.delivered_mask[i] = True
                    delivery_reward_this_step += self.delivery_reward
                    if self.marker_manager is not None:
                        self.marker_manager.mark_delivered(i)
 
        self._prev_nearest_dist = self._nearest_undelivered_distance(drone_pos)
 
        # ── Per-step reward ───────────────────────────────────────────────
        reward = self._compute_step_reward(shaping_reward) + delivery_reward_this_step
 
        # ── Termination / truncation ──────────────────────────────────────
        self.step_count += 1
        terminated = self._is_terminated()
        truncated  = self._is_truncated()
 
        # Terminal bonuses / penalties
        if terminated or truncated:
            all_done = bool(np.all(self.delivered_mask[: self.num_clients]))
            if all_done:
                reward += self.completion_bonus
                reward += self.current_energy * self.energy_bonus_coeff
            elif self.current_energy <= 0.0:
                reward += self.failure_penalty
                terminated = True
            elif is_colliding_with_obstacle(
                drone_pos, self.obstacle_positions, self.obstacle_radii
            ):
                reward += self.obstacle_collision_penalty
                terminated = True
            elif is_out_of_bounds(drone_pos, self.arena_size, z_max=5.0):
                reward += self.out_of_bounds_penalty
                terminated = True
 
        self.episode_reward += reward
 
        # ── Update HUD ────────────────────────────────────────────────────
        if self.gui and self.marker_manager is not None:
            remaining = self.num_clients - int(
                np.sum(self.delivered_mask[: self.num_clients])
            )
            undelivered_idx = np.where(~self.delivered_mask[: self.num_clients])[0]
            target = int(undelivered_idx[0]) if len(undelivered_idx) > 0 else None
            self.marker_manager.update_hud(
                self._energy_for_observation(),
                remaining,
                self.num_clients,
                target_index=target,
            )
 
        # ── Observation ───────────────────────────────────────────────────
        obs = self._compute_obs()
 
        info: Dict[str, Any] = {
            "energy":               self._energy_for_observation(),
            "deliveries_completed": int(np.sum(self.delivered_mask[: self.num_clients])),
            "total_deliveries":     self.num_clients,
            "episode_reward":       self.episode_reward,
            "shaping_reward":       shaping_reward,
            "delivery_radius":      float(self.delivery_radius),
            "wind_on":              self._wind_on,
            "obstacles_on":         self._obstacles_on,
            "num_obstacles":        int(len(self.obstacle_positions)),
        }
        if terminated or truncated:
            info["episode_success"] = bool(
                np.all(self.delivered_mask[: self.num_clients])
            )
 
        return obs, float(reward), terminated, truncated, info
 
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
 
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return None
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
 
    def close(self) -> None:
        """Release all RaiSim resources."""
        if self.marker_manager is not None:
            self.marker_manager.cleanup()
        if self._server is not None:
            try:
                self._server.killServer()
            except Exception:
                pass
        # The World destructor handles the rest.

    # ------------------------------------------------------------------
    # Helper for camera frames
    # ------------------------------------------------------------------

    def get_drone_pos(self) -> np.ndarray:
        return self._drone_pos.copy()

    def get_client_positions(self) -> np.ndarray:
        return self.client_positions[:self.num_clients].copy()

    def get_delivered_mask(self) -> np.ndarray:
        return self.delivered_mask[:self.num_clients].copy()
    
# ---------------------------------------------------------------------------
# Gym registration helper
# ---------------------------------------------------------------------------
 
def register_env() -> None:
    """Register DroneDelivery-v0 with the Gymnasium registry."""
    gym.register(
        id="DroneDelivery-v0",
        entry_point="drone_delivery_autonomous.env.delivery_env:DroneDeliveryEnv",
        max_episode_steps=2000,
    )