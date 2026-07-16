"""SAC agent wrapper around Stable-Baselines3 — mirrors PPOAgent's interface
(learn / predict / save / load / set_env / num_timesteps / get_policy_network)
so train.py can switch between algorithms without branching on the rest of
the training loop.

SAC is off-policy and needs a continuous action space (Box) — DroneDeliveryEnv
already uses spaces.Box, so no env changes are needed to use this.
"""

from typing import Optional, Any, List, Union
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv

# Re-exported so train.py can import one linear_schedule regardless of algo.
from .ppo import linear_schedule  # noqa: F401


class SACAgent:
    """Thin wrapper around SB3's SAC that matches the project's calling conventions."""

    def __init__(
        self,
        env: VecEnv,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        policy_hidden_sizes: Optional[List[int]] = None,
        verbose: int = 1,
        # NOTE: tensorboard_log intentionally omitted — same reasoning as
        # PPOAgent, W&B handles all metrics via WandbMetricsCallback.
    ) -> None:
        """
        Args:
            env: Vectorized training environment (continuous action space).
            learning_rate: Adam learning rate (actor, critics, and entropy coeff).
            buffer_size: Replay buffer capacity (transitions).
            learning_starts: Steps of random exploration before training starts.
            batch_size: Mini-batch size sampled from the replay buffer per update.
            tau: Polyak averaging coefficient for target-network updates.
            gamma: Discount factor.
            train_freq: Update the model every `train_freq` environment steps.
            gradient_steps: Gradient updates to run per training round (-1 = as
                many as steps collected this round, matching train_freq).
            ent_coef: Entropy-regularization coefficient. "auto" learns it
                automatically (recommended default); a float fixes it.
            target_update_interval: Steps between target-network updates.
            target_entropy: Target entropy for automatic ent_coef tuning; "auto"
                defaults to -dim(action_space).
            use_sde: Use generalized State-Dependent Exploration instead of
                independent action noise.
            policy_hidden_sizes: Hidden layer widths for both actor (pi) and
                critic (qf) networks.
        """
        if policy_hidden_sizes is None:
            policy_hidden_sizes = [256, 256]

        policy_kwargs = {
            "net_arch": {"pi": policy_hidden_sizes, "qf": policy_hidden_sizes}
        }

        self._env = env
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=(train_freq, "step"),
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None:
        """Run the SAC training loop (same signature as PPOAgent.learn)."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple:
        """Return (action, state) for a given observation."""
        return self.model.predict(observation, deterministic=deterministic)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model weights (and replay buffer state is NOT included —
        SB3 saves that separately via save_replay_buffer if you want it)."""
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        env: VecEnv,
        device: str = "auto",
        custom_objects: Optional[dict] = None,
    ) -> "SACAgent":
        """Load a previously saved model.

        Args:
            path: Path passed to ``SAC.load``.
            env:  Vectorized environment to attach to the loaded model.
            device: Device to load the model onto ("auto", "cpu", "cuda", ...).
            custom_objects: Passed through to ``SAC.load`` (e.g. to override
                the learning-rate schedule on resume).

        Returns:
            A new ``SACAgent`` instance wrapping the loaded model.
        """
        agent = cls.__new__(cls)
        agent._env = env
        agent.model = SAC.load(path, env=env, device=device, custom_objects=custom_objects)
        return agent

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        """Get the number of timesteps the agent has been trained for."""
        return self.model.num_timesteps

    def set_env(self, env: VecEnv) -> None:
        """
        Set a new environment for the model.

        Args:
            env: The new environment.
        """
        self.model.set_env(env)

    def get_policy_network(self):
        """Return the underlying SB3 policy (``SACPolicy``)."""
        return self.model.policy