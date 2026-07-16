"""PPO agent wrapper around Stable-Baselines3."""

from typing import Optional, Any, List, Callable
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def linear_schedule(initial_value: float, final_value: float = 5e-5) -> Callable[[float], float]:
    """Linear learning rate schedule.

    Args:
        initial_value: Starting learning rate.
        final_value: Minimum learning rate at the end of training.

    Returns:
        A function that takes the current progress (from 1 to 0) and returns
        the corresponding learning rate.
    """
    def _init(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return _init

class PPOAgent:
    """Thin wrapper around SB3's PPO that matches the project's calling conventions."""

    def __init__(
        self,
        env: VecEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_hidden_sizes: Optional[List[int]] = None,
        verbose: int = 1,
        # NOTE: tensorboard_log removed — logging is handled entirely by
        # WandbCallback (see callbacks.py).  SB3's built-in TensorBoard
        # writer is left unset so it doesn't create conflicting log files.
    ) -> None:
        """
        Args:
            env: Vectorized training environment.
            learning_rate: Adam learning rate.
            n_steps: Rollout length per environment per update.
            batch_size: Mini-batch size for the policy gradient update.
            n_epochs: Number of passes over each rollout buffer.
            gamma: Discount factor.
            gae_lambda: GAE-λ for advantage estimation.
            clip_range: PPO probability-ratio clipping ε.
            ent_coef: Entropy bonus coefficient (encourages exploration).
            vf_coef: Value-function loss weight in the combined loss.
            max_grad_norm: Gradient-norm clipping threshold.
            policy_hidden_sizes: Hidden layer widths for both π and V networks.
        """
        if policy_hidden_sizes is None:
            policy_hidden_sizes = [256, 256]

        policy_kwargs = {
            "net_arch": {"pi": policy_hidden_sizes, "vf": policy_hidden_sizes}
        }

        self._env = env
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            # tensorboard_log intentionally omitted — W&B handles all metrics
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
        """Run the PPO training loop.

        Args:
            total_timesteps: Total environment steps to collect.
            callback: SB3 callback (or CallbackList) for logging / checkpointing.
            log_interval: Number of updates between SB3's own console prints
                (irrelevant when verbose=0 — nothing prints either way).
            reset_num_timesteps: If False, keeps the existing step counter
                (e.g. when resuming) instead of restarting it at 0.
            progress_bar: Show a tqdm/rich progress bar instead of (or
                alongside) console table output. Pair with verbose=0 in the
                constructor for a clean progress-bar-only console.
        """
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
        """Save the model weights to *path* (extension added by SB3 if absent)."""
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        env: VecEnv,
        device: str = "auto",
        custom_objects: Optional[dict] = None,
    ) -> "PPOAgent":
        """Load a previously saved model.

        Args:
            path: Path passed to ``PPO.load``.
            env:  Vectorized environment to attach to the loaded model.
            device: Device to load the model onto ("auto", "cpu", "cuda", ...).
            custom_objects: Passed through to ``PPO.load`` (e.g. to override
                the learning-rate schedule on resume).

        Returns:
            A new ``PPOAgent`` instance wrapping the loaded model.
        """
        agent = cls.__new__(cls)
        agent._env = env
        agent.model = PPO.load(path, env=env, device=device, custom_objects=custom_objects)
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
        """Return the underlying SB3 policy (``ActorCriticPolicy``)."""
        return self.model.policy