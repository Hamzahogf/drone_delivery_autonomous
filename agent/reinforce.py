"""REINFORCE / vanilla policy gradient agent — built as a thin custom
Stable-Baselines3 algorithm (subclassing ``OnPolicyAlgorithm``, the same base
class ``PPO`` and ``A2C`` use) rather than a from-scratch training loop.

Why this design: subclassing OnPolicyAlgorithm gets rollout collection,
``.predict()``, ``.save()``/``.load()``, the SB3 logger, and — most
importantly — full compatibility with every existing callback
(WandbMetricsCallback, RaiSimVideoCallback, SuccessRateEvalCallback,
CustomCheckpointCallback) for free, since they only ever call generic
``model.*`` attributes/methods that OnPolicyAlgorithm already provides.

The only thing that actually differs from PPO is ``train()``: no probability-
ratio clipping, no multiple epochs / mini-batches over the rollout buffer —
just a single gradient step per rollout on the plain policy-gradient
objective, ``-log_prob(action) * advantage``, with a learned value-function
baseline for variance reduction (i.e. "REINFORCE with baseline", the
standard, still-called-REINFORCE variant — pure REINFORCE with *no* baseline
is rarely used in practice because its variance is prohibitively high).

Note: with ``gae_lambda=1.0`` (the default here), SB3's GAE advantage
computation collapses to the true Monte-Carlo return minus the value
baseline — i.e. authentic REINFORCE-with-baseline, not TD-bootstrapped like
A2C's usual lower gae_lambda. This is expected to be substantially weaker
than PPO/SAC on this task (see the caveats in the docstring below) — it's
included as a classical baseline for algorithm comparison, not because it's
expected to win.
"""

from typing import Optional, Any, List, Type, Union

import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
import numpy as np

# Re-exported so train.py can import one linear_schedule regardless of algo.
from .ppo import linear_schedule  # noqa: F401


class REINFORCE(OnPolicyAlgorithm):
    """Vanilla policy gradient (REINFORCE) with a learned value baseline.

    Structurally identical to PPO/A2C except for the loss in ``train()``:
    no clipping, no importance-sampling ratio, no multi-epoch passes over
    the rollout buffer — a single full-batch gradient step per rollout.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]] = ActorCriticPolicy,
        env: Optional[VecEnv] = None,
        learning_rate: float = 7e-4,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,  # 1.0 = true Monte-Carlo return (authentic REINFORCE)
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        policy_kwargs: Optional[dict] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.normalize_advantage = normalize_advantage
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            supported_action_spaces=(spaces.Box,),
            _init_setup_model=_init_setup_model,
        )

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Single full-batch pass over the just-collected rollout — that's the
        # "vanilla" in vanilla policy gradient. PPO's clipping/multi-epoch
        # reuse of the same data is precisely what this method omits.
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()
            value_loss = F.mse_loss(rollout_data.returns, values)
            entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class REINFORCEAgent:
    """Thin wrapper matching PPOAgent's / SACAgent's interface exactly, so
    train.py / evaluate.py can treat all three algorithms interchangeably."""

    def __init__(
        self,
        env: VecEnv,
        learning_rate: float = 7e-4,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        policy_hidden_sizes: Optional[List[int]] = None,
        verbose: int = 1,
    ) -> None:
        """
        Args:
            env: Vectorized training environment (continuous action space).
            learning_rate: Adam learning rate for the shared actor/critic optimizer.
            n_steps: Rollout length per environment per update (single gradient
                step is taken over the whole rollout — no mini-batching).
            gamma: Discount factor.
            gae_lambda: 1.0 recovers true Monte-Carlo returns (authentic
                REINFORCE-with-baseline); lower values bootstrap more (A2C-like).
            ent_coef: Entropy bonus coefficient (encourages exploration).
            vf_coef: Value-function (baseline) loss weight in the combined loss.
            max_grad_norm: Gradient-norm clipping threshold.
            normalize_advantage: Normalize advantages to zero-mean/unit-std
                before the policy-gradient step — helps stability a lot in
                practice; set False for the textbook-purest version.
            policy_hidden_sizes: Hidden layer widths for both actor (pi) and
                value/baseline (vf) networks.
        """
        if policy_hidden_sizes is None:
            policy_hidden_sizes = [256, 256]

        policy_kwargs = {
            "net_arch": {"pi": policy_hidden_sizes, "vf": policy_hidden_sizes}
        }

        self._env = env
        self.model = REINFORCE(
            policy=ActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantage=normalize_advantage,
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
        """Run the REINFORCE training loop (same signature as PPOAgent.learn)."""
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
    ) -> "REINFORCEAgent":
        """Load a previously saved model.

        Args:
            path: Path passed to ``REINFORCE.load``.
            env:  Vectorized environment to attach to the loaded model.
            device: Device to load the model onto ("auto", "cpu", "cuda", ...).
            custom_objects: Passed through to ``REINFORCE.load`` (e.g. to
                override the learning-rate schedule on resume).

        Returns:
            A new ``REINFORCEAgent`` instance wrapping the loaded model.
        """
        agent = cls.__new__(cls)
        agent._env = env
        agent.model = REINFORCE.load(path, env=env, device=device, custom_objects=custom_objects)
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