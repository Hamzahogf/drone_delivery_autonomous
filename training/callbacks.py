"""
Custom SB3 callbacks for W&B logging, curriculum control, and RaiSim video capture.

All TensorBoard references have been removed.  Every metric previously written
with ``self.logger.record(...)`` is now forwarded to W&B via ``wandb.log()``.
SB3's internal logger is left in its default (stdout) state.

Video workflow (RaiSim)
-----------------------
``RaiSimVideoCallback`` calls ``env_method("render", "rgb_array")`` on one of
the vectorised training environments every *video_every_n_episodes* evaluation
episodes, collects the resulting RGB frames, assembles them with ``imageio``,
and uploads the clip to W&B as a ``wandb.Video`` object.
Because RaiSim's headless renderer is used (no GUI window), this works on
remote training machines without a display.
"""

from __future__ import annotations

import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, sync_envs_normalization

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import imageio  # type: ignore
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# W&B logging shim
# ---------------------------------------------------------------------------

def _wlog(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log *metrics* to W&B if available, otherwise silently skip."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Main W&B metrics callback  (replaces TensorBoardCallback)
# ---------------------------------------------------------------------------

class WandbMetricsCallback(BaseCallback):
    """
    Logs per-step and per-episode diagnostics to Weights & Biases.

    Replaces the original ``TensorBoardCallback``.  The logged keys are
    identical so existing W&B dashboard charts work without modification.

    Logged metrics
    ~~~~~~~~~~~~~~
    * ``train/shaping_reward``   – rolling mean per-episode shaping reward
    * ``episode/energy_used``    – 100 − final energy (from Monitor info)
    * ``episode/deliveries_completed`` – deliveries finished in the episode
    * ``train/num_timesteps``    – global step counter (for axis alignment)
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_count = 0
        self._shaping_accumulator: Dict[int, float] = {}
        self._recent_episode_mean_shaping: Deque[float] = deque(maxlen=50)

    def _on_training_start(self) -> None:
        _wlog({"train/shaping_reward": 0.0}, step=0)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is None:
            return True

        # Accumulate per-step shaping reward for each parallel env
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            sh = float(info.get("shaping_reward", 0.0))
            self._shaping_accumulator[i] = (
                self._shaping_accumulator.get(i, 0.0) + sh
            )

        if not np.any(dones):
            return True

        self.episode_count += 1
        metrics: Dict[str, float] = {}

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            if isinstance(info, dict) and "episode" in info:
                ep_info = info["episode"]
                metrics["episode/energy_used"] = 100.0 - float(
                    ep_info.get("energy", 50)
                )
                metrics["episode/deliveries_completed"] = float(
                    ep_info.get("deliveries", 0)
                )
                ep_len = max(int(ep_info.get("l", 1)), 1)
            else:
                ep_len = 1

            total_sh = self._shaping_accumulator.pop(i, 0.0)
            self._recent_episode_mean_shaping.append(total_sh / ep_len)

        if self._recent_episode_mean_shaping:
            metrics["train/shaping_reward"] = float(
                np.mean(self._recent_episode_mean_shaping)
            )

        metrics["train/num_timesteps"] = float(self.num_timesteps)
        _wlog(metrics, step=self.num_timesteps)
        return True


# ---------------------------------------------------------------------------
# RaiSim video capture callback
# ---------------------------------------------------------------------------

class RaiSimVideoCallback(BaseCallback):
    def __init__(self, training_env: VecEnv, video_every_n_episodes: int = 5, fps: int = 30, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._venv = training_env
        self.video_every_n_episodes = max(1, int(video_every_n_episodes))
        self.fps = fps
        self._last_epoch = -1
        self._recording = False
        self._just_started = False
        self._frames: List[np.ndarray] = []
        self._video_index = 0

    #def _on_training_start(self) -> None:
    #    print(f"[DEBUG] unwrapped type: {type(self._venv.unwrapped)}")
    #    print(f"[DEBUG] envs[0] type: {type(self._venv.unwrapped.envs[0])}")
    #    print(f"[DEBUG] envs[0].unwrapped type: {type(self._venv.unwrapped.envs[0].unwrapped)}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        # Trigger recording at the start of each new epoch (every 8192 steps)
        epoch_size = 2048 * 4  # n_steps * num_envs
        current_epoch = self.num_timesteps // (epoch_size * 60)  # every 60 epochs
        if current_epoch > self._last_epoch:
            self._last_epoch = current_epoch
            if not self._recording:
                self._recording = True
                self._frames = []
                if self.verbose > 0:
                    print(f"[RaiSimVideoCallback] Recording epoch {current_epoch}...")

        if self._recording:
            self._collect_frame()
            if len(self._frames) >= 200:
                self._recording = False
                self._encode_and_upload(self._frames)
                self._frames = []
                self._video_index += 1

        return True

    def _collect_frame(self) -> None:
        try:
            base_env = self._venv.unwrapped.envs[0].unwrapped
            drone_pos = base_env.get_drone_pos()
            client_positions = base_env.get_client_positions()
            delivered_mask = base_env.get_delivered_mask()
            fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.set_facecolor("#111111"); fig.patch.set_facecolor("#111111")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#444444")
            for i, (pos, done) in enumerate(zip(client_positions, delivered_mask)):
                color = "#00ff00" if done else "#ff4444"
                ax.plot(pos[0], pos[1], "o", color=color, ms=10)
                ax.text(pos[0], pos[1] + 0.3, f"C{i}", color="white", fontsize=6, ha="center")
            ax.plot(drone_pos[0], drone_pos[1], "c^", ms=12, zorder=5)
            ax.set_title(f"step {self.num_timesteps:,}", color="white", fontsize=8)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            plt.close(fig)
            self._frames.append(frame.copy())
        except Exception as exc:
            print(f"[RaiSimVideoCallback] Frame capture FAILED: {exc}")  # always print

    def _encode_and_upload(self, frames: List[np.ndarray]) -> None:
        if not frames or not (_WANDB_AVAILABLE and wandb.run is not None):
            return
        try:
            import imageio
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            writer = imageio.get_writer(tmp_path, fps=self.fps, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            caption = f"step {self.num_timesteps:,} | clip #{self._video_index + 1}"
            wandb.log(
                {"video/raisim_episode": wandb.Video(tmp_path, fps=self.fps, format="mp4", caption=caption)},
                step=self.num_timesteps,
            )
            print(f"[RaiSimVideoCallback] Uploaded clip #{self._video_index + 1} ({len(frames)} frames).")
        except Exception as exc:
            print(f"[RaiSimVideoCallback] Video upload failed: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def start_recording(self) -> None:
        self._recording = True
        self._just_started = True
        self._frames = []

    def stop_and_upload(self) -> None:
        self._recording = False
        if self._frames:
            self._encode_and_upload(self._frames)
            self._frames = []
            self._video_index += 1


# ---------------------------------------------------------------------------
# Energy phasing
# ---------------------------------------------------------------------------

class CustomCheckpointCallback(CheckpointCallback):
    """CheckpointCallback with a W&B log line on each save.

    Also saves VecNormalize stats alongside each periodic checkpoint (named
    to exactly match train.py's resume lookup: '{checkpoint_stem}_norm.pkl').
    Without this, resuming from a periodic checkpoint (as opposed to
    best_model/final_model, which already got norm stats via
    SaveVecNormalizeCallback / env.save() at end of training) silently falls
    back to *fresh, uninitialized* normalization stats — which feeds the
    already-trained policy wildly differently-scaled observations and causes
    a temporary success-rate collapse until the fresh stats re-converge.
    """

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            checkpoint_stem = f"{self.name_prefix}_{self.num_timesteps}_steps"
            norm_path = Path(self.save_path) / f"{checkpoint_stem}_norm.pkl"
            try:
                self.training_env.save(str(norm_path))
            except Exception as exc:
                if self.verbose > 0:
                    print(f"[Checkpoint] WARNING: failed to save VecNormalize stats: {exc}")
            _wlog(
                {"checkpoint/step": float(self.n_calls)},
                step=self.num_timesteps,
            )
            if self.verbose > 0:
                print(f"[Checkpoint] saved at step {self.n_calls:,}")
        return result
    

from stable_baselines3.common.callbacks import BaseCallback

class SaveVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize stats every time EvalCallback finds a new best model."""
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.parent is not None and hasattr(self.parent, 'best_mean_reward'):
            self.training_env.save(self.save_path)
        return True
    
class SuccessRateEvalCallback(BaseCallback):
    """Evaluates the agent periodically and saves the best model by success rate.

    Unlike SB3's built-in EvalCallback (which ranks checkpoints by mean reward),
    this callback tracks the fraction of evaluation episodes where
    ``info['episode_success']`` is True and only overwrites the saved model when
    a new personal-best success rate is achieved.

    It also saves VecNormalize statistics alongside the best model so that
    evaluation at load-time receives correctly-scaled observations.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        train_env: VecEnv,
        best_model_save_path: str,
        eval_freq: int = 25000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        """
        Args:
            eval_env: Vectorised evaluation environment (should have evaluation_mode=True).
            train_env: The VecNormalize-wrapped training environment (for stat syncing).
            best_model_save_path: Directory where best_model.zip and best_model_norm.pkl are saved.
            eval_freq: Evaluate every ``eval_freq`` training steps (across all envs).
            n_eval_episodes: Number of episodes per evaluation round.
            deterministic: Use deterministic actions during evaluation.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self._train_env = train_env
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_success_rate: float = -1.0
        self.last_eval_step: int = 0
        self._eval_count: int = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True

        self._eval_count += 1

        # Sync normalisation stats from training to eval env
        if isinstance(self._train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            sync_envs_normalization(self._train_env, self.eval_env)

        successes: List[bool] = []
        rewards: List[float] = []
        episode_lengths: List[int] = []

        obs = self.eval_env.reset()
        n_envs = self.eval_env.num_envs
        ep_rewards = np.zeros(n_envs, dtype=np.float64)
        ep_lengths = np.zeros(n_envs, dtype=np.int32)

        while len(successes) < self.n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, dones, infos = self.eval_env.step(actions)
            ep_rewards += reward
            ep_lengths += 1

            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    success = bool(info.get("episode_success", False))
                    successes.append(success)
                    rewards.append(float(ep_rewards[i]))
                    episode_lengths.append(int(ep_lengths[i]))
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

        success_rate = float(np.mean(successes))
        mean_reward = float(np.mean(rewards))
        mean_length = float(np.mean(episode_lengths))

        # Log to TensorBoard
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_length)

        if self.verbose >= 1:
            print(
                f"[Eval #{self._eval_count}] step={self.num_timesteps} | "
                f"success_rate={success_rate:.2%} | mean_reward={mean_reward:.1f} | "
                f"mean_length={mean_length:.0f} | best={self.best_success_rate:.2%}",
                flush=True,
            )

        # Save if new best
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            os.makedirs(self.best_model_save_path, exist_ok=True)
            model_path = os.path.join(self.best_model_save_path, "best_model")
            self.model.save(model_path)
            # Also save normalisation stats
            if isinstance(self._train_env, VecNormalize):
                norm_path = os.path.join(self.best_model_save_path, "best_model_norm.pkl")
                self._train_env.save(norm_path)
            if self.verbose >= 1:
                print(
                    f"  >>> NEW BEST MODEL saved! success_rate={success_rate:.2%} "
                    f"at step {self.num_timesteps}",
                    flush=True,
                )

        return True