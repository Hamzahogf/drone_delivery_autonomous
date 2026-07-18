import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from stable_baselines3.common.monitor import Monitor
import yaml
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.ppo import linear_schedule
from drone_delivery_autonomous.env import DroneDeliveryEnv, register_env
from drone_delivery_autonomous.agent import PPOAgent, SACAgent, REINFORCEAgent
from drone_delivery_autonomous.training.callbacks import (
    WandbMetricsCallback,
    RaiSimVideoCallback,
    CustomCheckpointCallback,
    SuccessRateEvalCallback,
)

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — metrics will only be printed to stdout.")


"""
Training entry-point for the drone-delivery RL project (single-stage).

This replaces the old multi-stage curriculum (CurriculumCallback,
EnergyPhasingCallback, AdaptiveRadiusCallback, EnergyDrainStagingCallback,
WindAnnealCallback) with single-stage training at full difficulty from
step 0, plus per-episode domain randomization of wind and sphere obstacles
(handled inside DroneDeliveryEnv.reset() itself — no callback needed).

Removed vs. the previous version (all were either curriculum-only or
already broken / never defined anywhere in this file):
* --resume-stage, --wind-anneal, --baseline-report CLI flags
* _infer_callback_state (curriculum-only, no longer meaningful)
* TensorBoardCallback, NormSavingCheckpointCallback, BaselineReferenceCallback
  (referenced but never defined — this file could not have run to completion
  before this rewrite)
* SuccessRateEvalCallback is now actually imported (it was used at the
  bottom of this file previously but never imported — a NameError waiting
  to happen)
"""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return a YAML config file as a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _reward_kwargs(rewards: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the reward fields consumed by DroneDeliveryEnv.__init__."""
    keys = (
        "delivery_reward",
        "completion_bonus",
        "energy_bonus_coeff",
        "failure_penalty",
        "out_of_bounds_penalty",
        "dense_shaping_enabled",
        "shaping_coeff",
    )
    return {k: rewards[k] for k in keys if k in rewards}


def _delivery_env_kwargs(
    config: dict, evaluation_mode: bool = False
) -> Dict[str, Any]:
    """
    Build keyword arguments for DroneDeliveryEnv from the YAML config.

    Single-stage: delivery_radius / base_drain / speed_coefficient are fixed
    (no annealing). Domain randomization (wind + obstacles) is disabled in
    evaluation_mode so eval metrics come from the frozen dataset's explicit
    per-scenario wind/obstacle settings instead of a random draw.
    """
    env_c = config["env"]
    dr_c  = config.get("domain_randomization") or {}

    return {
        "num_clients_min":   env_c["num_clients_min"],
        "num_clients_max":   env_c["num_clients_max"],
        "max_clients":       env_c["max_clients"],
        "arena_size":        env_c["arena_size"],
        "delivery_radius":   env_c["delivery_radius"],
        "delivery_altitude": env_c["delivery_altitude"],
        "max_speed":         env_c["max_speed"],
        "initial_energy":    env_c["initial_energy"],
        "base_drain":        env_c["base_drain"],
        "speed_coefficient": env_c["speed_coefficient"],
        "max_episode_steps": env_c["max_episode_steps"],
        "evaluation_mode":   evaluation_mode,
        # ── Domain randomization (wind + obstacles) ──────────────────────
        "domain_randomization_enabled": bool(dr_c.get("enabled", True)),
        "wind_prob":               float(dr_c.get("wind_prob", 0.5)),
        "obstacles_prob":          float(dr_c.get("obstacles_prob", 0.5)),
        "wind_volatility":         float(dr_c.get("wind_volatility", 0.5)),
        "wind_mean_reversion":     float(dr_c.get("wind_mean_reversion", 0.1)),
        "wind_max_speed":          float(dr_c.get("wind_max_speed", 3.0)),
        "obstacles_min_k":         int(dr_c.get("obstacles_min_k", 2)),
        "obstacles_max_k":         int(dr_c.get("obstacles_max_k", 5)),
        "obstacle_radius_min":     float(dr_c.get("obstacle_radius_min", 0.3)),
        "obstacle_radius_max":     float(dr_c.get("obstacle_radius_max", 0.8)),
        "obstacle_collision_penalty": float(dr_c.get("obstacle_collision_penalty", -50.0)),
    }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def create_env(
    config: dict,
    gui: bool = False,
    num_envs: int = 1,
    log_dir: Optional[str] = None,
    for_evaluation: bool = False,
) -> VecNormalize:
    """
    Create a vectorised, normalised DroneDeliveryEnv.

    Returns
    -------
    VecNormalize
        Wrapped environment ready for SB3 training / evaluation.
    """
    register_env()

    env_kwargs = _delivery_env_kwargs(config, evaluation_mode=for_evaluation)
    reward_kw  = _reward_kwargs(config["rewards"])

    def make_env(rank: int = 0):
        def _init():
            env = DroneDeliveryEnv(gui=gui, **env_kwargs, **reward_kw)
            if log_dir is not None:
                monitor_file = Path(log_dir) / f"monitor_{rank}.csv"
                env = Monitor(env, str(monitor_file))
            return env
        return _init

    vec_env = DummyVecEnv([make_env(rank=i) for i in range(num_envs)])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0, # attention
        gamma=float(config["agent"].get("gamma", 0.995)),
    )
    return vec_env


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    resume_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_dir_override: Optional[str] = None,
    resume_lr: Optional[float] = None,
    wandb_run_id: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> None:
    config = load_config(config_path)

    training_config = config["training"]
    wandb_config    = config.get("wandb", {})

    algo = (algorithm or config.get("algorithm", "ppo")).lower()
    if algo not in ("ppo", "sac", "reinforce"):
        print(f"[ERROR] Unknown --algo '{algo}' — must be 'ppo', 'sac', or 'reinforce'.")
        sys.exit(1)
    agent_config = {"ppo": config.get("agent", {}), "sac": config.get("sac", {}),
                    "reinforce": config.get("reinforce", {})}[algo]
    if algo != "ppo" and not agent_config:
        print(f"[ERROR] algorithm='{algo}' but config.yaml has no '{algo}:' section.")
        sys.exit(1)

    if output_dir is not None:
        training_config["log_dir"] = str(Path(output_dir) / "logs")
        training_config["model_dir"] = str(Path(output_dir) / "models")

    # --model-dir applied after --output-dir so it wins if both are given.
    if model_dir_override is not None:
        training_config["model_dir"] = str(Path(model_dir_override))

    if total_timesteps is not None:
        training_config["total_timesteps"] = total_timesteps

    if seed is not None:
        training_config["seed"] = seed

    np.random.seed(training_config["seed"])

    # ── Directories ──────────────────────────────────────────────────────
    log_dir   = Path(training_config["log_dir"])
    model_dir = Path(training_config["model_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B initialisation ───────────────────────────────────────────────
    # Must happen BEFORE any callback is constructed so that wandb.run is
    # available when callbacks call wandb.log().
    run = None
    if _WANDB_AVAILABLE:
        run = wandb.init(
            project=wandb_config.get("project", "drone-delivery-rl"),
            entity=wandb_config.get("entity") or None,
            name=wandb_config.get("run_name") or None,
            tags=list(wandb_config.get("tags", [])) + [algo],
            id=wandb_run_id or None,
            resume="allow" if wandb_run_id else None,
            config={
                "algorithm":           algo,
                "env":                 config.get("env", {}),
                "domain_randomization": config.get("domain_randomization", {}),
                "rewards":             config.get("rewards", {}),
                "agent":               agent_config,
                "training":            training_config,
                "seed":                training_config["seed"],
                "total_timesteps":     training_config["total_timesteps"],
            },
            sync_tensorboard=False,   # we push metrics manually via wandb.log()
            monitor_gym=False,
            save_code=True,
        )
        print(f"W&B run initialised: {run.url}")
    else:
        print("W&B not available — running without experiment tracking.")

    num_envs = int(training_config.get("num_envs", 8))

    print("=" * 70)
    print(f"DRONE DELIVERY RL - TRAINING (single-stage, domain randomization, algo={algo.upper()})")
    print("=" * 70)
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Parallel environments: {num_envs}")
    print(f"Network architecture: {agent_config['policy_hidden_sizes']}")
    print(f"Learning rate: {agent_config['learning_rate']} -> {agent_config.get('learning_rate_end', 5e-5)}")
    if algo == "ppo":
        print(f"Batch size: {agent_config['batch_size']}")
        print(f"n_steps: {agent_config['n_steps']}")
        print(f"Entropy coeff: {agent_config['ent_coef']}")
    elif algo == "sac":
        print(f"Batch size: {agent_config['batch_size']}")
        print(f"Buffer size: {agent_config.get('buffer_size', 1_000_000):,}")
        print(f"Learning starts: {agent_config.get('learning_starts', 10_000):,}")
        print(f"Entropy coeff: {agent_config.get('ent_coef', 'auto')}")
    else:  # reinforce
        print(f"n_steps (full-batch per update): {agent_config.get('n_steps', 2048)}")
        print(f"gae_lambda: {agent_config.get('gae_lambda', 1.0)} (1.0 = true Monte-Carlo return)")
        print(f"Entropy coeff: {agent_config.get('ent_coef', 0.0)}")
    print(f"Gamma: {agent_config['gamma']}")
    dr_c = config.get("domain_randomization") or {}
    print(
        f"Domain randomization: enabled={dr_c.get('enabled', True)} | "
        f"wind_prob={dr_c.get('wind_prob', 0.5)} | "
        f"obstacles_prob={dr_c.get('obstacles_prob', 0.5)} | "
        f"obstacles K∈[{dr_c.get('obstacles_min_k', 2)},{dr_c.get('obstacles_max_k', 5)}]"
    )
    print()

    # ── Environments ─────────────────────────────────────────────────────
    print("Creating training environment...")
    env = create_env(
        config, gui=False, num_envs=num_envs, log_dir=str(log_dir)
    )

    # Build learning rate schedule
    lr_start = float(agent_config["learning_rate"])
    lr_end = float(agent_config.get("learning_rate_end", 5e-5))
    lr_schedule = linear_schedule(lr_start, lr_end)

    # ── Resume from checkpoint ────────────────────────────────────────────
    if resume_path is not None:
        p = Path(resume_path)
        if not p.exists() and not p.with_suffix(".zip").exists():
            print(f"[ERROR] Resume path does not exist: {resume_path}")
            sys.exit(1)

        resume_path = str(p.resolve())
        print(f"Resuming from checkpoint: {resume_path}")

        if resume_lr is not None:
            # Restart the LR decay from resume_lr instead of continuing the
            # original schedule's progress (useful after a long pause/tweak).
            print(f"[Resume] Overriding LR schedule: start={resume_lr}, end={lr_end}")

            def resume_lr_schedule(progress_remaining: float) -> float:
                return lr_end + (resume_lr - lr_end) * progress_remaining

            final_lr_schedule = resume_lr_schedule
        else:
            final_lr_schedule = lr_schedule

        agent_cls = {"ppo": PPOAgent, "sac": SACAgent, "reinforce": REINFORCEAgent}[algo]
        agent = agent_cls.load(
            resume_path,
            env=env,
            device="auto",
            custom_objects={"learning_rate": final_lr_schedule},
        )
        agent.model.tensorboard_log = str(log_dir)
        agent.model.verbose = 0  # quiet — progress bar + W&B/CSV logs cover this instead

        # Try to restore VecNormalize statistics saved alongside the checkpoint.
        checkpoint_stem = Path(resume_path).stem
        norm_candidates = [
            Path(resume_path).parent / f"{checkpoint_stem}_norm.pkl",
            model_dir / "final_model_norm.pkl",
            model_dir / "final_model_norm",
        ]
        for norm_path in norm_candidates:
            if norm_path.exists():
                print(f"Loading normalisation stats from: {norm_path}")
                env = VecNormalize.load(str(norm_path), env.venv)
                agent.set_env(env)
                break
        else:
            print("[WARN] No normalisation stats found – continuing with fresh stats.")
    else:
        if algo == "ppo":
            agent = PPOAgent(
                env=env,
                learning_rate=lr_schedule,
                n_steps=agent_config["n_steps"],
                batch_size=agent_config["batch_size"],
                n_epochs=agent_config["n_epochs"],
                gamma=agent_config["gamma"],
                gae_lambda=agent_config["gae_lambda"],
                clip_range=agent_config["clip_range"],
                ent_coef=agent_config["ent_coef"],
                vf_coef=agent_config["vf_coef"],
                max_grad_norm=agent_config["max_grad_norm"],
                policy_hidden_sizes=agent_config["policy_hidden_sizes"],
                verbose=0,  # quiet — progress bar + W&B/CSV logs cover this instead
            )
        elif algo == "sac":
            agent = SACAgent(
                env=env,
                learning_rate=lr_schedule,
                buffer_size=agent_config.get("buffer_size", 1_000_000),
                learning_starts=agent_config.get("learning_starts", 10_000),
                batch_size=agent_config["batch_size"],
                tau=agent_config.get("tau", 0.005),
                gamma=agent_config["gamma"],
                train_freq=agent_config.get("train_freq", 1),
                gradient_steps=agent_config.get("gradient_steps", 1),
                ent_coef=agent_config.get("ent_coef", "auto"),
                target_update_interval=agent_config.get("target_update_interval", 1),
                target_entropy=agent_config.get("target_entropy", "auto"),
                use_sde=agent_config.get("use_sde", False),
                policy_hidden_sizes=agent_config["policy_hidden_sizes"],
                verbose=0,  # quiet — progress bar + W&B/CSV logs cover this instead
            )
        else:  # reinforce
            agent = REINFORCEAgent(
                env=env,
                learning_rate=lr_schedule,
                n_steps=agent_config.get("n_steps", 2048),
                gamma=agent_config["gamma"],
                gae_lambda=agent_config.get("gae_lambda", 1.0),
                ent_coef=agent_config.get("ent_coef", 0.0),
                vf_coef=agent_config.get("vf_coef", 0.5),
                max_grad_norm=agent_config.get("max_grad_norm", 0.5),
                normalize_advantage=agent_config.get("normalize_advantage", True),
                policy_hidden_sizes=agent_config["policy_hidden_sizes"],
                verbose=0,  # quiet — progress bar + W&B/CSV logs cover this instead
            )

    # ── Callbacks ────────────────────────────────────────────────────────
    print("Setting up callbacks...")

    wandb_metrics_cb = WandbMetricsCallback(verbose=1)

    video_every = int(wandb_config.get("video_every_n_episodes", 5))
    video_fps   = int(wandb_config.get("video_fps", 30))
    video_cb = RaiSimVideoCallback(
        training_env=env,
        video_every_n_episodes=video_every,
        fps=video_fps,
        verbose=1,
    )

    eval_env = create_env(
        config,
        gui=False,
        num_envs=1,
        log_dir=str(log_dir / "eval"),
        for_evaluation=True,
    )

    eval_callback = SuccessRateEvalCallback(
        eval_env=eval_env,
        train_env=env,
        best_model_save_path=str(model_dir / "best_model"),
        eval_freq=training_config["eval_freq"],
        n_eval_episodes=training_config["eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CustomCheckpointCallback(
        save_freq=training_config["checkpoint_freq"],
        save_path=str(model_dir),
        name_prefix="checkpoint",
        verbose=1,
    )

    callbacks = CallbackList(
        [
            wandb_metrics_cb,
            video_cb,
            eval_callback,
            checkpoint_callback,
        ]
    )
    print("Registered callbacks: WandbMetrics, RaiSimVideo, SuccessRateEval, Checkpoint")

    # Train
    resuming = resume_path is not None
    if resuming:
        already_done = agent.num_timesteps
        remaining = training_config["total_timesteps"] - already_done
        print(f"\nResuming – already completed {already_done:,} steps.")
        print(f"Remaining steps: {remaining:,}")
    else:
        remaining = training_config["total_timesteps"]
        print(f"\nStarting training for {remaining:,} timesteps...")
    print("=" * 70)
    agent.learn(
        total_timesteps=remaining,
        callback=callbacks,
        log_interval=1,
        reset_num_timesteps=not resuming,
        progress_bar=True,
    )

    # Save final model
    print(f"\nSaving final model to {model_dir / 'final_model'}...")
    agent.save(str(model_dir / "final_model"))
    env.save(str(model_dir / "final_model_norm"))

    print("Training complete!")
    env.close()
    eval_env.close()
    if run is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drone delivery agent (single-stage)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to a checkpoint .zip file to resume training from (default: None)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for logs and models (sets log_dir=<dir>/logs, model_dir=<dir>/models)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        metavar="MODEL_DIR",
        help=(
            "Override the model save directory only (checkpoints, best model, final model). "
            "Does not affect the log directory. "
            "Applied after --output-dir, so both flags can be combined."
        ),
    )
    parser.add_argument(
        "--resume-lr",
        type=float,
        default=None,
        help="Optional: restart the LR schedule from this value on resume (e.g. 1e-4)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "W&B run id to resume logging into (the id in the run URL, e.g. "
            "'3oidznqe'). Pass the same id every time you --resume a checkpoint "
            "so all metrics land in one continuous W&B run instead of a new one."
        ),
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["ppo", "sac", "reinforce"],
        help="Which RL algorithm to train with. Overrides config.yaml's top-level "
             "'algorithm' key if both are given (default: config value, or 'ppo').",
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        resume_path=args.resume,
        output_dir=args.output_dir,
        model_dir_override=args.model_dir,
        resume_lr=args.resume_lr,
        wandb_run_id=args.wandb_run_id,
        algorithm=args.algo,
    )