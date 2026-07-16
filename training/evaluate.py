"""
Run one evaluation episode with the RaiSim GUI (raisimUnity) attached, so
you can visually watch the trained policy fly and deliver.

Usage (run from the repo root, e.g. ~/Downloads/drone_delivery_autonomous):

    uv run python training/evaluate.py \
        --config configs/config.yaml \
        --model models/final_model.zip \
        --norm  models/final_model_norm.pkl

Before running: launch raisimUnity and hit "Connect" to localhost:8080
*first*, then start this script — RaisimServer.launchServer() will block
briefly waiting for the client to attach.
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3.common.vec_env import VecNormalize

from drone_delivery_autonomous.training.train import create_env, load_config
from drone_delivery_autonomous.agent import PPOAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to final_model.zip / checkpoint .zip")
    parser.add_argument("--norm", type=str, default=None, help="Path to matching VecNormalize .pkl (optional but recommended)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Don't throttle to real time — run as fast as possible (hard to watch in the viewer).",
    )
    parser.add_argument("--force-wind", action="store_true", help="Guarantee wind is ON for these episode(s).")
    parser.add_argument(
        "--force-obstacles",
        type=int,
        default=None,
        metavar="MIN_K",
        help="Guarantee at least MIN_K sphere obstacles are spawned (up to the config's "
             "obstacles_max_k, which must NOT change — it fixes the observation-space size "
             "the model was trained with).",
    )
    args = parser.parse_args()

    # Matches delivery_env.py: _CTRL_EVERY (8) physics substeps at _PHYSICS_DT (1/240s)
    # per env.step() -> ~1/30s of sim-time per control step.
    CONTROL_DT = 8 * (1.0 / 240.0)

    config = load_config(args.config)

    # By default the eval env is built with for_evaluation=True, which forces
    # domain_randomization_enabled=False (see delivery_env.py) so eval numbers
    # are clean/comparable. To actually *see* wind/obstacles in the viewer, we
    # instead keep domain randomization enabled and crank the probabilities to
    # 1.0 so every reset spawns them (evaluation_mode has no other effect on
    # the env besides that one flag, so this is safe to flip here).
    #
    # IMPORTANT: obstacles_max_k is never overridden here — it fixes the size
    # of the obstacle-slot padding in the observation vector (see
    # delivery_env.py's self.max_obstacles), so it must stay exactly what the
    # loaded model/VecNormalize stats were trained with. Only obstacles_min_k
    # (a floor, clamped to not exceed the existing max_k) is adjustable.
    force_dr = args.force_wind or args.force_obstacles is not None
    if force_dr:
        dr = config.setdefault("domain_randomization", {})
        dr["enabled"] = True
        if args.force_wind:
            dr["wind_prob"] = 1.0
        if args.force_obstacles is not None:
            dr["obstacles_prob"] = 1.0
            existing_max_k = int(dr.get("obstacles_max_k", 5))
            dr["obstacles_min_k"] = min(args.force_obstacles, existing_max_k)

    # gui=True + a single env so it's a real-time, watchable rollout.
    env = create_env(config, gui=True, num_envs=1, for_evaluation=not force_dr)

    if args.norm:
        env = VecNormalize.load(args.norm, env.venv)
    env.training = False      # freeze running stats — don't keep updating them
    env.norm_reward = False   # see the *real* reward, not the normalized one

    agent = PPOAgent.load(args.model, env=env, device="auto")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            step_start = time.time()
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            if not args.no_realtime:
                elapsed = time.time() - step_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)
        print(f"[Episode {ep + 1}] steps={steps} total_reward={total_reward:.2f} info={info[0]}")

    env.close()


if __name__ == "__main__":
    main()