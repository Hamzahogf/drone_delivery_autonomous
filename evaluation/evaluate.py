"""
Run one evaluation episode with the RaiSim GUI (raisimUnity) attached, so
you can visually watch the trained policy fly and deliver.
Usage (run from the repo root, e.g. ~/Downloads/drone_delivery_autonomous):

    uv run python training/evaluate.py --config configs/config.yaml --model models/final_model.zip --norm  models/final_model_norm.pkl --episodes 50 --no-gui --force-wind --force-obstacles 4

Before running: launch raisimUnity and hit "Connect" to localhost:8080
*first*, then start this script — RaisimServer.launchServer() will block
briefly waiting for the client to attach.
"""

import sys
import time
import json
import argparse
from datetime import datetime, timezone
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
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip RaisimServer/viewer entirely — use for bulk runs (many episodes) where you "
             "just want the report, not to watch. Runs at full speed regardless of --no-realtime.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="PATH",
        help="Where to write the JSON results report. Defaults to "
             "reports/eval_report_<timestamp>.json.",
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

    # gui=True + a single env so it's a real-time, watchable rollout. Skipped
    # entirely with --no-gui for bulk report runs (no viewer, full speed).
    env = create_env(config, gui=not args.no_gui, num_envs=1, for_evaluation=not force_dr)

    if args.norm:
        env = VecNormalize.load(args.norm, env.venv)
    env.training = False      # freeze running stats — don't keep updating them
    env.norm_reward = False   # see the *real* reward, not the normalized one

    agent = PPOAgent.load(args.model, env=env, device="auto")

    episode_results = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = [{}]
        while not done:
            step_start = time.time()
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            if not args.no_realtime and not args.no_gui:
                elapsed = time.time() - step_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)

        ep_info = dict(info[0])
        ep_info.pop("terminal_observation", None)  # huge array, not report-worthy
        result = {
            "episode": ep + 1,
            "steps": steps,
            "total_reward": total_reward,
            "success": bool(ep_info.get("episode_success", False)),
            "termination_reason": ep_info.get("termination_reason"),
            "deliveries_completed": ep_info.get("deliveries_completed"),
            "total_deliveries": ep_info.get("total_deliveries"),
            "energy_remaining": ep_info.get("energy"),
            "wind_on": ep_info.get("wind_on"),
            "obstacles_on": ep_info.get("obstacles_on"),
            "num_obstacles": ep_info.get("num_obstacles"),
        }
        episode_results.append(result)
        print(f"[Episode {ep + 1}/{args.episodes}] steps={steps} total_reward={total_reward:.2f} "
              f"success={result['success']} deliveries={result['deliveries_completed']}/{result['total_deliveries']}")

    env.close()

    # ---- Aggregate summary ----
    n = len(episode_results)
    successes = sum(1 for r in episode_results if r["success"])
    success_rate = 100.0 * successes / n if n else 0.0
    mean_reward = sum(r["total_reward"] for r in episode_results) / n if n else 0.0
    mean_length = sum(r["steps"] for r in episode_results) / n if n else 0.0
    energies = [r["energy_remaining"] for r in episode_results if r["energy_remaining"] is not None]
    mean_energy_remaining = sum(energies) / len(energies) if energies else None

    obstacle_eps = [r for r in episode_results if r["obstacles_on"]]
    if obstacle_eps:
        collisions = sum(1 for r in obstacle_eps if r["termination_reason"] == "collision")
        collision_rate = round(100.0 * collisions / len(obstacle_eps), 2)
    else:
        collision_rate = "--"  # no obstacle episodes to measure collisions against

    summary = {
        "num_episodes": n,
        "success_rate_pct": round(success_rate, 2),
        "collision_rate_pct": collision_rate,
        "mean_reward": round(mean_reward, 3),
        "mean_episode_length": round(mean_length, 1),
        "mean_energy_remaining": round(mean_energy_remaining, 2) if mean_energy_remaining is not None else None,
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print("=" * 60)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "norm": args.norm,
        "config": args.config,
        "force_wind": args.force_wind,
        "force_obstacles": args.force_obstacles,
        "summary": summary,
        "episodes": episode_results,
    }

    if args.report:
        report_path = Path(args.report)
    else:
        model_stem = Path(args.model).stem
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = Path("reports") / f"eval_report_{model_stem}_{ts}.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()