import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_delivery_autonomous.env import DroneDeliveryEnv, register_env
from drone_delivery_autonomous.training.train import load_config, _reward_kwargs, _delivery_env_kwargs

def run_env(num_episodes: int = 3):
    """
    run the environment with a random policy
    """

    register_env()

    config_path = Path(__file__).parent / "configs" / "config.yaml"
    config = load_config(str(config_path))
    env_kw = _delivery_env_kwargs(config, evaluation_mode=False)
    reward_kw = _reward_kwargs(config["rewards"])

    print("=" * 70)
    print("Drone delivery environment TEst")
    print("=" * 70)
    print("\nINitializing enironment with GUI rendering...")
    print("close the pybullet winodw or press Ctrl+c to stop\n")
    env = DroneDeliveryEnv(
        gui=True,
        seed=42,
        **env_kw,
        **reward_kw
    )

    print(
        f"Config: delivery_radius={env.delivery_radius:.2f} m, "
        f"energy_phase={env.energy_phase},"
        f"shaping={'on' if env.dense_shaping_enabled else 'off'}\n"
    )

    try:
        for episode in range(num_episodes):
            print(f"\n{'=' * 70}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'=' * 70}")

            obs, info = env.reset()
            print(f"EPisode Initiliaed:")
            print(f"clients: {info['num_clients']}")
            print(f"delivery_radius: {info.get('delivery_radius', 'n/a')}")
            print(f"energy_phase: {info.get('energy_phase', 'n/a')}")
            print(f'client positons:\n{info["client_positions"] }')

            episode_reward = 0.0
            step=0
            done = False

            while not done:
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1

                env.render()

                if step % 50 == 0:
                    drone_pos = obs[:3]
                    energy = info["energy"]
                    deliveries = info["deliveries_completed"]
                    total = info["total_deliveries"]
                    sh = info.get("shaping_reward", 0.0)

                    print(
                        f"  Step {step:4d}: "
                        f"Pos=[{drone_pos[0]:6.2f}, {drone_pos[1]:6.2f}, {drone_pos[2]:6.2f}], "
                        f"Energy={energy:6.2f}%, "
                        f"Deliveries={deliveries}/{total}, "
                        f"Shaping={sh:7.4f}, "
                        f"Reward={reward:7.2f}"
                    )
            
            print(f"\nEpisode summary:")
            print(f' total reward: {episode_reward:.2f }')
            print(f'deliveries completed: {info["deliveries_completed"]}/{info["total_deliveries"]}')
            print(f"final energy: {info['energy']:.2f}%")
            print(f"total steps: {step} ")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        env.close()
        print("\nEnvironment closed. Test complete.")


if __name__ == "__main__":
    run_env(num_episodes=3)