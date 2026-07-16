"""Unified benchmarking and evaluation script for drone delivery agents."""

import argparse
import sys
import json
from pathlib import Path
import time
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drone_delivery_autonomous.env import DroneDeliveryEnv, register_env
from drone_delivery_autonomous.agent import PPOAgent

# Import from local evaluation module
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_wrapper import BenchmarkWrapper
from frozen_dataset import EvaluationDataset, EvaluationScenario

class UnifiedBenchmark:
    """
    Unified benchmarking system for evaluating multiple algorithms
    on identical evaluation scenarios.
    """
    
    def __init__(self, models_dir: str = "models", eval_dataset_path: Optional[str] = None):
        """
        Initialize the benchmark.
        
        Args:
            models_dir: Directory containing trained models
            eval_dataset_path: Path to frozen evaluation dataset (required, defaults to evaluation/frozen_dataset.json)
        """
        self.models_dir = Path(models_dir)
        self.register_env()
        
        # Use provided path or default
        if eval_dataset_path is None:
            eval_dataset_path = "evaluation/frozen_dataset.json"
        
        dataset_path = Path(eval_dataset_path)
        
        # Load evaluation dataset (MUST exist)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Frozen evaluation dataset not found at {dataset_path}\n"
                f"Please generate it first by running:\n"
                f"  python create_frozen_dataset.py\n"
                f"This only needs to be done once."
            )

        ext = dataset_path.suffix
        if ext == ".json":
            self.dataset = EvaluationDataset.load_json(dataset_path)
        else:
            self.dataset = EvaluationDataset.load_pickle(dataset_path)
        
        print(f"\nEvaluation Dataset Summary:")
        summary = self.dataset.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    @staticmethod
    def register_env():
        """Register the DroneDeliveryEnv with Gym."""
        register_env()
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available trained models.
        
        Returns:
            List of model names
        """
        if not self.models_dir.exists():
            return []
        
        models = []
        # Check for PPO models
        best_model_path = self.models_dir / "best_model"
        if best_model_path.exists():
            models.append("best_model")
        
        final_model_path = self.models_dir / "final_model.zip"
        if final_model_path.exists():
            models.append("final_model")
        
        # Check for checkpoint models
        for checkpoint in sorted(self.models_dir.glob("checkpoint_*_steps.zip")):
            models.append(checkpoint.stem)
        
        return models
    
    def load_model(self, model_name: str):
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded PPO model and optional normalization wrapper
        """
        # If a path-like string is provided (absolute or relative), try to load it directly
        maybe_path = Path(model_name)
        model_path = None

        if maybe_path.exists():
            # If it's a directory, try to find a .zip inside or load directly if supported
            if maybe_path.is_dir():
                # Look for a same-named zip inside the directory
                internal_zip = maybe_path / (maybe_path.name + ".zip")
                if internal_zip.exists():
                    model_path = internal_zip
                else:
                    # Try to find any .zip in the directory
                    zips = list(maybe_path.glob("*.zip"))
                    if len(zips) > 0:
                        model_path = zips[0]
            else:
                # It's a file - assume it's a model file
                model_path = maybe_path

        # If still none, fall back to searching in models_dir by name
        if model_path is None:
            # Try direct .zip file in models_dir
            direct_zip = self.models_dir / f"{model_name}.zip"
            if direct_zip.exists():
                model_path = direct_zip

            # Try directory with same-named .zip inside
            model_dir = self.models_dir / model_name
            if model_dir.is_dir():
                internal_zip = model_dir / f"{model_name}.zip"
                if internal_zip.exists():
                    model_path = internal_zip

        if model_path is None:
            raise FileNotFoundError(
                f"Model not found: {model_name}\n"
                f"Tried paths relative to models_dir and the literal path provided."
            )

        print(f"Loading model from {model_path}...")

        # Load model
        model = PPO.load(str(model_path))

        norm_stats_path = None
        for candidate in [
            Path(str(model_path).replace(".zip", "")) / "vecnormalize.pkl",
            Path("models") / "best_model" / "vecnormalize.pkl",
            Path("models") / "vecnormalize_final.pkl",
        ]:
            if candidate.exists():
                norm_stats_path = candidate
                print(f"Found VecNormalize stats at {norm_stats_path}")
                break

        if norm_stats_path is None:
            print("WARNING: No VecNormalize stats found — observations will not be normalized!")

        return model, norm_stats_path    
    
    def evaluate_scenario(
        self,
        model: PPO,
        scenario: EvaluationScenario,
        render: bool = False,
        env=None,
        obs_mean=None,
        obs_var=None,
        clip_obs=10.0,
    ) -> Dict[str, Any]:
        """Run one scenario. env must be a BenchmarkWrapper(DroneDeliveryEnv)."""
 
        created_locally = env is None
        if created_locally:
            env = BenchmarkWrapper(DroneDeliveryEnv(
                gui=render,
                num_clients_min=scenario.num_clients,
                num_clients_max=scenario.num_clients,
                max_clients=8,
                arena_size=10.0,
                delivery_altitude=1.0,
                delivery_radius=0.5,
                max_speed=2.0,
                initial_energy=scenario.initial_energy,
                base_drain=0.005,
                speed_coefficient=0.020,
                max_episode_steps=2000,
                evaluation_mode=True,
            ))
 
        # Apply this scenario's wind intensity (best-effort — only meaningful
        # when wind_on, but harmless to set either way).
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if scenario.wind_profile is not None and hasattr(base_env, "wind_model"):
            base_env.wind_model.volatility = float(scenario.wind_profile.get("volatility", base_env.wind_model.volatility))
            base_env.wind_model.mean_reversion = float(scenario.wind_profile.get("mean_reversion", base_env.wind_model.mean_reversion))
            base_env.wind_model.max_speed = float(scenario.wind_profile.get("max_speed", base_env.wind_model.max_speed))

        # IMPORTANT: reset with THIS scenario's clients + seed + wind/obstacles every time
        obs, info = env.reset(
            seed=scenario.seed,
            options={
                "client_positions": scenario.client_positions,
                "wind_on": scenario.wind_on,
                "obstacle_positions": scenario.obstacle_positions,
                "obstacle_radii": scenario.obstacle_radii,
            },
        )
 
        done = False
        step_count = 0
 
        while not done and step_count < 2500:
            # Normalize observation the same way VecNormalize does during training
            if obs_mean is not None and obs_var is not None:
                obs_n = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
                obs_n = np.clip(obs_n, -clip_obs, clip_obs).astype(np.float32)
            else:
                obs_n = obs.astype(np.float32)
 
            # model.predict expects (1, obs_dim) — add batch dim
            action, _ = model.predict(obs_n[np.newaxis], deterministic=True)
            action = action[0]  # remove batch dim before env.step
 
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
 
            if render:
                time.sleep(0.02)
 
        metrics = env.compute_benchmark_metrics()
        metrics["scenario_id"] = scenario.scenario_id
        metrics["num_clients"] = scenario.num_clients
 
        if created_locally:
            env.close()
 
        return metrics
 
    def benchmark_model(self, model_name: str, render: bool = False) -> Dict[str, Any]:
        import pickle
 
        model, norm_stats_path = self.load_model(model_name)
 
        # Load normalization stats ONCE
        obs_mean, obs_var, clip_obs = None, None, 10.0
        if norm_stats_path is not None and Path(str(norm_stats_path)).exists():
            with open(norm_stats_path, "rb") as f:
                vn = pickle.load(f)
            obs_mean = vn.obs_rms.mean.copy()
            obs_var  = vn.obs_rms.var.copy()
            clip_obs = float(getattr(vn, "clip_obs", 10.0))
            print(f"VecNormalize loaded: obs shape={obs_mean.shape}, clip={clip_obs}")
        else:
            print("WARNING: No VecNormalize stats — obs not normalized!")
 
        print(f"\nBenchmarking model: {model_name}")
        print(f"Running {len(self.dataset)} evaluation scenarios...\n")
 
        results = []
 
        # Create ONE persistent env (avoids repeated RaiSim init crashes)
        # Use max_clients config so it can handle any scenario size
        shared_env = BenchmarkWrapper(DroneDeliveryEnv(
            gui=render,
            num_clients_min=3,
            num_clients_max=8,      # wide range; reset() pins exact count
            max_clients=8,
            arena_size=10.0,
            delivery_altitude=1.0,
            delivery_radius=0.5,
            max_speed=2.0,
            initial_energy=100.0,  # overridden per scenario via reset seed
            base_drain=0.01,
            speed_coefficient=0.05,
            max_episode_steps=2000,
            evaluation_mode=True,
        ))
 
        for i, scenario in enumerate(self.dataset):
            # ← REMOVED: if i >= 1: break  (was limiting to 1 scenario!)
            if (i + 1) % 5 == 0 or i == 0:
                print(f"[{i+1}/{len(self.dataset)}] Scenario {scenario.scenario_id} "
                      f"({scenario.num_clients} clients)...", flush=True)
            try:
                metrics = self.evaluate_scenario(
                    model, scenario,
                    render=render,
                    env=shared_env,
                    obs_mean=obs_mean,
                    obs_var=obs_var,
                    clip_obs=clip_obs,
                )
                results.append(metrics)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  ✓ steps={metrics.get('episode_length','?')} "
                          f"success={metrics.get('success','?')} "
                          f"deliveries={metrics.get('deliveries_completed','?')}/"
                          f"{metrics.get('total_deliveries','?')} "
                          f"energy={metrics.get('energy_consumed',0):.1f}", flush=True)
            except Exception as e:
                print(f"  ✗ Scenario {scenario.scenario_id}: {e}")
                import traceback; traceback.print_exc()
                results.append({"scenario_id": scenario.scenario_id, "error": str(e)})
 
        try:
            shared_env.close()
        except Exception:
            pass
 
        print(f"\n✓ Done! {len(results)} scenarios.\n")
        df_results = pd.DataFrame(results)
        summary = self._compute_summary_statistics(df_results, model_name)
        return {
            "model_name": model_name,
            "num_scenarios": len(results),
            "detailed_results": df_results,
            "summary_statistics": summary,
        }

    @staticmethod
    def _compute_summary_statistics(df: pd.DataFrame, model_name: str) -> Dict[str, float]:
        """
        Compute summary statistics from benchmark results.
        
        Args:
            df: DataFrame with benchmark results
            model_name: Name of the model
            
        Returns:
            Dictionary with summary statistics
        """
        # Filter out error rows
        df = df[~df.isnull().any(axis=1)]
        
        if len(df) == 0:
            return {"error": "No valid results"}
        
        return {
            "success_rate": float(df["success"].mean()),
            "mean_reward": float(df["total_reward"].mean()),
            "std_reward": float(df["total_reward"].std()),
            "mean_energy_efficiency": float(df["energy_efficiency"].mean()),
            "mean_deliveries_per_energy": float(df["deliveries_per_energy"].mean()),
            "mean_time_per_delivery": float(df["time_per_delivery"].mean()),
            "mean_energy_consumed": float(df["energy_consumed"].mean()),
            "mean_episode_length": float(df["episode_length"].mean()),
        }
    
    def compare_models(self, model_names: List[str], render: bool = False) -> pd.DataFrame:
        """
        Compare multiple models and return unified results.
        
        Args:
            model_names: List of model names to compare
            render: Whether to render episodes
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name in model_names:
            try:
                benchmark_result = self.benchmark_model(model_name, render=render)
                summary = benchmark_result["summary_statistics"]
                summary["model_name"] = model_name
                comparison_results.append(summary)
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
        
        df_comparison = pd.DataFrame(comparison_results)
        return df_comparison
    
    def generate_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Optional[Path] = None,
    ):
        """
        Generate a comprehensive benchmark report.
        
        Args:
            benchmark_results: Results from benchmark_model()
            output_path: Path to save report (optional)
        """
        model_name = benchmark_results["model_name"]
        summary = benchmark_results["summary_statistics"]
        df_results = benchmark_results["detailed_results"]
        
        # Console report
        print("\n" + "="*80)
        print(f"BENCHMARK REPORT: {model_name}")
        print("="*80)
        print(f"\nTotal Scenarios Evaluated: {benchmark_results['num_scenarios']}")
        print("\n--- Summary Statistics ---")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Success breakdown by difficulty
        print("\n--- Success Rate by Difficulty ---")
        easy = df_results[df_results["num_clients"] == 3]
        medium = df_results[(df_results["num_clients"] >= 4) & (df_results["num_clients"] <= 5)]
        hard = df_results[df_results["num_clients"] >= 6]
        
        if len(easy) > 0:
            print(f"  Easy (3 clients): {easy['success'].mean():.1%}")
        if len(medium) > 0:
            print(f"  Medium (4-5 clients): {medium['success'].mean():.1%}")
        if len(hard) > 0:
            print(f"  Hard (6-8 clients): {hard['success'].mean():.1%}")
        
        print("\n" + "="*80)
        
        # Save JSON report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report_data = {
                "model_name": model_name,
                "num_scenarios": benchmark_results["num_scenarios"],
                "summary_statistics": summary,
                "results_by_scenario": df_results.to_dict(orient="records"),
            }
            
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)
            print(f"\nReport saved to {output_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Unified benchmarking for drone delivery agents"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to evaluate (if None, lists available models)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a model file or directory to evaluate (overrides --model)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Compare multiple models",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to frozen evaluation dataset",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the evaluation episodes via RaiSim",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save benchmark report JSON",
    )
   
    args = parser.parse_args()

    # Initialize benchmark
    benchmark = UnifiedBenchmark(
        models_dir=args.models_dir,
        eval_dataset_path=args.dataset,
    )

    # List available models
    available_models = benchmark.get_available_models()
    print(f"\nAvailable models: {available_models}")

    # If no model specified, default to best_model
    if args.model is None and not args.compare and args.model_path is None:
        print("No model specified. Defaulting to 'best_model'.")
        args.model = "best_model"

    if args.render:
        print("\n⚠️ RENDERING MODE: Rendering all scenarios sequentially")
        print("   (RaiSim GUI will display the evaluations)")

    if args.compare:
        # Compare multiple models
        print(f"\nComparing models: {args.compare}")
        df_comparison = benchmark.compare_models(args.compare, render=args.render)
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(df_comparison.to_string())

        if args.report:
            df_comparison.to_csv(args.report, index=False)
            print(f"\nComparison saved to {args.report}")

    elif args.model:
        # Benchmark single model
        # Allow overriding by explicit model path
        model_to_eval = args.model_path if args.model_path is not None else args.model

        try:
            result = benchmark.benchmark_model(model_to_eval, render=args.render)
            benchmark.generate_report(result, output_path=args.report)
        except Exception as e:
            print(f"Error benchmarking model '{model_to_eval}': {e}")
            return

        # Remove test_best_model.py from project root after completion (user requested)
        try:
            test_script = Path(__file__).parent.parent / "test_best_model.py"
            if test_script.exists():
                test_script.unlink()
                print(f"Removed {test_script}")
        except Exception as e:
            print(f"Warning: could not remove test_best_model.py: {e}")

    else:
        # No model specified, show usage
        print("\nUsage:")
        print("  Evaluate single model (no rendering):")
        print(f"    python evaluate.py --model best_model")
        print("\n  Evaluate and render all scenarios sequentially:")
        print(f"    python evaluate.py --model best_model --render")
        print("\n  Compare multiple models:")
        print(f"    python evaluate.py --compare best_model final_model checkpoint_1")
        print("\n  Generate report:")
        print(f"    python evaluate.py --model best_model --report reports/best_model_report.json")
        print("\n  Use custom dataset:")
        print(f"    python evaluate.py --model best_model --dataset custom_dataset.pkl")


if __name__ == "__main__":
    main()