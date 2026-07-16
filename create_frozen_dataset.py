#!usr/bin/env python3
"""
Standalone script to generate a frozen evaluation dataset.

This script should be run ONCE to create a fixed set of evaluation scenarios.
The generated dataset (evaluation/frozen_dataset.json) is then used by all
benchmarking and evaluation runs to ensure consistency.

Usage:
    python create_frozen_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from evaluation.frozen_dataset import EvaluationDataset

def main():
    """Generate and save frozen evaluation dataset."""
    
    print("="*80)
    print("FROZEN DATASET GENERATION")
    print("="*80)
    
    output_path = Path("evaluation/frozen_dataset.json")

    # check if data already exist
    if output_path.exists():
        print(f'\n warning dataset already exists at {output_path}.')
        response = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborting dataset generation.")
            return
        
    
    # generate frozen dataset
    print("\nGenerating 500 evaluation scenarios...")
    print(" - Client counts: 3-8 per scenario")
    print(" - Domain randomization: wind (on/off) x obstacles (2-5 spheres, on/off),")
    print("   cycled evenly across all four combinations")
    print(" - deterministic seeding: base_seed=42")
    print(" - format: json (human-readable)")

    # NOTE: was previously `EvaluationDataset(num_scenarios=..., ...)`, which
    # doesn't match EvaluationDataset.__init__ (it only takes `scenarios`).
    # The scenario-generating factory is the `generate()` staticmethod.
    dataset = EvaluationDataset.generate(
        num_scenarios=500,
        num_clients_range=(3, 8),
        base_seed=42,
        obstacles_k_range=(2, 5),
    )

    # save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_json(output_path)

    print(f'\n Dataset summary')
    print("-" * 80)
    summary = dataset.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n" + "="*80)
    print("✓ Frozen dataset created successfully!")
    print("="*80)
    print("\nYou can now use this dataset for all benchmark evaluations:")
    print("  python evaluation/evaluate.py --model best_model")
    print("  python evaluation/evaluate.py --compare best_model final_model")
    print("\nThe same dataset will be used automatically for all evaluations.")
    print("="*80)

if __name__ == "__main__":
    main()