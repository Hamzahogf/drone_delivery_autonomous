"""
Build comparison bar charts from the eval_report JSON files in `reports/`.

Expects files named exactly: {algo}_{variant}_{scenario}.json
    algo     : ppo | reinforce  (or sac, if you add it later)
    variant  : free | obstacles     (which model was trained on — domain-
               randomization-free vs. obstacle-trained)
    scenario : vanilla | wind | obstacles | combined

e.g. ppo_free_combined.json, reinforce_obstacles_wind.json ...

Produces two PNGs:
    success_rate_comparison.png   — grouped bars, one group per scenario
                                     (vanilla/wind/obstacles/combined), 4 bars
                                     per group (ppo/reinforce x free/obstacles)
    collision_rate_comparison.png — same grouping, but only for the
                                     "obstacles" and "combined" scenarios
                                     (collision rate is meaningless without
                                     obstacles present)

Usage:
    uv run python scripts/plot_comparison.py \
        --reports-dir reports \
        --output-dir reports/charts
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — just writes PNG files, no display needed
import matplotlib.pyplot as plt

ALGOS = ["ppo", "reinforce"]
VARIANTS = ["free", "obstacles"]
SCENARIOS = ["vanilla", "wind", "obstacles", "combined"]
COLLISION_SCENARIOS = ["obstacles", "combined"]  # only these have obstacles on

SCENARIO_LABELS = {
    "vanilla": "Vanilla",
    "wind": "Wind",
    "obstacles": "Obstacles",
    "combined": "Combined",
}
SERIES_LABELS = {
    ("ppo", "free"): "PPO (free-trained)",
    ("ppo", "obstacles"): "PPO (obstacles-trained)",
    ("reinforce", "free"): "REINFORCE (free-trained)",
    ("reinforce", "obstacles"): "REINFORCE (obstacles-trained)",
}
SERIES_COLORS = {
    ("ppo", "free"): "#4C72B0",
    ("ppo", "obstacles"): "#8DA9D6",
    ("reinforce", "free"): "#C44E52",
    ("reinforce", "obstacles"): "#E39A9D",
}

FILENAME_RE = re.compile(r"^(?P<algo>ppo|reinforce|sac)_(?P<variant>free|obstacles)_(?P<scenario>vanilla|wind|obstacles|combined)\.json$")


def load_reports(reports_dir: Path) -> dict:
    """Returns {(algo, variant, scenario): summary_dict}"""
    data = {}
    for path in reports_dir.glob("*.json"):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue  # skip anything that doesn't match the expected naming
        key = (m.group("algo"), m.group("variant"), m.group("scenario"))
        with open(path) as f:
            report = json.load(f)
        data[key] = report.get("summary", {})
    return data


def grouped_bar_chart(
    data: dict,
    scenarios: list,
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    series_keys = [(a, v) for a in ALGOS for v in VARIANTS]
    n_series = len(series_keys)
    n_groups = len(scenarios)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(2.5 + 2.2 * n_groups, 6))

    for i, series_key in enumerate(series_keys):
        algo, variant = series_key
        values = []
        for scenario in scenarios:
            summary = data.get((algo, variant, scenario))
            val = summary.get(metric_key) if summary else None
            values.append(val if isinstance(val, (int, float)) else 0.0)

        offsets = x - 0.4 + bar_width * (i + 0.5)
        bars = ax.bar(
            offsets, values, width=bar_width,
            label=SERIES_LABELS[series_key], color=SERIES_COLORS[series_key],
        )
        for rect, val, scenario in zip(bars, values, scenarios):
            summary = data.get((algo, variant, scenario))
            missing = summary is None or not isinstance(summary.get(metric_key), (int, float))
            label = "N/A" if missing else f"{val:.1f}"
            ax.annotate(
                label, xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_training_curve(csv_path: Path) -> tuple:
    """Reads a W&B-exported CSV. Auto-detects the step column and any column
    containing 'shaping_reward' (W&B names it '<run_name> - train/shaping_reward').
    Returns (steps, values) or (None, None) if not found."""
    df = pd.read_csv(csv_path)
    step_col = next((c for c in df.columns if c.lower() in ("step", "_step")), df.columns[0])
    value_col = next((c for c in df.columns if "shaping_reward" in c.lower()), None)
    if value_col is None:
        print(f"[WARN] No 'shaping_reward' column found in {csv_path} (columns: {list(df.columns)})")
        return None, None
    clean = df[[step_col, value_col]].dropna()
    return clean[step_col].to_numpy(), clean[value_col].to_numpy()


def shaping_reward_curve_chart(curves_dir: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    any_plotted = False

    for algo in ALGOS:
        for variant in VARIANTS:
            csv_path = curves_dir / f"{algo}_{variant}_training.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing training curve: {csv_path}")
                continue
            steps, values = load_training_curve(csv_path)
            if steps is None:
                continue
            key = (algo, variant)
            ax.plot(steps, values, label=SERIES_LABELS[key], color=SERIES_COLORS[key], linewidth=1.8)
            any_plotted = True

    if not any_plotted:
        print("[WARN] No training curves found — skipping shaping_reward_comparison.png")
        plt.close(fig)
        return

    ax.set_xlabel("Training step")
    ax.set_ylabel("train/shaping_reward")
    ax.set_title("Shaping Reward Over Training — PPO vs REINFORCE (free vs obstacles-trained)")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--output-dir", type=str, default="reports/charts")
    parser.add_argument(
        "--training-curves-dir", type=str, default="reports",
        help="Folder containing {algo}_{variant}_training.csv exports from W&B "
             "(used for the train/shaping_reward line chart). Defaults to the "
             "same folder as the eval JSONs.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_reports(reports_dir)

    expected = {(a, v, s) for a in ALGOS for v in VARIANTS for s in SCENARIOS}
    missing = expected - set(data.keys())
    if missing:
        print(f"[WARN] {len(missing)} expected report file(s) not found (will show as N/A / 0):")
        for algo, variant, scenario in sorted(missing):
            print(f"  - {algo}_{variant}_{scenario}.json")

    grouped_bar_chart(
        data, SCENARIOS, "success_rate_pct",
        title="Success Rate by Scenario — PPO vs REINFORCE (free vs obstacles-trained)",
        ylabel="Success Rate (%)",
        out_path=output_dir / "success_rate_comparison.png",
    )

    grouped_bar_chart(
        data, COLLISION_SCENARIOS, "collision_rate_pct",
        title="Collision Rate — Obstacle Scenarios Only — PPO vs REINFORCE (free vs obstacles-trained)",
        ylabel="Collision Rate (%)",
        out_path=output_dir / "collision_rate_comparison.png",
    )

    shaping_reward_curve_chart(Path(args.training_curves_dir), output_dir / "shaping_reward_comparison.png")


if __name__ == "__main__":
    main()