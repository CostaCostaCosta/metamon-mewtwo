#!/usr/bin/env python3
"""
Visualize tournament and ELO results.

Creates plots and tables to analyze:
- ELO progression over time
- Win rate heatmap (matchup matrix)
- Statistical comparisons
- Training correlation analysis

Usage:
    python scripts/visualize_results.py \
        --tournament_dir ~/gen1_tournament_results \
        --output_dir ~/gen1_tournament_results/visualizations
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_elo_data(tournament_dir: str) -> dict:
    """Load ELO ratings and progression data."""
    ratings_file = os.path.join(tournament_dir, "elo_results_ratings.json")
    progression_file = os.path.join(tournament_dir, "elo_results_progression.csv")

    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"ELO ratings file not found: {ratings_file}")

    with open(ratings_file, "r") as f:
        ratings_data = json.load(f)

    progression_df = None
    if os.path.exists(progression_file):
        progression_df = pd.read_csv(progression_file)

    return ratings_data, progression_df


def load_statistics(tournament_dir: str) -> pd.DataFrame:
    """Load battle statistics."""
    stats_file = os.path.join(tournament_dir, "elo_results_statistics.csv")

    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")

    return pd.read_csv(stats_file)


def load_matchup_matrix(tournament_dir: str) -> pd.DataFrame:
    """Load matchup matrix."""
    matrix_file = os.path.join(tournament_dir, "elo_results_matchup_matrix.csv")

    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Matchup matrix file not found: {matrix_file}")

    return pd.read_csv(matrix_file, index_col=0)


def plot_elo_rankings(ratings_data: dict, output_path: str):
    """Create bar chart of final ELO rankings."""
    ratings = ratings_data["ratings"]
    models = list(ratings.keys())
    elos = [ratings[m]["elo"] for m in models]
    ranks = [ratings[m]["rank"] for m in models]

    # Sort by rank
    sorted_idx = np.argsort(ranks)
    models = [models[i] for i in sorted_idx]
    elos = [elos[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, elos, color=plt.cm.viridis(np.linspace(0, 1, len(models))))

    # Add ELO values on bars
    for i, (model, elo) in enumerate(zip(models, elos)):
        plt.text(elo + 10, i, f"{elo:.0f}", va="center")

    # Add initial rating line
    initial_rating = ratings_data["initial_rating"]
    plt.axvline(initial_rating, color="red", linestyle="--", label=f"Initial Rating ({initial_rating})")

    plt.xlabel("ELO Rating", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.title("Final ELO Rankings", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Saved ELO rankings plot: {output_path}")


def plot_elo_progression(progression_df: pd.DataFrame, output_path: str):
    """Create line plot showing ELO progression over battles."""
    if progression_df is None or len(progression_df) == 0:
        print("⚠ Skipping ELO progression plot (no data)")
        return

    plt.figure(figsize=(12, 6))

    # Get unique models
    models = set(progression_df["player"].unique()) | set(
        progression_df["opponent"].unique()
    )

    # Plot each model's ELO trajectory
    for model in sorted(models):
        # Collect rating changes for this model
        battles = []
        ratings = []

        for idx, row in progression_df.iterrows():
            if row["player"] == model:
                battles.append(idx)
                ratings.append(row["player_rating_after"])
            elif row["opponent"] == model:
                battles.append(idx)
                ratings.append(row["opponent_rating_after"])

        if len(battles) > 0:
            plt.plot(battles, ratings, label=model, linewidth=2, alpha=0.8)

    plt.xlabel("Battle Number", fontsize=12)
    plt.ylabel("ELO Rating", fontsize=12)
    plt.title("ELO Rating Progression", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved ELO progression plot: {output_path}")


def plot_winrate_heatmap(matchup_matrix: pd.DataFrame, output_path: str):
    """Create heatmap showing win rates between models."""

    # Convert "W-L" strings to win rates
    winrate_matrix = matchup_matrix.copy()

    for i in winrate_matrix.index:
        for j in winrate_matrix.columns:
            val = str(winrate_matrix.loc[i, j])
            if val == "-" or val == "nan":
                winrate_matrix.loc[i, j] = np.nan
            elif "-" in val:
                wins, losses = map(int, val.split("-"))
                total = wins + losses
                winrate = wins / total if total > 0 else 0
                winrate_matrix.loc[i, j] = winrate * 100  # Convert to percentage
            else:
                winrate_matrix.loc[i, j] = 0

    # Convert to float
    winrate_matrix = winrate_matrix.astype(float)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        winrate_matrix,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Win Rate (%)"},
        linewidths=0.5,
    )

    plt.xlabel("Opponent", fontsize=12)
    plt.ylabel("Player", fontsize=12)
    plt.title("Head-to-Head Win Rates (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Saved win rate heatmap: {output_path}")


def plot_statistics_comparison(stats_df: pd.DataFrame, output_path: str):
    """Create comparison plots for battle statistics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parse win rates
    stats_df["WinRate_Numeric"] = (
        stats_df["Win Rate"].str.rstrip("%").astype(float) / 100
    )

    # Parse avg turn count
    stats_df["AvgTurns_Numeric"] = stats_df["Avg Turn Count"].astype(float)

    # 1. Win Rate comparison
    ax = axes[0, 0]
    models = stats_df["Model"]
    win_rates = stats_df["WinRate_Numeric"] * 100
    bars = ax.barh(models, win_rates, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    ax.axvline(50, color="red", linestyle="--", alpha=0.7, label="50%")
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("Overall Win Rate")
    ax.legend()

    # 2. Total Battles
    ax = axes[0, 1]
    total_battles = stats_df["Total Battles"]
    ax.barh(models, total_battles, color="skyblue")
    ax.set_xlabel("Total Battles")
    ax.set_title("Battle Count")

    # 3. Average Turn Count
    ax = axes[1, 0]
    avg_turns = stats_df["AvgTurns_Numeric"]
    ax.barh(models, avg_turns, color="orange")
    ax.set_xlabel("Average Turns")
    ax.set_title("Battle Length")

    # 4. Win/Loss Breakdown
    ax = axes[1, 1]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, stats_df["Wins"], width, label="Wins", color="green", alpha=0.7)
    ax.bar(
        x + width / 2,
        stats_df["Losses"],
        width,
        label="Losses",
        color="red",
        alpha=0.7,
    )
    ax.set_ylabel("Count")
    ax.set_title("Wins vs Losses")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Saved statistics comparison plot: {output_path}")


def create_summary_table(
    ratings_data: dict, stats_df: pd.DataFrame, output_path: str
):
    """Create comprehensive summary table."""

    ratings = ratings_data["ratings"]

    summary_data = []
    for _, row in stats_df.iterrows():
        model = row["Model"]
        if model in ratings:
            summary_data.append(
                {
                    "Model": model,
                    "Rank": ratings[model]["rank"],
                    "ELO": f"{ratings[model]['elo']:.0f}",
                    "ELO Change": f"{ratings[model]['elo'] - ratings_data['initial_rating']:+.0f}",
                    "Win Rate": row["Win Rate"],
                    "W-L": f"{row['Wins']}-{row['Losses']}",
                    "Avg Turns": row["Avg Turn Count"],
                }
            )

    summary_df = pd.DataFrame(summary_data).sort_values("Rank")

    # Save as CSV
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table: {output_path}")

    # Also print to console
    print(f"\n{'='*80}")
    print("TOURNAMENT SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tournament and ELO results"
    )
    parser.add_argument(
        "--tournament_dir",
        type=str,
        required=True,
        help="Directory containing tournament results (with ELO calculations)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations (defaults to tournament_dir/visualizations)",
    )

    args = parser.parse_args()

    tournament_dir = os.path.expanduser(args.tournament_dir)

    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    else:
        output_dir = os.path.join(tournament_dir, "visualizations")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"VISUALIZING TOURNAMENT RESULTS")
    print(f"Tournament directory: {tournament_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    ratings_data, progression_df = load_elo_data(tournament_dir)
    stats_df = load_statistics(tournament_dir)
    matchup_matrix = load_matchup_matrix(tournament_dir)

    # Generate visualizations
    print("\nGenerating visualizations...\n")

    plot_elo_rankings(
        ratings_data, os.path.join(output_dir, "elo_rankings.png")
    )

    if progression_df is not None:
        plot_elo_progression(
            progression_df, os.path.join(output_dir, "elo_progression.png")
        )

    plot_winrate_heatmap(
        matchup_matrix, os.path.join(output_dir, "winrate_heatmap.png")
    )

    plot_statistics_comparison(
        stats_df, os.path.join(output_dir, "statistics_comparison.png")
    )

    create_summary_table(
        ratings_data, stats_df, os.path.join(output_dir, "tournament_summary.csv")
    )

    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
