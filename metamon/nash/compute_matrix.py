#!/usr/bin/env python3
"""
Compute interaction matrix for Nash equilibrium training.

This script orchestrates a round-robin tournament to build the interaction matrix M,
then computes the Nash equilibrium meta-strategy σ over the policy population.

Usage:
    # Step 1: Define population
    python -m metamon.nash.compute_matrix \
        --population_file population.json \
        --battles_per_matchup 200 \
        --battle_format gen1ou \
        --team_set competitive \
        --output_dir ~/nash_phase0

    # Step 2: Script runs tournament and computes M and σ
    # Results saved to:
    #   - interaction_matrix.json (M)
    #   - interaction_matrix.csv (M as table)
    #   - meta_strategy.json (σ)
    #   - meta_game_analysis.json (full analysis)

Example population.json:
    {
        "policies": {
            "P0_SYN_V2_GEN1": {"policy_type": "pretrained", "model_class": "SyntheticRLV2", ...},
            "PokeEnvHeuristic": {"policy_type": "heuristic", "baseline_class": "PokeEnvHeuristic", ...},
            ...
        }
    }
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metamon.nash.population import PolicyPopulation
from metamon.nash.interaction_matrix import InteractionMatrix
from metamon.nash.solver import (
    solve_nash_mixture,
    analyze_meta_game,
    save_meta_strategy,
)
import subprocess


def create_initial_population(
    output_path: str,
    include_pretrained: List[str],
    include_heuristics: List[str],
) -> PolicyPopulation:
    """
    Create initial policy population and save to disk.

    Args:
        output_path: Path to save population.json
        include_pretrained: List of pretrained model names
        include_heuristics: List of heuristic baseline names

    Returns:
        PolicyPopulation instance
    """
    pop = PolicyPopulation()

    # Add pretrained models
    for model_name in include_pretrained:
        try:
            pop.add_pretrained_policy(
                model_name=model_name,
                description=f"Pretrained model: {model_name}",
            )
            print(f"✓ Added pretrained model: {model_name}")
        except Exception as e:
            print(f"⚠ Could not add model '{model_name}': {e}")

    # Add heuristic baselines
    for baseline_name in include_heuristics:
        try:
            pop.add_heuristic_policy(
                baseline_name=baseline_name,
                description=f"Heuristic baseline: {baseline_name}",
            )
            print(f"✓ Added heuristic baseline: {baseline_name}")
        except Exception as e:
            print(f"⚠ Could not add baseline '{baseline_name}': {e}")

    # Save to disk
    pop.save(output_path)
    print(f"\n✓ Saved population to: {output_path}")
    print(f"  Total policies: {pop.size()}")

    return pop


def main():
    parser = argparse.ArgumentParser(
        description="Compute interaction matrix and Nash equilibrium for policy population"
    )

    # Population definition
    pop_group = parser.add_argument_group("Population")
    pop_group.add_argument(
        "--population_file",
        type=str,
        help="Path to existing population.json (or where to save if creating new)",
    )
    pop_group.add_argument(
        "--create_population",
        action="store_true",
        help="Create new population instead of loading existing",
    )
    pop_group.add_argument(
        "--pretrained_models",
        nargs="+",
        default=[],
        help="Pretrained model names to include (when creating population)",
    )
    pop_group.add_argument(
        "--heuristic_baselines",
        nargs="+",
        default=["PokeEnvHeuristic", "Gen1BossAI", "GymLeader"],
        help="Heuristic baseline names to include (when creating population)",
    )

    # Tournament settings
    tournament_group = parser.add_argument_group("Tournament")
    tournament_group.add_argument(
        "--battles_per_matchup",
        type=int,
        default=200,
        help="Number of battles for each head-to-head matchup",
    )
    tournament_group.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Showdown battle format",
    )
    tournament_group.add_argument(
        "--team_set",
        type=str,
        default="competitive",
        help="Team set name",
    )
    tournament_group.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle backend",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    output_group.add_argument(
        "--skip_tournament",
        action="store_true",
        help="Skip tournament and load existing results (for recomputing Nash only)",
    )

    # Performance
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "--parallel_matchups",
        type=int,
        default=4,
        help="Number of matchups to run in parallel (default: 4)",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("NASH EQUILIBRIUM COMPUTATION")
    print("=" * 80)

    # Step 1: Load or create population
    print("\n[Step 1/4] Loading policy population...")

    if args.create_population:
        if not args.population_file:
            args.population_file = os.path.join(output_dir, "population.json")

        pop = create_initial_population(
            output_path=args.population_file,
            include_pretrained=args.pretrained_models,
            include_heuristics=args.heuristic_baselines,
        )
    else:
        if not args.population_file:
            raise ValueError("Must specify --population_file or use --create_population")

        pop = PolicyPopulation.load(args.population_file)
        print(f"✓ Loaded population from: {args.population_file}")
        print(f"  Total policies: {pop.size()}")

    print("\nPopulation:")
    print(pop)

    # If we just created the population and skip_tournament is set, exit here
    if args.create_population and args.skip_tournament:
        print("\n" + "=" * 80)
        print("POPULATION CREATED")
        print("=" * 80)
        print(f"\nPopulation saved to: {args.population_file}")
        print("\nNext step: Run tournament to compute interaction matrix")
        print(f"  python -m metamon.nash.compute_matrix \\")
        print(f"      --population_file {args.population_file} \\")
        print(f"      --battles_per_matchup 100 \\")
        print(f"      --battle_format {args.battle_format} \\")
        print(f"      --team_set {args.team_set} \\")
        print(f"      --output_dir {output_dir}")
        print()
        return 0

    # Step 2: Run tournament (or load existing results)
    print("\n[Step 2/4] Computing interaction matrix...")

    if args.skip_tournament:
        print("Skipping tournament (loading existing results)...")
        matrix = InteractionMatrix.from_tournament_results(pop, output_dir)
    else:
        print(f"Running round-robin tournament ({args.battles_per_matchup} battles per matchup)...")
        print("This may take several hours depending on population size.\n")

        # Run tournament using mixed population script (handles RL + heuristics)
        tournament_script = Path(__file__).parent.parent.parent / "scripts" / "run_mixed_population_tournament.py"
        result = subprocess.run(
            [
                sys.executable,
                str(tournament_script),
                "--population_file", args.population_file,
                "--battles_per_matchup", str(args.battles_per_matchup),
                "--battle_format", args.battle_format,
                "--team_set", args.team_set,
                "--output_dir", output_dir,
                "--parallel_matchups", str(args.parallel_matchups),
            ],
            check=True,
        )

        # Parse results into interaction matrix
        print("\nParsing tournament results into interaction matrix...")
        matrix = InteractionMatrix.from_tournament_results(pop, output_dir)

    print(f"✓ Interaction matrix computed")
    print(f"  Matchups recorded: {len(matrix.matchups)}")
    print(f"  Matrix complete: {matrix.is_complete()}")

    if not matrix.is_complete():
        missing = matrix.missing_matchups()
        print(f"  ⚠ Missing {len(missing)} matchups:")
        for p1, p2 in missing[:5]:
            print(f"    - {p1} vs {p2}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    # Save interaction matrix
    matrix_json_path = os.path.join(output_dir, "interaction_matrix.json")
    matrix_csv_path = os.path.join(output_dir, "interaction_matrix.csv")
    matrix.save(matrix_json_path)
    matrix.save_csv(matrix_csv_path)
    print(f"\n✓ Saved interaction matrix:")
    print(f"  JSON: {matrix_json_path}")
    print(f"  CSV: {matrix_csv_path}")

    # Display matrix
    print("\nInteraction Matrix (win-rates):")
    print(matrix.to_dataframe().to_string())

    # Step 3: Solve for Nash equilibrium
    print("\n[Step 3/4] Computing Nash equilibrium...")

    M = matrix.to_numpy()
    policy_names = pop.list_policies()

    nash_sigma = solve_nash_mixture(M)

    print(f"✓ Nash equilibrium computed")
    print("\nNash Mixture:")
    for i, (name, prob) in enumerate(zip(policy_names, nash_sigma)):
        if prob > 0.01:  # Only show policies with >1% mass
            print(f"  {name}: {prob:.1%}")

    # Save meta-strategy
    meta_strategy_path = os.path.join(output_dir, "meta_strategy.json")
    save_meta_strategy(
        sigma=nash_sigma,
        policy_names=policy_names,
        path=meta_strategy_path,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "battle_format": args.battle_format,
            "battles_per_matchup": args.battles_per_matchup,
        },
    )
    print(f"\n✓ Saved meta-strategy: {meta_strategy_path}")

    # Step 4: Full meta-game analysis
    print("\n[Step 4/4] Analyzing meta-game...")

    analysis = analyze_meta_game(M, policy_names)

    print(f"✓ Meta-game analysis complete")
    print(f"\nNash value: {analysis['nash_value']:.3f}")
    print(f"Exploitability: {analysis['exploitability']:.6f}")
    print(f"\nDominant policies:")
    for policy_info in analysis["dominant_policies"]:
        print(
            f"  {policy_info['policy']}: "
            f"{policy_info['probability']:.1%} probability, "
            f"{policy_info['expected_winrate_vs_nash']:.1%} expected win-rate vs Nash"
        )

    # Save analysis
    analysis_path = os.path.join(output_dir, "meta_game_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✓ Saved meta-game analysis: {analysis_path}")

    # Summary
    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    print(f"  - population.json (policy population)")
    print(f"  - interaction_matrix.json (M)")
    print(f"  - interaction_matrix.csv (M as table)")
    print(f"  - meta_strategy.json (σ)")
    print(f"  - meta_game_analysis.json (full analysis)")
    print(f"  - tournament_results.json (raw battle results)")
    print(f"  - team_results/ (battle logs)")
    print(f"  - trajectories/ (battle replays)")
    print("\nNext steps:")
    print("  1. Validate that Nash mixture makes sense (dominant policy gets high mass)")
    print("  2. Proceed to Phase 1: PSRO training (best-response to σ)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
