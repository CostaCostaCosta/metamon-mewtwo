"""
Interaction matrix computation for Nash equilibrium training.

The interaction matrix M is a K×K matrix where M[i,j] represents the empirical
win-rate of policy πᵢ against policy πⱼ. This is the "payoff matrix" for the
meta-game that we use to compute Nash equilibria.

M[i,j] = P(πᵢ beats πⱼ) ≈ (wins of πᵢ vs πⱼ) / (total battles)

Example 3×3 matrix:
             π₁    π₂    π₃
        π₁ [ 0.50  0.90  0.85 ]
        π₂ [ 0.10  0.50  0.60 ]
        π₃ [ 0.15  0.40  0.50 ]

Interpretation: π₁ beats π₂ 90% of the time, π₃ beats π₂ 60% of the time, etc.
Diagonal is always 0.5 (policy vs itself).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from metamon.nash.population import PolicyPopulation


@dataclass
class MatchupResult:
    """Results from a head-to-head matchup between two policies."""

    policy1: str
    policy2: str
    policy1_wins: int
    policy2_wins: int
    total_battles: int

    @property
    def policy1_winrate(self) -> float:
        """Win-rate of policy1 vs policy2."""
        if self.total_battles == 0:
            return 0.5
        return self.policy1_wins / self.total_battles

    @property
    def policy2_winrate(self) -> float:
        """Win-rate of policy2 vs policy1."""
        if self.total_battles == 0:
            return 0.5
        return self.policy2_wins / self.total_battles

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "policy1": self.policy1,
            "policy2": self.policy2,
            "policy1_wins": self.policy1_wins,
            "policy2_wins": self.policy2_wins,
            "total_battles": self.total_battles,
            "policy1_winrate": self.policy1_winrate,
            "policy2_winrate": self.policy2_winrate,
        }


class InteractionMatrix:
    """
    Manages interaction matrix M for a policy population.

    The matrix stores empirical win-rates between all pairs of policies.
    This is the fundamental data structure for computing Nash equilibria.

    Example:
        >>> matrix = InteractionMatrix(population)
        >>> # Add results from tournament
        >>> matrix.add_matchup_result("P0_SYN_V2_GEN1", "PokeEnvHeuristic", 95, 5, 100)
        >>> matrix.add_matchup_result("P0_SYN_V2_GEN1", "GymLeader", 98, 2, 100)
        >>> # Get win-rate matrix as numpy array
        >>> M = matrix.to_numpy()
        >>> # Save to disk
        >>> matrix.save("interaction_matrix.json")
    """

    def __init__(self, population: PolicyPopulation):
        """
        Initialize interaction matrix for a population.

        Args:
            population: PolicyPopulation instance defining policies
        """
        self.population = population
        self.matchups: Dict[tuple, MatchupResult] = {}

    def _matchup_key(self, policy1: str, policy2: str) -> tuple:
        """Canonical ordering for matchup keys."""
        return tuple(sorted([policy1, policy2]))

    def add_matchup_result(
        self,
        policy1: str,
        policy2: str,
        policy1_wins: int,
        policy2_wins: int,
        total_battles: int,
    ) -> None:
        """
        Add results from a head-to-head matchup.

        Args:
            policy1: Name of first policy
            policy2: Name of second policy
            policy1_wins: Number of wins for policy1
            policy2_wins: Number of wins for policy2
            total_battles: Total number of battles played

        Raises:
            ValueError: If policies not in population or if results are invalid
        """
        # Validate policies exist
        self.population.get_policy(policy1)
        self.population.get_policy(policy2)

        # Validate results
        if policy1_wins + policy2_wins != total_battles:
            raise ValueError(
                f"Wins don't sum to total battles: {policy1_wins} + {policy2_wins} != {total_battles}"
            )

        if total_battles <= 0:
            raise ValueError(f"Total battles must be positive, got {total_battles}")

        # Store result
        key = self._matchup_key(policy1, policy2)
        result = MatchupResult(
            policy1=policy1,
            policy2=policy2,
            policy1_wins=policy1_wins,
            policy2_wins=policy2_wins,
            total_battles=total_battles,
        )
        self.matchups[key] = result

    def get_winrate(self, policy1: str, policy2: str) -> float:
        """
        Get win-rate of policy1 vs policy2.

        Args:
            policy1: Name of first policy
            policy2: Name of second policy

        Returns:
            Win-rate of policy1 vs policy2 (0.0 to 1.0)
            Returns 0.5 for self-play or if matchup not found

        Example:
            >>> winrate = matrix.get_winrate("P0_SYN_V2_GEN1", "PokeEnvHeuristic")
            >>> print(f"P0 beats heuristic {winrate:.1%} of the time")
        """
        # Self-play is 50%
        if policy1 == policy2:
            return 0.5

        key = self._matchup_key(policy1, policy2)
        if key not in self.matchups:
            # Default to 50% if matchup hasn't been played
            return 0.5

        result = self.matchups[key]

        # Return win-rate from correct perspective
        if result.policy1 == policy1:
            return result.policy1_winrate
        else:
            return result.policy2_winrate

    def to_numpy(self, policy_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert interaction matrix to numpy array.

        Args:
            policy_order: Optional ordering of policies. If None, uses population.list_policies()

        Returns:
            K×K numpy array where M[i,j] = P(πᵢ beats πⱼ)

        Example:
            >>> M = matrix.to_numpy()
            >>> # M is now ready for Nash solver
            >>> sigma = solve_nash_mixture(M)
        """
        if policy_order is None:
            policy_order = self.population.list_policies()

        K = len(policy_order)
        M = np.zeros((K, K))

        for i, policy_i in enumerate(policy_order):
            for j, policy_j in enumerate(policy_order):
                M[i, j] = self.get_winrate(policy_i, policy_j)

        return M

    def to_dataframe(self, policy_order: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert interaction matrix to pandas DataFrame for visualization.

        Args:
            policy_order: Optional ordering of policies

        Returns:
            DataFrame with policies as both rows and columns

        Example:
            >>> df = matrix.to_dataframe()
            >>> print(df)
                              P0_SYN_V2_GEN1  PokeEnvHeuristic  GymLeader
            P0_SYN_V2_GEN1              0.50              0.95       0.98
            PokeEnvHeuristic            0.05              0.50       0.70
            GymLeader                   0.02              0.30       0.50
        """
        if policy_order is None:
            policy_order = self.population.list_policies()

        M = self.to_numpy(policy_order)
        return pd.DataFrame(M, index=policy_order, columns=policy_order)

    def is_complete(self) -> bool:
        """
        Check if all pairwise matchups have been played.

        Returns:
            True if matrix is complete (all pairs evaluated)
        """
        policies = self.population.list_policies()
        K = len(policies)

        # Number of unique pairs (excluding self-play)
        expected_matchups = K * (K - 1) // 2

        return len(self.matchups) >= expected_matchups

    def missing_matchups(self) -> List[tuple]:
        """
        Get list of matchups that haven't been played yet.

        Returns:
            List of (policy1, policy2) tuples for missing matchups
        """
        policies = self.population.list_policies()
        missing = []

        for i, policy1 in enumerate(policies):
            for policy2 in policies[i + 1 :]:
                key = self._matchup_key(policy1, policy2)
                if key not in self.matchups:
                    missing.append((policy1, policy2))

        return missing

    def save(self, path: str) -> None:
        """
        Save interaction matrix to JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "population": self.population.list_policies(),
            "matchups": [result.to_dict() for result in self.matchups.values()],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str) -> None:
        """
        Save interaction matrix to CSV file.

        Args:
            path: File path to save to
        """
        df = self.to_dataframe()
        df.to_csv(path)

    @classmethod
    def load(cls, path: str, population: PolicyPopulation) -> "InteractionMatrix":
        """
        Load interaction matrix from JSON file.

        Args:
            path: File path to load from
            population: PolicyPopulation instance (must match saved population)

        Returns:
            InteractionMatrix instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        matrix = cls(population)

        for matchup_dict in data["matchups"]:
            matrix.add_matchup_result(
                policy1=matchup_dict["policy1"],
                policy2=matchup_dict["policy2"],
                policy1_wins=matchup_dict["policy1_wins"],
                policy2_wins=matchup_dict["policy2_wins"],
                total_battles=matchup_dict["total_battles"],
            )

        return matrix

    @classmethod
    def from_tournament_results(
        cls,
        population: PolicyPopulation,
        tournament_dir: str,
    ) -> "InteractionMatrix":
        """
        Create interaction matrix from tournament results.

        Parses battle logs from a tournament directory. Supports two formats:
        1. tournament_results.json (from run_mixed_population_tournament.py)
        2. team_results/*.csv (from pure ladder-based tournaments)

        Args:
            population: PolicyPopulation instance
            tournament_dir: Directory containing tournament_results.json or team_results/*.csv

        Returns:
            InteractionMatrix populated with tournament results

        Example:
            >>> pop = PolicyPopulation.load("population.json")
            >>> matrix = InteractionMatrix.from_tournament_results(pop, "~/gen1_tournament_results")
            >>> print(matrix.to_dataframe())
        """
        from collections import defaultdict

        tournament_dir = Path(tournament_dir)

        # Try to load from tournament_results.json first (mixed population format)
        json_path = tournament_dir / "tournament_results.json"
        use_json = json_path.exists()

        if use_json:
            with open(json_path) as f:
                tournament_data = json.load(f)

            matrix = cls(population)
            battles_per_matchup = tournament_data.get("battles_per_matchup", 50)

            # Parse direct_eval results from JSON
            valid_policies = set(population.list_policies())
            for matchup in tournament_data.get("matchups", []):
                if matchup.get("status") != "success":
                    continue

                if "model1" not in matchup or "model2" not in matchup:
                    continue

                model1 = matchup["model1"]
                model2 = matchup["model2"]

                # Skip matchups with policies not in current population
                if model1 not in valid_policies or model2 not in valid_policies:
                    continue

                results = matchup.get("results", {})

                # Extract win rate from results (direct_eval matchups have this)
                win_rate_key = next((k for k in results.keys() if "Win Rate" in k), None)
                if win_rate_key:
                    win_rate = results[win_rate_key]
                    # Convert win rate to win/loss counts
                    p1_wins = int(win_rate * battles_per_matchup)
                    p2_wins = battles_per_matchup - p1_wins
                    matrix.add_matchup_result(model1, model2, p1_wins, p2_wins, battles_per_matchup)

            # Continue to CSV parsing to get ladder-based matchups (RL vs RL)
        else:
            matrix = cls(population)

        # Fall back to CSV parsing (pure ladder-based tournaments)
        team_results_dir = tournament_dir / "team_results"
        if not team_results_dir.exists():
            raise FileNotFoundError(f"Tournament results not found: {team_results_dir}")

        # Find all battle log CSV files
        csv_files = list(team_results_dir.glob("battle_log_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No battle log CSV files found in {team_results_dir}")

        # Build username -> model name mapping from log files (if available)
        username_to_model = {}
        logs_dir = tournament_dir / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.stderr.log"):
                # Format: {FullModelName}_{ShortUsername}.stderr.log
                name = log_file.stem.replace(".stderr", "")
                parts = name.split("_")
                if len(parts) >= 2:
                    # Last part is username, rest is model name
                    username = parts[-1]
                    model_name = "_".join(parts[:-1])
                    username_to_model[username] = model_name

        # Parse all CSVs
        all_battles = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, skipinitialspace=True)
            all_battles.append(df)

        battles_df = pd.concat(all_battles, ignore_index=True)

        # Extract model names from usernames
        def extract_model_name(username: str) -> str:
            # First try the mapping from log files
            if username in username_to_model:
                return username_to_model[username]

            # Fall back to parsing username format
            # New format: Short-HHMMSS-A (hyphen-separated, 6-digit timestamp)
            if "-" in username:
                parts = username.split("-")
                # Check if second-to-last part looks like a timestamp (6 digits)
                if len(parts) >= 3 and len(parts[-2]) == 6 and parts[-2].isdigit():
                    # Return everything except timestamp and suffix
                    return "-".join(parts[:-2])
            # Old format: ModelName_timestamp_A (underscore-separated)
            parts = username.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:-2])
            return username

        battles_df["Player_Model"] = battles_df["Player Username"].apply(extract_model_name)
        battles_df["Opponent_Model"] = battles_df["Opponent Username"].apply(extract_model_name)

        # Filter out battles where model names don't exist in population
        # (e.g., random usernames from old direct_eval battles)
        valid_policies = set(population.list_policies())
        battles_df = battles_df[
            battles_df["Player_Model"].isin(valid_policies) &
            battles_df["Opponent_Model"].isin(valid_policies)
        ]

        # Count wins for each matchup
        matchup_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})

        for _, row in battles_df.iterrows():
            player = row["Player_Model"]
            opponent = row["Opponent_Model"]
            result = row["Result"]

            key = (player, opponent)
            matchup_stats[key]["total"] += 1

            if result == "WIN":
                matchup_stats[key]["wins"] += 1
            elif result == "LOSS":
                matchup_stats[key]["losses"] += 1

        # Add CSV results to matrix (don't overwrite - matrix may already have JSON results)
        # Aggregate symmetric matchups
        processed = set()
        for (p1, p2), stats in matchup_stats.items():
            if (p1, p2) in processed or (p2, p1) in processed:
                continue

            # Get stats for both directions
            p1_vs_p2 = matchup_stats.get((p1, p2), {"wins": 0, "losses": 0, "total": 0})
            p2_vs_p1 = matchup_stats.get((p2, p1), {"wins": 0, "losses": 0, "total": 0})

            # Total battles from both perspectives
            total_battles = p1_vs_p2["total"] + p2_vs_p1["total"]

            # p1 wins = p1's wins as player + p2's losses as player
            p1_wins = p1_vs_p2["wins"] + p2_vs_p1["losses"]
            p2_wins = p1_vs_p2["losses"] + p2_vs_p1["wins"]

            if total_battles > 0:
                matrix.add_matchup_result(p1, p2, p1_wins, p2_wins, total_battles)

            processed.add((p1, p2))
            processed.add((p2, p1))

        return matrix

    def __repr__(self) -> str:
        """String representation."""
        policies = self.population.list_policies()
        complete = self.is_complete()
        return (
            f"InteractionMatrix (policies={len(policies)}, "
            f"matchups={len(self.matchups)}, complete={complete})"
        )
