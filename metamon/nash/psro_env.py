"""
PSRO self-play environment with opponent sampling from population.

In PSRO, we train a best-response (BR) policy against a distribution over opponents.
This module provides environment wrappers that:
1. Sample opponent policies from population according to meta-strategy σ
2. Load the appropriate opponent (PretrainedModel or Baseline)
3. Create battle environments with sampled opponents
4. Handle opponent switching between episodes

The core idea:
    For each training episode:
        opponent ~ σ  (sample from meta-strategy)
        run battle: trainee vs opponent
        collect trajectory for RL update

This is the "RL ORACLE" from the PSRO framework - we're training via RL (no search)
against a mixture of opponents, which approximates best-response to σ.
"""

import os
import random
import warnings
from typing import Optional, Dict, List, Callable
from pathlib import Path

import numpy as np
from poke_env.player import Player

from metamon.nash.population import PolicyPopulation
from metamon.rl.pretrained import get_pretrained_model, PretrainedModel
from metamon.baselines import get_baseline
from metamon.env import (
    PokeEnvWrapper,
    BattleAgainstBaseline,
    QueueOnLocalLadder,
    TeamSet,
)
from metamon.interface import ObservationSpace, ActionSpace, RewardFunction


class PopulationOpponentSampler:
    """
    Samples opponent policies from population according to meta-strategy σ.

    This is the core mechanism for PSRO training: instead of playing against a fixed
    opponent, we sample opponents from the population according to the Nash/meta-strategy
    distribution.

    Example:
        >>> pop = PolicyPopulation.load("population.json")
        >>> sigma = [0.7, 0.2, 0.1]  # 70% P0, 20% P1, 10% P2
        >>> sampler = PopulationOpponentSampler(pop, sigma)
        >>> opponent_name = sampler.sample()
        >>> opponent_policy = sampler.get_opponent(opponent_name, battle_format="gen1ou", ...)
    """

    def __init__(
        self,
        population: PolicyPopulation,
        meta_strategy: np.ndarray,
    ):
        """
        Initialize opponent sampler.

        Args:
            population: PolicyPopulation instance
            meta_strategy: Probability distribution over policies (must sum to 1)
        """
        self.population = population
        self.meta_strategy = np.array(meta_strategy)
        self.policy_names = population.list_policies()

        if len(self.meta_strategy) != len(self.policy_names):
            raise ValueError(
                f"Meta-strategy length ({len(self.meta_strategy)}) must match "
                f"population size ({len(self.policy_names)})"
            )

        if not np.isclose(self.meta_strategy.sum(), 1.0):
            raise ValueError(
                f"Meta-strategy must sum to 1.0 (got {self.meta_strategy.sum()})"
            )

        # Track sampling statistics for logging
        self.sample_counts = {name: 0 for name in self.policy_names}
        self.total_samples = 0

    def sample(self) -> str:
        """
        Sample an opponent name from the population according to meta-strategy.

        Returns:
            Policy name sampled from σ
        """
        opponent_name = np.random.choice(self.policy_names, p=self.meta_strategy)
        self.sample_counts[opponent_name] += 1
        self.total_samples += 1
        return opponent_name

    def get_opponent_player_class(
        self,
        opponent_name: str,
        battle_format: str,
        team_set: TeamSet,
    ) -> type:
        """
        Get the Player class for a sampled opponent.

        This returns a Player class (not an instance) that can be used with
        PokeEnvWrapper's opponent_type parameter.

        Args:
            opponent_name: Policy name from population
            battle_format: Showdown battle format
            team_set: Team set for opponent

        Returns:
            Player subclass for the opponent
        """
        policy = self.population.get_policy(opponent_name)

        if policy.policy_type == "heuristic":
            # Heuristic baselines are already Player classes
            baseline_class = get_baseline(policy.baseline_class)
            return baseline_class

        elif policy.policy_type == "pretrained":
            # Pretrained models need to be wrapped
            # We'll return a factory that creates a PretrainedModelPlayer
            pretrained_model = get_pretrained_model(policy.model_class)

            # Create a custom Player class that uses the pretrained model
            class PretrainedModelPlayer(Player):
                """Player that uses a pretrained model for move selection."""

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._pretrained_model = pretrained_model
                    # Initialize model if needed
                    if not hasattr(self._pretrained_model, '_ready'):
                        # Load model weights (assumes model is already loaded)
                        self._pretrained_model._ready = True

                def choose_move(self, battle):
                    """Use pretrained model to choose move."""
                    # Get action from pretrained model
                    # This uses the model's built-in inference
                    try:
                        action = self._pretrained_model.predict_action(battle)
                        return action
                    except Exception as e:
                        warnings.warn(f"Error in model prediction: {e}. Using random move.")
                        # Fall back to random legal move
                        return self.choose_random_move(battle)

            return PretrainedModelPlayer

        else:
            raise ValueError(f"Unknown policy type: {policy.policy_type}")

    def get_sampling_stats(self) -> Dict[str, float]:
        """
        Get statistics on opponent sampling.

        Returns:
            Dictionary with empirical sampling frequencies
        """
        if self.total_samples == 0:
            return {name: 0.0 for name in self.policy_names}

        return {
            name: count / self.total_samples
            for name, count in self.sample_counts.items()
        }


class PSROSelfPlayEnv(BattleAgainstBaseline):
    """
    PSRO self-play environment that samples opponents from population.

    This environment:
    - Samples a new opponent from the population according to meta-strategy σ
    - Creates a battle against that opponent
    - Switches opponents every N episodes (default: 1)

    Note: For Phase 1, we use a simpler approach where opponent is sampled once
    per environment reset. For Phase 2+, we may want more sophisticated opponent
    switching (e.g., prioritized sampling, curriculum learning, etc.).

    Example:
        >>> pop = PolicyPopulation.load("population.json")
        >>> sigma = np.array([0.7, 0.2, 0.1])
        >>> env = PSROSelfPlayEnv(
        ...     population=pop,
        ...     meta_strategy=sigma,
        ...     battle_format="gen1ou",
        ...     observation_space=obs_space,
        ...     action_space=action_space,
        ...     reward_function=reward_fn,
        ...     team_set=teams,
        ... )
        >>> obs, info = env.reset()
        >>> # Battle is now against a random opponent sampled from σ
    """

    def __init__(
        self,
        population: PolicyPopulation,
        meta_strategy: np.ndarray,
        battle_format: str,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        team_set: TeamSet,
        opponent_switch_interval: int = 1,
        turn_limit: int = 200,
        save_trajectories_to: Optional[str] = None,
        save_team_results_to: Optional[str] = None,
        battle_backend: str = "poke-env",
    ):
        """
        Initialize PSRO self-play environment.

        Args:
            population: PolicyPopulation instance
            meta_strategy: Probability distribution over policies
            battle_format: Showdown battle format (e.g., "gen1ou")
            observation_space: Observation space
            action_space: Action space
            reward_function: Reward function
            team_set: Team set for both players
            opponent_switch_interval: Switch opponent every N episodes
            turn_limit: Maximum turns per battle
            save_trajectories_to: Optional directory to save trajectories
            save_team_results_to: Optional directory to save battle logs
            battle_backend: Battle backend ("poke-env" or "metamon")
        """
        self.sampler = PopulationOpponentSampler(population, meta_strategy)
        self.opponent_switch_interval = opponent_switch_interval
        self.episode_count = 0
        self.current_opponent_name = None

        # Sample initial opponent
        self.current_opponent_name = self.sampler.sample()
        opponent_class = self.sampler.get_opponent_player_class(
            self.current_opponent_name,
            battle_format=battle_format,
            team_set=team_set,
        )

        # Initialize with first opponent
        super().__init__(
            battle_format=battle_format,
            observation_space=observation_space,
            action_space=action_space,
            reward_function=reward_function,
            team_set=team_set,
            opponent_type=opponent_class,
            turn_limit=turn_limit,
            save_trajectories_to=save_trajectories_to,
            save_team_results_to=save_team_results_to,
            battle_backend=battle_backend,
        )

        print(f"[PSRO Environment] Initialized with opponent: {self.current_opponent_name}")
        print(f"[PSRO Environment] Meta-strategy: {dict(zip(self.sampler.policy_names, meta_strategy))}")

    def reset(self, *args, **kwargs):
        """
        Reset environment and potentially switch opponent.

        Every opponent_switch_interval episodes, we sample a new opponent from
        the population according to meta-strategy σ.
        """
        self.episode_count += 1

        # Check if we should switch opponent
        if self.episode_count % self.opponent_switch_interval == 0:
            # Sample new opponent
            new_opponent_name = self.sampler.sample()

            if new_opponent_name != self.current_opponent_name:
                print(f"[PSRO Environment] Switching opponent: {self.current_opponent_name} → {new_opponent_name}")
                self.current_opponent_name = new_opponent_name

                # Recreate opponent (this is a bit hacky, but works for now)
                # In a production system, we'd want a cleaner opponent switching mechanism
                opponent_class = self.sampler.get_opponent_player_class(
                    self.current_opponent_name,
                    battle_format=self.metamon_battle_format,
                    team_set=self.metamon_team_set,
                )

                # Stop old opponent
                if self._current_opponent is not None:
                    try:
                        self._current_opponent.close()
                    except:
                        pass

                # Create new opponent instance
                from poke_env import AccountConfiguration
                opponent_account = AccountConfiguration(
                    f"PSRO_Opponent_{random.randint(1000, 9999)}",
                    None
                )

                self._current_opponent = opponent_class(
                    battle_format=self.metamon_battle_format,
                    team=self.metamon_team_set,
                    account_configuration=opponent_account,
                    server_configuration=self.server_configuration,
                )
                self.metamon_opponent_name = new_opponent_name

        return super().reset(*args, **kwargs)

    def get_opponent_sampling_stats(self) -> Dict[str, float]:
        """Get statistics on opponent sampling frequencies."""
        return self.sampler.get_sampling_stats()


def create_psro_env(
    population_file: str,
    meta_strategy_file: str,
    battle_format: str,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    team_set: TeamSet,
    **env_kwargs
) -> PSROSelfPlayEnv:
    """
    Convenience function to create PSRO environment from files.

    Args:
        population_file: Path to population.json
        meta_strategy_file: Path to meta_strategy.json
        battle_format: Showdown battle format
        observation_space: Observation space
        action_space: Action space
        reward_function: Reward function
        team_set: Team set
        **env_kwargs: Additional kwargs for PSROSelfPlayEnv

    Returns:
        PSROSelfPlayEnv instance
    """
    import json

    # Load population
    population = PolicyPopulation.load(population_file)

    # Load meta-strategy
    with open(meta_strategy_file, "r") as f:
        meta_data = json.load(f)

    meta_strategy = np.array(meta_data["meta_strategy"])

    return PSROSelfPlayEnv(
        population=population,
        meta_strategy=meta_strategy,
        battle_format=battle_format,
        observation_space=observation_space,
        action_space=action_space,
        reward_function=reward_function,
        team_set=team_set,
        **env_kwargs
    )
