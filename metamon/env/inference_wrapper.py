"""
Minimal inference wrapper for fast GPU-accelerated PyKMN battles.

This wrapper provides:
1. Numeric-only observations (no text/UniversalState overhead)
2. Direct integration with SafeBattleManager
3. Fast feature extraction via FastFeatureExtractor
4. Automatic reset handling for terminal battles
5. Type-safe tensor conversion for GPU inference

Performance targets:
- 1024 parallel battles
- >50 battles/sec throughput
- <100ms per step (full cycle: features → inference → update)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Result, Player, ResultType

from metamon.env.safe_battle_manager import SafeBattleManager, clone_pokemon_team
from metamon.env.fast_features import FastFeatureExtractor
from metamon.env.pykmn.action_mapper import ActionMappings, get_legal_mask_batch, metamon_action_to_choice

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single transition in a battle trajectory."""
    features_p1: Dict[str, np.ndarray]
    features_p2: Dict[str, np.ndarray]
    action_p1: int
    action_p2: int
    reward_p1: float
    reward_p2: float
    done: bool
    legal_mask_p1: np.ndarray
    legal_mask_p2: np.ndarray


@dataclass
class Trajectory:
    """Complete battle trajectory for saving."""
    transitions: List[Transition]
    winner: int  # 1 = P1 win, 2 = P2 win, 0 = tie


class InferenceWrapper:
    """
    Minimal wrapper for GPU inference with PyKMN battles.

    This class manages the full pipeline:
    1. Battle simulation (SafeBattleManager)
    2. Feature extraction (FastFeatureExtractor)
    3. Legal action masking
    4. Automatic resets

    Usage:
        # Create wrapper
        wrapper = InferenceWrapper(teams_p1, teams_p2, num_envs=128)

        # Reset
        obs_p1, obs_p2, legal_p1, legal_p2 = wrapper.reset()

        # Step
        obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = wrapper.step(
            actions_p1, actions_p2
        )

    Note:
        - Observations are numeric-only (no text)
        - All arrays are numpy (type-safe for torch conversion)
        - Automatic reset on terminal states
    """

    def __init__(
        self,
        teams_p1: List[List[Pokemon]],
        teams_p2: List[List[Pokemon]],
        num_envs: int,
        reward_fn=None,  # Optional reward function
        auto_reset: bool = True,
        enable_logging: bool = False,
        track_trajectories: bool = True,
    ):
        """
        Initialize inference wrapper.

        Args:
            teams_p1: List of teams for Player 1
            teams_p2: List of teams for Player 2
            num_envs: Number of parallel environments
            reward_fn: Optional reward function (UniversalState → float)
            auto_reset: Automatically reset terminal battles
            enable_logging: Enable detailed logging
            track_trajectories: Whether to save trajectories
        """
        self.num_envs = num_envs
        self.reward_fn = reward_fn
        self.auto_reset = auto_reset
        self.enable_logging = enable_logging
        self.track_trajectories = track_trajectories

        # Create battle manager
        self.battle_manager = SafeBattleManager(
            teams_p1=teams_p1,
            teams_p2=teams_p2,
            num_envs=num_envs,
            enable_logging=enable_logging,
        )

        # Create feature extractor
        self.feature_extractor = FastFeatureExtractor(num_envs=num_envs)

        # Create action mappings
        self.action_mappings = ActionMappings.create()

        # Trajectory tracking
        if self.track_trajectories:
            # Pre-allocate trajectory lists for each environment
            self.trajectories = [[] for _ in range(num_envs)]
            self.completed_trajectories = []

        # Statistics
        self.total_steps = 0
        self.total_resets = 0

        if self.enable_logging:
            logger.info(f"InferenceWrapper initialized with {num_envs} environments")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Reset all environments.

        Returns:
            (obs_p1, obs_p2, legal_masks_p1, legal_masks_p2):
                - obs_p1: Observations for P1 (dict with 'numbers' key)
                - obs_p2: Observations for P2 (dict with 'numbers' key)
                - legal_masks_p1: Legal action masks for P1 (shape: num_envs × 13)
                - legal_masks_p2: Legal action masks for P2 (shape: num_envs × 13)
        """
        # Reset all battles
        results_p1, results_p2 = self.battle_manager.reset_all()

        # Extract features
        obs_p1 = self.feature_extractor.extract_batch(
            battles=[s.battle for s in self.battle_manager.states],
            results_p1=results_p1,
            results_p2=results_p2,
            player=Player.P1,
        )

        obs_p2 = self.feature_extractor.extract_batch(
            battles=[s.battle for s in self.battle_manager.states],
            results_p1=results_p1,
            results_p2=results_p2,
            player=Player.P2,
        )

        # Get legal masks
        legal_masks_p1 = self._get_legal_masks(results_p1, Player.P1)
        legal_masks_p2 = self._get_legal_masks(results_p2, Player.P2)

        self.total_resets += 1

        return obs_p1, obs_p2, legal_masks_p1, legal_masks_p2

    def step(
        self,
        actions_p1: np.ndarray,
        actions_p2: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step all environments.

        Args:
            actions_p1: Actions for P1 (shape: num_envs,) in metamon action space [0-12]
            actions_p2: Actions for P2 (shape: num_envs,) in metamon action space [0-12]

        Returns:
            (obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info):
                - obs_p1: Observations for P1
                - obs_p2: Observations for P2
                - rewards_p1: Rewards for P1 (shape: num_envs,)
                - rewards_p2: Rewards for P2 (shape: num_envs,)
                - dones: Terminal flags (shape: num_envs,)
                - info: Additional info dict
        """
        # Validate inputs
        if actions_p1.shape != (self.num_envs,):
            raise ValueError(f"actions_p1 shape {actions_p1.shape} != ({self.num_envs},)")
        if actions_p2.shape != (self.num_envs,):
            raise ValueError(f"actions_p2 shape {actions_p2.shape} != ({self.num_envs},)")

        # Get current legal masks to filter illegal actions
        from metamon.env.pykmn.action_mapper import get_legal_mask

        # Get results for current state (before stepping)
        temp_results_p1 = [state.result_p1 for state in self.battle_manager.states]
        temp_results_p2 = [state.result_p2 for state in self.battle_manager.states]

        # Note: We don't extract observations here for trajectory tracking
        # Instead, we extract features in pykmn_to_features_raw format inline below
        # to avoid double extraction and format conversion issues

        # Get legal masks for trajectory tracking
        legal_masks_p1_before = self._get_legal_masks(temp_results_p1, Player.P1)
        legal_masks_p2_before = self._get_legal_masks(temp_results_p2, Player.P2)

        # Filter illegal actions (safety mechanism)
        filtered_actions_p1 = actions_p1.copy()
        filtered_actions_p2 = actions_p2.copy()

        for i in range(self.num_envs):
            state = self.battle_manager.states[i]
            if not state.is_healthy() or state.is_terminal:
                # Use default action (switch to first available)
                filtered_actions_p1[i] = 4  # Switch to slot 2
                filtered_actions_p2[i] = 4
                continue

            try:
                # Get legal masks
                legal_p1 = get_legal_mask(state.battle, temp_results_p1[i], Player.P1, self.action_mappings)
                legal_p2 = get_legal_mask(state.battle, temp_results_p2[i], Player.P2, self.action_mappings)

                # Filter illegal actions
                if not legal_p1[actions_p1[i]]:
                    # Find first legal action
                    legal_indices = np.where(legal_p1)[0]
                    if len(legal_indices) > 0:
                        filtered_actions_p1[i] = legal_indices[0]

                if not legal_p2[actions_p2[i]]:
                    legal_indices = np.where(legal_p2)[0]
                    if len(legal_indices) > 0:
                        filtered_actions_p2[i] = legal_indices[0]
            except:
                pass  # Keep original action if filtering fails

        # Convert metamon actions to PyKMN choices
        choices_p1 = np.array([
            metamon_action_to_choice(int(a), self.action_mappings)
            for a in filtered_actions_p1
        ], dtype=np.int32)

        choices_p2 = np.array([
            metamon_action_to_choice(int(a), self.action_mappings)
            for a in filtered_actions_p2
        ], dtype=np.int32)

        # Step all battles
        results_p1, results_p2, dones = self.battle_manager.step_all(choices_p1, choices_p2)

        # Calculate rewards (simplified - just win/loss)
        rewards_p1, rewards_p2 = self._calculate_rewards(results_p1, results_p2, dones)

        # Track trajectories BEFORE auto-reset
        if self.track_trajectories:
            from metamon.env.pykmn.features import pykmn_to_features_raw, precompute_mappings

            # Lazy load mappings for trajectory saving
            if not hasattr(self, '_trajectory_mappings'):
                self._trajectory_mappings = precompute_mappings()

            for i in range(self.num_envs):
                # Always save transition if the trajectory has started
                # This includes both ongoing transitions and the final done transition

                # Extract features in old format (with individual keys like 'active_species_id')
                # This is needed for trajectory saving
                state = self.battle_manager.states[i]
                try:
                    features_p1_i = pykmn_to_features_raw(
                        state.battle,
                        temp_results_p1[i],
                        Player.P1,
                        self._trajectory_mappings,
                    )
                    features_p2_i = pykmn_to_features_raw(
                        state.battle,
                        temp_results_p2[i],
                        Player.P2,
                        self._trajectory_mappings,
                    )
                except Exception as e:
                    # Skip this transition if feature extraction fails
                    logger.warning(f"Failed to extract features for trajectory {i}: {e}")
                    continue

                # Save transition (ongoing or just finished)
                transition = Transition(
                    features_p1=features_p1_i,
                    features_p2=features_p2_i,
                    action_p1=int(filtered_actions_p1[i]),
                    action_p2=int(filtered_actions_p2[i]),
                    reward_p1=float(rewards_p1[i]),
                    reward_p2=float(rewards_p2[i]),
                    done=dones[i],
                    legal_mask_p1=legal_masks_p1_before[i],
                    legal_mask_p2=legal_masks_p2_before[i],
                )
                self.trajectories[i].append(transition)

                # If battle just finished, save complete trajectory
                if dones[i] and len(self.trajectories[i]) > 0:
                    winner = self._get_winner(results_p1[i])
                    trajectory = Trajectory(
                        transitions=self.trajectories[i],
                        winner=winner,
                    )
                    self.completed_trajectories.append(trajectory)
                    self.trajectories[i] = []  # Clear for next episode

        # Auto-reset terminal battles BEFORE extracting features/masks
        if self.auto_reset:
            for i in range(self.num_envs):
                if dones[i]:
                    # Reset battle and get new result
                    new_result_p1, new_result_p2 = self.battle_manager.reset_battle(i)
                    # Update results for feature/mask extraction
                    results_p1[i] = new_result_p1
                    results_p2[i] = new_result_p2

        # Extract features (after auto-reset for done battles)
        obs_p1 = self.feature_extractor.extract_batch(
            battles=[s.battle for s in self.battle_manager.states],
            results_p1=results_p1,
            results_p2=results_p2,
            player=Player.P1,
        )

        obs_p2 = self.feature_extractor.extract_batch(
            battles=[s.battle for s in self.battle_manager.states],
            results_p1=results_p1,
            results_p2=results_p2,
            player=Player.P2,
        )

        # Get legal masks for next step (after auto-reset)
        legal_masks_p1 = self._get_legal_masks(results_p1, Player.P1)
        legal_masks_p2 = self._get_legal_masks(results_p2, Player.P2)

        # Build info dict
        info = {
            'legal_masks_p1': legal_masks_p1,
            'legal_masks_p2': legal_masks_p2,
        }

        if self.track_trajectories:
            info['completed_trajectories'] = len(self.completed_trajectories)

        self.total_steps += 1

        return obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info

    def _get_legal_masks(
        self,
        results: List[Result],
        player: Player,
    ) -> np.ndarray:
        """
        Get legal action masks for all environments.

        Args:
            results: List of Result objects
            player: Which player

        Returns:
            Boolean array of shape (num_envs, 13) with legal actions
        """
        masks = np.zeros((self.num_envs, 13), dtype=bool)

        for i in range(self.num_envs):
            state = self.battle_manager.states[i]
            if not state.is_healthy():
                # Errored battle - no legal actions
                if self.enable_logging:
                    logger.warning(f"Env {i} not healthy, skipping legal mask extraction")
                continue

            if state.is_terminal:
                # Terminal battle - should have been reset already
                # This indicates auto-reset didn't work properly
                if self.enable_logging:
                    logger.warning(f"Env {i} is terminal but not reset!")
                continue

            try:
                from metamon.env.pykmn.action_mapper import get_legal_mask
                mask = get_legal_mask(
                    state.battle,
                    results[i],
                    player,
                    self.action_mappings,
                )
                masks[i] = mask

                # Validate that we have at least one legal action
                if not mask.any():
                    if self.enable_logging:
                        logger.warning(f"Env {i} has no legal actions! Result type: {results[i].type()}")
            except Exception as e:
                if self.enable_logging:
                    logger.error(f"Failed to get legal mask for env {i}: {e}")
                # Leave as all False (no legal actions)

        return masks

    def _calculate_rewards(
        self,
        results_p1: List[Result],
        results_p2: List[Result],
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate rewards for both players.

        Simplified reward: +1 for win, -1 for loss, 0 otherwise.

        Args:
            results_p1: Results for P1
            results_p2: Results for P2
            dones: Terminal flags

        Returns:
            (rewards_p1, rewards_p2)
        """
        rewards_p1 = np.zeros(self.num_envs, dtype=np.float32)
        rewards_p2 = np.zeros(self.num_envs, dtype=np.float32)

        from pykmn.engine.common import ResultType

        for i in range(self.num_envs):
            if not dones[i]:
                continue

            result = results_p1[i]
            if result is None:
                continue

            result_type = result.type()

            if result_type == ResultType.PLAYER_1_WIN:
                # P1 wins
                rewards_p1[i] = 1.0
                rewards_p2[i] = -1.0
            elif result_type == ResultType.PLAYER_2_WIN:
                # P1 loses (P2 wins)
                rewards_p1[i] = -1.0
                rewards_p2[i] = 1.0
            elif result_type == ResultType.TIE:
                # Tie
                rewards_p1[i] = 0.0
                rewards_p2[i] = 0.0

        return rewards_p1, rewards_p2

    def _get_winner(self, result: Result) -> int:
        """
        Get winner from a result (1=P1, 2=P2, 0=tie).

        Args:
            result: Result object from PyKMN

        Returns:
            Winner (1, 2, or 0)
        """
        result_type = result.type()
        if result_type == ResultType.PLAYER_1_WIN:
            return 1
        elif result_type == ResultType.PLAYER_2_WIN:
            return 2
        elif result_type == ResultType.TIE:
            return 0
        else:
            return 0  # Shouldn't happen for finished battles

    def get_completed_trajectories(self) -> List[Trajectory]:
        """
        Get all completed trajectories and clear the buffer.

        Returns:
            List of Trajectory objects for finished battles.
        """
        if not self.track_trajectories:
            return []

        trajectories = self.completed_trajectories
        self.completed_trajectories = []
        return trajectories

    def get_statistics(self) -> dict:
        """Get wrapper statistics."""
        stats = self.battle_manager.get_statistics()
        stats.update({
            'total_steps': self.total_steps,
            'total_resets': self.total_resets,
        })
        if self.track_trajectories:
            stats['completed_trajectories'] = len(self.completed_trajectories)
        return stats

    def close(self):
        """Cleanup resources."""
        if self.battle_manager:
            del self.battle_manager
            self.battle_manager = None


def test_inference_wrapper():
    """Test the inference wrapper with a small batch."""
    print("Testing InferenceWrapper...")

    from pykmn.engine.gen1 import Pokemon

    # Create test team
    team = [
        Pokemon(species="Tauros", moves=("Body Slam", "Hyper Beam", "Blizzard", "Earthquake")),
        Pokemon(species="Snorlax", moves=("Body Slam", "Earthquake", "Rest", "Ice Beam")),
        Pokemon(species="Chansey", moves=("Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled")),
        Pokemon(species="Exeggutor", moves=("Psychic", "Sleep Powder", "Explosion", "Stun Spore")),
        Pokemon(species="Starmie", moves=("Thunderbolt", "Blizzard", "Thunder Wave", "Recover")),
        Pokemon(species="Alakazam", moves=("Psychic", "Seismic Toss", "Thunder Wave", "Recover")),
    ]

    num_envs = 16
    teams_p1 = [team] * num_envs
    teams_p2 = [team] * num_envs

    # Create wrapper
    wrapper = InferenceWrapper(
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        num_envs=num_envs,
        enable_logging=True,
    )

    # Reset
    print("\nResetting...")
    obs_p1, obs_p2, legal_p1, legal_p2 = wrapper.reset()
    print(f"  obs_p1 shape: {obs_p1['numbers'].shape}")
    print(f"  legal_p1 shape: {legal_p1.shape}")

    # Run a few steps
    print("\nRunning 10 steps...")
    for step in range(10):
        # Random legal actions
        actions_p1 = []
        actions_p2 = []

        for i in range(num_envs):
            legal_acts_p1 = np.where(legal_p1[i])[0]
            legal_acts_p2 = np.where(legal_p2[i])[0]

            action_p1 = np.random.choice(legal_acts_p1) if len(legal_acts_p1) > 0 else 0
            action_p2 = np.random.choice(legal_acts_p2) if len(legal_acts_p2) > 0 else 0

            actions_p1.append(action_p1)
            actions_p2.append(action_p2)

        actions_p1 = np.array(actions_p1, dtype=np.int32)
        actions_p2 = np.array(actions_p2, dtype=np.int32)

        # Step
        obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = wrapper.step(
            actions_p1, actions_p2
        )

        legal_p1 = info['legal_masks_p1']
        legal_p2 = info['legal_masks_p2']

        print(f"  Step {step}: {dones.sum()} done, rewards_p1 sum: {rewards_p1.sum():.2f}")

    # Statistics
    print("\nStatistics:")
    stats = wrapper.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✓ InferenceWrapper test passed!")


if __name__ == "__main__":
    test_inference_wrapper()
