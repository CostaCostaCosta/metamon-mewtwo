"""
Vectorized environment for pypkmn battles.

Provides batched simulation of N battles simultaneously with:
- Simultaneous actions (both players act per turn)
- Batched observations and legal masks
- Trajectory tracking for saving
- Compatible with metamon observation/action/reward spaces
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from pykmn.engine.gen1 import Battle, Pokemon
from pykmn.engine.common import Result, Player, ResultType

from metamon.interface import ObservationSpace, RewardFunction
from .team_parser import parse_showdown_team
from .features import Mappings, precompute_mappings, pykmn_to_features_raw
from .action_mapper import (
    ActionMappings,
    get_legal_mask,
    metamon_action_to_choice,
    TOTAL_ACTIONS,
)


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


class PyKMNVectorEnv:
    """
    Vectorized environment for fast pypkmn battle simulation.

    Simulates N battles simultaneously with batched operations.
    Designed for maximum throughput via:
    - Simultaneous action input (both players per turn)
    - Minimal Python overhead
    - Batched feature extraction
    - Optional trajectory tracking

    Example:
        >>> teams_p1 = [parse_showdown_team(t) for t in team_texts_p1]
        >>> teams_p2 = [parse_showdown_team(t) for t in team_texts_p2]
        >>> env = PyKMNVectorEnv(teams_p1, teams_p2, num_envs=64)
        >>> obs_p1, obs_p2, masks_p1, masks_p2 = env.reset()
        >>> obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)
    """

    def __init__(
        self,
        teams_p1: List[List[Pokemon]],
        teams_p2: List[List[Pokemon]],
        num_envs: int,
        obs_space: ObservationSpace,
        reward_fn: RewardFunction,
        battle_format: str = "gen1ou",
        track_trajectories: bool = True,
        use_trace: bool = False,
    ):
        """
        Initialize vectorized pypkmn environment.

        Args:
            teams_p1: List of teams for player 1 (length num_envs)
            teams_p2: List of teams for player 2 (length num_envs)
            num_envs: Number of parallel battles
            obs_space: ObservationSpace for feature extraction
            reward_fn: RewardFunction for computing rewards
            battle_format: Battle format string (e.g., "gen1ou")
            track_trajectories: Whether to save trajectory history
            use_trace: Whether to use trace logging (slower but useful for debugging)
        """
        if len(teams_p1) != num_envs or len(teams_p2) != num_envs:
            raise ValueError(
                f"Expected {num_envs} teams for each player, "
                f"got {len(teams_p1)} and {len(teams_p2)}"
            )

        self.num_envs = num_envs
        self.obs_space = obs_space
        self.reward_fn = reward_fn
        self.battle_format = battle_format
        self.track_trajectories = track_trajectories

        # Precompute mappings once
        self.mappings = precompute_mappings()
        self.action_mappings = ActionMappings.create()

        # Create per-environment observation spaces to prevent cross-battle state leaks
        # Each battle needs its own observation space to track revealed_opponents, sleep/freeze flags
        import copy
        self.obs_spaces = [copy.deepcopy(obs_space) for _ in range(num_envs)]

        # Store teams
        self.teams_p1 = teams_p1
        self.teams_p2 = teams_p2

        # Initialize battles (will be created in reset())
        self.battles: List[Optional[Battle]] = [None] * num_envs
        self.results: List[Optional[Result]] = [None] * num_envs
        self.dones: np.ndarray = np.zeros(num_envs, dtype=bool)

        # Track previous states for reward computation
        self.prev_states_p1: List[Optional[Any]] = [None] * num_envs
        self.prev_states_p2: List[Optional[Any]] = [None] * num_envs

        # Trajectory tracking
        self.trajectories: List[List[Transition]] = [[] for _ in range(num_envs)]
        self.completed_trajectories: List[Trajectory] = []

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset all battles to initial state.

        Returns:
            Tuple of (obs_p1, obs_p2, legal_masks_p1, legal_masks_p2)
            All arrays have shape (num_envs, ...).
        """
        # Explicitly clear old references before creating new battles
        # Let Python's reference counting handle cleanup naturally
        for i in range(self.num_envs):
            self.battles[i] = None
            self.results[i] = None
            self.prev_states_p1[i] = None
            self.prev_states_p2[i] = None

        # Create new battles
        for i in range(self.num_envs):
            self.battles[i] = Battle(
                p1_team=self.teams_p1[i],
                p2_team=self.teams_p2[i],
            )
            # Initial update (team preview / setup)
            # Pass raw choice integer 0 for PASS
            result, _ = self.battles[i].update_raw(0, 0)
            self.results[i] = result

        # Reset tracking
        self.dones = np.zeros(self.num_envs, dtype=bool)
        self.trajectories = [[] for _ in range(self.num_envs)]

        # Reset per-environment observation spaces at batch start
        for obs_space in self.obs_spaces:
            if hasattr(obs_space, 'reset'):
                obs_space.reset()

        # Extract initial observations and legal masks
        obs_p1, obs_p2 = self._extract_observations()
        legal_masks_p1, legal_masks_p2 = self._extract_legal_masks()

        # Initialize previous states for reward computation
        from .features import features_to_universal_state
        for i in range(self.num_envs):
            features_p1 = pykmn_to_features_raw(
                self.battles[i], self.results[i], Player.P1, self.mappings
            )
            features_p2 = pykmn_to_features_raw(
                self.battles[i], self.results[i], Player.P2, self.mappings
            )
            self.prev_states_p1[i] = features_to_universal_state(features_p1, self.mappings)
            self.prev_states_p2[i] = features_to_universal_state(features_p2, self.mappings)

        return obs_p1, obs_p2, legal_masks_p1, legal_masks_p2

    def step(
        self,
        actions_p1: np.ndarray,
        actions_p2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step all battles with simultaneous actions.

        Args:
            actions_p1: Array of shape (num_envs,) with P1 actions
            actions_p2: Array of shape (num_envs,) with P2 actions

        Returns:
            Tuple of:
            - obs_p1: (num_envs, obs_dim) observations for P1
            - obs_p2: (num_envs, obs_dim) observations for P2
            - rewards_p1: (num_envs,) rewards for P1
            - rewards_p2: (num_envs,) rewards for P2
            - dones: (num_envs,) boolean done flags
            - info: Dictionary with metadata
        """
        if len(actions_p1) != self.num_envs or len(actions_p2) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, "
                f"got {len(actions_p1)} and {len(actions_p2)}"
            )

        # Get legal masks before stepping (for trajectory saving)
        legal_masks_p1, legal_masks_p2 = self._extract_legal_masks()

        # Get current observations (before step, for trajectory saving)
        if self.track_trajectories:
            obs_before_p1, obs_before_p2 = self._extract_observations_raw()

        # Execute actions for all battles
        for i in range(self.num_envs):
            if self.dones[i]:
                continue  # Skip finished battles

            # Check if player must PASS (forced switch scenario)
            # If legal mask is all False, the only legal choice is PASS (0)
            if not legal_masks_p1[i].any():
                choice_p1 = 0  # PASS
            else:
                choice_p1 = metamon_action_to_choice(actions_p1[i], self.action_mappings)

            if not legal_masks_p2[i].any():
                choice_p2 = 0  # PASS
            else:
                choice_p2 = metamon_action_to_choice(actions_p2[i], self.action_mappings)

            # Update battle (choice_p1 and choice_p2 are already raw integers)
            result, trace = self.battles[i].update_raw(choice_p1, choice_p2)
            self.results[i] = result

            # Check if done
            if result.type() != ResultType.NONE:
                self.dones[i] = True

        # Extract new observations
        obs_p1, obs_p2 = self._extract_observations()

        # Compute rewards
        rewards_p1, rewards_p2 = self._compute_rewards()

        # Track trajectories
        if self.track_trajectories:
            for i in range(self.num_envs):
                if not self.dones[i] or len(self.trajectories[i]) == 0:
                    # Only save transition if battle is still ongoing
                    # (or just finished on this step)
                    transition = Transition(
                        features_p1=obs_before_p1[i],
                        features_p2=obs_before_p2[i],
                        action_p1=int(actions_p1[i]),
                        action_p2=int(actions_p2[i]),
                        reward_p1=float(rewards_p1[i]),
                        reward_p2=float(rewards_p2[i]),
                        done=self.dones[i],
                        legal_mask_p1=legal_masks_p1[i],
                        legal_mask_p2=legal_masks_p2[i],
                    )
                    self.trajectories[i].append(transition)

                # If battle just finished, save complete trajectory
                if self.dones[i] and len(self.trajectories[i]) > 0:
                    winner = self._get_winner(i)
                    trajectory = Trajectory(
                        transitions=self.trajectories[i],
                        winner=winner,
                    )
                    self.completed_trajectories.append(trajectory)
                    self.trajectories[i] = []  # Clear for next episode

        # Build info dict
        info = {
            "num_done": int(self.dones.sum()),
            "completed_trajectories": len(self.completed_trajectories),
        }

        return obs_p1, obs_p2, rewards_p1, rewards_p2, self.dones.copy(), info

    def _extract_observations(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract observations for both players (batched).

        Returns:
            Tuple of dictionaries with batched observations.
            Each dict has keys matching obs_space format (e.g., "numbers", "text").
        """
        from .features import features_to_universal_state

        obs_list_p1 = []
        obs_list_p2 = []

        for i in range(self.num_envs):
            # Extract raw features
            features_p1 = pykmn_to_features_raw(
                self.battles[i], self.results[i], Player.P1, self.mappings
            )
            features_p2 = pykmn_to_features_raw(
                self.battles[i], self.results[i], Player.P2, self.mappings
            )

            # Convert to UniversalState
            state_p1 = features_to_universal_state(features_p1, self.mappings)
            state_p2 = features_to_universal_state(features_p2, self.mappings)

            # Convert to observation format using per-environment ObservationSpace
            # Each environment has its own obs_space that maintains state across steps
            obs_p1 = self.obs_spaces[i](state_p1)
            obs_p2 = self.obs_spaces[i](state_p2)

            obs_list_p1.append(obs_p1)
            obs_list_p2.append(obs_p2)

        # Stack into batched observations
        # obs_space returns a dict with "numbers" and "text" keys
        batched_obs_p1 = {
            key: np.stack([obs[key] for obs in obs_list_p1])
            for key in obs_list_p1[0].keys()
        }
        batched_obs_p2 = {
            key: np.stack([obs[key] for obs in obs_list_p2])
            for key in obs_list_p2[0].keys()
        }

        return batched_obs_p1, batched_obs_p2

    def _extract_observations_raw(self) -> Tuple[List[Dict], List[Dict]]:
        """Extract raw feature dictionaries (for trajectory saving)."""
        obs_p1 = []
        obs_p2 = []

        for i in range(self.num_envs):
            features_p1 = pykmn_to_features_raw(
                self.battles[i], self.results[i], Player.P1, self.mappings
            )
            features_p2 = pykmn_to_features_raw(
                self.battles[i], self.results[i], Player.P2, self.mappings
            )
            obs_p1.append(features_p1)
            obs_p2.append(features_p2)

        return obs_p1, obs_p2

    def _extract_legal_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract legal action masks for both players (batched)."""
        masks_p1 = np.zeros((self.num_envs, TOTAL_ACTIONS), dtype=bool)
        masks_p2 = np.zeros((self.num_envs, TOTAL_ACTIONS), dtype=bool)

        for i in range(self.num_envs):
            if not self.dones[i]:
                masks_p1[i] = get_legal_mask(
                    self.battles[i], self.results[i], Player.P1, self.action_mappings
                )
                masks_p2[i] = get_legal_mask(
                    self.battles[i], self.results[i], Player.P2, self.action_mappings
                )

        return masks_p1, masks_p2

    def _compute_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rewards for both players using the reward function."""
        from .features import features_to_universal_state

        rewards_p1 = np.zeros(self.num_envs, dtype=np.float32)
        rewards_p2 = np.zeros(self.num_envs, dtype=np.float32)

        for i in range(self.num_envs):
            if self.dones[i]:
                # For finished battles, use the reward function
                # Extract current states
                features_p1 = pykmn_to_features_raw(
                    self.battles[i], self.results[i], Player.P1, self.mappings
                )
                features_p2 = pykmn_to_features_raw(
                    self.battles[i], self.results[i], Player.P2, self.mappings
                )
                current_state_p1 = features_to_universal_state(features_p1, self.mappings)
                current_state_p2 = features_to_universal_state(features_p2, self.mappings)

                # Compute rewards (comparing prev_state to current_state)
                rewards_p1[i] = self.reward_fn(self.prev_states_p1[i], current_state_p1)
                rewards_p2[i] = self.reward_fn(self.prev_states_p2[i], current_state_p2)
            else:
                # Battle ongoing, compute shaped rewards
                features_p1 = pykmn_to_features_raw(
                    self.battles[i], self.results[i], Player.P1, self.mappings
                )
                features_p2 = pykmn_to_features_raw(
                    self.battles[i], self.results[i], Player.P2, self.mappings
                )
                current_state_p1 = features_to_universal_state(features_p1, self.mappings)
                current_state_p2 = features_to_universal_state(features_p2, self.mappings)

                # Compute rewards
                rewards_p1[i] = self.reward_fn(self.prev_states_p1[i], current_state_p1)
                rewards_p2[i] = self.reward_fn(self.prev_states_p2[i], current_state_p2)

                # Update previous states for next step
                self.prev_states_p1[i] = current_state_p1
                self.prev_states_p2[i] = current_state_p2

        return rewards_p1, rewards_p2

    def _get_winner(self, env_idx: int) -> int:
        """Get winner of a finished battle (1=P1, 2=P2, 0=tie)."""
        result_type = self.results[env_idx].type()
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
        trajectories = self.completed_trajectories.copy()
        self.completed_trajectories = []

        # Force garbage collection to free trajectory data immediately
        # This helps prevent memory fragmentation from large trajectory buffers
        import gc
        gc.collect()

        return trajectories

    def close(self):
        """Clean up resources."""
        # Explicitly clear all battle references to help Python GC
        # free C++ PyKMN objects immediately
        for i in range(self.num_envs):
            self.battles[i] = None
            self.results[i] = None
            self.prev_states_p1[i] = None
            self.prev_states_p2[i] = None

        # Clear trajectories
        self.trajectories = [[] for _ in range(self.num_envs)]
        self.completed_trajectories = []

        # Force garbage collection to free C++ memory
        import gc
        gc.collect()
