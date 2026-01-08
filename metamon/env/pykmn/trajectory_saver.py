"""
Save pypkmn trajectories to .json.lz4 format compatible with metamon training.

Converts vectorized battle trajectories to the ParsedReplay format
used by metamon's offline training pipeline.
"""

import json
import lz4.frame
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime

from .vector_env import Trajectory
from .features import Mappings, features_to_universal_state


def save_trajectories(
    trajectories: List[Trajectory],
    output_dir: Path | str,
    mappings: Mappings,
    battle_format: str = "gen1ou",
    verbose: bool = False,
    start_id: int = 0,
):
    """
    Save trajectories to .json.lz4 files in metamon format.

    IMPORTANT: This function saves BOTH player perspectives for each battle,
    creating 2 files per battle for balanced win/loss data.

    Args:
        trajectories: List of Trajectory objects from pypkmn battles
        output_dir: Directory to save trajectory files
        mappings: Precomputed mappings for feature conversion
        battle_format: Battle format string (e.g., "gen1ou")
        verbose: Whether to print progress
        start_id: Starting battle ID for filename generation (default: 0)

    Output format:
        output_dir/
            {battle_format}/
                metamon-{format}-{id}_Unrated_PyKMNP1_vs_PyKMNP2_{date}_{result}_P1.json.lz4
                metamon-{format}-{id}_Unrated_PyKMNP1_vs_PyKMNP2_{date}_{result}_P2.json.lz4
                ...

    Each .json.lz4 file contains a single-perspective battle replay:
        {
            "states": [...],  # List of UniversalState dicts (from one player's view)
            "actions": [...],  # List of action indices (from that player's view)
            "rewards": [...],  # List of rewards (from that player's view)
            "winner": 1 or 2,
            "metadata": {...}
        }

    Note: Saving both perspectives doubles the dataset size but ensures balanced
          win/loss distribution for self-play training.
    """
    output_dir = Path(output_dir).expanduser()
    format_dir = output_dir / battle_format
    format_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for i, trajectory in enumerate(trajectories):
        try:
            # Save P1 perspective
            replay_data_p1 = _trajectory_to_replay(trajectory, mappings, battle_format, player=1)
            _save_single_replay(
                replay_data_p1,
                format_dir,
                battle_format,
                start_id + i,
                trajectory.winner,
                player_perspective=1,
            )
            total_saved += 1

            # Save P2 perspective
            replay_data_p2 = _trajectory_to_replay(trajectory, mappings, battle_format, player=2)
            _save_single_replay(
                replay_data_p2,
                format_dir,
                battle_format,
                start_id + i,
                trajectory.winner,
                player_perspective=2,
            )
            total_saved += 1

            if verbose and (i + 1) % 100 == 0:
                print(f"Saved {i + 1}/{len(trajectories)} battles ({total_saved} files)")

        except Exception as e:
            print(f"Error saving trajectory {i}: {e}")
            continue

    if verbose:
        print(f"Saved {total_saved} files from {len(trajectories)} battles to {format_dir}")


def _save_single_replay(
    replay_data: Dict[str, Any],
    format_dir: Path,
    battle_format: str,
    battle_id: int,
    winner: int,
    player_perspective: int,
):
    """
    Save a single replay file to disk.

    Args:
        replay_data: Replay data dictionary
        format_dir: Directory to save to
        battle_format: Battle format string
        battle_id: Battle ID number
        winner: Winner (1 or 2)
        player_perspective: Which player's perspective (1 or 2)
    """
    # Generate filename
    battle_id_str = f"metamon-{battle_format}-{battle_id:06d}"
    rating = "Unrated"
    p1 = "PyKMNP1"
    p2 = "PyKMNP2"
    date = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    # Result from this player's perspective
    result = "WIN" if winner == player_perspective else "LOSS"

    # Add player suffix to distinguish perspectives
    filename = f"{battle_id_str}_{rating}_{p1}_vs_{p2}_{date}_{result}_P{player_perspective}.json.lz4"
    filepath = format_dir / filename

    # Save as compressed JSON
    json_str = json.dumps(replay_data)
    compressed = lz4.frame.compress(json_str.encode("utf-8"))

    with open(filepath, "wb") as f:
        f.write(compressed)


def _trajectory_to_replay(
    trajectory: Trajectory,
    mappings: Mappings,
    battle_format: str,
    player: int = 1,
) -> Dict[str, Any]:
    """
    Convert Trajectory to metamon ParsedReplay format from a player's perspective.

    Args:
        trajectory: Trajectory object from pypkmn battle
        mappings: Precomputed mappings
        battle_format: Battle format string
        player: Which player's perspective (1 or 2)

    Returns:
        Dictionary representing a ParsedReplay in metamon format.
        Format matches what parsed_replay_dset.py expects:
        - "states": list of UniversalState dicts (from specified player's perspective)
        - "actions": list of action indices (from specified player's perspective)
        - "rewards": list of rewards (from specified player's perspective)
    """
    states = []
    actions = []
    rewards = []

    for transition in trajectory.transitions:
        # Select features based on player perspective
        if player == 1:
            features = transition.features_p1
            action = transition.action_p1
            reward = transition.reward_p1
        else:  # player == 2
            features = transition.features_p2
            action = transition.action_p2
            reward = transition.reward_p2

        # Convert features to UniversalState
        state = features_to_universal_state(
            features,
            mappings,
            battle_format,
        )
        states.append(_universal_state_to_dict(state))
        actions.append(action)
        rewards.append(reward)

    # Build replay data structure (matches ParsedReplay format)
    replay_data = {
        "format": battle_format,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "winner": trajectory.winner,
        "num_turns": len(trajectory.transitions),
        "timestamp": datetime.now().isoformat(),
        "source": "pypkmn",
    }

    return replay_data


def _universal_state_to_dict(state) -> Dict[str, Any]:
    """
    Convert UniversalState to dictionary for JSON serialization.

    Args:
        state: UniversalState object

    Returns:
        Dictionary representation of the state.
    """
    return {
        "format": state.format,
        "player_active_pokemon": _universal_pokemon_to_dict(state.player_active_pokemon),
        "opponent_active_pokemon": _universal_pokemon_to_dict(state.opponent_active_pokemon),
        "available_switches": [
            _universal_pokemon_to_dict(p) for p in state.available_switches
        ],
        "player_prev_move": _universal_move_to_dict(state.player_prev_move),
        "opponent_prev_move": _universal_move_to_dict(state.opponent_prev_move),
        "opponents_remaining": state.opponents_remaining,
        "player_conditions": state.player_conditions,
        "opponent_conditions": state.opponent_conditions,
        "weather": state.weather,
        "battle_field": state.battle_field,
        "forced_switch": state.forced_switch,
        "battle_won": state.battle_won,
        "battle_lost": state.battle_lost,
        "can_tera": state.can_tera,
        "opponent_teampreview": state.opponent_teampreview,
    }


def _universal_pokemon_to_dict(pokemon) -> Dict[str, Any]:
    """Convert UniversalPokemon to dictionary."""
    return {
        "name": pokemon.name,
        "hp_pct": pokemon.hp_pct,
        "types": pokemon.types,
        "item": pokemon.item,
        "ability": pokemon.ability,
        "lvl": pokemon.lvl,
        "status": pokemon.status,
        "effect": pokemon.effect,
        "moves": [_universal_move_to_dict(m) for m in pokemon.moves],
        "atk_boost": pokemon.atk_boost,
        "spa_boost": pokemon.spa_boost,
        "def_boost": pokemon.def_boost,
        "spd_boost": pokemon.spd_boost,
        "spe_boost": pokemon.spe_boost,
        "accuracy_boost": pokemon.accuracy_boost,
        "evasion_boost": pokemon.evasion_boost,
        "base_atk": pokemon.base_atk,
        "base_spa": pokemon.base_spa,
        "base_def": pokemon.base_def,
        "base_spd": pokemon.base_spd,
        "base_spe": pokemon.base_spe,
        "base_hp": pokemon.base_hp,
        "tera_type": pokemon.tera_type,
        "base_species": pokemon.base_species,
    }


def _universal_move_to_dict(move) -> Dict[str, Any]:
    """Convert UniversalMove to dictionary."""
    return {
        "name": move.name,
        "move_type": move.move_type,
        "category": move.category,
        "base_power": move.base_power,
        "accuracy": move.accuracy,
        "priority": move.priority,
        "current_pp": move.current_pp,
        "max_pp": move.max_pp,
    }


def load_trajectory(filepath: Path | str) -> Dict[str, Any]:
    """
    Load a trajectory from .json.lz4 file.

    Args:
        filepath: Path to .json.lz4 file

    Returns:
        Dictionary with trajectory data.
    """
    filepath = Path(filepath).expanduser()

    with open(filepath, "rb") as f:
        compressed = f.read()

    decompressed = lz4.frame.decompress(compressed)
    replay_data = json.loads(decompressed.decode("utf-8"))

    return replay_data
