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
):
    """
    Save trajectories to .json.lz4 files in metamon format.

    Args:
        trajectories: List of Trajectory objects from pypkmn battles
        output_dir: Directory to save trajectory files
        mappings: Precomputed mappings for feature conversion
        battle_format: Battle format string (e.g., "gen1ou")
        verbose: Whether to print progress

    Output format:
        output_dir/
            {battle_format}/
                {uuid}_pypkmn.json.lz4
                {uuid}_pypkmn.json.lz4
                ...

    Each .json.lz4 file contains a battle replay in the format:
        {
            "states": [...],  # List of UniversalState dicts
            "actions_p1": [...],  # List of action indices
            "actions_p2": [...],
            "winner": 1 or 2,
            "metadata": {...}
        }
    """
    output_dir = Path(output_dir).expanduser()
    format_dir = output_dir / battle_format
    format_dir.mkdir(parents=True, exist_ok=True)

    for i, trajectory in enumerate(trajectories):
        try:
            # Convert trajectory to metamon format
            replay_data = _trajectory_to_replay(trajectory, mappings, battle_format)

            # Generate unique filename
            replay_id = str(uuid.uuid4())[:8]
            filename = f"{replay_id}_pypkmn.json.lz4"
            filepath = format_dir / filename

            # Save as compressed JSON
            json_str = json.dumps(replay_data)
            compressed = lz4.frame.compress(json_str.encode("utf-8"))

            with open(filepath, "wb") as f:
                f.write(compressed)

            if verbose and (i + 1) % 100 == 0:
                print(f"Saved {i + 1}/{len(trajectories)} trajectories")

        except Exception as e:
            print(f"Error saving trajectory {i}: {e}")
            continue

    if verbose:
        print(f"Saved {len(trajectories)} trajectories to {format_dir}")


def _trajectory_to_replay(
    trajectory: Trajectory,
    mappings: Mappings,
    battle_format: str,
) -> Dict[str, Any]:
    """
    Convert Trajectory to metamon ParsedReplay format.

    Args:
        trajectory: Trajectory object from pypkmn battle
        mappings: Precomputed mappings
        battle_format: Battle format string

    Returns:
        Dictionary representing a ParsedReplay in metamon format.
    """
    # Convert features to UniversalState for each transition
    states_p1 = []
    states_p2 = []
    actions_p1 = []
    actions_p2 = []
    rewards_p1 = []
    rewards_p2 = []

    for transition in trajectory.transitions:
        # Convert P1 features to UniversalState
        state_p1 = features_to_universal_state(
            transition.features_p1,
            mappings,
            battle_format,
        )
        states_p1.append(_universal_state_to_dict(state_p1))

        # Convert P2 features to UniversalState
        state_p2 = features_to_universal_state(
            transition.features_p2,
            mappings,
            battle_format,
        )
        states_p2.append(_universal_state_to_dict(state_p2))

        # Store actions and rewards
        actions_p1.append(transition.action_p1)
        actions_p2.append(transition.action_p2)
        rewards_p1.append(transition.reward_p1)
        rewards_p2.append(transition.reward_p2)

    # Build replay data structure
    replay_data = {
        "format": battle_format,
        "winner": trajectory.winner,
        "num_turns": len(trajectory.transitions),
        "timestamp": datetime.now().isoformat(),
        "source": "pypkmn",
        # P1 perspective
        "states_p1": states_p1,
        "actions_p1": actions_p1,
        "rewards_p1": rewards_p1,
        # P2 perspective
        "states_p2": states_p2,
        "actions_p2": actions_p2,
        "rewards_p2": rewards_p2,
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
