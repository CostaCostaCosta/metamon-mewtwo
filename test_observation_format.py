#!/usr/bin/env python3
"""
Check the format of observations from PyKMN vector env
"""

import os
import sys
import numpy as np

# Add metamon to path
sys.path.insert(0, '/home/eddie/repos/metamon')

from pykmn.engine.gen1 import Battle, Pokemon
from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.interface import ExpandedObservationSpace, DefaultShapedReward


def create_simple_team():
    """Create a simple hardcoded Gen1 team."""
    team = [
        Pokemon(species="Tauros", moves=["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"]),
        Pokemon(species="Chansey", moves=["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
        Pokemon(species="Snorlax", moves=["Body Slam", "Hyper Beam", "Earthquake", "Self-Destruct"]),
        Pokemon(species="Starmie", moves=["Psychic", "Blizzard", "Thunder Wave", "Recover"]),
        Pokemon(species="Exeggutor", moves=["Psychic", "Sleep Powder", "Explosion", "Stun Spore"]),
        Pokemon(species="Alakazam", moves=["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
    ]
    return team


def main():
    """Check observation format."""
    print("=" * 70)
    print("OBSERVATION FORMAT CHECK")
    print("=" * 70)

    # Create small environment
    team = create_simple_team()
    teams = [team] * 2  # Just 2 envs

    env = PyKMNVectorEnv(
        num_envs=2,
        teams_p1=teams,
        teams_p2=teams,
        obs_space=ExpandedObservationSpace(),
        reward_fn=DefaultShapedReward(),
        battle_format="gen1ou",
        track_trajectories=False,
    )

    # Reset and get observations
    obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()

    print("\nObservation P1 keys:")
    for key in obs_p1.keys():
        value = obs_p1[key]
        print(f"  '{key}': shape={value.shape if hasattr(value, 'shape') else 'N/A'}, "
              f"dtype={value.dtype if hasattr(value, 'dtype') else type(value).__name__}")

        # Show first element if it's problematic
        if hasattr(value, 'dtype'):
            if value.dtype == np.object_ or 'str' in str(value.dtype):
                print(f"    First element: {value[0] if len(value.shape) > 0 else value}")
                print(f"    Type of first element: {type(value.flat[0] if hasattr(value, 'flat') else value)}")

    print("\nLegal masks P1:")
    print(f"  Shape: {legal_masks_p1.shape}, dtype: {legal_masks_p1.dtype}")
    print(f"  First env legal actions: {np.where(legal_masks_p1[0])[0]}")

    # Try to convert each observation type
    import torch

    print("\n" + "=" * 70)
    print("TORCH CONVERSION TEST")
    print("=" * 70)

    for key, value in obs_p1.items():
        try:
            if hasattr(value, 'dtype') and 'float' in str(value.dtype):
                tensor = torch.from_numpy(value)
                print(f"✓ '{key}': Successfully converted to torch tensor")
            elif hasattr(value, 'dtype') and 'int' in str(value.dtype):
                tensor = torch.from_numpy(value)
                print(f"✓ '{key}': Successfully converted to torch tensor")
            elif hasattr(value, 'dtype') and 'bool' in str(value.dtype):
                tensor = torch.from_numpy(value)
                print(f"✓ '{key}': Successfully converted to torch tensor")
            else:
                print(f"✗ '{key}': Cannot convert type {value.dtype if hasattr(value, 'dtype') else type(value)}")
        except Exception as e:
            print(f"✗ '{key}': Conversion failed - {e}")

    # Check what the observation space expects
    print("\n" + "=" * 70)
    print("EXPECTED OBSERVATION FORMAT")
    print("=" * 70)

    obs_space = ExpandedObservationSpace()
    print(f"Observation space type: {type(obs_space)}")

    # Get a single observation to check format
    from metamon.env.pykmn.feature_extractor import (
        pykmn_to_features_raw,
        features_to_universal_state,
        create_gen1_mappings
    )
    from pykmn.engine.common import Player

    battle = Battle(p1_team=team, p2_team=team)
    result, _ = battle.update_raw(0, 0)

    mappings = create_gen1_mappings()
    features = pykmn_to_features_raw(battle, result, Player.P1, mappings)
    state = features_to_universal_state(features, mappings)

    single_obs = obs_space(state, None)

    print("\nSingle observation keys (before batching):")
    for key, value in single_obs.items():
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            print(f"  '{key}': shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  '{key}': type={type(value)}")


if __name__ == "__main__":
    # Set environment variable for cache
    if "METAMON_CACHE_DIR" not in os.environ:
        os.environ["METAMON_CACHE_DIR"] = os.path.expanduser("~/metamon_cache")

    main()