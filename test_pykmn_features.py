#!/usr/bin/env python3
"""Quick test of pypkmn feature extraction."""

from metamon.env.pykmn import (
    parse_showdown_team,
    precompute_mappings,
    pykmn_to_features_raw,
)
from pykmn.engine.gen1 import Battle, Choice
from pykmn.engine.common import Player

# Sample teams
team1_text = """
Tauros
- Body Slam
- Hyper Beam
- Blizzard
- Earthquake

Chansey
- Ice Beam
- Thunderbolt
- Thunder Wave
- Soft-Boiled

Exeggutor
- Sleep Powder
- Psychic
- Double-Edge
- Explosion

Starmie
- Psychic
- Blizzard
- Thunder Wave
- Recover

Alakazam
- Psychic
- Seismic Toss
- Thunder Wave
- Recover

Snorlax
- Body Slam
- Reflect
- Earthquake
- Rest
"""

team2_text = """
Jynx
- Lovely Kiss
- Blizzard
- Psychic
- Rest

Starmie
- Psychic
- Thunderbolt
- Thunder Wave
- Recover

Alakazam
- Psychic
- Seismic Toss
- Thunder Wave
- Recover

Chansey
- Seismic Toss
- Reflect
- Thunder Wave
- Soft-Boiled

Snorlax
- Body Slam
- Reflect
- Self-Destruct
- Rest

Tauros
- Body Slam
- Hyper Beam
- Blizzard
- Earthquake
"""

print("Parsing teams...")
team1 = parse_showdown_team(team1_text)
team2 = parse_showdown_team(team2_text)
print(f"Parsed {len(team1)} vs {len(team2)} Pokemon")

print("\nInitializing battle...")
battle = Battle(p1_team=team1, p2_team=team2)

# Initial pass (team preview)
result, _ = battle.update(Choice.PASS(), Choice.PASS())

print("\nPrecomputing mappings...")
mappings = precompute_mappings()

print("\nExtracting features for P1...")
features_p1 = pykmn_to_features_raw(battle, result, Player.P1, mappings)

print("\nP1 Features:")
print(f"  Active species ID: {features_p1['active_species_id']}")
print(f"  Active HP %: {features_p1['active_hp_pct']}")
print(f"  Active status: {features_p1['active_status']}")
print(f"  Active moves: {features_p1['active_moves']}")
print(f"  Active PP: {features_p1['active_move_pp']}")
print(f"  Active max PP: {features_p1['active_move_max_pp']}")
print(f"  Active boosts: ATK={features_p1['active_atk_boost']}, DEF={features_p1['active_def_boost']}")
print(f"  Team species IDs: {features_p1['team_species_ids']}")
print(f"  Team HP %: {features_p1['team_hp_pct']}")
print(f"  Side conditions: {features_p1['side_condition']}")
print(f"  Forced switch: {features_p1['forced_switch']}")

print("\nExtracting features for P2...")
features_p2 = pykmn_to_features_raw(battle, result, Player.P2, mappings)

print("\nP2 Features:")
print(f"  Active species ID: {features_p2['active_species_id']}")
print(f"  Active HP %: {features_p2['active_hp_pct']}")

print("\n✅ Feature extraction successful!")
