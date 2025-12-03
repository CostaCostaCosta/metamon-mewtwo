#!/usr/bin/env python3
"""Minimal test to debug remote policy integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.rl.gen1_binary_models import *
from metamon.rl.evaluate_gpu import pretrained_vs_local_ladder_gpu
from metamon.env import get_metamon_teams

print("=" * 80)
print("REMOTE POLICY DEBUG TEST")
print("=" * 80)

team_set = get_metamon_teams('gen1ou', 'joint_teams1')

print("\nStarting evaluation with remote server...")
results = pretrained_vs_local_ladder_gpu(
    pretrained_model_name='DampedBinarySuperV1_Epoch4',
    server_address='tcp://localhost:5555',
    username='DebugTest',
    battle_format='gen1ou',
    team_set=team_set,
    total_battles=1,  # Just 1 battle for quick debug
    battle_backend='poke-env',
    save_trajectories_to='/tmp/debug_test',
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Client stats: {results.get('client_stats', {})}")
print("=" * 80)
