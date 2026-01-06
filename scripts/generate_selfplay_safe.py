#!/usr/bin/env python3
"""
Safer version of selfplay generation with better memory management.
Runs batches in chunks with cleanup between them.
"""

import os
import sys
import gc
import argparse
import time
from pathlib import Path
import numpy as np
import torch

# Set cache directory before imports
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")

from metamon.env.pykmn import (
    load_random_teams,
    PyKMNVectorEnv,
    LocalPolicyRunner,
    save_trajectories,
    precompute_mappings,
)
from metamon.interface import TokenizedObservationSpace, DefaultObservationSpace
from metamon.tokenizer import PokemonTokenizer
from metamon.rl.pretrained import get_pretrained_model


def generate_batch_safely(
    model_name: str,
    batch_size: int,
    battles_per_batch: int,
    format_name: str,
    team_set: str,
    save_dir: str,
    device: str = "cuda",
    temperature: float = 1.0,
):
    """Generate one batch of battles with full cleanup afterward."""

    # Load teams for both players
    teams_p1 = load_random_teams(team_set, format_name, batch_size)
    teams_p2 = load_random_teams(team_set, format_name, batch_size)
    print(f"  Loaded {len(teams_p1)} teams for P1, {len(teams_p2)} teams for P2")

    # Create observation space
    tokenizer = PokemonTokenizer()
    vocab_path = os.path.join(os.environ["METAMON_CACHE_DIR"], "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer.load_tokens_from_disk(vocab_path)

    base_obs_space = DefaultObservationSpace()
    obs_space = TokenizedObservationSpace(base_obs_space, tokenizer)

    # Create environment
    env = PyKMNVectorEnv(
        num_envs=batch_size,
        teams_p1=teams_p1,
        teams_p2=teams_p2,
        obs_space=obs_space,
        reward_fn=None,  # Use default
        battle_format=format_name,
        track_trajectories=True,
    )

    # Create policy runner
    policy = LocalPolicyRunner(
        model_name=model_name,
        device=device,
        temperature=temperature,
        use_amp=(device == "cuda"),
        verbose=False,
    )

    # Reset
    obs_p1, obs_p2, legal_masks_p1, legal_masks_p2 = env.reset()
    policy.reset(batch_size)

    # Collect data
    battles_completed = 0
    max_steps = battles_per_batch * 500  # Max steps per battle estimate

    for step in range(max_steps):
        # Get actions
        actions_p1 = policy.infer(obs_p1, legal_masks_p1)
        actions_p2 = policy.infer(obs_p2, legal_masks_p2)

        # Step environment
        obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = env.step(actions_p1, actions_p2)

        # Update legal masks
        legal_masks_p1 = obs_p1.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))
        legal_masks_p2 = obs_p2.get("legal_actions_mask", np.ones((batch_size, 13), dtype=bool))

        # Update policy
        policy.update_rewards(rewards_p1)
        policy.reset_hidden_state_for_dones(dones)

        battles_completed += dones.sum()

        if battles_completed >= battles_per_batch:
            break

    # Get trajectories
    trajectories = env.completed_trajectories

    # Save
    if trajectories:
        save_trajectories(trajectories, save_dir, format_name)
        print(f"  Saved {len(trajectories)} trajectories")

    # CRITICAL: Cleanup
    del policy
    env._cleanup_battles_incremental()
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return len(trajectories)


def main():
    parser = argparse.ArgumentParser(description="Safe selfplay generation")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--num_battles", type=int, default=512, help="Total battles to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Parallel battles per batch")
    parser.add_argument("--battles_per_chunk", type=int, default=100, help="Battles before cleanup")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--format", type=str, default="gen1ou", help="Battle format")
    parser.add_argument("--team_set", type=str, default="smogon_pass2", help="Team set")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    print("=" * 70)
    print("SAFE BATCHED SELF-PLAY GENERATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Target: {args.num_battles} battles")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.battles_per_chunk} battles")
    print(f"Output: {args.save_dir}")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate in chunks
    total_generated = 0
    chunk_num = 0

    while total_generated < args.num_battles:
        chunk_num += 1
        battles_this_chunk = min(args.battles_per_chunk, args.num_battles - total_generated)

        print(f"\n[Chunk {chunk_num}] Generating {battles_this_chunk} battles...")

        try:
            trajectories = generate_batch_safely(
                model_name=args.model,
                batch_size=args.batch_size,
                battles_per_batch=battles_this_chunk,
                format_name=args.format,
                team_set=args.team_set,
                save_dir=args.save_dir,
                device=args.device,
                temperature=args.temperature,
            )

            total_generated += trajectories
            print(f"  Total progress: {total_generated}/{args.num_battles} battles")

        except Exception as e:
            print(f"  ⚠️  Chunk failed: {e}")
            print("  Continuing with next chunk...")

        # Rest between chunks
        time.sleep(1)
        gc.collect()

    print(f"\n{'='*70}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total battles generated: {total_generated}")
    print(f"Output directory: {args.save_dir}")


if __name__ == "__main__":
    main()