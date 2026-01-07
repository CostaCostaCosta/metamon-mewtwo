#!/usr/bin/env python3
"""
Generate battles between two different models (cross-play evaluation).

This script runs head-to-head battles between two models to:
1. Evaluate relative strength
2. Generate diverse training data
3. Identify exploitable weaknesses
4. Measure win rates and ELO

Usage:
    python scripts/generate_crossplay_battles.py \
        --team_dir ~/metamon_cache/teams/modern_replays_v2 \
        --num_battles 1000 \
        --num_envs 256 \
        --save_dir ~/metamon/trajectories/crossplay \
        --format gen1ou \
        --model_p1 Kakuna \
        --model_p2 SyntheticRLV2 \
        --device cuda \
        --verbose
"""

import argparse
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from metamon.env.inference_wrapper import InferenceWrapper
from metamon.env.pykmn import load_random_teams, save_trajectories, precompute_mappings
from metamon.env.pykmn.policy_runner import LocalPolicyRunner
from metamon.rl.pretrained import get_pretrained_model


def main():
    parser = argparse.ArgumentParser(description="Cross-play battles between two models")
    parser.add_argument("--team_dir", type=str, required=True, help="Team directory")
    parser.add_argument("--num_battles", type=int, default=1000, help="Number of battles")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--format", type=str, default="gen1ou", help="Battle format")

    # Player 1 model
    parser.add_argument("--model_p1", type=str, required=True, help="Model for Player 1 (e.g., Kakuna)")
    parser.add_argument("--checkpoint_p1", type=int, default=None, help="Player 1 checkpoint")
    parser.add_argument("--temperature_p1", type=float, default=1.0, help="Player 1 temperature")

    # Player 2 model
    parser.add_argument("--model_p2", type=str, required=True, help="Model for Player 2 (e.g., SyntheticRLV2)")
    parser.add_argument("--checkpoint_p2", type=int, default=None, help="Player 2 checkpoint")
    parser.add_argument("--temperature_p2", type=float, default=1.0, help="Player 2 temperature")

    # Other options
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per battle")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--swap_sides", action="store_true",
                       help="Run battles with swapped sides (doubles data, tests side bias)")

    args = parser.parse_args()

    # Expand paths
    team_dir = Path(args.team_dir).expanduser()
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create format-specific subdirectory
    format_dir = save_dir / args.format
    format_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CROSS-PLAY BATTLE GENERATION")
    print("=" * 70)
    print(f"Team directory: {team_dir}")
    print(f"Number of battles: {args.num_battles:,}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Battle format: {args.format}")
    print(f"Player 1: {args.model_p1} (temp={args.temperature_p1})")
    print(f"Player 2: {args.model_p2} (temp={args.temperature_p2})")
    print(f"Swap sides: {args.swap_sides}")
    print(f"Output directory: {format_dir}")
    print("=" * 70)

    # Calculate number of batches needed
    total_battles = args.num_battles * (2 if args.swap_sides else 1)
    num_batches = (total_battles + args.num_envs - 1) // args.num_envs
    battles_per_batch = args.num_envs

    print(f"\nBatch configuration:")
    print(f"  Total battles (including swaps): {total_battles:,}")
    print(f"  Batches: {num_batches}")
    print(f"  Battles per batch: {battles_per_batch}")

    # Precompute mappings for trajectory saving
    print(f"\nPrecomputing feature mappings...")
    mappings = precompute_mappings()

    # Statistics
    total_battles_collected = 0
    total_battles_saved = 0
    total_steps = 0
    all_trajectories = []

    print(f"\nStarting cross-play battles...")
    print(f"Target: {total_battles:,} battles")
    print("=" * 70)

    # Progress bar
    pbar = tqdm(total=total_battles, desc="Battles", unit="battle")

    batch_start_time = time.time()

    # Run battles with original sides
    for swap_idx in range(2 if args.swap_sides else 1):
        if swap_idx == 1:
            print(f"\n{'=' * 70}")
            print("SWAPPING SIDES - P1 ↔ P2")
            print(f"{'=' * 70}\n")

        # Determine which model is which player for this set
        if swap_idx == 0:
            # Normal: P1=model_p1, P2=model_p2
            model_p1_name, checkpoint_p1, temp_p1 = args.model_p1, args.checkpoint_p1, args.temperature_p1
            model_p2_name, checkpoint_p2, temp_p2 = args.model_p2, args.checkpoint_p2, args.temperature_p2
        else:
            # Swapped: P1=model_p2, P2=model_p1
            model_p1_name, checkpoint_p1, temp_p1 = args.model_p2, args.checkpoint_p2, args.temperature_p2
            model_p2_name, checkpoint_p2, temp_p2 = args.model_p1, args.checkpoint_p1, args.temperature_p1

        # Create policy runners (load once per side configuration)
        print(f"\nInitializing Player 1: {model_p1_name}")
        print(f"  Device: {args.device}")
        print(f"  Temperature: {temp_p1}")
        print(f"  Checkpoint: {checkpoint_p1 or 'default'}")

        start_time = time.time()
        policy_p1 = LocalPolicyRunner(
            model_name=model_p1_name,
            checkpoint=checkpoint_p1,
            device=args.device,
            temperature=temp_p1,
            verbose=args.verbose,
        )
        p1_load_time = time.time() - start_time
        print(f"  Loaded in {p1_load_time:.2f}s")

        print(f"\nInitializing Player 2: {model_p2_name}")
        print(f"  Device: {args.device}")
        print(f"  Temperature: {temp_p2}")
        print(f"  Checkpoint: {checkpoint_p2 or 'default'}")

        start_time = time.time()
        policy_p2 = LocalPolicyRunner(
            model_name=model_p2_name,
            checkpoint=checkpoint_p2,
            device=args.device,
            temperature=temp_p2,
            verbose=args.verbose,
        )
        p2_load_time = time.time() - start_time
        print(f"  Loaded in {p2_load_time:.2f}s")

        # Verify models are compatible
        if policy_p1.action_dim != policy_p2.action_dim:
            print(f"\n{'=' * 70}")
            print("ERROR: Incompatible Action Spaces")
            print(f"{'=' * 70}")
            print(f"Player 1 ({model_p1_name}): {policy_p1.action_dim} actions")
            print(f"Player 2 ({model_p2_name}): {policy_p2.action_dim} actions")
            print("\nModels must have the same action space to battle.")
            print("Use models from the same generation or architecture.")
            print(f"{'=' * 70}")
            return

        # Run battles for this configuration
        battles_this_config = args.num_battles
        battles_collected_this_config = 0

        while battles_collected_this_config < battles_this_config:
            # Determine how many battles we still need
            battles_remaining = battles_this_config - battles_collected_this_config
            current_batch_size = min(args.num_envs, battles_remaining)

            if current_batch_size <= 0:
                break

            # Load fresh teams for this batch
            teams_p1 = load_random_teams(team_dir, args.format, current_batch_size)
            teams_p2 = load_random_teams(team_dir, args.format, current_batch_size)

            # Create wrapper with trajectory tracking enabled
            wrapper = InferenceWrapper(
                teams_p1=teams_p1,
                teams_p2=teams_p2,
                num_envs=current_batch_size,
                track_trajectories=True,
            )

            # Reset
            obs_p1, obs_p2, legal_p1, legal_p2 = wrapper.reset()
            policy_p1.reset(batch_size=current_batch_size)
            policy_p2.reset(batch_size=current_batch_size)

            # Run until all battles complete
            batch_steps = 0
            battles_done = 0

            while battles_done < current_batch_size and batch_steps < args.max_steps:
                # Infer actions for both players
                actions_p1 = policy_p1.infer(obs_p1, legal_p1)
                actions_p2 = policy_p2.infer(obs_p2, legal_p2)

                # Step environment
                obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = wrapper.step(
                    actions_p1, actions_p2
                )

                # Update policy state
                policy_p1.update_rewards(rewards_p1)
                policy_p2.update_rewards(rewards_p2)
                policy_p1.reset_hidden_state_for_dones(dones)
                policy_p2.reset_hidden_state_for_dones(dones)

                # Extract legal masks
                legal_p1 = info['legal_masks_p1']
                legal_p2 = info['legal_masks_p2']

                batch_steps += 1
                battles_done = info.get('num_done', 0)

            # Collect completed trajectories from this batch
            completed = wrapper.get_completed_trajectories()
            to_keep = min(len(completed), battles_remaining)
            batch_trajectories = completed[:to_keep]

            # Save this batch immediately
            if len(batch_trajectories) > 0:
                save_start = time.time()
                save_trajectories(
                    trajectories=batch_trajectories,
                    output_dir=save_dir,
                    mappings=mappings,
                    battle_format=args.format,
                    verbose=False,
                )
                save_time = time.time() - save_start
                total_battles_saved += len(batch_trajectories)
                all_trajectories.extend(batch_trajectories)

                if args.verbose:
                    side_label = "(swapped)" if swap_idx == 1 else "(normal)"
                    pbar.write(f"  💾 {side_label} Saved {len(batch_trajectories)} trajectories in {save_time:.1f}s (total: {total_battles_saved:,})")

            battles_collected_this_config += to_keep
            total_battles_collected += to_keep
            total_steps += batch_steps
            pbar.update(to_keep)

            # Stats
            elapsed = time.time() - batch_start_time
            battles_per_sec = total_battles_collected / elapsed
            steps_per_sec = total_steps / elapsed
            pbar.set_postfix({
                'battles/s': f'{battles_per_sec:.1f}',
                'steps/s': f'{steps_per_sec:.0f}',
                'saved': f'{total_battles_saved:,}',
            })

    pbar.close()

    # Final statistics
    total_time = time.time() - batch_start_time

    print("\n" + "=" * 70)
    print("CROSS-PLAY COMPLETE")
    print("=" * 70)
    print(f"Battles collected: {total_battles_collected:,} / {total_battles:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Performance:")
    print(f"  Battles/sec: {total_battles_collected / total_time:.1f}")
    print(f"  Steps/sec: {total_steps / total_time:.0f}")
    print(f"  Avg steps/battle: {total_steps / total_battles_collected:.1f}")
    print("=" * 70)

    print(f"\nTotal trajectories saved to disk: {total_battles_saved:,}")
    print(f"Output directory: {format_dir}")

    # Compute win statistics
    if len(all_trajectories) > 0:
        p1_wins = sum(1 for t in all_trajectories if t.winner == 1)
        p2_wins = sum(1 for t in all_trajectories if t.winner == 2)
        ties = sum(1 for t in all_trajectories if t.winner == 0)

        print(f"\nWin statistics:")
        if args.swap_sides:
            # With swapped sides, need to attribute wins to actual models
            # First half: P1=model_p1, second half: P1=model_p2
            half = args.num_battles
            first_half = all_trajectories[:half]
            second_half = all_trajectories[half:]

            # model_p1 wins = (P1 wins in first half) + (P2 wins in second half)
            model_p1_wins = (
                sum(1 for t in first_half if t.winner == 1) +
                sum(1 for t in second_half if t.winner == 2)
            )
            # model_p2 wins = (P2 wins in first half) + (P1 wins in second half)
            model_p2_wins = (
                sum(1 for t in first_half if t.winner == 2) +
                sum(1 for t in second_half if t.winner == 1)
            )

            print(f"  {args.model_p1} wins: {model_p1_wins:,} ({100 * model_p1_wins / len(all_trajectories):.1f}%)")
            print(f"  {args.model_p2} wins: {model_p2_wins:,} ({100 * model_p2_wins / len(all_trajectories):.1f}%)")
            print(f"  Ties: {ties:,} ({100 * ties / len(all_trajectories):.1f}%)")

            # Estimate ELO difference (rough approximation)
            if model_p1_wins + model_p2_wins > 0:
                win_rate_p1 = model_p1_wins / (model_p1_wins + model_p2_wins)
                if 0.001 < win_rate_p1 < 0.999:
                    elo_diff = -400 * np.log10((1 - win_rate_p1) / win_rate_p1)
                    print(f"\nEstimated ELO difference: {elo_diff:+.0f} ({args.model_p1} vs {args.model_p2})")
        else:
            print(f"  P1 ({args.model_p1}) wins: {p1_wins:,} ({100 * p1_wins / len(all_trajectories):.1f}%)")
            print(f"  P2 ({args.model_p2}) wins: {p2_wins:,} ({100 * p2_wins / len(all_trajectories):.1f}%)")
            print(f"  Ties: {ties:,} ({100 * ties / len(all_trajectories):.1f}%)")
            print(f"\n  ⚠️  Consider using --swap_sides to control for side bias")

    # Verify file count
    saved_files = list(format_dir.glob("*.json.lz4"))
    print(f"\nVerification:")
    print(f"  Files on disk: {len(saved_files):,}")
    print(f"  Expected: {total_battles_saved:,}")
    if len(saved_files) == total_battles_saved:
        print(f"  ✓ File count matches!")
    else:
        print(f"  ⚠️  Mismatch detected!")

    print("=" * 70)
    print("✓ COMPLETE - Cross-play trajectories saved successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
