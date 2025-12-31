"""
Test full battles with pretrained models to ensure they complete successfully.
"""

import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.env.pykmn.policy_runner import LocalPolicyRunner, RandomPolicyRunner
from metamon.env.pykmn.team_parser import load_random_teams
from metamon.rl.pretrained import get_pretrained_model


def test_full_battles(num_battles=5):
    """Test complete battles with pretrained model vs random."""
    print("=" * 60)
    print(f"Testing {num_battles} Full Battles (Pretrained vs Random)")
    print("=" * 60)

    # Load pretrained model
    print("\nLoading SyntheticRLV2...")
    pretrained_cls = get_pretrained_model("SyntheticRLV2")
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    policy_p1 = LocalPolicyRunner(
        model_name="SyntheticRLV2",
        checkpoint=48,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        temperature=1.0,
    )
    policy_p2 = RandomPolicyRunner()

    # Load teams
    print("Loading teams...")
    teams = load_random_teams(
        "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou",
        "gen1ou",
        num_battles * 2
    )

    print(f"\nRunning {num_battles} battles...\n")

    successes = 0
    total_steps = 0

    for i in range(num_battles):
        print(f"Battle {i+1}/{num_battles}:")

        # Create environment
        vec_env = PyKMNVectorEnv(
            teams_p1=[teams[i * 2]],
            teams_p2=[teams[i * 2 + 1]],
            num_envs=1,
            obs_space=obs_space,
            reward_fn=reward_fn,
            track_trajectories=False,
        )

        # Reset
        obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()
        policy_p1.reset()
        # RandomPolicyRunner doesn't need reset

        # Run battle
        step_count = 0
        max_steps = 1000

        while step_count < max_steps:
            # Get actions
            action_p1 = policy_p1.infer(obs_p1, masks_p1)
            action_p2 = policy_p2.infer(obs_p2, masks_p2)

            # Step
            obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = vec_env.step(
                action_p1, action_p2
            )

            # Update rewards
            policy_p1.update_rewards(rewards_p1)

            # Update masks
            masks_p1, masks_p2 = vec_env._extract_legal_masks()

            step_count += 1

            if dones[0]:
                print(f"  ✓ Completed at step {step_count}")
                print(f"    Final rewards: P1={rewards_p1[0]:.1f}, P2={rewards_p2[0]:.1f}")
                successes += 1
                total_steps += step_count
                break

        if step_count >= max_steps:
            print(f"  ✗ TIMEOUT at {max_steps} steps")

        vec_env.close()

    avg_steps = total_steps / successes if successes > 0 else 0

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Battles completed: {successes}/{num_battles}")
    print(f"  Average battle length: {avg_steps:.1f} steps")
    print("=" * 60)

    return successes == num_battles


if __name__ == "__main__":
    success = test_full_battles(num_battles=5)

    print()
    print("=" * 60)
    print("FINAL RESULT:")
    if success:
        print("  ✓ ALL BATTLES COMPLETED SUCCESSFULLY!")
        print("  ✓ PyKMN integration with pretrained models is working!")
    else:
        print("  ✗ Some battles failed to complete")
    print("=" * 60)
