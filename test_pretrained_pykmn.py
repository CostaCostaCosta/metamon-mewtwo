"""
Test script for pretrained model inference with pykmn.

This script tests that we can:
1. Load a pretrained AMAGO agent
2. Run inference on pykmn observations
3. Get valid actions back

This is the minimal checkpoint before trying full episodes.
"""

import os
import numpy as np

# Set cache directory
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

from metamon.env.pykmn.vector_env import PyKMNVectorEnv
from metamon.env.pykmn.policy_runner import LocalPolicyRunner, RandomPolicyRunner
from metamon.env.pykmn.team_parser import parse_showdown_team
from metamon.interface import get_observation_space, get_reward_function


def test_single_step():
    """Test single-step inference with pretrained model."""
    print("=" * 60)
    print("Testing Pretrained Model Inference with PyKMN")
    print("=" * 60)

    # Load teams
    print("\n1. Loading teams...")
    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ][:2]  # Just 2 teams for minimal test

    teams = [parse_showdown_team(open(f).read()) for f in team_files]
    print(f"   ✓ Loaded {len(teams)} teams")

    # Load pretrained model first to get its observation space
    print("\n2. Loading pretrained model (this may take a minute)...")
    try:
        from metamon.rl.pretrained import get_pretrained_model

        pretrained_cls = get_pretrained_model("SyntheticRLV2")
        policy = LocalPolicyRunner(
            model_name="SyntheticRLV2",
            checkpoint=48,  # Default best checkpoint
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            temperature=1.0,
        )
        print(f"   ✓ Model loaded successfully!")

        # Use the pretrained model's observation space and action space
        obs_space = pretrained_cls.observation_space
        action_space = pretrained_cls.action_space
        reward_fn = pretrained_cls.reward_function
        print(f"   ✓ Using {obs_space.__class__.__name__}")
        print(f"   ✓ Using {action_space.__class__.__name__} ({action_space.gym_space.n} actions)")
        print(f"   ✓ Using {reward_fn.__class__.__name__}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create environment (single battle)
    print("\n3. Creating pykmn environment...")
    vec_env = PyKMNVectorEnv(
        teams_p1=[teams[0]],
        teams_p2=[teams[1]],
        num_envs=1,
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=False,  # Don't need trajectories for this test
    )
    print(f"   ✓ Environment created with 1 parallel battle")

    # Reset environment
    print("\n4. Resetting environment and getting initial observation...")
    obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()
    policy.reset()
    print(f"   ✓ Observation keys: {list(obs_p1.keys())}")
    print(f"   ✓ Observation shapes:")
    for key, value in obs_p1.items():
        print(f"      - {key}: {value.shape}")
    print(f"   ✓ Legal mask shape: {masks_p1.shape}")
    print(f"   ✓ Num legal actions: {masks_p1.sum()}")

    # Run single inference step
    print("\n5. Running single inference step...")
    try:
        action = policy.infer(obs_p1, masks_p1)
        print(f"   ✓ Action selected: {action}")
        print(f"   ✓ Action shape: {action.shape}")
        print(f"   ✓ Action is legal: {masks_p1[0, action[0]]}")

        # Verify action is valid
        assert action.shape == (1,), f"Expected shape (1,), got {action.shape}"
        assert 0 <= action[0] < 13, f"Action {action[0]} out of range [0, 13)"
        assert masks_p1[0, action[0]], f"Action {action[0]} is illegal!"

        print(f"\n   ✓ Single-step inference works!")
        return True

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_step():
    """Test multi-step inference (5 steps) to verify state tracking."""
    print("\n" + "=" * 60)
    print("Testing Multi-Step Inference")
    print("=" * 60)

    # Setup (same as above)
    from metamon.rl.pretrained import get_pretrained_model

    team_dir = "/home/eddie/metamon_cache/teams/modern_replays_v2/gen1ou"
    team_files = [
        os.path.join(team_dir, f)
        for f in os.listdir(team_dir)
        if f.endswith(".gen1ou_team")
    ][:2]
    teams = [parse_showdown_team(open(f).read()) for f in team_files]

    # Load pretrained model and use its observation space
    pretrained_cls = get_pretrained_model("SyntheticRLV2")
    obs_space = pretrained_cls.observation_space
    reward_fn = pretrained_cls.reward_function

    vec_env = PyKMNVectorEnv(
        teams_p1=[teams[0]],
        teams_p2=[teams[1]],
        num_envs=1,
        obs_space=obs_space,
        reward_fn=reward_fn,
        track_trajectories=False,
    )

    policy = LocalPolicyRunner(
        model_name="SyntheticRLV2",
        checkpoint=48,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    # Reset
    obs_p1, obs_p2, masks_p1, masks_p2 = vec_env.reset()
    policy.reset()

    # Create random opponent
    random_policy = RandomPolicyRunner()

    # Run 5 steps
    print("\nRunning 5 steps...")
    for step in range(5):
        # Get actions
        action_p1 = policy.infer(obs_p1, masks_p1)
        action_p2 = random_policy.infer(obs_p2, masks_p2)

        # Step environment
        obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = vec_env.step(
            action_p1, action_p2
        )

        # Update policy rewards
        policy.update_rewards(rewards_p1)

        # Get new masks
        masks_p1, masks_p2 = vec_env._extract_legal_masks()

        print(
            f"   Step {step+1}: P1 action={action_p1[0]}, P2 action={action_p2[0]}, "
            f"rewards=({rewards_p1[0]:.2f}, {rewards_p2[0]:.2f}), done={dones[0]}"
        )

        if dones[0]:
            print(f"   Battle finished at step {step+1}!")
            break

    print("\n   ✓ Multi-step inference works!")
    return True


if __name__ == "__main__":
    # Run tests
    success = test_single_step()

    if success:
        print("\n" + "=" * 60)
        print("✓ Single-step test PASSED")
        print("=" * 60)

        try:
            test_multi_step()
            print("\n" + "=" * 60)
            print("✓ Multi-step test PASSED")
            print("✓ ALL TESTS PASSED!")
            print("=" * 60)
        except Exception as e:
            print(f"\n✗ Multi-step test FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "=" * 60)
        print("✗ Single-step test FAILED")
        print("=" * 60)
