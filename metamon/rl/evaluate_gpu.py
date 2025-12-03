"""
GPU-Accelerated Evaluation using Remote Policy Server

This module provides evaluation functions that use a remote policy server
instead of loading models locally, enabling efficient batched GPU inference
across multiple battle workers.

Based on metamon/rl/evaluate.py but modified to use PolicyClient.
"""

import json
import collections
import functools
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

import metamon
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
    PretrainedModel,
)
from metamon.rl.policy_client import PolicyClient, RemoteAMAGOAgent
from metamon.baselines import get_baseline
from metamon.rl.metamon_to_amago import (
    make_baseline_env,
    make_local_ladder_env,
    make_pokeagent_ladder_env,
)


def pretrained_vs_local_ladder_gpu(
    pretrained_model_name: str,
    server_address: str,
    username: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    battle_backend: str = "poke-env",
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a pretrained model on the local ladder using remote GPU inference.

    Uses AMAGO's evaluate_test() method with a RemoteAMAGOAgent that queries
    a policy server. This ensures proper Timestep preprocessing and AMAGO state
    management while using remote GPU inference.

    Args:
        pretrained_model_name: Name of pretrained model (e.g., "SyntheticRLV2")
        server_address: Address of policy server (e.g., "tcp://localhost:5555")
        username: Showdown username
        battle_format: Battle format (e.g., "gen1ou")
        team_set: Team set to use
        total_battles: Number of battles to run
        avatar: Showdown avatar
        battle_backend: Battle backend ("poke-env" or "metamon")
        save_trajectories_to: Directory to save trajectories
        save_team_results_to: Directory to save team results

    Returns:
        Dictionary with evaluation results
    """
    # Get pretrained model metadata (for observation/action/reward spaces)
    pretrained_model = get_pretrained_model(pretrained_model_name)

    # Initialize AMAGO agent on CPU to save GPU memory
    # Workers load model structure on CPU, policy server handles GPU inference
    print(f"[evaluate_gpu] Initializing AMAGO agent on CPU (GPU inference via server)...")
    base_agent = pretrained_model.initialize_agent(checkpoint=None, log=False, device='cpu')

    # Create policy client (connects to remote server)
    print(f"[evaluate_gpu] Connecting to policy server at {server_address}")
    client = PolicyClient(server_address, timeout_ms=5000, max_retries=3)

    # Wrap agent to redirect policy calls to remote server
    # This keeps all AMAGO infrastructure but uses remote GPU for inference
    agent = RemoteAMAGOAgent(base_agent, client, verbose=False)

    # Configure agent for single synchronous environment (like evaluate.py)
    agent.env_mode = "sync"
    agent.parallel_actors = 1

    # Create environment factory
    make_env = functools.partial(
        make_local_ladder_env,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        num_battles=total_battles,
        player_username=username,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_team_results_to=save_team_results_to,
    )

    # Use AMAGO's evaluate_test() method (handles all Timestep preprocessing)
    print(f"[evaluate_gpu] Starting {total_battles} battles as {username}")
    results = agent.evaluate_test(
        [make_env],
        timesteps=total_battles * 1000,
        episodes=total_battles,
    )

    # Print client statistics
    client_stats = client.get_stats()
    print(f"\n[evaluate_gpu] Client stats:")
    print(f"  Total requests: {client_stats['total_requests']}")
    print(f"  Avg latency: {client_stats['avg_latency_ms']:.1f}ms")
    print(f"  Error rate: {client_stats['error_rate']:.2%}")

    # Cleanup
    client.close()

    # Add client stats to results
    results["client_stats"] = client_stats

    return results


def pretrained_vs_pokeagent_ladder_gpu(
    pretrained_model_name: str,
    server_address: str,
    username: str,
    password: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    battle_backend: str = "poke-env",
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate on PokéAgent Challenge ladder using remote GPU inference.

    Uses AMAGO's evaluate_test() method with a RemoteAMAGOAgent that queries
    a policy server. This ensures proper Timestep preprocessing and AMAGO state
    management while using remote GPU inference.
    """
    # Get pretrained model metadata
    pretrained_model = get_pretrained_model(pretrained_model_name)

    # Initialize AMAGO agent on CPU to save GPU memory
    # Workers load model structure on CPU, policy server handles GPU inference
    print(f"[evaluate_gpu] Initializing AMAGO agent on CPU (GPU inference via server)...")
    base_agent = pretrained_model.initialize_agent(checkpoint=None, log=False, device='cpu')

    # Create policy client
    print(f"[evaluate_gpu] Connecting to policy server at {server_address}")
    client = PolicyClient(server_address, timeout_ms=5000, max_retries=3)

    # Wrap agent to redirect policy calls to remote server
    agent = RemoteAMAGOAgent(base_agent, client, verbose=False)

    # Configure agent for single synchronous environment
    agent.env_mode = "sync"
    agent.parallel_actors = 1

    # Create environment factory
    make_env = functools.partial(
        make_pokeagent_ladder_env,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        num_battles=total_battles,
        player_username=username,
        player_password=password,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_team_results_to=save_team_results_to,
    )

    # Use AMAGO's evaluate_test() method
    print(f"[evaluate_gpu] Starting {total_battles} battles on PokéAgent ladder as {username}")
    results = agent.evaluate_test(
        [make_env],
        timesteps=total_battles * 1000,
        episodes=total_battles,
    )

    # Print client statistics
    client_stats = client.get_stats()
    print(f"\n[evaluate_gpu] Client stats:")
    print(f"  Total requests: {client_stats['total_requests']}")
    print(f"  Avg latency: {client_stats['avg_latency_ms']:.1f}ms")
    print(f"  Error rate: {client_stats['error_rate']:.2%}")

    # Cleanup
    client.close()

    # Add client stats to results
    results["client_stats"] = client_stats

    return results


def add_cli(parser):
    """Add CLI arguments for GPU-accelerated evaluation."""
    parser.add_argument(
        "--model",
        required=True,
        choices=get_pretrained_model_names(),
        help="Pretrained model name",
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Policy server address (e.g., tcp://localhost:5555)",
    )
    parser.add_argument(
        "--eval-type",
        required=True,
        choices=["ladder", "pokeagent"],
        help="Evaluation type: 'ladder' for local Showdown, 'pokeagent' for PokéAgent Challenge",
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Showdown username",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Password (required for pokeagent ladder)",
    )
    parser.add_argument(
        "--battle-format",
        default="gen1ou",
        help="Battle format (e.g., gen1ou, gen2ou, etc.)",
    )
    parser.add_argument(
        "--team-set",
        default="competitive",
        help="Team set name",
    )
    parser.add_argument(
        "--total-battles",
        type=int,
        default=100,
        help="Number of battles to run",
    )
    parser.add_argument(
        "--avatar",
        default="red-gen1main",
        help="Showdown avatar",
    )
    parser.add_argument(
        "--battle-backend",
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle backend",
    )
    parser.add_argument(
        "--save-trajectories-to",
        default=None,
        help="Directory to save trajectories",
    )
    parser.add_argument(
        "--save-team-results-to",
        default=None,
        help="Directory to save team results",
    )

    return parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="GPU-accelerated evaluation using remote policy server"
    )
    add_cli(parser)
    args = parser.parse_args()

    # Load team set
    team_set_type = (
        metamon.env.PokeAgentTeamSet
        if args.eval_type == "pokeagent"
        else metamon.env.TeamSet
    )
    team_set = metamon.env.get_metamon_teams(
        args.battle_format, args.team_set, set_type=team_set_type
    )

    # Run evaluation
    if args.eval_type == "ladder":
        results = pretrained_vs_local_ladder_gpu(
            pretrained_model_name=args.model,
            server_address=args.server,
            username=args.username,
            battle_format=args.battle_format,
            team_set=team_set,
            total_battles=args.total_battles,
            avatar=args.avatar,
            battle_backend=args.battle_backend,
            save_trajectories_to=args.save_trajectories_to,
            save_team_results_to=args.save_team_results_to,
        )
    elif args.eval_type == "pokeagent":
        if not args.password:
            raise ValueError("--password required for pokeagent ladder")

        results = pretrained_vs_pokeagent_ladder_gpu(
            pretrained_model_name=args.model,
            server_address=args.server,
            username=args.username,
            password=args.password,
            battle_format=args.battle_format,
            team_set=team_set,
            total_battles=args.total_battles,
            avatar=args.avatar,
            battle_backend=args.battle_backend,
            save_trajectories_to=args.save_trajectories_to,
            save_team_results_to=args.save_team_results_to,
        )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
