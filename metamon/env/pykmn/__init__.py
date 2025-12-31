"""
PyKMN integration for fast offline policy evaluation and self-play data generation.

This package provides a high-performance alternative to the Showdown-based simulation
backend using the pypkmn (libpkmn) C/Zig engine.

Key components:
- team_parser: Convert Showdown team format to pypkmn Pokemon objects
- features: Two-tier state representation (fast numeric + slow UniversalState)
- action_mapper: Convert metamon actions to pypkmn choices with legal masks
- vector_env: Vectorized environment for batched simulation
- policy_runner: Abstraction for policy inference
- trajectory_saver: Save battles to .json.lz4 format

Performance targets:
- Sim-only: 100x+ faster than Showdown subprocess
- End-to-end: 10-100x faster including inference + serialization
"""

# Import all implemented modules
from .team_parser import parse_showdown_team, parse_team_file, load_random_teams
from .features import (
    Mappings,
    precompute_mappings,
    pykmn_to_features_raw,
    features_to_universal_state,
)
from .action_mapper import (
    get_legal_mask,
    metamon_action_to_choice,
    ActionMappings,
    filter_illegal_actions,
)
from .vector_env import PyKMNVectorEnv, Trajectory, Transition
from .policy_runner import (
    PolicyRunner,
    LocalPolicyRunner,
    RandomPolicyRunner,
    SelfPlayRunner,
    EvaluationRunner,
)
from .trajectory_saver import save_trajectories, load_trajectory

__all__ = [
    # Team parsing
    "parse_showdown_team",
    "parse_team_file",
    "load_random_teams",
    # State features
    "Mappings",
    "precompute_mappings",
    "pykmn_to_features_raw",
    "features_to_universal_state",
    # Action mapping
    "get_legal_mask",
    "metamon_action_to_choice",
    "ActionMappings",
    "filter_illegal_actions",
    # Environment
    "PyKMNVectorEnv",
    "Trajectory",
    "Transition",
    # Policy
    "PolicyRunner",
    "LocalPolicyRunner",
    "RandomPolicyRunner",
    "SelfPlayRunner",
    "EvaluationRunner",
    # Saving
    "save_trajectories",
    "load_trajectory",
]
