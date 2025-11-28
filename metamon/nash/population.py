"""
Policy population management for Nash equilibrium training.

In the Nash framework, we maintain a population Π = {π₁, π₂, ..., π_K} of policies
representing different strategies. This module provides infrastructure for:
- Registering policies (RL checkpoints, heuristics, external baselines)
- Sampling opponents according to meta-strategy distributions
- Tracking policy metadata (name, type, checkpoint path)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

from metamon.rl.pretrained import get_pretrained_model, PretrainedModel
from metamon.baselines import get_baseline


@dataclass
class PolicyInfo:
    """Metadata for a policy in the population."""

    name: str
    policy_type: str  # "pretrained", "heuristic", "checkpoint"
    description: str = ""
    checkpoint: Optional[int] = None
    model_class: Optional[str] = None  # For pretrained models
    baseline_class: Optional[str] = None  # For heuristic baselines

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyInfo":
        """Load from dictionary."""
        return cls(**data)


class PolicyPopulation:
    """
    Manages a population of policies for Nash equilibrium training.

    The population Π contains different strategies that can be:
    1. Pretrained RL models (e.g., SyntheticRLV2, Gen1BinaryV0_Epoch2)
    2. Heuristic baselines (e.g., PokeEnvHeuristic, GymLeader)
    3. Checkpoints from PSRO training (best-responses)

    Example:
        >>> pop = PolicyPopulation()
        >>> pop.add_pretrained_policy("P0_SYN_V2_GEN1", "Gen1 baseline from SyntheticRLV2")
        >>> pop.add_heuristic_policy("PokeEnvHeuristic", "Simple heuristic baseline")
        >>> pop.size()
        2
        >>> sigma = [0.8, 0.2]  # Meta-strategy: 80% RL, 20% heuristic
        >>> opponent = pop.sample_policy(sigma)
    """

    def __init__(self):
        self.policies: Dict[str, PolicyInfo] = {}

    def add_policy(self, policy_info: PolicyInfo) -> None:
        """Add a policy to the population."""
        if policy_info.name in self.policies:
            raise ValueError(f"Policy '{policy_info.name}' already exists in population")
        self.policies[policy_info.name] = policy_info

    def add_pretrained_policy(
        self,
        model_name: str,
        description: str = "",
        checkpoint: Optional[int] = None,
    ) -> None:
        """
        Add a pretrained RL model to the population.

        Args:
            model_name: Name of the pretrained model (must be registered in metamon.rl.pretrained)
            description: Human-readable description
            checkpoint: Optional checkpoint number to load
        """
        # Validate model exists
        try:
            get_pretrained_model(model_name)
        except Exception as e:
            raise ValueError(f"Model '{model_name}' not found: {e}")

        policy_info = PolicyInfo(
            name=model_name,
            policy_type="pretrained",
            description=description,
            checkpoint=checkpoint,
            model_class=model_name,
        )
        self.add_policy(policy_info)

    def add_heuristic_policy(
        self,
        baseline_name: str,
        description: str = "",
    ) -> None:
        """
        Add a heuristic baseline to the population.

        Args:
            baseline_name: Name of the baseline (must be registered in metamon.baselines)
            description: Human-readable description
        """
        # Validate baseline exists
        try:
            get_baseline(baseline_name)
        except Exception as e:
            raise ValueError(f"Baseline '{baseline_name}' not found: {e}")

        policy_info = PolicyInfo(
            name=baseline_name,
            policy_type="heuristic",
            description=description,
            baseline_class=baseline_name,
        )
        self.add_policy(policy_info)

    def get_policy(self, name: str) -> PolicyInfo:
        """Get policy metadata by name."""
        if name not in self.policies:
            raise KeyError(f"Policy '{name}' not in population. Available: {self.list_policies()}")
        return self.policies[name]

    def list_policies(self) -> List[str]:
        """Get list of all policy names in population."""
        return list(self.policies.keys())

    def size(self) -> int:
        """Get number of policies in population."""
        return len(self.policies)

    def get_pretrained_model(self, name: str) -> PretrainedModel:
        """
        Load a pretrained model from the population.

        Args:
            name: Policy name

        Returns:
            PretrainedModel instance ready for evaluation

        Raises:
            ValueError: If policy is not a pretrained model
        """
        policy = self.get_policy(name)
        if policy.policy_type != "pretrained":
            raise ValueError(f"Policy '{name}' is not a pretrained model (type: {policy.policy_type})")

        return get_pretrained_model(policy.model_class)

    def get_baseline_class(self, name: str):
        """
        Get a heuristic baseline class from the population.

        Args:
            name: Policy name

        Returns:
            Baseline class ready for instantiation

        Raises:
            ValueError: If policy is not a heuristic baseline
        """
        policy = self.get_policy(name)
        if policy.policy_type != "heuristic":
            raise ValueError(f"Policy '{name}' is not a heuristic baseline (type: {policy.policy_type})")

        return get_baseline(policy.baseline_class)

    def save(self, path: str) -> None:
        """
        Save population to JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "policies": {name: info.to_dict() for name, info in self.policies.items()}
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PolicyPopulation":
        """
        Load population from JSON file.

        Args:
            path: File path to load from

        Returns:
            PolicyPopulation instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        pop = cls()
        for name, policy_dict in data["policies"].items():
            policy_info = PolicyInfo.from_dict(policy_dict)
            pop.add_policy(policy_info)

        return pop

    def sample_policy(self, meta_strategy: List[float]) -> str:
        """
        Sample a policy name according to meta-strategy distribution.

        Args:
            meta_strategy: Probability distribution over policies (must sum to 1)

        Returns:
            Policy name sampled from distribution

        Example:
            >>> pop = PolicyPopulation()
            >>> pop.add_pretrained_policy("Model1")
            >>> pop.add_heuristic_policy("Heuristic1")
            >>> # Meta-strategy: 70% Model1, 30% Heuristic1
            >>> opponent = pop.sample_policy([0.7, 0.3])
        """
        import numpy as np

        if len(meta_strategy) != self.size():
            raise ValueError(
                f"Meta-strategy length ({len(meta_strategy)}) must match "
                f"population size ({self.size()})"
            )

        if not np.isclose(sum(meta_strategy), 1.0):
            raise ValueError(f"Meta-strategy must sum to 1.0 (got {sum(meta_strategy)})")

        policy_names = self.list_policies()
        return np.random.choice(policy_names, p=meta_strategy)

    def __repr__(self) -> str:
        """String representation of population."""
        lines = [f"PolicyPopulation (size={self.size()})"]
        for name, info in self.policies.items():
            lines.append(f"  - {name} ({info.policy_type}): {info.description}")
        return "\n".join(lines)
