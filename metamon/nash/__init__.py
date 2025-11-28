"""
Nash equilibrium infrastructure for multi-agent RL in Pokémon battles.

This package implements the framework from "A Unified Game-Theoretic Approach to
Multiagent Reinforcement Learning" for achieving superhuman Gen1 performance through:

- Policy populations (Π): Managing multiple strategies/checkpoints
- Interaction matrices (M): Empirical win-rate payoff matrices
- Meta-strategy solvers: Computing Nash mixtures over populations
- PSRO (Policy Space Response Oracles): Iterative best-response training

Key components:
- PolicyPopulation: Manage and sample from policy populations
- InteractionMatrix: Compute and store empirical win-rates
- NashSolver: Solve for Nash equilibrium mixtures over policies
"""

from metamon.nash.population import PolicyPopulation
from metamon.nash.interaction_matrix import InteractionMatrix
from metamon.nash.solver import solve_nash_mixture

__all__ = [
    "PolicyPopulation",
    "InteractionMatrix",
    "solve_nash_mixture",
]
