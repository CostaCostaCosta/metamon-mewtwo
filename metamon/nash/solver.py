"""
Nash equilibrium solver for meta-game strategy optimization.

Given an interaction matrix M (K×K win-rates), compute the Nash equilibrium
mixed strategy σ that is unexploitable. This is the core of the Nash-first
approach: we're not just training better policies, we're finding optimal
mixtures over our policy population.

For 2-player zero-sum games, the Nash equilibrium can be found by solving
a linear program. We treat the population as the strategy space and solve
for the probability distribution that maximizes worst-case performance.

Mathematical formulation:
    maximize v
    subject to:
        Σᵢ M[i,j]·σᵢ ≥ v  ∀j  (our mixture beats all opponent pure strategies)
        Σᵢ σᵢ = 1
        σᵢ ≥ 0

This is a max-min LP that scipy.optimize.linprog can solve efficiently.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Optional, Tuple
import json


def solve_nash_mixture(
    M: np.ndarray,
    method: str = "highs",
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve for Nash equilibrium mixed strategy over policy population.

    Args:
        M: K×K interaction matrix where M[i,j] = P(πᵢ beats πⱼ)
        method: Solver method for scipy.optimize.linprog
        verbose: Print solver output

    Returns:
        σ: K-dimensional probability distribution (Nash mixture)

    Example:
        >>> M = np.array([
        ...     [0.5, 0.9, 0.8],  # Policy 0 beats 1 (90%), beats 2 (80%)
        ...     [0.1, 0.5, 0.6],  # Policy 1 loses to 0, beats 2 (60%)
        ...     [0.2, 0.4, 0.5],  # Policy 2 loses to both
        ... ])
        >>> sigma = solve_nash_mixture(M)
        >>> print(f"Nash mixture: {sigma}")
        Nash mixture: [0.85, 0.05, 0.10]  # Mostly play policy 0

    Notes:
        - If one policy dominates, σ will put ~100% mass on it
        - If rock-paper-scissors dynamics exist, σ will spread probability
        - The resulting σ minimizes exploitability by definition
    """
    K = M.shape[0]

    if M.shape != (K, K):
        raise ValueError(f"M must be square, got shape {M.shape}")

    # Convert to zero-sum game (payoff = win_rate - 0.5)
    # This centers the game around 0 for numerical stability
    M_zerosum = M - 0.5

    # Linear program formulation:
    # Variables: [σ₁, σ₂, ..., σ_K, v]
    # Minimize: -v (equivalent to maximizing v)
    # Subject to:
    #   M^T @ σ - v·1 ≥ 0  (mixture beats all opponent strategies)
    #   1^T @ σ = 1        (probabilities sum to 1)
    #   σ ≥ 0              (non-negative probabilities)

    # Objective: minimize -v (i.e., maximize v)
    c = np.zeros(K + 1)
    c[-1] = -1.0  # Coefficient for v

    # Inequality constraints: M^T @ σ - v·1 ≥ 0
    # Rewrite as: -M^T @ σ + v·1 ≤ 0
    A_ub = np.hstack([-M_zerosum.T, np.ones((K, 1))])
    b_ub = np.zeros(K)

    # Equality constraint: σ sums to 1
    A_eq = np.zeros((1, K + 1))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    # Bounds: σᵢ ≥ 0, v unbounded
    bounds = [(0, None) for _ in range(K)] + [(None, None)]

    # Solve LP
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        options={"disp": verbose},
    )

    if not result.success:
        raise RuntimeError(f"Nash solver failed: {result.message}")

    # Extract mixture (first K components)
    sigma = result.x[:K]

    # Numerical cleanup: ensure valid probability distribution
    sigma = np.maximum(sigma, 0)  # Force non-negative
    sigma = sigma / sigma.sum()  # Renormalize to sum to 1

    return sigma


def compute_exploitability(
    M: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Compute exploitability of a mixed strategy.

    Exploitability measures how much better a best-response can do against
    σ compared to Nash equilibrium play. Lower is better.

    Exploitability = max_j (Σᵢ M[i,j]·σᵢ) - v*

    where v* is the value of the Nash equilibrium.

    Args:
        M: K×K interaction matrix
        sigma: K-dimensional mixed strategy

    Returns:
        Exploitability value (≥ 0, lower is better)

    Example:
        >>> sigma_nash = solve_nash_mixture(M)
        >>> exp_nash = compute_exploitability(M, sigma_nash)
        >>> sigma_uniform = np.ones(K) / K
        >>> exp_uniform = compute_exploitability(M, sigma_uniform)
        >>> assert exp_nash <= exp_uniform  # Nash is less exploitable
    """
    K = M.shape[0]

    if len(sigma) != K:
        raise ValueError(f"Sigma length ({len(sigma)}) must match M size ({K})")

    if not np.isclose(sigma.sum(), 1.0):
        raise ValueError(f"Sigma must sum to 1, got {sigma.sum()}")

    # Expected value against each pure strategy
    values_vs_pure = M.T @ sigma  # Shape: (K,)

    # Best-response value
    best_response_value = values_vs_pure.max()

    # Nash equilibrium value (minimax)
    nash_sigma = solve_nash_mixture(M)
    nash_value = (M.T @ nash_sigma).min()

    # Exploitability
    return best_response_value - nash_value


def analyze_meta_game(
    M: np.ndarray,
    policy_names: List[str],
) -> Dict:
    """
    Comprehensive analysis of the meta-game.

    Computes Nash equilibrium, exploitability, dominance relationships, etc.

    Args:
        M: K×K interaction matrix
        policy_names: Names of the K policies

    Returns:
        Dictionary with analysis results:
        - "nash_mixture": Nash equilibrium distribution
        - "nash_value": Value of Nash equilibrium
        - "exploitability": Exploitability of Nash mixture (should be ~0)
        - "dominant_policies": Policies with highest Nash probabilities
        - "expected_winrates": Expected win-rate of each policy vs Nash mixture

    Example:
        >>> analysis = analyze_meta_game(M, ["P0", "Heuristic1", "Heuristic2"])
        >>> print(f"Nash mixture: {analysis['nash_mixture']}")
        >>> print(f"Dominant policy: {analysis['dominant_policies'][0]}")
    """
    K = len(policy_names)

    if M.shape != (K, K):
        raise ValueError(f"M shape {M.shape} doesn't match policy count {K}")

    # Solve for Nash equilibrium
    nash_sigma = solve_nash_mixture(M)

    # Nash value (expected win-rate)
    nash_value = (M.T @ nash_sigma).min()

    # Exploitability (should be ~0 for Nash)
    exploitability = compute_exploitability(M, nash_sigma)

    # Expected win-rate of each policy vs Nash mixture
    expected_winrates = M @ nash_sigma

    # Find dominant policies (highest Nash probabilities)
    sorted_indices = np.argsort(nash_sigma)[::-1]
    dominant_policies = [
        {
            "policy": policy_names[idx],
            "probability": float(nash_sigma[idx]),
            "expected_winrate_vs_nash": float(expected_winrates[idx]),
        }
        for idx in sorted_indices
        if nash_sigma[idx] > 0.01  # Only report policies with >1% mass
    ]

    return {
        "nash_mixture": nash_sigma.tolist(),
        "nash_value": float(nash_value),
        "exploitability": float(exploitability),
        "dominant_policies": dominant_policies,
        "expected_winrates": expected_winrates.tolist(),
        "policy_names": policy_names,
    }


def save_meta_strategy(
    sigma: np.ndarray,
    policy_names: List[str],
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save meta-strategy to JSON file.

    Args:
        sigma: K-dimensional probability distribution
        policy_names: Names of the K policies
        path: File path to save to
        metadata: Optional additional metadata (iteration number, etc.)
    """
    data = {
        "meta_strategy": sigma.tolist(),
        "policy_names": policy_names,
        "metadata": metadata or {},
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_meta_strategy(path: str) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load meta-strategy from JSON file.

    Args:
        path: File path to load from

    Returns:
        Tuple of (sigma, policy_names, metadata)
    """
    with open(path, "r") as f:
        data = json.load(f)

    sigma = np.array(data["meta_strategy"])
    policy_names = data["policy_names"]
    metadata = data.get("metadata", {})

    return sigma, policy_names, metadata


if __name__ == "__main__":
    # Example usage
    print("Nash Solver Example\n" + "=" * 50)

    # Example 1: Dominant policy
    print("\nExample 1: One policy dominates")
    M1 = np.array([
        [0.5, 0.9, 0.85],  # P0 strong
        [0.1, 0.5, 0.55],  # P1 weak
        [0.15, 0.45, 0.5],  # P2 weak
    ])
    sigma1 = solve_nash_mixture(M1)
    print(f"Nash mixture: {sigma1}")
    print(f"Interpretation: Policy 0 dominates, gets ~{sigma1[0]:.1%} probability")

    # Example 2: Rock-paper-scissors
    print("\nExample 2: Cyclic dominance (rock-paper-scissors)")
    M2 = np.array([
        [0.5, 0.7, 0.3],  # P0 beats P1, loses to P2
        [0.3, 0.5, 0.7],  # P1 beats P2, loses to P0
        [0.7, 0.3, 0.5],  # P2 beats P0, loses to P1
    ])
    sigma2 = solve_nash_mixture(M2)
    print(f"Nash mixture: {sigma2}")
    print(f"Interpretation: Near-uniform mixture due to cyclic dynamics")

    # Example 3: Full analysis
    print("\nExample 3: Full meta-game analysis")
    policy_names = ["P0_SYN_V2_GEN1", "PokeEnvHeuristic", "GymLeader"]
    analysis = analyze_meta_game(M1, policy_names)
    print(json.dumps(analysis, indent=2))
