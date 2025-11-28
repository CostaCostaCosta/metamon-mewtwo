"""
Dynamic damping for RL training: reverse-KL regularization with adaptive schedules.

This module implements dynamic damping to stabilize oracle training in self-play
and PSRO scenarios. It consists of three main components:

1. Reverse-KL regularization: Keep new policy close to reference policy π_ref
2. Power-law schedules: Gradually relax entropy and KL coefficients
3. Adaptive update control: Monitor KL divergence and adjust LR/coefficients

Reference: Based on techniques from Ataraxos and OpenAI's PPO-KL implementations.
"""

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DynamicDampingConfig:
    """Configuration for dynamic damping.

    Attributes:
        enabled: Whether to enable dynamic damping

        KL regularization schedule:
            kl_coef_init: Initial KL coefficient (weight on KL loss term)
            kl_coef_max: Maximum KL coefficient (hard cap)
            kl_power_alpha: Power-law exponent for KL schedule decay
            kl_schedule_steps: Number of steps over which to decay KL coefficient

        Entropy regularization schedule:
            ent_coef_init: Initial entropy coefficient
            ent_coef_min: Minimum entropy coefficient (floor)
            ent_power_alpha: Power-law exponent for entropy schedule decay
            ent_schedule_steps: Number of steps over which to decay entropy coefficient

        Adaptive control:
            target_kl_per_step: Target KL divergence per optimization step
            kl_tolerance: Tolerance multiplier (e.g., 1.5 = 50% tolerance band)
            lr_shrink_factor: Factor to shrink LR when KL is too high
            lr_grow_factor: Factor to grow LR when KL is too low
            kl_coef_growth_factor: Factor to increase kl_coef when KL is too high
            kl_coef_decay_factor: Factor to decrease kl_coef when KL is too low
            min_lr: Minimum learning rate (hard floor)
            max_lr: Maximum learning rate (hard cap)
    """
    enabled: bool = True

    # KL schedule
    kl_coef_init: float = 0.05
    kl_coef_max: float = 0.5
    kl_power_alpha: float = 0.5
    kl_schedule_steps: int = 1_000_000

    # Entropy schedule
    ent_coef_init: float = 0.01
    ent_coef_min: float = 0.001
    ent_power_alpha: float = 0.7
    ent_schedule_steps: int = 1_000_000

    # Adaptive control
    target_kl_per_step: float = 0.01
    kl_tolerance: float = 1.5
    lr_shrink_factor: float = 0.5
    lr_grow_factor: float = 1.1
    kl_coef_growth_factor: float = 1.5
    kl_coef_decay_factor: float = 0.9
    min_lr: float = 1e-6
    max_lr: float = 1e-3


class DynamicDampingState:
    """Manages dynamic damping state during training.

    This class maintains:
    - A frozen reference policy (snapshot at iteration start)
    - Current step counter
    - Current KL and entropy coefficients (updated by schedules)
    - Current learning rate (for adaptive adjustment)

    Usage:
        # At start of training iteration:
        dd_state = DynamicDampingState(model, config)

        # Each training step:
        dd_state.update_schedules()
        kl_loss = dd_state.compute_kl_loss(new_logits, ref_logits, legal_mask)
        total_loss = policy_loss + kl_loss

        # After each epoch (or periodically):
        dd_state.adapt_from_observed_kl(optimizer, observed_kl)
    """

    def __init__(self, base_model: nn.Module, config: DynamicDampingConfig):
        """Initialize dynamic damping state.

        Args:
            base_model: The policy model to train (will be copied as reference)
            config: Dynamic damping configuration
        """
        self.config = config
        self.step = 0

        # Create frozen copy of model as reference policy
        self.ref_model = copy.deepcopy(base_model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        # Initialize coefficients
        self.kl_coef = config.kl_coef_init
        self.ent_coef = config.ent_coef_init
        self.current_lr = None  # Will be set from optimizer

    def update_schedules(self):
        """Update KL and entropy coefficients using power-law schedules.

        Power-law schedule: coef_t = coef_0 * (1 + t / T)^(-alpha)

        This provides smooth decay that starts strong and gradually relaxes,
        avoiding sudden "entropy cliffs" that can destabilize training.
        """
        c = self.config
        t = self.step

        # KL coefficient schedule (with upper cap)
        kl_scale = (1.0 + t / max(1, c.kl_schedule_steps)) ** (-c.kl_power_alpha)
        self.kl_coef = min(c.kl_coef_init * kl_scale, c.kl_coef_max)

        # Entropy coefficient schedule (with lower floor)
        ent_scale = (1.0 + t / max(1, c.ent_schedule_steps)) ** (-c.ent_power_alpha)
        self.ent_coef = max(c.ent_coef_init * ent_scale, c.ent_coef_min)

        self.step += 1

    def adapt_from_observed_kl(self, optimizer: torch.optim.Optimizer,
                                observed_kl: Optional[float]):
        """Adapt learning rate and KL coefficient based on observed KL divergence.

        Strategy:
        - If observed KL > tolerance * target: updates are too aggressive
          → Shrink LR and increase KL coefficient to dampen updates
        - If observed KL < target / tolerance: updates are too conservative
          → Grow LR and decrease KL coefficient to enable larger updates

        This provides adaptive control to keep update sizes stable throughout training.

        Args:
            optimizer: The optimizer whose learning rate should be adjusted
            observed_kl: Mean KL divergence observed over recent updates (can be None)
        """
        c = self.config
        target = c.target_kl_per_step
        tol = c.kl_tolerance

        # Get current LR from optimizer
        self.current_lr = optimizer.param_groups[0]["lr"]

        if observed_kl is None:
            return  # Nothing to adapt

        if observed_kl > tol * target:
            # Updates too aggressive: shrink LR, increase KL damping
            new_lr = max(self.current_lr * c.lr_shrink_factor, c.min_lr)
            optimizer.param_groups[0]["lr"] = new_lr

            self.kl_coef = min(
                self.kl_coef * c.kl_coef_growth_factor,
                c.kl_coef_max
            )

        elif observed_kl < target / tol:
            # Updates too conservative: grow LR, decrease KL damping
            new_lr = min(self.current_lr * c.lr_grow_factor, c.max_lr)
            optimizer.param_groups[0]["lr"] = new_lr

            self.kl_coef = max(
                self.kl_coef * c.kl_coef_decay_factor,
                1e-6  # Small floor to avoid zero
            )

    def compute_kl_loss(self, new_logits: torch.Tensor,
                        legal_mask: torch.Tensor) -> torch.Tensor:
        """Compute reverse-KL loss: KL(π_new || π_ref).

        Args:
            new_logits: Logits from current policy [batch_size, num_actions]
            legal_mask: Binary mask for legal actions [batch_size, num_actions]

        Returns:
            Weighted KL loss (scalar): kl_coef * mean(KL(π_new || π_ref))
        """
        # Get reference logits (no gradients needed for ref model)
        with torch.no_grad():
            # Assuming ref_model has same forward signature as base model
            # This will need to be adapted based on actual model interface
            # For now, we'll compute this in the experiment class where we have context
            raise NotImplementedError(
                "compute_kl_loss should be called from experiment class "
                "where we have access to full observation context"
            )

    def get_metrics(self) -> dict:
        """Get current damping metrics for logging.

        Returns:
            Dictionary of metrics: kl_coef, ent_coef, step, current_lr
        """
        return {
            "damping/kl_coef": self.kl_coef,
            "damping/ent_coef": self.ent_coef,
            "damping/step": self.step,
            "damping/lr": self.current_lr if self.current_lr is not None else 0.0,
        }


def compute_masked_reverse_kl(new_logits: torch.Tensor,
                               ref_logits: torch.Tensor,
                               legal_mask: torch.Tensor,
                               eps: float = 1e-8,
                               already_masked: bool = True) -> torch.Tensor:
    """Compute reverse KL divergence: KL(π_new || π_ref) with action masking.

    Reverse KL is defined as:
        KL(π_new || π_ref) = Σ_a π_new(a) * [log π_new(a) - log π_ref(a)]

    This measures how much the new policy differs from the reference policy,
    weighted by the new policy's probabilities. It penalizes new policy for
    placing mass where reference policy has low probability.

    Important: This function properly handles illegal actions by:
    1. Masking logits with -inf for illegal actions (if not already masked)
    2. Computing log-probabilities only over legal actions
    3. Ensuring KL computation respects the constrained action space

    Args:
        new_logits: Logits from current/new policy [batch_size, num_actions]
        ref_logits: Logits from reference policy [batch_size, num_actions]
        legal_mask: Binary mask (1=legal, 0=illegal) [batch_size, num_actions]
        eps: Small constant for numerical stability
        already_masked: If True, assumes logits are already masked with -inf for illegal actions

    Returns:
        KL divergence per example [batch_size]
    """
    # Only mask if not already masked (avoids double masking issues)
    if not already_masked:
        minus_inf = -1e9
        mask_value = (~legal_mask).float() * minus_inf  # 0 for legal, -inf for illegal
        new_logits = new_logits + mask_value
        ref_logits = ref_logits + mask_value

    # Compute log probabilities (softmax automatically handles -inf masking)
    log_probs_new = torch.log_softmax(new_logits, dim=-1)
    log_probs_ref = torch.log_softmax(ref_logits, dim=-1)

    # Compute probabilities from new policy
    probs_new = torch.exp(log_probs_new)

    # Replace -inf with 0 in log probs to avoid NaN from 0 * -inf
    log_probs_new = torch.where(legal_mask, log_probs_new, torch.zeros_like(log_probs_new))
    log_probs_ref = torch.where(legal_mask, log_probs_ref, torch.zeros_like(log_probs_ref))
    probs_new = torch.where(legal_mask, probs_new, torch.zeros_like(probs_new))

    # Reverse KL: KL(new || ref) = Σ p_new * (log p_new - log p_ref)
    # Only sum over legal actions (illegal actions contribute 0)
    kl_per_example = (probs_new * (log_probs_new - log_probs_ref)).sum(dim=-1)

    return kl_per_example


def compute_policy_entropy(logits: torch.Tensor,
                           legal_mask: torch.Tensor,
                           eps: float = 1e-8,
                           already_masked: bool = True) -> torch.Tensor:
    """Compute policy entropy with action masking.

    Entropy measures the randomness/exploration of the policy:
        H(π) = -Σ_a π(a) * log π(a)

    Higher entropy = more exploration/randomness
    Lower entropy = more deterministic/exploitative

    Args:
        logits: Policy logits [batch_size, num_actions]
        legal_mask: Binary mask (1=legal, 0=illegal) [batch_size, num_actions]
        eps: Small constant for numerical stability
        already_masked: If True, assumes logits are already masked with -inf for illegal actions

    Returns:
        Entropy per example [batch_size]
    """
    # Only mask if not already masked
    if not already_masked:
        minus_inf = -1e9
        mask_value = (~legal_mask).float() * minus_inf
        logits = logits + mask_value

    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Replace -inf with 0 to avoid NaN from 0 * -inf
    log_probs = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(legal_mask, probs, torch.zeros_like(probs))

    # Entropy: H = -Σ p * log(p) (only over legal actions, illegal actions contribute 0)
    entropy_per_example = -(probs * log_probs).sum(dim=-1)

    return entropy_per_example
