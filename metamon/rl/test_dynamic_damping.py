"""
Unit tests for dynamic damping controller.

These tests validate the adaptive KL/entropy control and learning rate adjustment
logic without requiring full training runs.
"""

import pytest
import torch
import torch.nn as nn

from metamon.rl.dynamic_damping import (
    DynamicDampingConfig,
    DynamicDampingState,
    compute_masked_reverse_kl,
    compute_policy_entropy,
)


class DummyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, num_actions=13):
        super().__init__()
        self.linear = nn.Linear(10, num_actions)

    def forward(self, x):
        return self.linear(x)


class DummyOptimizer:
    """Minimal optimizer mock for testing LR adjustments."""

    def __init__(self, lr=1e-4):
        self.param_groups = [{"lr": lr}]


# =============================================================================
# Power-Law Schedule Tests
# =============================================================================

def test_power_law_schedule_initialization():
    """Test that schedules initialize correctly."""
    config = DynamicDampingConfig(
        kl_coef_init=0.10,
        ent_coef_init=0.02,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)

    # Initial values should match config
    assert state.kl_coef == 0.10
    assert state.ent_coef == 0.02
    assert state.step == 0


def test_power_law_schedule_decay():
    """Test that power-law schedules decay over time."""
    config = DynamicDampingConfig(
        kl_coef_init=0.10,
        kl_power_alpha=0.5,
        kl_schedule_steps=1000,
        ent_coef_init=0.02,
        ent_power_alpha=0.7,
        ent_schedule_steps=1000,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)

    initial_kl = state.kl_coef
    initial_ent = state.ent_coef

    # Run schedule for several steps
    for _ in range(500):
        state.update_schedules()

    # Both coefficients should decay
    assert state.kl_coef < initial_kl
    assert state.ent_coef < initial_ent

    # KL coefficient: coef_t = 0.10 * (1 + 500/1000)^(-0.5) = 0.10 * 1.5^(-0.5) ≈ 0.0816
    expected_kl = 0.10 * (1.5 ** -0.5)
    assert abs(state.kl_coef - expected_kl) < 1e-4

    # Entropy coefficient: coef_t = 0.02 * (1 + 500/1000)^(-0.7) = 0.02 * 1.5^(-0.7) ≈ 0.0146
    expected_ent = 0.02 * (1.5 ** -0.7)
    assert abs(state.ent_coef - expected_ent) < 1e-4


def test_power_law_schedule_caps_and_floors():
    """Test that schedules respect min/max bounds."""
    config = DynamicDampingConfig(
        kl_coef_init=0.10,
        kl_coef_max=0.15,  # Cap
        kl_power_alpha=0.0,  # No decay (to test cap)
        ent_coef_init=0.02,
        ent_coef_min=0.005,  # Floor
        ent_power_alpha=10.0,  # Aggressive decay (to test floor)
        ent_schedule_steps=10,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)

    # Run for many steps
    for _ in range(1000):
        state.update_schedules()

    # KL should not exceed max (stays at init since alpha=0)
    assert state.kl_coef <= config.kl_coef_max

    # Entropy should not go below min
    assert state.ent_coef >= config.ent_coef_min


def test_power_law_schedule_disabled():
    """Test that zero alpha disables decay."""
    config = DynamicDampingConfig(
        kl_coef_init=0.10,
        kl_power_alpha=0.0,
        ent_coef_init=0.02,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)

    initial_kl = state.kl_coef
    initial_ent = state.ent_coef

    # Run schedule
    for _ in range(1000):
        state.update_schedules()

    # Coefficients should not change (alpha=0 means no decay)
    assert state.kl_coef == initial_kl
    assert state.ent_coef == initial_ent


# =============================================================================
# Adaptive Controller Tests
# =============================================================================

def test_adaptive_controller_kl_too_high():
    """Test that controller shrinks LR and grows kl_coef when KL is too high."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,  # Acceptable range: [0.0077, 0.013]
        lr_shrink_factor=0.5,
        kl_coef_growth_factor=2.0,
        kl_coef_init=0.10,
        kl_power_alpha=0.0,  # Disable schedule to isolate controller
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_kl_coef = state.kl_coef

    # Simulate observing KL that's too high
    observed_kl = 0.05  # Way above 0.01 * 1.3 = 0.013
    state.adapt_from_observed_kl(optimizer, observed_kl)

    # LR should shrink
    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr < initial_lr
    assert abs(new_lr - initial_lr * config.lr_shrink_factor) < 1e-9

    # KL coefficient should grow
    assert state.kl_coef > initial_kl_coef
    assert abs(state.kl_coef - initial_kl_coef * config.kl_coef_growth_factor) < 1e-9


def test_adaptive_controller_kl_too_low():
    """Test that controller grows LR and shrinks kl_coef when KL is too low."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,  # Acceptable range: [0.0077, 0.013]
        lr_grow_factor=1.1,
        kl_coef_decay_factor=0.9,
        kl_coef_init=0.10,
        kl_power_alpha=0.0,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_kl_coef = state.kl_coef

    # Simulate observing KL that's too low
    observed_kl = 0.002  # Way below 0.01 / 1.3 ≈ 0.0077
    state.adapt_from_observed_kl(optimizer, observed_kl)

    # LR should grow
    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr > initial_lr
    assert abs(new_lr - initial_lr * config.lr_grow_factor) < 1e-9

    # KL coefficient should shrink
    assert state.kl_coef < initial_kl_coef
    assert abs(state.kl_coef - initial_kl_coef * config.kl_coef_decay_factor) < 1e-9


def test_adaptive_controller_kl_in_range():
    """Test that controller doesn't adjust when KL is in acceptable range."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,  # Acceptable range: [0.0077, 0.013]
        kl_power_alpha=0.0,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_kl_coef = state.kl_coef

    # Simulate observing KL in acceptable range
    observed_kl = 0.01  # Right at target
    state.adapt_from_observed_kl(optimizer, observed_kl)

    # Neither LR nor kl_coef should change
    assert optimizer.param_groups[0]["lr"] == initial_lr
    assert state.kl_coef == initial_kl_coef


def test_adaptive_controller_respects_lr_bounds():
    """Test that LR adjustments respect min/max bounds."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,
        lr_shrink_factor=0.1,
        lr_grow_factor=10.0,
        min_lr=1e-6,
        max_lr=1e-3,
        kl_power_alpha=0.0,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)

    # Test lower bound
    optimizer = DummyOptimizer(lr=2e-6)  # Close to min
    state.adapt_from_observed_kl(optimizer, observed_kl=1.0)  # Way too high
    assert optimizer.param_groups[0]["lr"] >= config.min_lr

    # Test upper bound
    optimizer = DummyOptimizer(lr=5e-4)  # Mid range
    state.adapt_from_observed_kl(optimizer, observed_kl=0.0001)  # Way too low
    assert optimizer.param_groups[0]["lr"] <= config.max_lr


def test_adaptive_controller_respects_kl_coef_bounds():
    """Test that kl_coef adjustments respect bounds."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,
        kl_coef_init=0.10,
        kl_coef_max=0.20,
        kl_coef_growth_factor=5.0,
        kl_coef_decay_factor=0.01,
        kl_power_alpha=0.0,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    # Test upper bound
    state.adapt_from_observed_kl(optimizer, observed_kl=1.0)  # Way too high
    assert state.kl_coef <= config.kl_coef_max

    # Test lower bound (small floor to avoid zero)
    state.kl_coef = 0.001
    state.adapt_from_observed_kl(optimizer, observed_kl=0.0001)  # Way too low
    assert state.kl_coef >= 1e-6


# =============================================================================
# Integration Tests: Synthetic KL Sequences
# =============================================================================

def test_controller_with_oscillating_kl():
    """Test controller behavior with oscillating KL values."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,
        lr_shrink_factor=0.5,
        lr_grow_factor=1.2,
        kl_coef_growth_factor=1.5,
        kl_coef_decay_factor=0.9,
        kl_coef_init=0.10,
        kl_power_alpha=0.0,  # Disable schedule
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    # Simulate oscillating KL: high, low, high, low
    kl_sequence = [0.05, 0.002, 0.05, 0.002]
    lr_history = [optimizer.param_groups[0]["lr"]]
    kl_coef_history = [state.kl_coef]

    for kl_val in kl_sequence:
        state.adapt_from_observed_kl(optimizer, kl_val)
        lr_history.append(optimizer.param_groups[0]["lr"])
        kl_coef_history.append(state.kl_coef)

    # LR should decrease then increase (oscillating)
    assert lr_history[1] < lr_history[0]  # After high KL
    assert lr_history[2] > lr_history[1]  # After low KL
    assert lr_history[3] < lr_history[2]  # After high KL again
    assert lr_history[4] > lr_history[3]  # After low KL again

    # KL coef should increase then decrease (oscillating)
    assert kl_coef_history[1] > kl_coef_history[0]  # After high KL
    assert kl_coef_history[2] < kl_coef_history[1]  # After low KL
    assert kl_coef_history[3] > kl_coef_history[2]  # After high KL again
    assert kl_coef_history[4] < kl_coef_history[3]  # After low KL again


def test_controller_with_gradually_increasing_kl():
    """Test controller dampens policy when KL gradually increases."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,
        lr_shrink_factor=0.7,
        kl_coef_growth_factor=1.3,
        kl_coef_init=0.10,
        kl_power_alpha=0.0,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_kl_coef = state.kl_coef

    # Simulate gradually increasing KL (policy drifting)
    kl_sequence = [0.015, 0.020, 0.030, 0.050]

    for kl_val in kl_sequence:
        state.adapt_from_observed_kl(optimizer, kl_val)

    final_lr = optimizer.param_groups[0]["lr"]
    final_kl_coef = state.kl_coef

    # LR should have decreased significantly
    assert final_lr < initial_lr * 0.5

    # KL coefficient should have increased significantly
    assert final_kl_coef > initial_kl_coef * 1.5


def test_controller_with_stable_kl():
    """Test controller maintains settings when KL is stable in target range."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,
        kl_coef_init=0.10,
        kl_power_alpha=0.0,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_kl_coef = state.kl_coef

    # Simulate stable KL in target range
    kl_sequence = [0.009, 0.010, 0.011, 0.010, 0.009]

    for kl_val in kl_sequence:
        state.adapt_from_observed_kl(optimizer, kl_val)

    # Settings should remain stable
    assert optimizer.param_groups[0]["lr"] == initial_lr
    assert state.kl_coef == initial_kl_coef


def test_controller_with_schedule_interaction():
    """Test interaction between power-law schedule and adaptive controller."""
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,
        kl_coef_init=0.10,
        kl_power_alpha=0.3,  # Enable schedule
        kl_schedule_steps=1000,
        lr_shrink_factor=0.5,
        kl_coef_growth_factor=1.5,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    # Run for a while with stable KL (schedule should decay kl_coef)
    for i in range(100):
        state.update_schedules()
        if i % 10 == 0:
            state.adapt_from_observed_kl(optimizer, observed_kl=0.01)

    scheduled_kl_coef = state.kl_coef

    # Now observe high KL (controller should increase kl_coef)
    state.adapt_from_observed_kl(optimizer, observed_kl=0.05)

    # Controller should override schedule decay
    assert state.kl_coef > scheduled_kl_coef


def test_full_step_update_order():
    """Test full training loop simulation: mimics actual training step order.

    This test validates the bug we saw in the actual training run:
    - After a long stretch of high KL, kl_coef must be higher than at start
    - LR should be lower than at start
    - This ensures the controller actually responds to persistent high KL
    """
    config = DynamicDampingConfig(
        target_kl_per_step=0.01,
        kl_tolerance=1.3,  # Acceptable range: [0.0077, 0.013]
        kl_coef_init=0.10,
        kl_coef_max=0.20,
        kl_power_alpha=0.3,  # Enable schedule decay
        kl_schedule_steps=1000,
        lr_shrink_factor=0.5,
        lr_grow_factor=1.1,
        kl_coef_growth_factor=1.5,
        kl_coef_decay_factor=0.9,
        min_lr=1e-6,
        max_lr=1e-3,
        ent_power_alpha=0.0,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_kl_coef = state.kl_coef

    # Simulate 100 training steps with realistic KL pattern:
    # - Start with low KL (updates conservative)
    # - Spike to medium KL (updates getting larger)
    # - Long tail of high KL (policy drifting, needs damping)
    kl_sequence = [0.002] * 20 + [0.02] * 20 + [0.05] * 60

    kl_coef_history = []
    lr_history = []

    for kl_val in kl_sequence:
        # Mimic actual training order: schedule update -> adapt based on observed KL
        state.update_schedules()
        state.adapt_from_observed_kl(optimizer, kl_val)

        kl_coef_history.append(state.kl_coef)
        lr_history.append(optimizer.param_groups[0]["lr"])

    final_metrics = state.get_metrics()

    # Validate that controller responded to persistent high KL:

    # 1. KL coefficient should have increased significantly despite schedule decay
    # (controller should override schedule when KL is high)
    assert state.kl_coef > initial_kl_coef, \
        f"kl_coef should increase with high KL: {state.kl_coef:.4f} <= {initial_kl_coef:.4f}"

    # 2. KL coefficient should be in reasonable range (not at max cap)
    assert 0.01 <= final_metrics["damping/kl_coef"] <= 0.20, \
        f"kl_coef out of range: {final_metrics['damping/kl_coef']:.4f}"

    # 3. Learning rate should have shrunk on the high-KL tail
    assert optimizer.param_groups[0]["lr"] < initial_lr, \
        f"LR should shrink with high KL: {optimizer.param_groups[0]['lr']:.6f} >= {initial_lr:.6f}"

    # 4. LR should be significantly reduced (not just epsilon change)
    lr_ratio = optimizer.param_groups[0]["lr"] / initial_lr
    assert lr_ratio < 0.5, \
        f"LR should shrink significantly: ratio={lr_ratio:.3f} (expected < 0.5)"

    # 5. Verify we actually adapted multiple times (not stuck)
    # Check that kl_coef changed over the sequence
    assert len(set(kl_coef_history)) > 5, \
        "kl_coef should vary over time (controller should be active)"

    # 6. Verify LR adapted multiple times
    assert len(set(lr_history)) > 3, \
        "LR should vary over time (controller should be active)"

    # 7. In the high-KL tail, kl_coef should be consistently elevated
    avg_kl_coef_early = sum(kl_coef_history[:20]) / 20  # Low KL period
    avg_kl_coef_late = sum(kl_coef_history[-20:]) / 20   # High KL period
    assert avg_kl_coef_late > avg_kl_coef_early, \
        f"kl_coef should be higher during high-KL period: {avg_kl_coef_late:.4f} <= {avg_kl_coef_early:.4f}"

    # 8. In the high-KL tail, LR should be consistently depressed
    avg_lr_early = sum(lr_history[:20]) / 20
    avg_lr_late = sum(lr_history[-20:]) / 20
    assert avg_lr_late < avg_lr_early, \
        f"LR should be lower during high-KL period: {avg_lr_late:.6f} >= {avg_lr_early:.6f}"


# =============================================================================
# KL and Entropy Computation Tests
# =============================================================================

def test_masked_reverse_kl_basic():
    """Test basic reverse KL computation."""
    batch_size = 4
    num_actions = 13

    # Create identical logits → KL should be zero
    new_logits = torch.randn(batch_size, num_actions)
    ref_logits = new_logits.clone()
    legal_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)

    kl = compute_masked_reverse_kl(new_logits, ref_logits, legal_mask)

    assert kl.shape == (batch_size,)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_masked_reverse_kl_with_masking():
    """Test reverse KL respects action masking."""
    batch_size = 2
    num_actions = 13

    new_logits = torch.randn(batch_size, num_actions)
    ref_logits = torch.randn(batch_size, num_actions)

    # Some actions illegal
    legal_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
    legal_mask[0, 5:] = False  # First example: only first 5 actions legal
    legal_mask[1, :3] = False   # Second example: first 3 actions illegal

    # Mask logits with -inf for illegal actions
    minus_inf = torch.tensor(-1e9)
    new_logits = torch.where(legal_mask, new_logits, minus_inf)
    ref_logits = torch.where(legal_mask, ref_logits, minus_inf)

    kl = compute_masked_reverse_kl(new_logits, ref_logits, legal_mask, already_masked=True)

    assert kl.shape == (batch_size,)
    assert torch.all(kl >= 0)  # KL should be non-negative
    assert torch.all(torch.isfinite(kl))  # No NaN or inf


def test_masked_reverse_kl_divergence_direction():
    """Test that KL increases when policies diverge."""
    batch_size = 4
    num_actions = 13

    # Reference policy
    ref_logits = torch.zeros(batch_size, num_actions)
    legal_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)

    # New policy slightly different
    new_logits_close = ref_logits + torch.randn(batch_size, num_actions) * 0.1
    kl_close = compute_masked_reverse_kl(new_logits_close, ref_logits, legal_mask)

    # New policy very different
    new_logits_far = ref_logits + torch.randn(batch_size, num_actions) * 2.0
    kl_far = compute_masked_reverse_kl(new_logits_far, ref_logits, legal_mask)

    # Larger divergence should have larger KL
    assert torch.all(kl_far > kl_close)


def test_policy_entropy_basic():
    """Test basic entropy computation."""
    batch_size = 4
    num_actions = 13

    # Uniform distribution should have high entropy
    uniform_logits = torch.zeros(batch_size, num_actions)
    legal_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)

    entropy_uniform = compute_policy_entropy(uniform_logits, legal_mask)

    # Deterministic distribution should have low entropy
    deterministic_logits = torch.full((batch_size, num_actions), -1e9)
    deterministic_logits[:, 0] = 10.0  # All mass on first action

    entropy_deterministic = compute_policy_entropy(deterministic_logits, legal_mask)

    assert torch.all(entropy_uniform > entropy_deterministic)


def test_policy_entropy_with_masking():
    """Test entropy computation respects action masking."""
    batch_size = 2
    num_actions = 13

    logits = torch.zeros(batch_size, num_actions)  # Uniform

    # Different legal action sets
    legal_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
    legal_mask[0, 10:] = False  # First example: 10 legal actions
    legal_mask[1, 5:] = False   # Second example: 5 legal actions

    # Mask logits
    minus_inf = torch.tensor(-1e9)
    logits = torch.where(legal_mask, logits, minus_inf)

    entropy = compute_policy_entropy(logits, legal_mask, already_masked=True)

    # Entropy should be log(num_legal_actions) for uniform distribution
    expected_entropy_0 = torch.log(torch.tensor(10.0))
    expected_entropy_1 = torch.log(torch.tensor(5.0))

    assert torch.allclose(entropy[0], expected_entropy_0, atol=1e-5)
    assert torch.allclose(entropy[1], expected_entropy_1, atol=1e-5)


# =============================================================================
# Metrics and Logging Tests
# =============================================================================

def test_get_metrics():
    """Test that metrics are properly returned."""
    config = DynamicDampingConfig(
        kl_coef_init=0.10,
        ent_coef_init=0.02,
    )
    model = DummyModel()
    state = DynamicDampingState(model, config)
    optimizer = DummyOptimizer(lr=1e-4)

    # Initial metrics
    metrics = state.get_metrics()
    assert "damping/kl_coef" in metrics
    assert "damping/ent_coef" in metrics
    assert "damping/step" in metrics
    assert "damping/lr" in metrics

    assert metrics["damping/kl_coef"] == 0.10
    assert metrics["damping/ent_coef"] == 0.02
    assert metrics["damping/step"] == 0

    # After some updates
    state.update_schedules()
    state.adapt_from_observed_kl(optimizer, observed_kl=0.05)

    metrics = state.get_metrics()
    assert metrics["damping/step"] == 1

    # Note: adapt_from_observed_kl() captures current_lr at the START of the call,
    # before making adjustments. So we need to call it again (even with None) to
    # refresh current_lr, or just verify the optimizer was actually updated.
    expected_lr = 1e-4 * 0.5  # Initial LR * shrink_factor (high KL triggers shrink)
    assert optimizer.param_groups[0]["lr"] == expected_lr

    # Call adapt again with None to refresh current_lr in state
    state.adapt_from_observed_kl(optimizer, observed_kl=None)
    metrics = state.get_metrics()
    assert metrics["damping/lr"] == expected_lr


# =============================================================================
# Reference Model Tests
# =============================================================================

def test_reference_model_frozen():
    """Test that reference model is properly frozen."""
    config = DynamicDampingConfig()
    model = DummyModel()

    # Set some specific values in base model
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.5)

    state = DynamicDampingState(model, config)

    # Reference model should be in eval mode
    assert not state.ref_model.training

    # Reference model parameters should not require gradients
    for param in state.ref_model.parameters():
        assert not param.requires_grad

    # Reference model should have same initial values
    assert torch.allclose(
        state.ref_model.linear.weight,
        model.linear.weight
    )

    # Modify base model
    with torch.no_grad():
        model.linear.weight.fill_(2.0)

    # Reference model should remain unchanged
    assert torch.allclose(
        state.ref_model.linear.weight,
        torch.ones_like(state.ref_model.linear.weight)
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
