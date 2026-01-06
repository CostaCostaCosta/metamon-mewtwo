"""
Policy runner abstraction for pypkmn environments.

Provides a unified interface for policy inference with:
- Local pretrained models
- Remote policy servers
- Heuristic baselines
- Batched inference for efficiency
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch

from .vector_env import PyKMNVectorEnv, Trajectory
from .action_mapper import filter_illegal_actions


class PolicyRunner(ABC):
    """
    Abstract base class for policy inference.

    Subclasses implement different policy types (local model, remote server, etc.)
    """

    @abstractmethod
    def infer(
        self,
        obs_batch: np.ndarray,
        legal_mask_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Run inference on a batch of observations.

        Args:
            obs_batch: Observations, shape (batch_size, obs_dim)
            legal_mask_batch: Legal action masks, shape (batch_size, num_actions)

        Returns:
            Selected actions, shape (batch_size,)
        """
        pass


class LocalPolicyRunner(PolicyRunner):
    """
    Run inference using a local pretrained metamon model.

    Loads model from HuggingFace or local checkpoint and runs
    inference on GPU or CPU with batched inference support.

    IMPROVEMENTS:
    - Per-environment time indexing (not global counter)
    - Buffer preallocation for zero-copy operations
    - Mixed precision support (bfloat16 autocast)
    - Proper hidden state reset for episodic boundaries
    """

    def __init__(
        self,
        model_name: str,
        checkpoint: Optional[int] = None,
        device: str = "cuda",
        temperature: float = 1.0,
        use_amp: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize local policy runner.

        Args:
            model_name: Name of pretrained model (e.g., "SyntheticRLV2")
            checkpoint: Checkpoint number (None for default)
            device: Device to run inference on ("cuda" or "cpu")
            temperature: Sampling temperature (1.0 = unmodified, higher = more random)
            use_amp: Use mixed precision (bfloat16) for inference (default: True)
            verbose: Print debug info (default: False)
        """
        from metamon.rl.pretrained import get_pretrained_model

        self.model_name = model_name
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.temperature = temperature
        self.verbose = verbose
        self.use_amp = use_amp and (device == "cuda")

        # Load pretrained model
        print(f"Loading pretrained model: {model_name}")
        pretrained_cls = get_pretrained_model(model_name)

        # initialize_agent returns the experiment object
        experiment = pretrained_cls.initialize_agent(
            checkpoint=checkpoint,
            log=False,
            action_temperature=temperature,
        )

        # The actual agent is at experiment.policy
        self.agent = experiment.policy
        self.agent.eval()  # Set to evaluation mode

        # Enable TF32 for better performance on Ampere+ GPUs (RTX 30/40 series)
        if self.use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Mixed precision (bfloat16) enabled with TF32 matmul")

        # Get action space size from the pretrained model
        self.action_dim = pretrained_cls.action_space.gym_space.n
        print(f"Action space size: {self.action_dim}")

        # State will be initialized on first infer()
        self.hidden_state = None
        self.prev_actions = None  # (N,) tensor
        self.prev_rewards = None  # (N,) tensor
        self.time_idxs = None     # (N,) tensor - PER-ENV counters (not global!)

        # Preallocated buffers (allocated on first infer to avoid overhead)
        self.rl2_buffer = None                   # (N, action_dim+1)
        self.prev_action_onehot_buffer = None    # (N, action_dim)

        print(f"Model loaded successfully! Action dim: {self.action_dim}")

    def reset(self, batch_size: Optional[int] = None):
        """
        Reset episode state (call when starting new episodes).

        Args:
            batch_size: If provided, preallocate buffers for this batch size
        """
        self.hidden_state = None
        self.prev_actions = None
        self.prev_rewards = None
        self.time_idxs = None

        # Preallocate buffers if batch_size known
        if batch_size is not None:
            self.time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
            self.rl2_buffer = torch.zeros((batch_size, self.action_dim + 1),
                                          dtype=torch.float32, device=self.device)
            self.prev_action_onehot_buffer = torch.zeros((batch_size, self.action_dim),
                                                          dtype=torch.float32, device=self.device)

    def infer(
        self,
        obs_dict: dict[str, np.ndarray],
        legal_mask_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Run batched inference with the pretrained model.

        OPTIMIZED FOR BATCHING:
        - Per-env time indexing (not global counter)
        - Preallocated buffers (no per-step allocations)
        - Mixed precision (bfloat16) support
        - Async GPU transfers

        Args:
            obs_dict: Dictionary of observations with keys like "numbers", "text_tokens"
                Each value has shape (batch_size, feature_dim)
            legal_mask_batch: Legal action masks, shape (batch_size, num_actions)
                True = legal action

        Returns:
            Selected actions, shape (batch_size,)
        """
        import torch.nn.functional as F

        batch_size = legal_mask_batch.shape[0]

        # Initialize on first call
        if self.hidden_state is None:
            self.hidden_state = self.agent.traj_encoder.init_hidden_state(
                batch_size, self.device
            )
        if self.time_idxs is None:
            # First inference: initialize per-env time counters and buffers
            self.time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
            self.rl2_buffer = torch.zeros((batch_size, self.action_dim + 1),
                                          dtype=torch.float32, device=self.device)
            self.prev_action_onehot_buffer = torch.zeros((batch_size, self.action_dim),
                                                          dtype=torch.float32, device=self.device)

        # Convert observations to torch (async GPU transfer)
        # Skip text fields that cannot be converted to tensors
        obs_torch = {}
        for k, v in obs_dict.items():
            # Skip string/text fields (e.g., "text", "text_raw")
            if v.dtype.kind in ('U', 'S', 'O'):  # Unicode, bytes, or object dtypes
                continue
            obs_torch[k] = torch.from_numpy(v).to(self.device, non_blocking=True)

        # ✅ FIX #1: Legal action masking (already embedded in observations)
        # MetamonMaskedActor will apply mask internally via straight_from_obs["illegal_actions"]
        illegal_mask = ~legal_mask_batch  # Invert: True = illegal
        illegal_mask_trimmed = illegal_mask[:, :self.action_dim]

        # Ensure bool dtype for consistency
        obs_torch["illegal_actions"] = torch.from_numpy(illegal_mask_trimmed).to(
            self.device, non_blocking=True
        ).bool()

        # ✅ FIX #3 + #8: Build RL2 input (reuse preallocated buffers, no allocations!)
        if self.prev_actions is None:
            # First step: zeros
            self.rl2_buffer.zero_()
        else:
            # Scatter prev_actions into onehot buffer (avoids F.one_hot allocation)
            self.prev_action_onehot_buffer.zero_()
            self.prev_action_onehot_buffer.scatter_(
                dim=1,
                index=self.prev_actions.long().unsqueeze(1),
                value=1.0
            )
            # Concatenate onehot + reward (in-place via slicing)
            self.rl2_buffer[:, :self.action_dim] = self.prev_action_onehot_buffer
            self.rl2_buffer[:, self.action_dim] = self.prev_rewards

        # Add sequence dimension: (N, ...) -> (N, 1, ...)
        obs_torch_seq = {k: v.unsqueeze(1) for k, v in obs_torch.items()}
        rl2s_seq = self.rl2_buffer.unsqueeze(1)

        # ✅ FIX #4: Per-env time indexing (not global!)
        time_idxs_seq = self.time_idxs.unsqueeze(1).unsqueeze(2)  # (N,) -> (N, 1, 1)

        # ✅ FIX #7: Mixed precision via autocast (not .half())
        with torch.inference_mode():  # Faster than no_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    actions, self.hidden_state = self.agent.get_actions(
                        obs=obs_torch_seq,
                        rl2s=rl2s_seq,
                        time_idxs=time_idxs_seq,
                        hidden_state=self.hidden_state,
                        sample=True,
                    )
            else:
                actions, self.hidden_state = self.agent.get_actions(
                    obs=obs_torch_seq,
                    rl2s=rl2s_seq,
                    time_idxs=time_idxs_seq,
                    hidden_state=self.hidden_state,
                    sample=True,
                )

        # Detach hidden state to prevent computational graph retention
        # This prevents memory leaks from accumulated gradients across thousands of steps
        if isinstance(self.hidden_state, torch.Tensor):
            self.hidden_state = self.hidden_state.detach()
        elif isinstance(self.hidden_state, (tuple, list)):
            self.hidden_state = type(self.hidden_state)(
                h.detach() if isinstance(h, torch.Tensor) else h
                for h in self.hidden_state
            )

        # Extract actions: (N, 1, 1) -> (N,)
        actions_np = actions.squeeze(-1).squeeze(1).cpu().numpy().astype(np.int32)

        if self.verbose:
            legal_actions = np.where(legal_mask_batch[0])[0]
            print(f"[PolicyRunner] Step {self.time_idxs[0].item()}: action={actions_np[0]}, "
                  f"legal_actions={legal_actions[:5]}{'...' if len(legal_actions) > 5 else ''}, "
                  f"num_legal={len(legal_actions)}")

        # Update RL2 state (clone to allow inplace updates later)
        # Note: actions comes from inference_mode() so we need to clone
        self.prev_actions = actions.squeeze(-1).squeeze(1).clone()
        if self.prev_rewards is None:
            self.prev_rewards = torch.zeros((batch_size,), device=self.device)

        # ✅ FIX #4: Increment per-env time counters
        self.time_idxs += 1

        return actions_np

    def update_rewards(self, rewards: np.ndarray):
        """
        ✅ FIX #3: Update stored rewards for RL2 state.

        Call this after stepping the environment to provide rewards
        for the next inference step.

        Args:
            rewards: Rewards from environment, shape (batch_size,)
        """
        self.prev_rewards = torch.from_numpy(rewards).float().to(self.device, non_blocking=True)

    def reset_hidden_state_for_dones(self, dones: np.ndarray):
        """
        ✅ FIX #2 + #4: Reset hidden state AND time indices for finished episodes.

        This method handles episodic boundaries in vectorized environments by:
        1. Resetting hidden states for finished envs (structure-aware)
        2. Resetting RL2 state (prev_actions, prev_rewards)
        3. Resetting per-env time counters

        Args:
            dones: Boolean array (batch_size,) indicating which envs finished
        """
        if not dones.any():
            return

        # FIX #2: Structure-aware hidden state reset
        # AMAGO's reset_hidden_state() handles tensor/tuple/dict structures
        if self.hidden_state is not None:
            self.hidden_state = self.agent.traj_encoder.reset_hidden_state(
                self.hidden_state,
                dones  # Expects numpy array
            )

        # Reset RL2 state for done envs
        done_mask = torch.from_numpy(dones).to(self.device)
        if self.prev_actions is not None:
            self.prev_actions[done_mask] = 0
        if self.prev_rewards is not None:
            self.prev_rewards[done_mask] = 0.0

        # FIX #4: Reset per-env time counters
        if self.time_idxs is not None:
            self.time_idxs[done_mask] = 0


class RandomPolicyRunner(PolicyRunner):
    """Random policy that selects uniformly from legal actions."""

    def infer(
        self,
        obs_batch: np.ndarray,
        legal_mask_batch: np.ndarray,
    ) -> np.ndarray:
        """Select random legal actions."""
        batch_size, num_actions = legal_mask_batch.shape
        actions = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            legal_actions = np.where(legal_mask_batch[i])[0]
            if len(legal_actions) > 0:
                actions[i] = np.random.choice(legal_actions)
            else:
                actions[i] = 0

        return actions


class SelfPlayRunner:
    """
    Run vectorized self-play with one or two policies.

    Handles the main loop of:
    1. Reset environments
    2. Infer actions for both players
    3. Step environments
    4. Collect trajectories
    """

    def __init__(
        self,
        vec_env: PyKMNVectorEnv,
        policy_p1: PolicyRunner,
        policy_p2: Optional[PolicyRunner] = None,
    ):
        """
        Initialize self-play runner.

        Args:
            vec_env: Vectorized pypkmn environment
            policy_p1: Policy for player 1
            policy_p2: Policy for player 2 (if None, use policy_p1 for both)
        """
        self.vec_env = vec_env
        self.policy_p1 = policy_p1
        self.policy_p2 = policy_p2 if policy_p2 is not None else policy_p1

    def collect_trajectories(
        self,
        num_battles: int,
        max_steps_per_battle: int = 1000,
        verbose: bool = False,
    ) -> list[Trajectory]:
        """
        Collect trajectories from self-play battles.

        OPTIMIZED: Runs full batches to completion before resetting,
        maximizing throughput from batched inference.

        Args:
            num_battles: Number of complete battles to collect
            max_steps_per_battle: Maximum steps before timeout
            verbose: Whether to print progress

        Returns:
            List of Trajectory objects (length = num_battles)
        """
        collected_trajectories = []

        while len(collected_trajectories) < num_battles:
            # Determine batch size (might be smaller for last batch)
            battles_remaining = num_battles - len(collected_trajectories)
            current_batch_size = min(self.vec_env.num_envs, battles_remaining)

            # Reset environment for new batch
            obs_p1, obs_p2, masks_p1, masks_p2 = self.vec_env.reset()

            # Reset policy states
            if hasattr(self.policy_p1, "reset"):
                self.policy_p1.reset(current_batch_size)
            if hasattr(self.policy_p2, "reset"):
                self.policy_p2.reset(current_batch_size)

            total_steps = 0
            batch_complete = False

            # Run until ALL environments in batch finish or timeout
            while not batch_complete and total_steps < max_steps_per_battle:
                # Infer actions for both players
                actions_p1 = self.policy_p1.infer(obs_p1, masks_p1)
                actions_p2 = self.policy_p2.infer(obs_p2, masks_p2)

                # Filter illegal actions (safety check)
                actions_p1 = filter_illegal_actions(actions_p1, masks_p1)
                actions_p2 = filter_illegal_actions(actions_p2, masks_p2)

                # Step environment
                obs_p1, obs_p2, rewards_p1, rewards_p2, dones, info = self.vec_env.step(
                    actions_p1, actions_p2
                )

                # Update policy rewards for RL2 state
                if hasattr(self.policy_p1, "update_rewards"):
                    self.policy_p1.update_rewards(rewards_p1)
                if hasattr(self.policy_p2, "update_rewards"):
                    self.policy_p2.update_rewards(rewards_p2)

                # Reset hidden states for finished episodes
                if dones.any():
                    if hasattr(self.policy_p1, "reset_hidden_state_for_dones"):
                        self.policy_p1.reset_hidden_state_for_dones(dones)
                    if hasattr(self.policy_p2, "reset_hidden_state_for_dones"):
                        self.policy_p2.reset_hidden_state_for_dones(dones)

                # Extract legal masks for next step
                masks_p1, masks_p2 = self.vec_env._extract_legal_masks()

                total_steps += 1

                # Check if batch is complete (all environments done)
                if info["num_done"] == current_batch_size:
                    batch_complete = True

            # Collect all completed trajectories from this batch
            completed = self.vec_env.get_completed_trajectories()
            collected_trajectories.extend(completed[:battles_remaining])

            if verbose:
                print(
                    f"  ✓ Batch complete: {len(completed)} battles in {total_steps} steps "
                    f"(total: {len(collected_trajectories)}/{num_battles})"
                )

            # Timeout warning
            if total_steps >= max_steps_per_battle and not batch_complete:
                if verbose:
                    print(
                        f"  ⚠️  Batch timeout at {max_steps_per_battle} steps "
                        f"({info['num_done']}/{current_batch_size} completed)"
                    )

        return collected_trajectories[:num_battles]


class EvaluationRunner:
    """
    Run head-to-head evaluation between two policies.

    Measures win rates, average game length, etc.
    """

    def __init__(
        self,
        vec_env: PyKMNVectorEnv,
        policy_p1: PolicyRunner,
        policy_p2: PolicyRunner,
    ):
        """
        Initialize evaluation runner.

        Args:
            vec_env: Vectorized pypkmn environment
            policy_p1: Policy for player 1
            policy_p2: Policy for player 2
        """
        self.vec_env = vec_env
        self.policy_p1 = policy_p1
        self.policy_p2 = policy_p2

    def evaluate(
        self,
        num_battles: int,
        verbose: bool = False,
    ) -> dict:
        """
        Evaluate policies head-to-head.

        Args:
            num_battles: Number of battles to run
            verbose: Whether to print progress

        Returns:
            Dictionary with evaluation metrics:
            - p1_wins: Number of P1 wins
            - p2_wins: Number of P2 wins
            - ties: Number of ties
            - avg_length: Average battle length (steps)
        """
        runner = SelfPlayRunner(self.vec_env, self.policy_p1, self.policy_p2)
        trajectories = runner.collect_trajectories(num_battles, verbose=verbose)

        # Compute statistics
        p1_wins = sum(1 for t in trajectories if t.winner == 1)
        p2_wins = sum(1 for t in trajectories if t.winner == 2)
        ties = sum(1 for t in trajectories if t.winner == 0)
        avg_length = np.mean([len(t.transitions) for t in trajectories])

        return {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "ties": ties,
            "win_rate_p1": p1_wins / num_battles,
            "win_rate_p2": p2_wins / num_battles,
            "avg_length": avg_length,
        }
