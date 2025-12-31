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
    inference on GPU or CPU.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint: Optional[int] = None,
        device: str = "cuda",
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        """
        Initialize local policy runner.

        Args:
            model_name: Name of pretrained model (e.g., "SyntheticRLV2")
            checkpoint: Checkpoint number (None for default)
            device: Device to run inference on ("cuda" or "cpu")
            temperature: Sampling temperature (1.0 = unmodified, higher = more random)
        """
        from metamon.rl.pretrained import get_pretrained_model
        import torch.nn.functional as F

        self.model_name = model_name
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.temperature = temperature
        self.verbose = verbose

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

        # Get action space size from the pretrained model
        self.action_dim = pretrained_cls.action_space.gym_space.n
        print(f"Action space size: {self.action_dim}")

        # Initialize recurrent hidden state (required for inference mode)
        # Following pattern from amago.experiment.interact()
        self.hidden_state = None  # Will be initialized on first infer() call

        # Track RL2 state (prev action + reward)
        self.prev_actions = None  # shape: (batch_size,)
        self.prev_rewards = None  # shape: (batch_size,)
        self.time_idx = 0

        print(f"Model loaded successfully! Action dim: {self.action_dim}")

    def reset(self):
        """Reset episode state (call when starting new episodes)."""
        # Reset hidden state (will be re-initialized on next infer call)
        self.hidden_state = None
        self.prev_actions = None
        self.prev_rewards = None
        self.time_idx = 0

    def infer(
        self,
        obs_dict: dict[str, np.ndarray],
        legal_mask_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Run inference with the pretrained model.

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

        # Initialize hidden state on first call (required for inference mode)
        if self.hidden_state is None:
            self.hidden_state = self.agent.traj_encoder.init_hidden_state(
                batch_size, self.device
            )

        # Convert observations to torch tensors on device
        obs_torch = {}
        for key, value in obs_dict.items():
            obs_torch[key] = torch.from_numpy(value).to(self.device)

        # Add illegal action mask to observations (AMAGO expects this)
        # Note: AMAGO uses "illegal_actions" where True = illegal
        illegal_mask = ~legal_mask_batch  # Invert: True = illegal

        # Trim mask to match model's action space (e.g., 9 for MinimalActionSpace, 13 for full)
        illegal_mask_trimmed = illegal_mask[:, :self.action_dim]

        if self.verbose and batch_size > 0:
            print(f"[PolicyRunner] legal_mask_batch shape: {legal_mask_batch.shape}, "
                  f"raw_legal={legal_mask_batch[0]}, "
                  f"trimmed_legal={legal_mask_batch[0, :self.action_dim]}")

        obs_torch["illegal_actions"] = torch.from_numpy(illegal_mask_trimmed).to(self.device)

        # Build RL2 input (prev action + reward)
        if self.prev_actions is None:
            # First step: zero init
            rl2s = torch.zeros(
                (batch_size, self.action_dim + 1), device=self.device, dtype=torch.float32
            )
            time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        else:
            # Concatenate prev action (one-hot) + prev reward
            prev_action_onehot = F.one_hot(
                self.prev_actions.long(), self.action_dim
            ).float()
            rl2s = torch.cat(
                [prev_action_onehot, self.prev_rewards.unsqueeze(-1)], dim=-1
            )
            time_idxs = torch.full(
                (batch_size,), self.time_idx, dtype=torch.long, device=self.device
            )

        # Add sequence dimension (AMAGO expects [batch, length, ...])
        # Following the pattern from amago.experiment.interact()
        obs_torch_seq = {k: v.unsqueeze(1) for k, v in obs_torch.items()}
        rl2s_seq = rl2s.unsqueeze(1)
        # time_idxs needs shape (B, L, 1) so squeeze(-1) in position embedding gives (B, L)
        time_idxs_seq = time_idxs.unsqueeze(1).unsqueeze(2)  # (batch,) -> (batch, 1, 1)

        # Get actions from agent
        with torch.no_grad():
            actions, self.hidden_state = self.agent.get_actions(
                obs=obs_torch_seq,
                rl2s=rl2s_seq,
                time_idxs=time_idxs_seq,
                hidden_state=self.hidden_state,
                sample=True,
            )

        # Remove sequence and action dimensions, convert to numpy
        # actions shape: (batch, length, 1) -> squeeze to (batch,)
        actions_np = actions.squeeze(-1).squeeze(1).cpu().numpy().astype(np.int32)

        if self.verbose:
            legal_actions = np.where(legal_mask_batch[0])[0]
            print(f"[PolicyRunner] Step {self.time_idx}: action={actions_np[0]}, "
                  f"legal_actions={legal_actions[:5]}{'...' if len(legal_actions) > 5 else ''}, "
                  f"num_legal={len(legal_actions)}")

        # Store for next RL2 step (shape: batch,)
        self.prev_actions = actions.squeeze(-1).squeeze(1)

        # Initialize prev_rewards to zeros if not set yet
        # (will be updated with actual rewards via update_rewards())
        if self.prev_rewards is None:
            self.prev_rewards = torch.zeros((batch_size,), device=self.device)

        self.time_idx += 1

        return actions_np

    def update_rewards(self, rewards: np.ndarray):
        """
        Update stored rewards for RL2 state.

        Call this after stepping the environment to provide rewards
        for the next inference step.

        Args:
            rewards: Rewards from environment, shape (batch_size,)
        """
        self.prev_rewards = torch.from_numpy(rewards).float().to(self.device)


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

        Args:
            num_battles: Number of complete battles to collect
            max_steps_per_battle: Maximum steps before timeout
            verbose: Whether to print progress

        Returns:
            List of Trajectory objects (length = num_battles)
        """
        collected_trajectories = []
        total_steps = 0
        battles_completed = 0

        # Reset environment
        obs_p1, obs_p2, masks_p1, masks_p2 = self.vec_env.reset()

        # Reset policy states if they support it
        if hasattr(self.policy_p1, "reset"):
            self.policy_p1.reset()
        if hasattr(self.policy_p2, "reset"):
            self.policy_p2.reset()

        while battles_completed < num_battles:
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

            if verbose and total_steps % 100 == 0:
                print(f"  Step {total_steps}: rewards=({rewards_p1[0]:.2f}, {rewards_p2[0]:.2f}), "
                      f"done={dones[0]}, battles_completed={battles_completed}")

            # Update policy rewards for RL2 state (if they support it)
            if hasattr(self.policy_p1, "update_rewards"):
                self.policy_p1.update_rewards(rewards_p1)
            if hasattr(self.policy_p2, "update_rewards"):
                self.policy_p2.update_rewards(rewards_p2)

            # Extract legal masks for next step
            masks_p1, masks_p2 = self.vec_env._extract_legal_masks()

            total_steps += 1

            # Check for completed battles
            if info["num_done"] > 0:
                completed = self.vec_env.get_completed_trajectories()
                collected_trajectories.extend(completed)
                battles_completed = len(collected_trajectories)

                if verbose:
                    print(
                        f"  ✓ Battle completed! {battles_completed}/{num_battles} done "
                        f"({total_steps} total steps)"
                    )

                # Reset finished environments
                # TODO: Implement partial reset for only finished envs
                # For now, reset all when any finish (simplified)
                if battles_completed < num_battles:
                    obs_p1, obs_p2, masks_p1, masks_p2 = self.vec_env.reset()
                    total_steps = 0

                    # Reset policy states
                    if hasattr(self.policy_p1, "reset"):
                        self.policy_p1.reset()
                    if hasattr(self.policy_p2, "reset"):
                        self.policy_p2.reset()

            # Timeout check
            if total_steps >= max_steps_per_battle:
                if verbose:
                    print(
                        f"Warning: Reached max steps ({max_steps_per_battle}), "
                        "resetting environments"
                    )
                obs_p1, obs_p2, masks_p1, masks_p2 = self.vec_env.reset()
                total_steps = 0

                # Reset policy states
                if hasattr(self.policy_p1, "reset"):
                    self.policy_p1.reset()
                if hasattr(self.policy_p2, "reset"):
                    self.policy_p2.reset()

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
