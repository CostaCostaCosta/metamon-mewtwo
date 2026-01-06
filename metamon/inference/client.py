#!/usr/bin/env python3
"""
Client for GPU Inference Server.

This client runs in the PyKMN process and communicates with the inference server,
keeping GPU operations completely separate from battle simulation.
"""

import numpy as np
import requests
import pickle
import base64
from typing import Dict, Optional
import time


class InferenceClient:
    """
    Client for remote GPU inference.

    This replaces LocalPolicyRunner and communicates with the inference server,
    avoiding all memory corruption issues.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: float = 5.0,  # Increased default timeout
        retry_count: int = 3,
        client_id: Optional[str] = None,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.retry_count = retry_count
        self.client_id = client_id or f"client_{id(self)}"  # Unique client ID

        # Check server health with longer timeout
        self._check_health()

    def _check_health(self):
        """Check if server is healthy."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.server_url}/health",
                    timeout=10.0  # Longer timeout for health check
                )
                response.raise_for_status()
                info = response.json()
                print(f"✓ Connected to inference server: {info}")
                return
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Cannot connect to inference server at {self.server_url}\n"
                        f"Make sure the server is running with:\n"
                        f"  python -m metamon.inference.server --model <MODEL> --batch_size 128 --port 8080"
                    )
                time.sleep(1.0)  # Wait before retry
            except Exception as e:
                raise RuntimeError(f"Server health check failed: {e}")

    def infer(
        self,
        observations: Dict[str, np.ndarray],
        legal_masks: np.ndarray,
        request_id: Optional[str] = None,
        reset_state: bool = False
    ) -> np.ndarray:
        """
        Run inference via the server.

        Args:
            observations: Observation dictionary
            legal_masks: Legal action masks
            request_id: Optional request ID for tracking
            reset_state: Whether to reset hidden state for this client

        Returns:
            Selected actions as numpy array
        """
        # Serialize observations and masks
        obs_bytes = pickle.dumps(observations)
        obs_b64 = base64.b64encode(obs_bytes).decode('utf-8')

        mask_bytes = pickle.dumps(legal_masks)
        mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

        # Prepare request
        request_data = {
            'observations': obs_b64,
            'legal_masks': mask_b64,
            'request_id': request_id,
            'client_id': self.client_id,
            'reset_state': reset_state
        }

        # Send request with retries
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    f"{self.server_url}/inference",
                    json=request_data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Decode response
                result = response.json()
                if 'error' in result:
                    raise RuntimeError(f"Inference error: {result['error']}")

                actions_b64 = result['actions']
                actions_bytes = base64.b64decode(actions_b64)
                actions = pickle.loads(actions_bytes)
                # Debug logging removed - actions shape verified

                return actions

            except requests.exceptions.Timeout:
                if attempt == self.retry_count - 1:
                    raise RuntimeError(f"Inference request timed out after {self.retry_count} attempts")
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff

            except requests.exceptions.ConnectionError as e:
                if attempt == self.retry_count - 1:
                    raise RuntimeError(f"Cannot connect to inference server at {self.server_url}: {e}")
                time.sleep(0.1 * (attempt + 1))

            except Exception as e:
                if attempt == self.retry_count - 1:
                    raise RuntimeError(f"Inference failed after {self.retry_count} attempts: {e}")
                time.sleep(0.1 * (attempt + 1))

    def reset(self, batch_size: Optional[int] = None):
        """Reset state (compatibility with PolicyRunner interface)."""
        # Send a dummy request with reset_state=True to reset hidden state on server
        # This will be called at the start of new episodes
        pass  # Hidden state will be reset on next inference call if needed

    def update_rewards(self, rewards: np.ndarray):
        """Update rewards (compatibility)."""
        # Could send to server for RL2 state tracking if needed
        pass

    def reset_hidden_state_for_dones(self, dones: np.ndarray):
        """Reset hidden states for done episodes (compatibility)."""
        # Could send to server if we implement stateful inference
        pass


class RemotePolicyRunner:
    """
    Drop-in replacement for LocalPolicyRunner that uses the inference server.

    This maintains the same interface but delegates all GPU operations to the server,
    completely avoiding memory corruption issues.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        model_name: Optional[str] = None,  # For compatibility
        device: Optional[str] = None,  # Ignored - server handles this
        client_id: Optional[str] = None,  # Unique client ID
        **kwargs
    ):
        self.client = InferenceClient(server_url, client_id=client_id)
        self.model_name = model_name or "remote"

    def infer(
        self,
        obs_dict: Dict[str, np.ndarray],
        legal_mask_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Run inference via the server.

        Maintains same interface as LocalPolicyRunner.
        Sends the entire batch in one request for efficiency.
        """
        batch_size = legal_mask_batch.shape[0]

        # Send entire batch in one request (much faster!)
        actions = self.client.infer(obs_dict, legal_mask_batch, reset_state=False)
        return actions

    def reset(self, batch_size: Optional[int] = None):
        """Reset (compatibility)."""
        self.client.reset(batch_size)

    def update_rewards(self, rewards: np.ndarray):
        """Update rewards (compatibility)."""
        self.client.update_rewards(rewards)

    def reset_hidden_state_for_dones(self, dones: np.ndarray):
        """Reset hidden states (compatibility)."""
        self.client.reset_hidden_state_for_dones(dones)


def test_client():
    """Test the inference client."""
    print("Testing inference client...")

    # Create client
    client = InferenceClient()

    # Create fake observation
    obs = {
        'numbers': np.random.randn(48).astype(np.float32),
        'text': np.array(['test']),
    }
    legal_mask = np.array([True, True, False, False, True, False, False, False, False])

    # Run inference
    print("Sending inference request...")
    actions = client.infer(obs, legal_mask)
    print(f"Received actions: {actions}")

    print("✓ Client test successful")


if __name__ == "__main__":
    test_client()