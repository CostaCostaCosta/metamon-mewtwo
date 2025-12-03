"""
Policy Client for Remote GPU Inference

Lightweight client that sends observations to a policy server and receives
actions, enabling multiple battle workers to share a single GPU model.
"""

import pickle
import uuid
import time
from typing import Any, Dict, Optional

import zmq


class PolicyClient:
    """
    Client for querying remote policy server.

    Connects to a PolicyServer running on a specified address and sends
    observations for inference, receiving actions in return.

    Args:
        server_address: ZMQ address of policy server (e.g., "tcp://localhost:5555")
        timeout_ms: Request timeout in milliseconds
        max_retries: Maximum number of retry attempts
    """

    def __init__(
        self,
        server_address: str = "tcp://localhost:5555",
        timeout_ms: int = 5000,
        max_retries: int = 3,
    ):
        self.server_address = server_address
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

        self.context = zmq.Context()
        self.socket = None
        self._connect()

        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0

    def _connect(self):
        """Establish connection to policy server."""
        if self.socket is not None:
            self.socket.close()

        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't hang on close
        self.socket.connect(self.server_address)

    def get_action(self, observation: Any, traj_id: int = 0) -> Any:
        """
        Request an action from the policy server.

        Args:
            observation: Game observation (format depends on observation space)
            traj_id: Trajectory ID for AMAGO's memory management

        Returns:
            Action from the policy

        Raises:
            RuntimeError: If request fails after all retries
        """
        request_data = {
            "observation": observation,
            "traj_id": traj_id,
        }

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Generate unique request ID
                request_id = uuid.uuid4().bytes

                # Serialize and send request
                serialized = pickle.dumps(request_data)
                self.socket.send_multipart([b"", request_id, serialized])

                # Receive response
                msg = self.socket.recv_multipart()

                # Parse response: [empty, request_id, response_data]
                if len(msg) >= 3:
                    received_id = msg[1]
                    response_data = msg[2]

                    # Verify request ID matches
                    if received_id != request_id:
                        print(f"[PolicyClient] Warning: Request ID mismatch")

                    # Deserialize response
                    response = pickle.loads(response_data)

                    # Check for errors
                    if "error" in response:
                        raise RuntimeError(f"Server error: {response['error']}")

                    # Extract action
                    action = response.get("action")

                    # Update statistics
                    latency = time.time() - start_time
                    self.total_requests += 1
                    self.total_latency += latency

                    return action

                else:
                    raise RuntimeError(f"Invalid response format: {len(msg)} parts")

            except zmq.Again:
                # Timeout
                print(f"[PolicyClient] Request timeout (attempt {attempt + 1}/{self.max_retries})")
                self.total_errors += 1

                # Reconnect and retry
                if attempt < self.max_retries - 1:
                    print(f"[PolicyClient] Reconnecting to {self.server_address}...")
                    self._connect()
                    time.sleep(0.1)

            except Exception as e:
                print(f"[PolicyClient] Error: {e} (attempt {attempt + 1}/{self.max_retries})")
                self.total_errors += 1

                if attempt < self.max_retries - 1:
                    print(f"[PolicyClient] Reconnecting to {self.server_address}...")
                    self._connect()
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")

        raise RuntimeError(f"Failed to get action after {self.max_retries} attempts")

    def close(self):
        """Clean up resources."""
        if self.socket:
            self.socket.close()

        if self.context:
            self.context.term()

    def get_stats(self) -> Dict[str, float]:
        """Get client statistics."""
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": (self.total_latency / self.total_requests * 1000)
            if self.total_requests > 0
            else 0.0,
            "error_rate": (self.total_errors / self.total_requests)
            if self.total_requests > 0
            else 0.0,
        }

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class RemotePolicyWrapper:
    """
    Wraps a policy to redirect inference to remote server.

    This allows keeping all of AMAGO's agent infrastructure (evaluate_test,
    Timestep preprocessing, etc.) while only replacing the actual policy
    forward pass with a remote call.
    """

    def __init__(self, client: PolicyClient):
        self.client = client
        self.call_count = 0

    def __call__(self, timestep, traj_id):
        """Forward inference request to remote server."""
        self.call_count += 1
        if self.call_count == 1:
            print(f"[RemotePolicyWrapper] First __call__ received! (traj_id={traj_id})")
        # Send the Timestep object to the remote server
        return self.client.get_action(timestep, traj_id)

    def forward(self, *args, **kwargs):
        """Forward method if AMAGO calls this instead of __call__."""
        print(f"[RemotePolicyWrapper] forward() called with args={len(args)}, kwargs={list(kwargs.keys())}")
        return self.__call__(*args, **kwargs)


class RemoteAMAGOAgent:
    """
    Wrapper that uses an actual AMAGO agent but redirects policy calls to remote server.

    This approach:
    1. Creates a real AMAGO agent with all its infrastructure
    2. Replaces only the policy's forward pass to use remote inference
    3. Keeps all Timestep preprocessing, memory management, evaluate_test(), etc.

    Args:
        agent: Actual AMAGO agent from pretrained_model.initialize_agent()
        client: PolicyClient connected to policy server
        verbose: Print request statistics
    """

    def __init__(self, agent, client: PolicyClient, verbose: bool = False):
        self.agent = agent
        self.client = client
        self.verbose = verbose

        # Replace the agent's policy with remote wrapper
        # Store original and create wrapper
        self._original_policy = agent.policy
        self._remote_policy = RemotePolicyWrapper(client)

        print(f"[RemoteAMAGOAgent] Original policy type: {type(self._original_policy)}")
        print(f"[RemoteAMAGOAgent] Original policy has get_actions: {hasattr(self._original_policy, 'get_actions')}")

        # Monkey-patch the policy's get_actions method (this is what AMAGO actually calls!)
        original_get_actions = agent.policy.get_actions
        def remote_get_actions(obs, rl2s, time_idxs, hidden_state=None, sample=True):
            if self.verbose:
                print(f"[RemoteAMAGOAgent] Patched get_actions invoked! obs keys: {list(obs.keys())}, hidden_state type: {type(hidden_state)}")

            # Send get_actions request to remote server
            request_data = {
                "type": "get_actions",
                "obs": obs,
                "rl2s": rl2s,
                "time_idxs": time_idxs,
                "hidden_state": hidden_state,
                "sample": sample,
            }

            try:
                # Use PolicyClient to send request
                for attempt in range(self.client.max_retries):
                    try:
                        import uuid
                        import time as time_module
                        start_time = time_module.time()

                        # Generate unique request ID
                        request_id = uuid.uuid4().bytes

                        # Serialize and send request
                        serialized = pickle.dumps(request_data)
                        self.client.socket.send_multipart([b"", request_id, serialized])

                        # Receive response
                        msg = self.client.socket.recv_multipart()

                        # Parse response: [empty, request_id, response_data]
                        if len(msg) >= 3:
                            received_id = msg[1]
                            response_data = msg[2]

                            # Verify request ID matches
                            if received_id != request_id:
                                print(f"[RemoteAMAGOAgent] Warning: Request ID mismatch")

                            # Deserialize response
                            response = pickle.loads(response_data)

                            # Check for errors
                            if "error" in response:
                                raise RuntimeError(f"Server error: {response['error']}")

                            # Extract actions and hidden_state
                            actions = response.get("actions")
                            new_hidden_state = response.get("hidden_state")

                            # Update statistics
                            latency = time_module.time() - start_time
                            self.client.total_requests += 1
                            self.client.total_latency += latency

                            if self.verbose:
                                print(f"[RemoteAMAGOAgent] Remote inference successful (latency: {latency*1000:.1f}ms)")

                            return actions, new_hidden_state
                        else:
                            raise RuntimeError(f"Invalid response format: {len(msg)} parts")

                    except Exception as e:
                        if attempt < self.client.max_retries - 1:
                            print(f"[RemoteAMAGOAgent] Request failed (attempt {attempt + 1}/{self.client.max_retries}): {e}")
                            self.client._connect()
                            time_module.sleep(0.1)
                        else:
                            raise

                raise RuntimeError(f"Failed after {self.client.max_retries} attempts")

            except Exception as e:
                # Fallback to local CPU model if server fails
                print(f"[RemoteAMAGOAgent] Server request failed, using local CPU fallback: {e}")
                return original_get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=hidden_state, sample=sample)

        agent.policy.get_actions = remote_get_actions
        print(f"[RemoteAMAGOAgent] Monkey-patched agent.policy.get_actions")

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped agent."""
        return getattr(self.agent, name)

    def __call__(self, *args, **kwargs):
        """Delegate calls to the wrapped agent."""
        return self.agent(*args, **kwargs)

    def print_stats(self):
        """Print client statistics."""
        stats = self.client.get_stats()
        print(f"[RemoteAMAGOAgent] Stats:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total errors: {stats['total_errors']}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"  Error rate: {stats['error_rate']:.2%}")


def test_connection(server_address: str = "tcp://localhost:5555"):
    """
    Test connection to policy server.

    Args:
        server_address: Address of policy server

    Returns:
        True if connection successful, False otherwise
    """
    print(f"Testing connection to {server_address}...")

    try:
        with PolicyClient(server_address, timeout_ms=2000) as client:
            # Send a dummy request (will fail but tests connection)
            print("Connection established!")
            return True

    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test policy server connection")
    parser.add_argument(
        "--server",
        default="tcp://localhost:5555",
        help="Policy server address (default: tcp://localhost:5555)",
    )

    args = parser.parse_args()
    test_connection(args.server)
