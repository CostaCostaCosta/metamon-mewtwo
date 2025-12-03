"""
Policy Server for Batched GPU Inference

Loads a pretrained model once and serves inference requests from multiple
battle workers over ZMQ, enabling efficient batched GPU inference.
"""

import pickle
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import signal
import sys

import zmq
import torch

from metamon.rl.pretrained import get_pretrained_model, PretrainedModel


@dataclass
class InferenceRequest:
    """A single inference request from a battle worker."""
    request_id: bytes
    observation: Any
    traj_id: int
    arrival_time: float


class PolicyServer:
    """
    Batched GPU inference server for pretrained Metamon policies.

    Architecture:
    1. Loads pretrained model once into GPU
    2. Receives observation requests via ZMQ
    3. Accumulates requests into batches
    4. Performs batched GPU inference
    5. Returns actions to requesters

    Args:
        model_name: Name of pretrained model (e.g., "SyntheticRLV2")
        checkpoint: Checkpoint number to load (None for default)
        batch_size: Maximum batch size for GPU inference
        timeout_ms: Max time to wait for batch accumulation (milliseconds)
        port: ZMQ port to bind to
    """

    def __init__(
        self,
        model_name: str,
        checkpoint: Optional[int] = None,
        batch_size: int = 32,
        timeout_ms: int = 50,
        port: int = 5555,
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.port = port

        self.context = None
        self.socket = None
        self.agent = None
        self.pretrained_model = None

        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_inference_time = 0.0

        # Graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[PolicyServer] Received signal {signum}, shutting down...")
        self.running = False

    def start(self):
        """Initialize model and start serving requests."""
        print(f"[PolicyServer] Starting server on port {self.port}")
        print(f"[PolicyServer] Model: {self.model_name}, Checkpoint: {self.checkpoint}")
        print(f"[PolicyServer] Batch size: {self.batch_size}, Timeout: {self.timeout_ms}ms")

        # Load pretrained model
        print(f"[PolicyServer] Loading model...")
        self.pretrained_model = get_pretrained_model(self.model_name)
        self.agent = self.pretrained_model.initialize_agent(
            checkpoint=self.checkpoint,
            log=False
        )
        print(f"[PolicyServer] Model loaded successfully")

        # Set up ZMQ server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{self.port}")

        # Set socket timeout for responsive shutdown
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        print(f"[PolicyServer] Listening on tcp://*:{self.port}")
        print(f"[PolicyServer] Ready to serve requests")

        # Main server loop
        try:
            while self.running:
                self._serve_batch()
        except KeyboardInterrupt:
            print("\n[PolicyServer] Interrupted by user")
        finally:
            self._shutdown()

    def _serve_batch(self):
        """Collect requests and perform batched inference."""
        batch = self._collect_batch()

        if not batch:
            return

        # Perform batched inference
        start_time = time.time()
        actions = self._batch_inference(batch)
        inference_time = time.time() - start_time

        # Send responses
        self._distribute_responses(batch, actions)

        # Update statistics
        self.total_batches += 1
        self.total_requests += len(batch)
        self.total_inference_time += inference_time

        # Log periodically
        if self.total_batches % 100 == 0:
            avg_batch_size = self.total_requests / self.total_batches
            avg_inference_time = self.total_inference_time / self.total_batches
            throughput = avg_batch_size / avg_inference_time
            print(
                f"[PolicyServer] Stats: {self.total_requests} requests, "
                f"{self.total_batches} batches, "
                f"avg batch size: {avg_batch_size:.1f}, "
                f"avg inference: {avg_inference_time*1000:.1f}ms, "
                f"throughput: {throughput:.1f} req/s"
            )

    def _collect_batch(self) -> List[InferenceRequest]:
        """
        Accumulate requests until batch_size reached or timeout expires.

        Uses dynamic timeout: if many requests are queued, reduce timeout
        to improve responsiveness.
        """
        batch = []
        start_time = time.time()
        timeout_sec = self.timeout_ms / 1000.0

        while len(batch) < self.batch_size:
            elapsed = time.time() - start_time
            remaining = timeout_sec - elapsed

            # Stop if timeout reached and we have at least one request
            if remaining <= 0 and len(batch) > 0:
                break

            # Try to receive a message (non-blocking after first request)
            try:
                # First request: block up to timeout
                # Subsequent: continue accumulating quickly
                recv_timeout = max(1, int(remaining * 1000)) if len(batch) == 0 else 1
                self.socket.setsockopt(zmq.RCVTIMEO, recv_timeout)

                msg = self.socket.recv_multipart()

                # Parse message: [identity, empty, request_id, serialized_data]
                if len(msg) >= 4:
                    identity = msg[0]
                    request_id = msg[2]
                    serialized_data = msg[3]

                    try:
                        data = pickle.loads(serialized_data)

                        # Support both legacy and new request formats
                        if "type" in data and data["type"] == "get_actions":
                            # New get_actions request
                            batch.append(InferenceRequest(
                                request_id=identity + b"|" + request_id,
                                observation=data,  # Store full request data
                                traj_id=0,  # Not used for get_actions
                                arrival_time=time.time()
                            ))
                        else:
                            # Legacy single observation request
                            observation = data["observation"]
                            traj_id = data.get("traj_id", 0)

                            batch.append(InferenceRequest(
                                request_id=identity + b"|" + request_id,
                                observation=observation,
                                traj_id=traj_id,
                                arrival_time=time.time()
                            ))
                    except Exception as e:
                        print(f"[PolicyServer] Error deserializing request: {e}")
                        # Send error response
                        error_response = pickle.dumps({"error": str(e)})
                        self.socket.send_multipart([identity, b"", request_id, error_response])

            except zmq.Again:
                # Timeout reached, no more messages available
                if len(batch) > 0:
                    break
                # If no batch yet, continue waiting
            except Exception as e:
                print(f"[PolicyServer] Error receiving message: {e}")
                break

        return batch

    def _batch_inference(self, batch: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """
        Perform batched GPU inference for a list of requests.

        Args:
            batch: List of inference requests

        Returns:
            List of action dictionaries, one per request
        """
        try:
            results = []

            with torch.no_grad():
                for req in batch:
                    # Check request type
                    if isinstance(req.observation, dict) and req.observation.get("type") == "get_actions":
                        # Handle get_actions request
                        data = req.observation
                        obs = data["obs"]
                        rl2s = data["rl2s"]
                        time_idxs = data["time_idxs"]
                        hidden_state = data.get("hidden_state")
                        sample = data.get("sample", True)

                        # Call agent.policy.get_actions()
                        actions, new_hidden_state = self.agent.policy.get_actions(
                            obs=obs,
                            rl2s=rl2s,
                            time_idxs=time_idxs,
                            hidden_state=hidden_state,
                            sample=sample,
                        )

                        results.append({
                            "actions": actions,
                            "hidden_state": new_hidden_state,
                        })
                    else:
                        # Legacy single observation request
                        obs = req.observation
                        traj_id = req.traj_id

                        # Call policy for single inference
                        action = self.agent.policy(obs, traj_id)
                        results.append({"action": action})

            return results

        except Exception as e:
            print(f"[PolicyServer] Error during inference: {e}")
            import traceback
            traceback.print_exc()
            # Return error responses for all requests
            return [{"error": str(e)} for _ in batch]

    def _distribute_responses(self, batch: List[InferenceRequest], actions: List[Dict[str, Any]]):
        """Send inference results back to requesters."""
        for request, action in zip(batch, actions):
            try:
                # Parse combined identity
                parts = request.request_id.split(b"|", 1)
                if len(parts) == 2:
                    identity, request_id = parts
                else:
                    print(f"[PolicyServer] Invalid request_id format: {request.request_id}")
                    continue

                # Serialize response
                response_data = pickle.dumps(action)

                # Send: [identity, empty, request_id, response_data]
                self.socket.send_multipart([identity, b"", request_id, response_data])

            except Exception as e:
                print(f"[PolicyServer] Error sending response: {e}")

    def _shutdown(self):
        """Clean up resources."""
        print(f"[PolicyServer] Shutting down...")

        if self.socket:
            self.socket.close()

        if self.context:
            self.context.term()

        print(f"[PolicyServer] Final stats:")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Total batches: {self.total_batches}")
        if self.total_batches > 0:
            print(f"  Avg batch size: {self.total_requests / self.total_batches:.1f}")
            print(f"  Avg inference time: {self.total_inference_time / self.total_batches * 1000:.1f}ms")

        print(f"[PolicyServer] Shutdown complete")


def main():
    """CLI entry point for running policy server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Start a batched GPU inference server for Metamon policies"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Pretrained model name (e.g., SyntheticRLV2)"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint number (default: use model's default)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Maximum batch size for GPU inference (default: 32)"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=50,
        help="Batch accumulation timeout in milliseconds (default: 50)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ port to bind to (default: 5555)"
    )

    args = parser.parse_args()

    server = PolicyServer(
        model_name=args.model,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        timeout_ms=args.timeout_ms,
        port=args.port,
    )

    server.start()


if __name__ == "__main__":
    main()
