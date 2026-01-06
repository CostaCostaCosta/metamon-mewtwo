#!/usr/bin/env python3
"""
GPU Inference Server for Metamon Models.

This server runs the model on GPU and provides a clean API for inference,
completely separating PyKMN battle simulation from neural network inference.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import aiohttp
from aiohttp import web
import json
import base64
import pickle
from pathlib import Path

# Try to import msgpack for faster serialization
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Ensure cache directory is set
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = str(Path.home() / "metamon_cache")


@dataclass
class InferenceRequest:
    """Request for batch inference."""
    observations: Dict[str, np.ndarray]  # Observation dictionaries
    legal_masks: np.ndarray  # Legal action masks
    request_id: Optional[str] = None
    client_id: Optional[str] = None  # For per-client hidden state tracking
    reset_state: bool = False  # Whether to reset hidden state for this client


@dataclass
class InferenceResponse:
    """Response from inference server."""
    actions: np.ndarray  # Selected actions
    request_id: Optional[str] = None


class InferenceServer:
    """
    GPU Inference Server that maintains model in memory.

    Key design decisions:
    - Fixed internal batch size to avoid memory issues
    - Queues requests and batches them efficiently
    - Clean separation from battle simulation
    """

    def __init__(
        self,
        model_name: str,
        checkpoint: Optional[int] = None,
        device: str = "cuda",
        max_batch_size: int = 64,
        port: int = 8080,
        use_msgpack: bool = False,
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.device = device
        self.max_batch_size = max_batch_size
        self.port = port
        self.use_msgpack = use_msgpack and HAS_MSGPACK

        # Model will be loaded on start
        self.model = None
        self.agent = None
        self.hidden_states = {}  # Track hidden states per client

        # Request queue for batching
        self.request_queue = asyncio.Queue()
        self.response_futures = {}

    def load_model(self):
        """Load the model once at startup."""
        from metamon.rl.pretrained import get_pretrained_model

        print(f"Loading model {self.model_name} on {self.device}...")

        pretrained_cls = get_pretrained_model(self.model_name)

        # Initialize agent
        experiment = pretrained_cls.initialize_agent(
            checkpoint=self.checkpoint,
            log=False,
            action_temperature=1.0,
        )

        self.agent = experiment.policy
        self.agent.eval()

        # Get action dimension
        self.action_dim = pretrained_cls.action_space.gym_space.n

        print(f"Model loaded successfully! Action dim: {self.action_dim}")

        # Enable mixed precision for better performance
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    async def process_batch(self):
        """Process a batch of requests."""
        # Collect requests up to max_batch_size
        batch_requests = []
        batch_futures = []

        # Wait for at least one request
        request, future = await self.request_queue.get()
        batch_requests.append(request)
        batch_futures.append(future)

        # Collect more requests if available (with timeout)
        try:
            while len(batch_requests) < self.max_batch_size:
                request, future = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=0.001  # 1ms timeout for batching
                )
                batch_requests.append(request)
                batch_futures.append(future)
        except asyncio.TimeoutError:
            pass  # Process what we have

        # Run inference
        try:
            actions = self._run_inference(batch_requests)
            # Actions shape verified

            # Send responses
            # If single request with batched data, return all actions
            if len(batch_requests) == 1 and actions.shape[0] > 1:
                # Single batched request - return all actions
                # Returning batched response
                response = InferenceResponse(
                    actions=actions,
                    request_id=batch_requests[0].request_id
                )
                batch_futures[0].set_result(response)
            else:
                # Multiple single requests - split actions
                for i, (request, future) in enumerate(zip(batch_requests, batch_futures)):
                    response = InferenceResponse(
                        actions=actions[i:i+1],  # Single action for this request
                        request_id=request.request_id
                    )
                    future.set_result(response)

        except Exception as e:
            # Send error to all requests in batch
            for future in batch_futures:
                future.set_exception(e)

    def _run_inference(self, requests: List[InferenceRequest]) -> np.ndarray:
        """Run inference on a batch of requests."""
        num_requests = len(requests)

        # Stack observations and masks
        obs_list = [r.observations for r in requests]
        masks_list = [r.legal_masks for r in requests]

        # Input shapes verified

        # Combine into batched tensors
        obs_batch = self._stack_observations(obs_list)

        # Handle both single-env and multi-env requests
        # If masks_list[0] is 2D (already batched), don't stack again
        if len(masks_list) == 1 and masks_list[0].ndim == 2:
            # Single request with multiple environments (batched)
            legal_mask_batch = masks_list[0]
            batch_size = legal_mask_batch.shape[0]
            # Detected batched request
        else:
            # Multiple requests, each with single environment
            legal_mask_batch = np.stack(masks_list)
            batch_size = num_requests
            # Multiple single requests

        # Convert to torch tensors on GPU
        obs_torch = {}
        for k, v in obs_batch.items():
            if k == "text" or (hasattr(v, 'dtype') and 'str' in str(v.dtype)):
                continue
            elif isinstance(v, np.ndarray):
                obs_torch[k] = torch.from_numpy(v).to(self.device, non_blocking=True)

        # Add illegal actions mask
        illegal_mask = ~legal_mask_batch
        illegal_mask_trimmed = illegal_mask[:, :self.action_dim]
        obs_torch["illegal_actions"] = torch.from_numpy(illegal_mask_trimmed).to(
            self.device, non_blocking=True
        ).bool()

        # Add sequence dimension
        obs_torch_seq = {k: v.unsqueeze(1) for k, v in obs_torch.items()}

        # Create RL2 inputs (simplified - no state tracking for now)
        rl2_buffer = torch.zeros((batch_size, self.action_dim + 1),
                                dtype=torch.float32, device=self.device)
        rl2s_seq = rl2_buffer.unsqueeze(1)

        time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        time_idxs_seq = time_idxs.unsqueeze(1).unsqueeze(2)

        # Handle per-client hidden states
        # For now, use a single shared hidden state per batch
        # TODO: Implement proper per-client state tracking with state concatenation
        client_ids = [r.client_id or f"default_{i}" for i, r in enumerate(requests)]

        # Check if any request needs reset
        needs_reset = any(r.reset_state for r in requests)

        # Initialize or retrieve batch hidden state
        batch_key = f"batch_{batch_size}"
        if needs_reset or batch_key not in self.hidden_states:
            batch_hidden_state = self.agent.traj_encoder.init_hidden_state(
                batch_size, self.device
            )
            self.hidden_states[batch_key] = batch_hidden_state
        else:
            batch_hidden_state = self.hidden_states[batch_key]

        # Run inference
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                actions, new_hidden_state = self.agent.get_actions(
                    obs=obs_torch_seq,
                    rl2s=rl2s_seq,
                    time_idxs=time_idxs_seq,
                    hidden_state=batch_hidden_state,
                    sample=True,
                )

        # Update batch hidden state
        self.hidden_states[batch_key] = new_hidden_state

        # Convert back to numpy
        actions_np = actions.squeeze(-1).squeeze(1).cpu().numpy().astype(np.int32)

        return actions_np

    def _stack_observations(self, obs_list: List[Dict]) -> Dict[str, np.ndarray]:
        """Stack list of observation dicts into batched dict."""
        if not obs_list:
            return {}

        # If there's only one observation dict and it's already batched, return as-is
        if len(obs_list) == 1:
            first_obs = obs_list[0]
            # Check if observations are already batched (2D arrays)
            for key, value in first_obs.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    # Already batched, return as-is
                    return first_obs
            # Not batched, wrap in list for normal processing

        # Stack multiple single observations
        batched = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                batched[key] = np.stack(values)
            else:
                batched[key] = values

        return batched

    def _serialize(self, data: Any) -> bytes:
        """Serialize data using msgpack or pickle."""
        if self.use_msgpack:
            # Convert numpy arrays to lists for msgpack
            if isinstance(data, np.ndarray):
                data = data.tolist()
            elif isinstance(data, dict):
                data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
            return msgpack.packb(data, use_bin_type=True)
        else:
            return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data using msgpack or pickle."""
        if self.use_msgpack:
            obj = msgpack.unpackb(data, raw=False)
            # Convert lists back to numpy arrays where appropriate
            if isinstance(obj, dict):
                obj = {k: np.array(v) if isinstance(v, list) else v for k, v in obj.items()}
            elif isinstance(obj, list):
                obj = np.array(obj)
            return obj
        else:
            return pickle.loads(data)

    async def handle_inference(self, request):
        """Handle HTTP inference request."""
        try:
            data = await request.json()

            # Decode observations and masks
            obs_bytes = base64.b64decode(data['observations'])
            observations = self._deserialize(obs_bytes)

            mask_bytes = base64.b64decode(data['legal_masks'])
            legal_masks = self._deserialize(mask_bytes)

            request_id = data.get('request_id')
            client_id = data.get('client_id') or request.headers.get('X-Client-ID')
            reset_state = data.get('reset_state', False)

            # Create inference request
            inf_request = InferenceRequest(
                observations=observations,
                legal_masks=legal_masks,
                request_id=request_id,
                client_id=client_id,
                reset_state=reset_state
            )

            # Queue request and wait for response
            future = asyncio.Future()
            await self.request_queue.put((inf_request, future))
            response = await future

            # Encode response
            actions_bytes = self._serialize(response.actions)
            actions_b64 = base64.b64encode(actions_bytes).decode('utf-8')

            return web.json_response({
                'actions': actions_b64,
                'request_id': response.request_id
            })

        except Exception as e:
            import traceback
            import sys
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"ERROR in handle_inference: {error_msg}", file=sys.stderr, flush=True)
            print(error_trace, file=sys.stderr, flush=True)
            return web.json_response(
                {'error': error_msg},
                status=500
            )

    async def handle_health(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'model': self.model_name,
            'device': self.device,
            'max_batch_size': self.max_batch_size,
            'serialization': 'msgpack' if self.use_msgpack else 'pickle',
            'num_clients': len(self.hidden_states)
        })

    async def start(self, host: str = '0.0.0.0'):
        """Start the inference server."""
        # Load model
        self.load_model()

        # Start batch processor
        asyncio.create_task(self._batch_processor())

        # Setup web server
        app = web.Application()
        app.router.add_post('/inference', self.handle_inference)
        app.router.add_get('/health', self.handle_health)

        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, self.port)
        await site.start()

        print(f"Inference server running on http://{host}:{self.port}")
        print(f"Health check: http://{host}:{self.port}/health")
        print(f"Inference endpoint: http://{host}:{self.port}/inference")

        # Keep the server running forever
        try:
            await asyncio.Event().wait()  # Wait forever
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            await runner.cleanup()

    async def _batch_processor(self):
        """Continuously process batches of requests."""
        while True:
            try:
                await self.process_batch()
            except Exception as e:
                import traceback
                print(f"ERROR in batch processor: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)  # Brief pause on error to avoid tight loop


def main():
    """Run the inference server."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Inference Server for Metamon")
    parser.add_argument("--model", default="SyntheticRLV2", help="Model name")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=128, help="Max batch size (default: 128)")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--use_msgpack", action="store_true", help="Use msgpack for serialization (faster, requires msgpack-python)")

    args = parser.parse_args()

    if args.use_msgpack and not HAS_MSGPACK:
        print("WARNING: --use_msgpack specified but msgpack not installed. Falling back to pickle.")
        print("Install with: pip install msgpack-python")

    server = InferenceServer(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=args.device,
        max_batch_size=args.batch_size,
        port=args.port,
        use_msgpack=args.use_msgpack
    )

    # Run server
    asyncio.run(server.start(host=args.host))


if __name__ == "__main__":
    main()