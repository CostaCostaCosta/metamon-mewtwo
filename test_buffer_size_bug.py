"""
Test to reproduce the buffer size corruption bug.

Scenario:
1. First batch: 64 envs (allocates buffers of size 64)
2. Second batch: 32 envs (should reallocate or handle properly)
3. Third batch: 64 envs again (might write out of bounds if buffers were shrunk)
"""
import os
os.environ["METAMON_CACHE_DIR"] = "/home/eddie/metamon_cache"

import torch
import numpy as np

# Simulate the LocalPolicyRunner buffer management
class MockPolicyRunner:
    def __init__(self, action_dim=13, device="cpu"):
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Buffers (initially None)
        self.rl2_buffer = None
        self.prev_action_onehot_buffer = None
        self.time_idxs = None
        self.prev_actions = None
        self.prev_rewards = None
    
    def reset(self, batch_size):
        """Reset for new batch - allocates buffers."""
        print(f"  reset(batch_size={batch_size})")
        self.time_idxs = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        self.rl2_buffer = torch.zeros((batch_size, self.action_dim + 1), 
                                      dtype=torch.float32, device=self.device)
        self.prev_action_onehot_buffer = torch.zeros((batch_size, self.action_dim),
                                                      dtype=torch.float32, device=self.device)
        self.prev_actions = None
        self.prev_rewards = None
        print(f"    Allocated buffers: rl2_buffer.shape={self.rl2_buffer.shape}, "
              f"time_idxs.shape={self.time_idxs.shape}")
    
    def infer(self, batch_size):
        """Simulate inference with current batch size."""
        print(f"  infer(batch_size={batch_size})")
        
        # Check if batch size exceeds buffer capacity
        if self.time_idxs is not None and batch_size > self.time_idxs.shape[0]:
            print(f"    ⚠️  BUFFER OVERFLOW DETECTED!")
            print(f"       Trying to access [:{ batch_size}] but buffer size is {self.time_idxs.shape[0]}")
            print(f"       This would write out of bounds!")
            return False
        
        # Simulate buffer operations (from lines 214-232)
        if self.prev_actions is None:
            self.rl2_buffer[:batch_size].zero_()
        else:
            prev_batch_size = min(len(self.prev_actions), batch_size)
            self.prev_action_onehot_buffer[:prev_batch_size].zero_()
            self.rl2_buffer[:prev_batch_size, :self.action_dim] = self.prev_action_onehot_buffer[:prev_batch_size]
            if prev_batch_size < batch_size:
                self.rl2_buffer[prev_batch_size:batch_size].zero_()
        
        # Simulate creating prev_actions/rewards
        self.prev_actions = torch.randint(0, self.action_dim, (batch_size,), device=self.device)
        self.prev_rewards = torch.zeros((batch_size,), device=self.device)
        
        # THE BUG: This line increments ALL elements, not just [:batch_size]!
        print(f"    Incrementing time_idxs (shape={self.time_idxs.shape})")
        self.time_idxs += 1  # BUG: increments all elements!
        print(f"    time_idxs after increment: {self.time_idxs[:min(10, len(self.time_idxs))]}")
        
        return True

# Test scenario that causes corruption
print("=" * 70)
print("TESTING BUFFER SIZE MISMATCH SCENARIO")
print("=" * 70)

runner = MockPolicyRunner()

print("\n1. First batch: 64 environments")
runner.reset(64)
for i in range(3):
    print(f"  Step {i+1}:")
    runner.infer(64)

print("\n2. Second batch: 32 environments (smaller)")
runner.reset(32)
for i in range(3):
    print(f"  Step {i+1}:")
    runner.infer(32)

print("\n3. Third batch: 64 environments (back to original size)")
runner.reset(64)
for i in range(3):
    print(f"  Step {i+1}:")
    if not runner.infer(64):
        print("  ❌ CORRUPTION DETECTED!")
        break

print("\n4. Fourth batch: 128 environments (larger than ever before)")
runner.reset(64)  # Reset with 64
print("  Trying to infer with 128 (larger than buffer)...")
if not runner.infer(128):
    print("  ❌ WOULD WRITE OUT OF BOUNDS!")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
