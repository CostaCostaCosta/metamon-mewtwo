"""
Test that checkpoint loading works correctly on CPU with the fixed LocalPretrainedModel.
"""

import sys
sys.path.insert(0, '/home/eddie/repos/metamon')

from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.gen1_binary_models import *

print("[Test] Loading DampedBinarySuperV1_Epoch4 model...")
pretrained_model = get_pretrained_model("DampedBinarySuperV1_Epoch4")

print("[Test] Initializing agent on CPU with checkpoint=None (should use default)...")
agent = pretrained_model.initialize_agent(checkpoint=None, log=False, device='cpu')

print(f"[Test] Agent policy type: {type(agent.policy)}")
print(f"[Test] Agent policy exists: {agent.policy is not None}")
print(f"[Test] Agent has traj_encoder: {hasattr(agent.policy, 'traj_encoder')}")

if agent.policy is not None and hasattr(agent.policy, 'traj_encoder'):
    print(f"[Test] traj_encoder type: {type(agent.policy.traj_encoder)}")
    print(f"[Test] Testing init_hidden_state...")
    try:
        hidden_state = agent.policy.traj_encoder.init_hidden_state(1, 'cpu')
        print(f"[Test] init_hidden_state returned: {type(hidden_state)}, len={len(hidden_state) if hasattr(hidden_state, '__len__') else 'N/A'}")
        print("[Test] SUCCESS! Checkpoint loaded correctly on CPU.")
    except Exception as e:
        print(f"[Test] ERROR initializing hidden state: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[Test] FAILED - policy not properly initialized")
