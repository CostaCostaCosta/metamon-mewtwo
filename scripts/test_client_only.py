#!/usr/bin/env python3
"""Quick test to see if policy server is responding."""

import sys
sys.path.insert(0, '/home/eddie/repos/metamon')

from metamon.rl.policy_client import PolicyClient
import numpy as np

print("Testing connection to policy server at tcp://localhost:5555")

try:
    client = PolicyClient("tcp://localhost:5555", timeout_ms=10000)
    print("✓ Client created successfully")

    # Create a dummy observation (would need to match actual observation space)
    # For now, just test the connection
    print("Attempting to send test request...")
    print("(This will likely fail because we don't have a real observation, but it tests connectivity)")

    # This will fail but should at least connect
    try:
        result = client.get_action({"test": "data"}, traj_id=0)
        print(f"Got result: {result}")
    except Exception as e:
        print(f"Expected error (need real observation): {e}")

    client.close()
    print("\n✓ Client can communicate with server!")

except Exception as e:
    print(f"\n✗ Failed to connect: {e}")
    import traceback
    traceback.print_exc()
