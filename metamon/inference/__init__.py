"""
Inference Server Architecture for Metamon.

This module provides a clean separation between PyKMN battle simulation
and GPU model inference, eliminating memory corruption issues.
"""

from .client import InferenceClient, RemotePolicyRunner
from .server import InferenceServer

__all__ = [
    'InferenceClient',
    'RemotePolicyRunner',
    'InferenceServer',
]