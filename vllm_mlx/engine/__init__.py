# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for vllm-mlx inference.

Provides two engine implementations:
- SimpleEngine: Direct model calls for maximum single-user throughput
- BatchedEngine: Continuous batching for multiple concurrent users

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
from .base import BaseEngine, GenerationOutput
from .batched import BatchedEngine
from .hybrid import HybridEngine
from .simple import SimpleEngine

__all__ = [
    "BaseEngine",
    "GenerationOutput",
    "SimpleEngine",
    "BatchedEngine",
    "HybridEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
