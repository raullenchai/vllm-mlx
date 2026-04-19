# SPDX-License-Identifier: Apache-2.0
"""Prefix cache persistence — load/save KV cache to disk."""

from __future__ import annotations

import logging
import os

from ..config import get_config

logger = logging.getLogger(__name__)


def load_prefix_cache_from_disk() -> None:
    """Load prefix cache from disk during startup."""
    cfg = get_config()
    if cfg.engine is None:
        return
    try:
        d = get_cache_dir()
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = cfg.engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


def save_prefix_cache_to_disk() -> None:
    """Save prefix cache to disk during shutdown."""
    cfg = get_config()
    if cfg.engine is None:
        return
    try:
        d = get_cache_dir()
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = cfg.engine.save_cache_to_disk(d)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


def get_cache_dir() -> str:
    """Get cache persistence directory based on actual model path."""
    cfg = get_config()
    model_name = cfg.model_path or cfg.model_name or "default"
    safe_name = str(model_name).replace("/", "--").replace("\\", "--")
    return os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", safe_name
    )
