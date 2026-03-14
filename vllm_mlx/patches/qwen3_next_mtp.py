# SPDX-License-Identifier: Apache-2.0
"""
Runtime MTP (Multi-Token Prediction) validation for Qwen3-Next models.

Qwen3-Next models may include a built-in MTP head that predicts token n+2
from hidden states + token n+1.  When MTP weights have been added to the
quantized MLX model (via scripts/add_mtp_weights.py), mlx_lm.load()
automatically instantiates the MTP module (model.mtp).

This module provides a lightweight validation function that checks whether
a loaded model has a working MTP head and logs diagnostic information.
The actual MTP logic lives in:
  - mlx_lm/models/qwen3_next.py  (Model.__call__ with return_hidden,
    Model.mtp_forward, Model.make_mtp_cache)
  - vllm_mlx/scheduler.py  (_install_mtp, _mtp_step, _mtp_next)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_mtp_support(model: Any) -> bool:
    """Validate that a loaded model has working MTP support.

    Checks:
    1. model.mtp exists and is not None (MTP module instantiated)
    2. model.mtp has layers with loaded weights
    3. model has return_hidden support in __call__
    4. model has mtp_forward method
    5. model has make_mtp_cache method

    Args:
        model: A model loaded via mlx_lm.load()

    Returns:
        True if MTP is fully functional, False otherwise.
    """
    # Check 1: MTP module exists
    mtp = getattr(model, "mtp", None)
    if mtp is None:
        num_mtp = 0
        args = getattr(model, "args", None)
        if args is not None:
            num_mtp = getattr(args, "num_nextn_predict_layers", 0)
        if num_mtp > 0:
            logger.warning(
                "[MTP] Model config has num_nextn_predict_layers=%d but "
                "model.mtp is None. MTP weights may not be in the model files. "
                "Run scripts/add_mtp_weights.py to add them.",
                num_mtp,
            )
        else:
            logger.info(
                "[MTP] Model does not have MTP config (num_nextn_predict_layers=0)."
            )
        return False

    # Check 2: MTP layers have weights
    mtp_layers = getattr(mtp, "layers", [])
    if not mtp_layers:
        logger.warning("[MTP] model.mtp exists but has no layers.")
        return False

    # Check 3: return_hidden support
    import inspect

    call_sig = inspect.signature(type(model).__call__)
    if "return_hidden" not in call_sig.parameters:
        logger.warning(
            "[MTP] Model.__call__ does not accept return_hidden parameter. "
            "The mlx_lm model implementation may be outdated."
        )
        return False

    # Check 4: mtp_forward method
    if not hasattr(model, "mtp_forward") or not callable(model.mtp_forward):
        logger.warning("[MTP] Model does not have mtp_forward() method.")
        return False

    # Check 5: make_mtp_cache method
    if not hasattr(model, "make_mtp_cache") or not callable(model.make_mtp_cache):
        logger.warning("[MTP] Model does not have make_mtp_cache() method.")
        return False

    # All checks passed
    args = getattr(model, "args", None)
    num_layers = getattr(args, "num_nextn_predict_layers", 0) if args else 0
    logger.info(
        "[MTP] Model has working MTP support: "
        "%d MTP layer(s), %d predictor decoder layer(s)",
        num_layers,
        len(mtp_layers),
    )
    return True
