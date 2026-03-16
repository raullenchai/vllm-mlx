# SPDX-License-Identifier: Apache-2.0
"""
Monkey-patch qwen3_next.Model to add MTP (Multi-Token Prediction) support.

Upstream mlx-lm (0.31.x) does not have return_hidden or mtp_forward.
This module patches the loaded model instance to add these methods when
MTP weights are present (added via scripts/add_mtp_weights.py or
scripts/gen_synthetic_mtp_weights.py).

The MTP predictor architecture (from Qwen3-Next):
1. Norm hidden: pre_fc_norm_hidden(hidden_states)
2. Norm embed:  pre_fc_norm_embedding(embed_tokens(token_ids))
3. Combine:     fc(concat([hidden, embed], dim=-1))
4. Decoder:     layers[0](x, mask, cache)  # single attention + MoE layer
5. Final norm:  norm(x)
6. Logits:      lm_head(x)  # shared with main model
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class MTPPredictor(nn.Module):
    """MTP predictor module matching Qwen3-Next architecture."""

    def __init__(self, args):
        super().__init__()
        from mlx_lm.models.qwen3_next import (
            Qwen3NextAttention,
            Qwen3NextSparseMoeBlock,
        )

        hidden_size = args.hidden_size

        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layers = [_MTPDecoderLayer(args)]
        self.norm = nn.RMSNorm(hidden_size, eps=args.rms_norm_eps)


class _MTPDecoderLayer(nn.Module):
    """Single MTP decoder layer: attention + MoE MLP."""

    def __init__(self, args):
        super().__init__()
        from mlx_lm.models.qwen3_next import (
            Qwen3NextAttention,
            Qwen3NextSparseMoeBlock,
        )

        self.self_attn = Qwen3NextAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3NextSparseMoeBlock(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


def _find_model_dir(model_name: str) -> Optional[Path]:
    """Resolve model name to a directory path."""
    p = Path(model_name)
    if p.is_dir() and (p / "config.json").exists():
        return p
    # Try HF cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    cache_dir = hf_cache / slug
    if cache_dir.exists():
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            snaps = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
            if snaps:
                return snaps[-1]
    return None


def _has_mtp_weights(model_dir: Path) -> bool:
    """Check if model directory has MTP weight files."""
    mtp_file = model_dir / "model-mtp.safetensors"
    if mtp_file.exists():
        return True
    # Check index for mtp.* entries
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                index = json.load(f)
            return any(k.startswith("mtp.") for k in index.get("weight_map", {}))
        except Exception:
            pass
    return False


def _load_mtp_weights(model_dir: Path) -> dict:
    """Load MTP weights from model directory."""
    mtp_file = model_dir / "model-mtp.safetensors"
    if mtp_file.exists():
        weights = mx.load(str(mtp_file))
        return {k: v for k, v in weights.items() if k.startswith("mtp.")}
    return {}


def patch_model_for_mtp(model, model_name: str = None) -> bool:
    """Monkey-patch a loaded qwen3_next.Model to support MTP.

    Adds return_hidden, mtp_forward, make_mtp_cache methods.
    Loads MTP weights from model-mtp.safetensors.

    Args:
        model: Loaded model instance
        model_name: Model path/name for finding weight files

    Returns True if patch succeeded, False otherwise.
    """
    from mlx_lm.models.qwen3_next import Model as Qwen3NextModel

    if not isinstance(model, Qwen3NextModel):
        logger.warning("[MTP-patch] Model is not qwen3_next.Model, cannot patch")
        return False

    # Find model directory and check for MTP weights
    model_dir = None
    if model_name:
        model_dir = _find_model_dir(model_name)

    if model_dir is None or not _has_mtp_weights(model_dir):
        logger.info("[MTP-patch] No MTP weight files found")
        return False

    # Create MTP module
    args = model.args
    try:
        mtp = MTPPredictor(args)
    except Exception as e:
        logger.warning(f"[MTP-patch] Failed to create MTP module: {e}")
        return False

    # Load MTP weights from disk
    mtp_weights = _load_mtp_weights(model_dir)
    if not mtp_weights:
        logger.warning("[MTP-patch] No MTP weights found in safetensors")
        return False

    # Strip "mtp." prefix — MTPPredictor's parameters don't have it
    stripped = {k[4:]: v for k, v in mtp_weights.items() if k.startswith("mtp.")}
    if not stripped:
        stripped = mtp_weights  # fallback if keys don't have prefix

    # Quantize the MTP module to match model's quantization config
    # This converts nn.Linear → nn.QuantizedLinear so weights load properly
    try:
        quant_config = getattr(args, "quantization", None)
        if quant_config is None:
            # Try reading from config.json directly
            config_path = model_dir / "config.json"
            if config_path.exists():
                import json as _json

                with open(config_path) as f:
                    cfg = _json.load(f)
                quant_config = cfg.get("quantization", {})

        if quant_config:
            bits = quant_config.get("bits", 4)
            group_size = quant_config.get("group_size", 64)

            def _quant_predicate(path, module):
                # shared_expert_gate is tiny (1, hidden) — keep FP
                if "shared_expert_gate" in path:
                    return False
                if isinstance(module, nn.Linear):
                    # MoE gate uses 8-bit
                    if "gate" in path and "gate_proj" not in path:
                        return {"group_size": 64, "bits": 8}
                    return {"group_size": group_size, "bits": bits}
                # SwitchLinear (expert stacked weights) — also quantize
                if hasattr(module, "to_quantized") and not isinstance(
                    module, nn.Linear
                ):
                    return {"group_size": group_size, "bits": bits}
                return False

            nn.quantize(mtp, class_predicate=_quant_predicate)
            logger.info(
                f"[MTP-patch] Quantized MTP module: {bits}-bit, group_size={group_size}"
            )
    except Exception as e:
        logger.warning(f"[MTP-patch] Failed to quantize MTP module: {e}")
        return False

    # Load weights into the MTP module
    try:
        mtp.load_weights(list(stripped.items()))
        mx.eval(mtp.parameters())
        logger.info(f"[MTP-patch] Loaded {len(stripped)} MTP weights")
    except Exception as e:
        logger.warning(f"[MTP-patch] Failed to load MTP weights: {e}")
        return False

    # Attach MTP module to model
    model.mtp = mtp

    # Patch __call__ to support return_hidden
    def patched_call(self, inputs, cache=None, return_hidden=False):
        hidden = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = self.lm_head(hidden)
        if return_hidden:
            return out, hidden
        return out

    type(model).__call__ = patched_call

    # Add mtp_forward method
    def mtp_forward(self, hidden_states, token_ids, mtp_cache=None):
        """Predict next-next token from hidden states + current token."""
        from mlx_lm.models.base import create_attention_mask

        emb = self.model.embed_tokens(token_ids)
        h = self.mtp.pre_fc_norm_hidden(hidden_states)
        e = self.mtp.pre_fc_norm_embedding(emb)
        x = self.mtp.fc(mx.concatenate([h, e], axis=-1))

        mask = None
        if mtp_cache is not None:
            mask = create_attention_mask(x, mtp_cache)

        for layer in self.mtp.layers:
            x = layer(x, mask=mask, cache=mtp_cache)

        x = self.mtp.norm(x)

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    type(model).mtp_forward = mtp_forward

    def make_mtp_cache(self):
        from mlx_lm.models.cache import KVCache
        return KVCache()

    type(model).make_mtp_cache = make_mtp_cache

    logger.info("[MTP-patch] Successfully patched model with MTP support")
    return True
