"""
Tree verification forward pass through the hybrid Qwen 3.5 model.

Processes all tree nodes through the model:
- Attention layers: per-token RoPE (batch-reshape trick) + tree attention mask
- Linear layers: parent-state forking for exact tree-aware recurrence, or the
  legacy DFS sequential path when DDTREE_TREE_AWARE_LINEAR=0.

The tree-aware path keeps logits exact for every branch and commits arbitrary
accepted paths without re-forwarding. The legacy DFS path keeps only the DFS
prefix exact and re-forwards divergent suffixes before committing them.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g, gated_delta_update

from .compile import CompiledTree
from .kernels import tree_conv1d_kernel, tree_gated_delta_kernel

# Use dflash-mlx's model navigation helpers (handles VL models, nested wrappers)
from dflash_mlx.runtime import (
    _target_text_model,
    _lm_head_logits,
    _split_sdpa_output,
    _HYBRID_SDPA_EXACT_KV_THRESHOLD,
)


_TREE_KERNEL_ENABLED = os.environ.get("DDTREE_TREE_KERNEL", "1").lower() not in (
    "",
    "0",
    "false",
)
_TREE_CONV_KERNEL_ENABLED = os.environ.get("DDTREE_TREE_CONV_KERNEL", "1").lower() not in (
    "",
    "0",
    "false",
)
_EXACT_TREE_ATTENTION = os.environ.get(
    "DDTREE_EXACT_TREE_ATTENTION", "auto"
).lower()
_EXACT_TREE_ATTENTION_MIN_PREFIX = int(
    os.environ.get("DDTREE_EXACT_TREE_ATTENTION_MIN_PREFIX", "8192")
)


def _profile_detail_enabled(profile_timings: Optional[dict[str, Any]]) -> bool:
    return bool(profile_timings is not None and profile_timings.get("_detail"))


def _profile_eval(
    profile_timings: Optional[dict[str, Any]],
    key: str,
    *arrays: mx.array,
) -> None:
    if not _profile_detail_enabled(profile_timings):
        return
    start_ns = time.perf_counter_ns()
    mx.eval(*arrays)
    profile_timings[key] = profile_timings.get(key, 0) + (
        time.perf_counter_ns() - start_ns
    )


def _exact_tree_attention_enabled(prefix_len: int) -> bool:
    if int(prefix_len) <= 0:
        return False
    if _EXACT_TREE_ATTENTION in ("", "0", "false", "off", "none"):
        return False
    if _EXACT_TREE_ATTENTION in ("1", "true", "on"):
        return True
    return int(prefix_len) >= _EXACT_TREE_ATTENTION_MIN_PREFIX


def _rope_with_positions(
    x: mx.array,
    position_ids: mx.array,
    rope_fn: Any,
) -> mx.array:
    """Apply RoPE with per-token positions via batch-reshape trick.

    Reshapes [1, H, T, D] → [T, H, 1, D], applies per-batch offsets,
    then reshapes back. Verified to produce zero diff vs individual application.

    Args:
        x: (1, n_heads, T, head_dim) query or key tensor.
        position_ids: (T,) int32 absolute positions per token.
        rope_fn: The model's rope module (nn.RoPE or variant).
    """
    _, H, T, D = x.shape
    # [1, H, T, D] → [T, H, 1, D]
    x_reshaped = x.transpose(0, 2, 1, 3).reshape(T, H, 1, D)
    # Apply rope with per-batch offsets
    x_roped = rope_fn(x_reshaped, offset=position_ids)
    # [T, H, 1, D] → [1, H, T, D]
    return x_roped.reshape(1, T, H, D).transpose(0, 2, 1, 3)


def _repeat_kv_heads_for_queries(
    keys: mx.array,
    values: mx.array,
    query_heads: int,
) -> tuple[mx.array, mx.array]:
    kv_heads = int(keys.shape[1])
    if kv_heads == query_heads:
        return keys, values
    if query_heads % kv_heads != 0:
        raise ValueError(
            f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})"
        )
    repeat_factor = query_heads // kv_heads
    return mx.repeat(keys, repeat_factor, axis=1), mx.repeat(values, repeat_factor, axis=1)


def _split_prefix_tree_attention_exact(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    tree_mask: mx.array,
    cached_prefix_len: int,
) -> mx.array:
    """Exact attention over prefix + tree without allocating a prefix-width mask."""
    query_heads = int(queries.shape[1])
    keys, values = _repeat_kv_heads_for_queries(keys, values, query_heads)

    prefix_len = int(cached_prefix_len)
    prefix_keys = keys[:, :, :prefix_len, :].astype(mx.float32)
    prefix_values = values[:, :, :prefix_len, :].astype(mx.float32)
    tree_keys = keys[:, :, prefix_len:, :].astype(mx.float32)
    tree_values = values[:, :, prefix_len:, :].astype(mx.float32)
    q = queries.astype(mx.float32)

    prefix_scores = mx.matmul(q, prefix_keys.transpose(0, 1, 3, 2)) * scale
    tree_scores = mx.matmul(q, tree_keys.transpose(0, 1, 3, 2)) * scale
    tree_scores = tree_scores + tree_mask.astype(mx.float32)

    prefix_max = mx.max(prefix_scores, axis=-1, keepdims=True)
    tree_max = mx.max(tree_scores, axis=-1, keepdims=True)
    joint_max = mx.maximum(prefix_max, tree_max)

    prefix_weights = mx.exp(prefix_scores - joint_max)
    tree_weights = mx.exp(tree_scores - joint_max)
    denom = (
        mx.sum(prefix_weights, axis=-1, keepdims=True)
        + mx.sum(tree_weights, axis=-1, keepdims=True)
    )
    output = (
        mx.matmul(prefix_weights, prefix_values)
        + mx.matmul(tree_weights, tree_values)
    ) / denom
    return output.astype(queries.dtype)


def _attention_forward_with_tree(
    attn: Any,
    x: mx.array,
    position_ids: mx.array,
    mask: mx.array,
    cache: Any,
    *,
    cached_prefix_len: Optional[int] = None,
    tree_mask_only: bool = False,
    profile_timings: Optional[dict[str, Any]] = None,
) -> mx.array:
    """Replicate Qwen3NextAttention.__call__ with per-token RoPE and tree mask.

    Based on mlx_lm/models/qwen3_next.py:120-158.
    """
    B, L, D = x.shape
    if cached_prefix_len is None:
        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0

    # Q projection + split into queries and gate
    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)

    # K, V projections
    keys, values = attn.k_proj(x), attn.v_proj(x)
    _profile_eval(
        profile_timings,
        "attention_qkv_proj_ns",
        q_proj_output,
        keys,
        values,
    )

    # Reshape and normalize
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)  # (B, H, L, head_dim)
    keys = attn.k_norm(
        keys.reshape(B, L, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    _profile_eval(profile_timings, "attention_qk_norm_ns", queries, keys)

    # Per-token RoPE (batch-reshape trick instead of scalar offset)
    queries = _rope_with_positions(queries, position_ids, attn.rope)
    keys = _rope_with_positions(keys, position_ids, attn.rope)
    _profile_eval(profile_timings, "attention_rope_ns", queries, keys)

    # Update KV cache (appends tree nodes)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    _profile_eval(profile_timings, "attention_cache_update_ns", keys, values)

    if tree_mask_only:
        if int(mask.shape[-1]) != L:
            raise ValueError(
                f"tree-only attention mask width {mask.shape[-1]} does not match tree length {L}"
            )
        output = _split_prefix_tree_attention_exact(
            queries=queries,
            keys=keys,
            values=values,
            scale=attn.scale,
            tree_mask=mask,
            cached_prefix_len=int(cached_prefix_len),
        )
        _profile_eval(profile_timings, "attention_exact_prefix_tree_ns", output)
    else:
        # SDPA with tree attention mask — mask must cover full KV length
        kv_len = keys.shape[2]
        mask_kv_len = mask.shape[-1]
        if mask_kv_len != kv_len:
            raise ValueError(
                f"tree attention mask width {mask_kv_len} does not match KV length {kv_len}"
            )

        should_split = (
            cache is not None
            and cached_prefix_len >= _HYBRID_SDPA_EXACT_KV_THRESHOLD
            and isinstance(mask, mx.array)
        )
        if should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=attn.scale,
                mask=mask,
                cache=cache,
                chunk_size=1,
                cached_prefix_len=cached_prefix_len,
            )
            _profile_eval(profile_timings, "attention_split_sdpa_ns", output)
        else:
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=attn.scale, mask=mask
            )
            _profile_eval(profile_timings, "attention_sdpa_ns", output)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # Gate and output projection
    projected = attn.o_proj(output * mx.sigmoid(gate))
    _profile_eval(profile_timings, "attention_out_proj_ns", projected)
    return projected


def _tree_depth_groups(depths: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for index, depth in enumerate(depths):
        while len(groups) <= depth:
            groups.append([])
        groups[depth].append(index)
    return groups


def _linear_forward_tree_aware(
    linear_attn: Any,
    inputs: mx.array,
    cache: Any,
    *,
    parents: list[int],
    depth_groups: list[list[int]],
    profile_timings: Optional[dict[str, Any]] = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run one GatedDeltaNet layer with parent-state tree forking.

    Returns:
        (output, node_states, node_conv_states), where node_states and
        node_conv_states are indexed by tree index and contain the exact cache
        state after processing that node.
    """
    B, T, _ = inputs.shape
    if B != 1:
        raise ValueError("tree-aware GatedDelta verification only supports batch size 1")

    if linear_attn.sharding_group is not None:
        from mlx.nn.layers.distributed import sum_gradients

        inputs = sum_gradients(linear_attn.sharding_group)(inputs)

    qkv = linear_attn.in_proj_qkv(inputs)
    z = linear_attn.in_proj_z(inputs).reshape(
        B, T, linear_attn.num_v_heads, linear_attn.head_v_dim
    )
    b = linear_attn.in_proj_b(inputs)
    a = linear_attn.in_proj_a(inputs)
    _profile_eval(profile_timings, "linear_proj_ns", qkv, z, b, a)

    keep = int(linear_attn.conv_kernel_size) - 1
    if cache is not None and cache[0] is not None:
        base_conv_state = cache[0]
    else:
        base_conv_state = mx.zeros(
            (B, keep, linear_attn.conv_dim),
            dtype=inputs.dtype,
        )

    if cache is not None and cache[1] is not None:
        base_state = cache[1]
    else:
        base_state = mx.zeros(
            (
                B,
                linear_attn.num_v_heads,
                linear_attn.head_v_dim,
                linear_attn.head_k_dim,
            ),
            dtype=mx.float32,
        )

    raw_outputs: list[mx.array | None] = [None] * T
    node_states: list[mx.array | None] = [None] * T
    node_conv_states: list[mx.array | None] = [None] * T
    conv_weight = linear_attn.conv1d.weight[:, :, 0].T

    if _TREE_KERNEL_ENABLED and not linear_attn.training:
        kernel_result = _linear_forward_tree_aware_kernel(
            linear_attn,
            qkv=qkv,
            z=z,
            a=a,
            b=b,
            base_state=base_state,
            base_conv_state=base_conv_state,
            parents=parents,
            depth_groups=depth_groups,
            conv_weight=conv_weight,
            keep=keep,
            input_dtype=inputs.dtype,
            profile_timings=profile_timings,
        )
        if kernel_result is not None:
            return kernel_result

    for indices in depth_groups:
        if not indices:
            continue
        index_array = mx.array(indices, dtype=mx.int32)
        group_size = len(indices)

        parent_states = []
        parent_conv_states = []
        for tree_index in indices:
            parent_index = int(parents[tree_index])
            if parent_index < 0:
                parent_states.append(base_state)
                parent_conv_states.append(base_conv_state)
            else:
                parent_state = node_states[parent_index]
                parent_conv_state = node_conv_states[parent_index]
                if parent_state is None or parent_conv_state is None:
                    raise ValueError("parent state missing during tree-aware verify")
                parent_states.append(parent_state)
                parent_conv_states.append(parent_conv_state)

        state_in = mx.concatenate(parent_states, axis=0)
        conv_state = mx.concatenate(parent_conv_states, axis=0)
        qkv_step = mx.take(qkv, index_array, axis=1).reshape(
            group_size, 1, linear_attn.conv_dim
        )
        conv_input = mx.concatenate([conv_state, qkv_step], axis=1)
        new_conv_state = (
            mx.contiguous(conv_input[:, -keep:, :])
            if keep > 0
            else mx.zeros((group_size, 0, linear_attn.conv_dim), dtype=inputs.dtype)
        )
        conv_out = nn.silu(
            (conv_input * conv_weight[None, :, :]).sum(axis=1)[:, None, :]
        )
        _profile_eval(
            profile_timings,
            "linear_conv_fallback_ns",
            conv_out,
            new_conv_state,
        )

        q, k, v = [
            tensor.reshape(group_size, 1, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                [
                    linear_attn.num_k_heads,
                    linear_attn.num_k_heads,
                    linear_attn.num_v_heads,
                ],
                [
                    linear_attn.head_k_dim,
                    linear_attn.head_k_dim,
                    linear_attn.head_v_dim,
                ],
                strict=True,
            )
        ]

        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        _profile_eval(profile_timings, "linear_qk_norm_ns", q, k)
        a_step = mx.take(a, index_array, axis=1).reshape(
            group_size, 1, linear_attn.num_v_heads
        )
        b_step = mx.take(b, index_array, axis=1).reshape(
            group_size, 1, linear_attn.num_v_heads
        )

        out, state_out = gated_delta_update(
            q,
            k,
            v,
            a_step,
            b_step,
            linear_attn.A_log,
            linear_attn.dt_bias,
            state_in,
            None,
            use_kernel=not linear_attn.training,
        )
        _profile_eval(profile_timings, "linear_recurrence_fallback_ns", out, state_out)
        for group_pos, tree_index in enumerate(indices):
            raw_outputs[tree_index] = out[group_pos : group_pos + 1]
            node_states[tree_index] = state_out[group_pos : group_pos + 1]
            node_conv_states[tree_index] = new_conv_state[group_pos : group_pos + 1]

    if any(output is None for output in raw_outputs):
        raise ValueError("tree-aware verify did not produce every node output")

    out = mx.concatenate(raw_outputs, axis=1)  # type: ignore[arg-type]
    out = linear_attn.norm(out, z)
    out = linear_attn.out_proj(out.reshape(B, T, -1))
    _profile_eval(profile_timings, "linear_out_proj_ns", out)

    if linear_attn.sharding_group is not None:
        out = mx.distributed.all_sum(out, group=linear_attn.sharding_group)

    return (
        out,
        mx.concatenate(node_states, axis=0),  # type: ignore[arg-type]
        mx.concatenate(node_conv_states, axis=0),  # type: ignore[arg-type]
    )


def _linear_forward_tree_aware_kernel(
    linear_attn: Any,
    *,
    qkv: mx.array,
    z: mx.array,
    a: mx.array,
    b: mx.array,
    base_state: mx.array,
    base_conv_state: mx.array,
    parents: list[int],
    depth_groups: list[list[int]],
    conv_weight: mx.array,
    keep: int,
    input_dtype: mx.Dtype,
    profile_timings: Optional[dict[str, Any]] = None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Tree-aware GatedDelta path using one Metal recurrence launch.

    The depth loop is still used for the causal depthwise convolution because
    each node's conv state depends on its parent. The heavier recurrent state
    update runs once over the parent-indexed tree.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if linear_attn.head_k_dim % 32 != 0:
        return None
    if any(parent >= idx for idx, parent in enumerate(parents) if parent >= 0):
        return None

    B, T, _ = qkv.shape
    conv_result = None
    if _TREE_CONV_KERNEL_ENABLED:
        conv_result = tree_conv1d_kernel(
            qkv,
            base_conv_state,
            conv_weight,
            mx.array(parents, dtype=mx.int32),
        )

    if conv_result is not None:
        conv_out, node_conv_states_full = conv_result
        _profile_eval(
            profile_timings,
            "linear_conv_kernel_ns",
            conv_out,
            node_conv_states_full,
        )
        q_all, k_all, v_all = [
            tensor.reshape(B, T, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                [
                    linear_attn.num_k_heads,
                    linear_attn.num_k_heads,
                    linear_attn.num_v_heads,
                ],
                [
                    linear_attn.head_k_dim,
                    linear_attn.head_k_dim,
                    linear_attn.head_v_dim,
                ],
                strict=True,
            )
        ]
        node_conv_states_out = node_conv_states_full[0]
    else:
        q_parts: list[mx.array | None] = [None] * T
        k_parts: list[mx.array | None] = [None] * T
        v_parts: list[mx.array | None] = [None] * T
        node_conv_states: list[mx.array | None] = [None] * T

        for indices in depth_groups:
            if not indices:
                continue

            index_array = mx.array(indices, dtype=mx.int32)
            group_size = len(indices)

            parent_conv_states = []
            for tree_index in indices:
                parent_index = int(parents[tree_index])
                if parent_index < 0:
                    parent_conv_states.append(base_conv_state)
                else:
                    parent_conv_state = node_conv_states[parent_index]
                    if parent_conv_state is None:
                        return None
                    parent_conv_states.append(parent_conv_state)

            conv_state = mx.concatenate(parent_conv_states, axis=0)
            qkv_step = mx.take(qkv, index_array, axis=1).reshape(
                group_size, 1, linear_attn.conv_dim
            )
            conv_input = mx.concatenate([conv_state, qkv_step], axis=1)
            new_conv_state = (
                mx.contiguous(conv_input[:, -keep:, :])
                if keep > 0
                else mx.zeros((group_size, 0, linear_attn.conv_dim), dtype=input_dtype)
            )
            conv_out = nn.silu(
                (conv_input * conv_weight[None, :, :]).sum(axis=1)[:, None, :]
            )
            _profile_eval(
                profile_timings,
                "linear_conv_fallback_ns",
                conv_out,
                new_conv_state,
            )

            q, k, v = [
                tensor.reshape(group_size, 1, heads, dim)
                for tensor, heads, dim in zip(
                    mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                    [
                        linear_attn.num_k_heads,
                        linear_attn.num_k_heads,
                        linear_attn.num_v_heads,
                    ],
                    [
                        linear_attn.head_k_dim,
                        linear_attn.head_k_dim,
                        linear_attn.head_v_dim,
                    ],
                    strict=True,
                )
            ]

            for group_pos, tree_index in enumerate(indices):
                q_parts[tree_index] = q[group_pos : group_pos + 1]
                k_parts[tree_index] = k[group_pos : group_pos + 1]
                v_parts[tree_index] = v[group_pos : group_pos + 1]
                node_conv_states[tree_index] = new_conv_state[group_pos : group_pos + 1]

        if (
            any(part is None for part in q_parts)
            or any(part is None for part in k_parts)
            or any(part is None for part in v_parts)
            or any(state is None for state in node_conv_states)
        ):
            return None

        q_all = mx.concatenate(q_parts, axis=1)  # type: ignore[arg-type]
        k_all = mx.concatenate(k_parts, axis=1)  # type: ignore[arg-type]
        v_all = mx.concatenate(v_parts, axis=1)  # type: ignore[arg-type]
        node_conv_states_out = mx.concatenate(node_conv_states, axis=0)  # type: ignore[arg-type]

    inv_scale = k_all.shape[-1] ** -0.5
    q_all = (inv_scale**2) * mx.fast.rms_norm(q_all, None, 1e-6)
    k_all = inv_scale * mx.fast.rms_norm(k_all, None, 1e-6)
    _profile_eval(profile_timings, "linear_qk_norm_ns", q_all, k_all)

    g = compute_g(linear_attn.A_log, a, linear_attn.dt_bias)
    beta = mx.sigmoid(b)
    _profile_eval(profile_timings, "linear_g_beta_ns", g, beta)
    kernel_result = tree_gated_delta_kernel(
        q_all,
        k_all,
        v_all,
        g,
        beta,
        base_state,
        mx.array(parents, dtype=mx.int32),
    )
    if kernel_result is None:
        return None

    raw_out, node_states = kernel_result
    _profile_eval(profile_timings, "linear_recurrence_kernel_ns", raw_out, node_states)
    out = linear_attn.norm(raw_out, z)
    out = linear_attn.out_proj(out.reshape(B, T, -1))
    _profile_eval(profile_timings, "linear_out_proj_ns", out)

    if linear_attn.sharding_group is not None:
        out = mx.distributed.all_sum(out, group=linear_attn.sharding_group)

    return (
        out,
        node_states[0],
        node_conv_states_out,
    )


def tree_verify_forward(
    target_model: Any,
    *,
    compiled_tree: CompiledTree,
    cache: list[Any],
    capture_layer_ids: Optional[set[int]] = None,
    profile_timings: Optional[dict[str, Any]] = None,
    tree_aware_linear: bool = False,
    tree_cache_state: Optional[dict[str, Any]] = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    """Run the target model on all tree nodes with tree attention.

    Tree-aware mode processes tokens in tree-index order and forks recurrent
    state by parent. Legacy mode reorders to DFS for sequential recurrence.
    Attention layers use tree masks + per-token RoPE for correct scoring.

    Args:
        target_model: The loaded MLX target model (TextModel).
        compiled_tree: CompiledTree from compile_tree().
        cache: List of per-layer caches (KVCache for attention, ArraysCache for linear).
        capture_layer_ids: Set of layer indices to capture hidden states (for draft conditioning).
        profile_timings: Optional dict updated with synchronized layer timings.
        tree_aware_linear: Use parent-state forking for GatedDeltaNet layers.
        tree_cache_state: Optional dict populated with per-node linear states.

    Returns:
        (logits, captured_hidden_states):
        - logits: (1, tree_size, vocab_size) in tree-index order
        - captured_hidden_states: {layer_id: (1, tree_size, hidden_dim)} in tree-index order
    """
    ct = compiled_tree

    # Get model internals (handles VL models, nested wrappers)
    inner = _target_text_model(target_model)

    # Find actual KV cache offset from an attention layer cache
    fa_idx = getattr(inner, "fa_idx", None)
    if fa_idx is not None and cache[fa_idx] is not None:
        actual_prefix = int(getattr(cache[fa_idx], "offset", 0) or 0)
    else:
        # Fallback: find first KVCache
        actual_prefix = 0
        for c in cache:
            if hasattr(c, "offset") and not hasattr(c, "cache"):
                actual_prefix = int(c.offset or 0)
                break

    dfs = ct.dfs_order
    inv_dfs = ct.inv_dfs_order
    depth_groups = _tree_depth_groups(ct.depths)

    # Embed tokens in tree-index order.
    h = inner.embed_tokens(ct.input_ids)  # (1, tree_size, hidden_dim)

    # Build attention mask from tree visibility, using actual cache prefix length
    # Tree visibility additive mask (tree-index order): (N+1, N+1)
    tree_vis = ct.attention_mask

    if tree_aware_linear:
        position_ids = ct.position_ids
        tree_mask = tree_vis
    else:
        # Reorder tokens and mask to DFS order for the legacy recurrent path.
        h = h[:, dfs, :]
        position_ids = ct.position_ids[dfs]
        tree_mask = tree_vis[:, :, dfs, :][:, :, :, dfs]

    use_exact_tree_attention = _exact_tree_attention_enabled(actual_prefix)
    if use_exact_tree_attention:
        attention_mask = tree_mask.astype(h.dtype)
    else:
        # All tree nodes attend to entire prefix → zeros (attend).
        # Use actual_prefix from cache, not compiled prefix_len.
        prefix_mask_dfs = mx.zeros((1, 1, ct.tree_size, actual_prefix), dtype=mx.float32)
        attention_mask = mx.concatenate([prefix_mask_dfs, tree_mask], axis=-1)
        attention_mask = attention_mask.astype(h.dtype)
    _profile_eval(profile_timings, "attention_mask_build_ns", attention_mask)

    # SSM mask for linear layers (None for standard ArraysCache)
    ssm_cache_idx = getattr(inner, "ssm_idx", 0)
    from mlx_lm.models.base import create_ssm_mask
    ssm_mask = create_ssm_mask(h, cache[ssm_cache_idx])

    # Track captured hidden states
    captured: dict[int, mx.array] = {}
    if capture_layer_ids and 0 in capture_layer_ids:
        captured[0] = h if tree_aware_linear else h[:, inv_dfs, :]

    if tree_cache_state is not None:
        tree_cache_state["linear_layers"] = {}
        tree_cache_state["attention_append_order"] = (
            "tree" if tree_aware_linear else "dfs"
        )

    # Process through each layer
    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        layer_start_ns = time.perf_counter_ns() if profile_timings is not None else 0
        if layer.is_linear:
            linear_input = layer.input_layernorm(h)
            _profile_eval(profile_timings, "linear_input_norm_ns", linear_input)
            if tree_aware_linear:
                r, node_states, node_conv_states = _linear_forward_tree_aware(
                    layer.linear_attn,
                    linear_input,
                    layer_cache,
                    parents=ct.parents,
                    depth_groups=depth_groups,
                    profile_timings=profile_timings,
                )
                if tree_cache_state is not None:
                    tree_cache_state["linear_layers"][layer_idx] = {
                        "states": node_states,
                        "conv_states": node_conv_states,
                    }
            else:
                # Linear layer: process in DFS order (sequential recurrent)
                r = layer.linear_attn(linear_input, ssm_mask, layer_cache)
            h = h + r
            mlp_input = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(mlp_input)
            _profile_eval(profile_timings, "linear_mlp_ns", mlp_out)
            h = h + mlp_out
        else:
            # Attention layer: custom forward with per-token RoPE + tree mask
            attn_input = layer.input_layernorm(h)
            _profile_eval(profile_timings, "attention_input_norm_ns", attn_input)
            r = _attention_forward_with_tree(
                layer.self_attn,
                attn_input,
                position_ids,
                attention_mask,
                layer_cache,
                cached_prefix_len=actual_prefix,
                tree_mask_only=use_exact_tree_attention,
                profile_timings=profile_timings,
            )
            h = h + r
            mlp_input = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(mlp_input)
            _profile_eval(profile_timings, "attention_mlp_ns", mlp_out)
            h = h + mlp_out

        if profile_timings is not None:
            mx.eval(h)
            key = "linear_ns" if layer.is_linear else "attention_ns"
            profile_timings[key] = profile_timings.get(key, 0) + (
                time.perf_counter_ns() - layer_start_ns
            )

        if capture_layer_ids and (layer_idx + 1) in capture_layer_ids:
            captured[layer_idx + 1] = h if tree_aware_linear else h[:, inv_dfs, :]

    # Final norm and LM head
    normalized = inner.norm(h)
    _profile_eval(profile_timings, "final_norm_ns", normalized)

    if not tree_aware_linear:
        # Reorder back to tree-index order for logits
        normalized = normalized[:, inv_dfs, :]

    logits = _lm_head_logits(target_model, normalized)
    _profile_eval(profile_timings, "lm_head_ns", logits)

    return logits, captured
