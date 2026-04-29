"""
Compile a DDTree into MLX tensors for the verification forward pass.

Takes the tree structure (Python/NumPy) and produces:
- input_ids: token IDs for all tree nodes
- position_ids: absolute positions for per-token RoPE
- tree_attention_mask: additive tree-to-tree mask for SDPA
- dfs_order / inv_dfs_order: reordering indices for linear layers
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import numpy as np

from .tree import DDTree, compute_dfs_order


class CompiledTree(NamedTuple):
    """MLX tensors ready for tree verification."""

    input_ids: mx.array       # (1, N+1) uint32 — root token + tree node tokens
    position_ids: mx.array    # (N+1,) int32 — absolute positions for RoPE
    attention_mask: mx.array  # (1, 1, N+1, N+1) float32 — tree-only additive mask
    dfs_order: mx.array       # (N+1,) int32 — indices to reorder to DFS
    inv_dfs_order: mx.array   # (N+1,) int32 — indices to reorder back from DFS
    parents: list[int]        # (N+1,) — parent tree index for each node
    depths: list[int]         # (N+1,) — root depth 0, drafted nodes 1..L
    tree_size: int             # N+1 (root + nodes)


def compile_tree(
    tree: DDTree,
    root_token_id: int,
    prefix_len: int,
) -> CompiledTree:
    """Compile a DDTree into MLX tensors for verification.

    Args:
        tree: DDTree from build_ddtree_tree.
        root_token_id: The bonus token (root of the tree).
        prefix_len: Number of tokens already in KV cache (context length).

    Returns:
        CompiledTree with all tensors needed for tree_verify_forward.
    """
    tree_size = 1 + tree.node_count  # root + nodes

    # 1. Input IDs: [root_token, node_0_token, node_1_token, ...]
    token_ids = np.empty(tree_size, dtype=np.int32)
    token_ids[0] = root_token_id
    if tree.node_count > 0:
        token_ids[1:] = tree.node_token_ids
    input_ids = mx.array(token_ids, dtype=mx.uint32)[None]  # (1, tree_size)

    # 2. Position IDs: root is at prefix_len, each node at prefix_len + depth
    positions = np.empty(tree_size, dtype=np.int32)
    positions[0] = prefix_len
    if tree.node_count > 0:
        positions[1:] = prefix_len + tree.node_depths
    position_ids = mx.array(positions, dtype=mx.int32)
    depths = [0]
    if tree.node_count > 0:
        depths.extend(int(depth) for depth in tree.node_depths.tolist())

    # 3. Attention mask: tree-to-tree visibility only. Prefix attention is
    # rebuilt in verify from the actual cache offset to avoid prefix-sized
    # allocations during compile.
    mask = np.where(tree.visibility, 0.0, -np.inf).astype(np.float32)
    attention_mask = mx.array(mask)[None, None, :, :]  # (1, 1, T, T)

    # 4. DFS ordering for linear layers
    dfs, inv_dfs = compute_dfs_order(tree)
    dfs_order = mx.array(dfs, dtype=mx.int32)
    inv_dfs_order = mx.array(inv_dfs, dtype=mx.int32)

    return CompiledTree(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        dfs_order=dfs_order,
        inv_dfs_order=inv_dfs_order,
        parents=list(tree.parents),
        depths=depths,
        tree_size=tree_size,
    )


def is_dfs_prefix(accepted_indices: list[int], dfs_order: list[int] | mx.array) -> bool:
    """Check if the accepted path is a prefix of the DFS traversal order.

    When True, we can use the fast-path commit (tape rollback for linear layers).
    When False, we must use the slow-path commit (re-forward from snapshot).

    Args:
        accepted_indices: Node indices on the accepted path (from follow_verified_tree).
        dfs_order: DFS traversal order from compute_dfs_order.

    Returns:
        True if accepted_indices == dfs_order[:len(accepted_indices)].
    """
    n = len(accepted_indices)
    if isinstance(dfs_order, mx.array):
        dfs_prefix = dfs_order[:n].tolist()
    else:
        dfs_prefix = dfs_order[:n]
    return accepted_indices == dfs_prefix
