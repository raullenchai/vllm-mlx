"""
DDTree tree building and walking — ported from liranringel/ddtree.

Pure Python/NumPy, no framework dependencies. The heap-based tree
construction implements Algorithm 1 from the DDTree paper: best-first
search over per-position draft distributions under a fixed node budget.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple

import numpy as np


class DDTree(NamedTuple):
    """Result of tree construction.

    All indices are 0-based with index 0 = root (the bonus token).
    Nodes 1..N are the drafted tree nodes.
    """

    node_token_ids: np.ndarray  # (N,) int64 — token ID for each tree node
    node_depths: np.ndarray     # (N,) int64 — depth of each node (1-based: root's children are depth 1)
    parents: list[int]          # (N+1,) — parent index for each node; parents[0] = -1 (root)
    child_maps: list[dict[int, int]]  # (N+1,) — {token_id: child_index} for each node
    visibility: np.ndarray      # (N+1, N+1) bool — ancestor-only attention mask
    node_count: int             # number of tree nodes (excluding root)


def build_ddtree_tree_from_topk(
    top_token_ids: np.ndarray,
    top_log_probs: np.ndarray,
    budget: int,
) -> DDTree:
    """Build a DDTree from precomputed per-position top-k log-probs.

    Args:
        top_token_ids: (L, K) int array sorted by descending log-probability.
        top_log_probs: (L, K) float array aligned with top_token_ids.
        budget: Maximum number of tree nodes (excluding root).
    """
    if budget <= 0 or top_token_ids.shape[0] == 0 or top_token_ids.shape[1] == 0:
        visibility = np.zeros((1, 1), dtype=np.bool_)
        visibility[0, 0] = True
        return DDTree(
            node_token_ids=np.empty(0, dtype=np.int64),
            node_depths=np.empty(0, dtype=np.int64),
            parents=[-1],
            child_maps=[{}],
            visibility=visibility,
            node_count=0,
        )

    top_token_ids = np.asarray(top_token_ids, dtype=np.int64)
    top_log_probs = np.asarray(top_log_probs, dtype=np.float32)
    topk = min(int(budget), int(top_token_ids.shape[1]))
    depth_limit = int(top_token_ids.shape[0])

    # Best-first heap search (Algorithm 1)
    # Heap entries: (-logw, ranks_tuple, parent_index, depth, rank, logw)
    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids = np.empty(budget, dtype=np.int64)
    node_depths = np.empty(budget, dtype=np.int64)
    parents = np.empty(budget + 1, dtype=np.int32)
    parents[0] = -1
    child_maps: list[dict[int, int]] = [{}]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids[node_count] = token_id
        node_depths[node_count] = depth
        parents[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Push sibling (next rank at same depth)
        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs[depth - 1, rank]) + float(top_log_probs[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        # Push first child (rank 0 at next depth)
        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    # Build visibility matrix (ancestor-only attention mask)
    current_length = 1 + node_count
    visibility = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents[index])
        visibility[index, :index] = visibility[parent_index, :index]
        visibility[index, index] = True

    return DDTree(
        node_token_ids=node_token_ids[:node_count],
        node_depths=node_depths[:node_count],
        parents=parents[:current_length].tolist(),
        child_maps=child_maps,
        visibility=visibility,
        node_count=node_count,
    )


def build_ddtree_tree(
    draft_logits: np.ndarray,
    budget: int,
) -> DDTree:
    """Build an optimal draft tree from block diffusion per-position logits.

    Implements Algorithm 1 from the DDTree paper. Uses a max-heap keyed by
    log-probability to greedily select the B highest-probability prefixes.

    Args:
        draft_logits: (L, vocab_size) float32 — per-position logits from
            the DFlash draft model for positions 1..L after the bonus token.
        budget: Maximum number of tree nodes (excluding root).

    Returns:
        DDTree namedtuple with tree structure and visibility matrix.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )

    topk = min(budget, draft_logits.shape[-1])

    # Compute top-K log-probabilities per position
    logits = draft_logits.astype(np.float32)
    # Partial sort for top-K indices
    top_indices = np.argpartition(-logits, topk - 1, axis=-1)[:, :topk]
    top_logits = np.take_along_axis(logits, top_indices, axis=-1)
    # Sort within top-K for descending order
    sort_order = np.argsort(-top_logits, axis=-1)
    top_token_ids = np.take_along_axis(top_indices, sort_order, axis=-1)
    top_logits = np.take_along_axis(top_logits, sort_order, axis=-1)

    # Log-softmax for log-probabilities
    log_z = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) + logits.max(axis=-1, keepdims=True)
    top_log_probs = top_logits - log_z

    return build_ddtree_tree_from_topk(
        top_token_ids=top_token_ids,
        top_log_probs=top_log_probs,
        budget=budget,
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
) -> tuple[list[int], int]:
    """Walk the verified tree following the target model's greedy tokens.

    Starting at the root (index 0), check if the target model's chosen token
    matches a child in the tree. If so, accept and continue. The walk stops
    at the first mismatch; that token becomes the bonus for the next round.

    Args:
        child_maps: Per-node {token_id: child_index} maps from DDTree.
        posterior_tokens: Greedy argmax token for each tree node (length = tree_size).

    Returns:
        (accepted_indices, bonus_token): indices of accepted nodes and the
        first unmatched target token (bonus for next round).
    """
    accepted_indices = [0]
    current_index = 0
    next_token = posterior_tokens[current_index]

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = posterior_tokens[current_index]

    return accepted_indices, next_token


def compute_dfs_order(tree: DDTree) -> tuple[list[int], list[int]]:
    """Compute DFS traversal order for tree nodes, highest-probability child first.

    The heap construction already produces nodes roughly in probability order,
    but we need explicit DFS ordering for linear layer processing.

    Args:
        tree: DDTree from build_ddtree_tree.

    Returns:
        (dfs_order, inv_dfs_order): dfs_order[i] = tree index at position i
        in DFS traversal. inv_dfs_order[tree_index] = position in DFS.
    """
    if tree.node_count == 0:
        return [0], [0]

    # Build children list per node (ordered by probability — first child in
    # child_maps is highest prob due to heap construction order)
    n = 1 + tree.node_count
    children: list[list[int]] = [[] for _ in range(n)]
    for idx in range(1, n):
        parent = tree.parents[idx]
        children[parent].append(idx)

    # DFS from root
    dfs_order: list[int] = []
    stack = [0]
    while stack:
        node = stack.pop()
        dfs_order.append(node)
        # Push children in reverse so first child is popped first
        for child in reversed(children[node]):
            stack.append(child)

    # Inverse mapping
    inv_dfs_order = [0] * n
    for pos, idx in enumerate(dfs_order):
        inv_dfs_order[idx] = pos

    return dfs_order, inv_dfs_order
