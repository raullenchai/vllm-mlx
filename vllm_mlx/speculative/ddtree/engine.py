from __future__ import annotations

import os
import copy
import time
from typing import Any

import mlx.core as mx
import numpy as np
from dflash.model_mlx import (
    AdaptiveBlockSizeConfig,
    PromptPrefillState,
    _patch_model,
    derive_prefill_prefix_state,
    next_adaptive_block_size,
    prefill_prompt,
    tokenize_prompt,
)
from dflash_mlx.runtime import (
    build_suppress_token_mask,
    extract_context_feature_from_dict,
    greedy_tokens_with_mask,
    target_forward_with_hidden_states,
)
from mlx_lm.models.cache import make_prompt_cache

from ..prompt_lookup import PromptLookupDecoder


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tokenizer_eos_token_ids(tokenizer: Any) -> set[int]:
    eos_values = getattr(tokenizer, "eos_token_ids", None)
    if eos_values is None:
        eos_values = []
    elif isinstance(eos_values, int):
        eos_values = [eos_values]

    eos_ids = {int(token_id) for token_id in eos_values if token_id is not None}
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_ids.add(int(eos_token_id))
        else:
            eos_ids.update(int(token_id) for token_id in eos_token_id if token_id is not None)
    return eos_ids


def _truncate_for_stop_text(
    text: str,
    *,
    stop_strings: list[str] | None = None,
    stop_after_strings: list[str] | None = None,
) -> tuple[str, bool]:
    best_index: int | None = None
    best_end: int | None = None

    for stop in stop_strings or ():
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
            best_end = index

    for stop in stop_after_strings or ():
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
            best_end = index + len(stop)

    if best_index is None or best_end is None:
        return text, False
    return text[:best_end], True


_TOOL_CALL_MARKERS = (
    "<tool_call",
    "</tool_call",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)


def _looks_like_tool_call_draft(
    tokenizer: Any,
    history: list[int],
    pending_token: int,
    draft_tokens: list[int],
) -> bool:
    probe = list(history[-96:]) + [int(pending_token), *[int(t) for t in draft_tokens]]
    try:
        text = tokenizer.decode(probe)
    except TypeError:
        text = tokenizer.decode(probe, skip_special_tokens=False)
    except Exception:
        return False
    return any(marker in text for marker in _TOOL_CALL_MARKERS)


def _import_ddtree_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from .cache import (
            slow_path_commit,
            snapshot_caches,
            tree_aware_path_commit,
        )
        from .compile import compile_tree
        from .tree import build_ddtree_tree_from_topk, follow_verified_tree
        from .verify import tree_verify_forward
    except ImportError as exc:
        raise RuntimeError(
            "DDTree engine requested, but Rapid-MLX DDTree modules could not be loaded."
        ) from exc
    return (
        build_ddtree_tree_from_topk,
        follow_verified_tree,
        compile_tree,
        tree_verify_forward,
        tree_aware_path_commit,
        snapshot_caches,
        slow_path_commit,
    )


def _can_tree_aware_commit(cache_entries: list[Any]) -> bool:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            continue
        if hasattr(cache_entry, "state") and not hasattr(cache_entry, "offset"):
            continue
        if hasattr(cache_entry, "offset") and not (
            hasattr(cache_entry, "keys") and hasattr(cache_entry, "values")
        ):
            return False
    return True


def _tree_token_id(tree: Any, root_token: int, tree_index: int) -> int:
    if tree_index == 0:
        return int(root_token)
    return int(tree.node_token_ids[tree_index - 1])


def _tree_token_ids(tree: Any, root_token: int, indices: list[int]) -> list[int]:
    return [_tree_token_id(tree, root_token, idx) for idx in indices]


def _tree_node_count(tree: Any) -> int:
    node_token_ids = getattr(tree, "node_token_ids", None)
    return 1 + (0 if node_token_ids is None else len(node_token_ids))


def _extract_context_feature_for_indices(
    captured_hidden_states: dict[int, mx.array],
    target_layer_ids: list[int],
    indices: mx.array | None = None,
) -> mx.array:
    selected = []
    for layer_id in target_layer_ids:
        key = int(layer_id) + 1
        hidden = captured_hidden_states.get(key)
        if hidden is None:
            raise KeyError(f"Missing captured hidden state for layer {key}")
        if indices is not None:
            hidden = hidden[:, indices, :]
        selected.append(hidden)
    if not selected:
        raise ValueError("target_layer_ids must not be empty")
    return mx.concatenate(selected, axis=-1)


def _build_tree_from_mlx_logits(
    draft_logits: mx.array,
    *,
    budget: int,
    build_ddtree_tree_from_topk: Any,
    suppress_mask: mx.array | None = None,
) -> Any:
    if budget <= 0 or int(draft_logits.shape[0]) == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )

    logits = draft_logits.astype(mx.float32)
    if suppress_mask is not None:
        floor = mx.array(-1e9, dtype=logits.dtype)
        logits = mx.where(suppress_mask, floor, logits)

    topk = min(int(budget), int(logits.shape[-1]))
    top_indices = mx.argpartition(-logits, kth=topk - 1, axis=-1)[:, :topk]
    top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
    sort_order = mx.argsort(-top_logits, axis=-1)
    top_token_ids = mx.take_along_axis(top_indices, sort_order, axis=-1)
    top_logits = mx.take_along_axis(top_logits, sort_order, axis=-1)
    if _env_bool("LOCAL_DFLASH_DDTREE_APPROX_LOGPROBS", True):
        normalizer = mx.logsumexp(top_logits, axis=-1, keepdims=True)
    else:
        normalizer = mx.logsumexp(logits, axis=-1, keepdims=True)
    top_log_probs = top_logits - normalizer
    mx.eval(top_token_ids, top_log_probs)
    return build_ddtree_tree_from_topk(
        np.array(top_token_ids, copy=False),
        np.array(top_log_probs, copy=False),
        budget,
    )


def _make_draft_cache(draft_model: Any) -> list[Any]:
    try:
        from dflash_mlx.model import ContextOnlyDraftKVCache
    except ImportError:
        if hasattr(draft_model, "make_cache"):
            return draft_model.make_cache()
        return make_prompt_cache(draft_model)

    draft_sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64"))
    draft_window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024"))
    return [
        ContextOnlyDraftKVCache(sink_size=draft_sink, window_size=draft_window)
        for _ in range(len(draft_model.layers))
    ]


def generate_ddtree(
    *,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_tokens: str | list[int] | mx.array,
    max_new_tokens: int,
    tree_budget: int,
    block_size: int | None = None,
    adaptive_block_size: AdaptiveBlockSizeConfig | None = None,
    prefix_state: PromptPrefillState | None = None,
    capture_prefill_state: bool = False,
    target_turboquant_bits: float | None = None,
    stop_strings: list[str] | None = None,
    stop_after_strings: list[str] | None = None,
    should_stop: Any = None,
    prefix_boundary: int = 0,
    on_step: Any = None,
    ngram_num_draft_tokens: int | None = None,
    ngram_size: int | None = None,
    ngram_min_matches: int | None = None,
    ngram_disable_threshold: float | None = None,
    ngram_disable_window: int | None = None,
    ngram_disable_cooldown: int | None = None,
) -> dict[str, Any]:
    (
        build_ddtree_tree_from_topk,
        follow_verified_tree,
        compile_tree,
        tree_verify_forward,
        tree_aware_path_commit,
        snapshot_caches,
        slow_path_commit,
    ) = _import_ddtree_modules()

    prompt_array = tokenize_prompt(tokenizer, prompt_tokens)
    prompt_token_ids = prompt_array.tolist()
    prompt_len = len(prompt_token_ids)

    if not hasattr(draft_model, "config"):
        raise RuntimeError("DDTree engine requires a DFlash MLX draft model with .config")
    draft_model.bind(target_model)

    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.config.target_layer_ids}
    _patch_model(target_model, list(draft_model.config.target_layer_ids))
    prefill = prefill_prompt(
        target_model,
        tokenizer,
        prompt_array,
        target_turboquant_bits=target_turboquant_bits,
        prefix_state=prefix_state,
        capture_prefill_state=capture_prefill_state,
    )
    target_cache = prefill.target_cache
    prefix_boundary_state = None
    if capture_prefill_state and prefix_boundary > 0 and prefill.prefill_state is not None:
        try:
            boundary = min(int(prefix_boundary), prompt_len)
            if boundary > 0:
                prefix_boundary_state = derive_prefill_prefix_state(
                    prefill.prefill_state,
                    boundary,
                )
        except Exception:
            prefix_boundary_state = None
    tree_aware_commit = _can_tree_aware_commit(target_cache)
    draft_cache = _make_draft_cache(draft_model)
    lm_holder = getattr(target_model, "language_model", target_model)
    lm_head = getattr(target_model, "lm_head", None) or getattr(lm_holder, "lm_head", None)
    vocab_size = getattr(getattr(lm_head, "weight", None), "shape", (0,))[0]
    if not vocab_size:
        vocab_size = getattr(tokenizer, "vocab_size", 0)
    if not vocab_size:
        raise RuntimeError("Could not infer vocabulary size for DDTree generation")
    suppress_mask = build_suppress_token_mask(int(vocab_size), None)
    tree_aware_linear = _env_bool("DDTREE_TREE_AWARE_LINEAR", True)
    if not tree_aware_linear:
        raise RuntimeError("This integration currently requires DDTREE_TREE_AWARE_LINEAR=1")
    if _env_bool("DDTREE_EXACT_COMMIT", False):
        raise RuntimeError("This integration does not currently support DDTREE_EXACT_COMMIT=1")

    started = time.perf_counter()
    prefill_logits = prefill.logits
    prefill_seconds = prefill.prefill_seconds
    prompt_tps = prefill.prompt_tps
    target_hidden = prefill.hidden
    staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_mask).reshape(-1)

    generated_token_ids: list[int] = []
    generated_hidden_chunks: list[mx.array] = []
    acceptance_lengths: list[int] = []
    acceptance_ratios: list[float] = []
    block_size_history: list[int] = []
    tree_node_count_history: list[int] = []
    cycles_completed = 0
    phase_timings_us = {
        "draft": 0.0,
        "ngram_verify": 0.0,
        "tree_build": 0.0,
        "tree_verify": 0.0,
        "commit": 0.0,
    }
    fast_path_count = 0
    ddtree_cycles_completed = 0
    stop_hit = False
    cancelled = False
    ngram_decoder: PromptLookupDecoder | None = None
    ngram_recent: list[tuple[int, int]] = []
    ngram_cooldown_remaining = 0
    ngram_cycles_completed = 0
    ngram_fallback_cycles = 0
    ngram_disabled_cycles = 0
    ngram_tool_guard_cycles = 0
    ngram_proposed_tokens = 0
    ngram_accepted_tokens = 0
    ngram_disabled_for_request = False
    if ngram_num_draft_tokens is not None:
        ngram_decoder = PromptLookupDecoder(
            num_draft_tokens=max(1, int(ngram_num_draft_tokens)),
            ngram_size=max(1, int(ngram_size or 3)),
            min_matches=max(1, int(ngram_min_matches or 1)),
        )
        ngram_decoder.add_prompt_tokens(prompt_token_ids)
    ngram_threshold = float(
        0.55 if ngram_disable_threshold is None else ngram_disable_threshold
    )
    ngram_window = max(1, int(ngram_disable_window or 4))
    ngram_cooldown = max(0, int(ngram_disable_cooldown or 8))

    def stop_requested() -> bool:
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:
            return False

    current_block_size = max(
        1,
        int(block_size if block_size is not None else draft_model.config.block_size),
    )
    _adaptive_hysteresis: dict[str, int] = {"grow": 0, "shrink": 0}
    eos_token_ids = _tokenizer_eos_token_ids(tokenizer)
    check_text_stops = bool(stop_strings or stop_after_strings)
    max_stop_chars = (
        max((len(s) for s in [*(stop_strings or ()), *(stop_after_strings or ())]), default=0)
        + 32
    )
    stop_text_tail = ""

    while len(generated_token_ids) < max_new_tokens:
        if stop_requested():
            cancelled = True
            break
        remaining = max_new_tokens - len(generated_token_ids)
        block_len = max(1, min(current_block_size, remaining))
        root_token = int(staged_first[0].item() if staged_first.ndim > 0 else staged_first.item())
        used_ngram = False
        tree_node_count = 0

        if ngram_decoder is not None:
            ngram_decoder.add_generated_token(root_token)

        draft_tokens: list[int] = []
        if ngram_decoder is not None:
            if ngram_disabled_for_request:
                ngram_disabled_cycles += 1
            elif ngram_cooldown_remaining > 0:
                ngram_disabled_cycles += 1
                ngram_cooldown_remaining -= 1
            else:
                draft_tokens = ngram_decoder.get_draft_tokens()

        if (
            draft_tokens
            and _looks_like_tool_call_draft(
                tokenizer,
                getattr(ngram_decoder, "_token_history", []),
                root_token,
                draft_tokens,
            )
        ):
            ngram_tool_guard_cycles += 1
            draft_tokens = []

        if draft_tokens and remaining > 1:
            used_ngram = True
            proposed_count = 1 + min(len(draft_tokens), remaining - 1)
            draft_tokens = draft_tokens[: max(0, proposed_count - 1)]
            verify_ids = mx.array([[root_token, *draft_tokens]], dtype=mx.uint32)
            cache_snapshot = snapshot_caches(target_cache)
            verify_started = time.perf_counter_ns()
            verify_logits, verify_hidden_raw = target_forward_with_hidden_states(
                target_model,
                input_ids=verify_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            posterior_mx = greedy_tokens_with_mask(verify_logits[0], suppress_mask)
            mx.eval(posterior_mx)
            phase_timings_us["ngram_verify"] += (
                time.perf_counter_ns() - verify_started
            ) / 1_000.0

            posterior_tokens = posterior_mx.tolist()
            ngram_draft_accepted = 0
            for draft_token, posterior_token in zip(draft_tokens, posterior_tokens):
                if int(draft_token) != int(posterior_token):
                    break
                ngram_draft_accepted += 1

            accepted_token_ids = [root_token, *draft_tokens[:ngram_draft_accepted]]
            bonus_token = int(posterior_tokens[ngram_draft_accepted])
            acceptance_len = len(accepted_token_ids)
            ngram_cycles_completed += 1
            ngram_proposed_tokens += len(draft_tokens)
            ngram_accepted_tokens += ngram_draft_accepted
            ngram_decoder.record_accepted(ngram_draft_accepted)
            ngram_recent.append((ngram_draft_accepted, len(draft_tokens)))
            if len(ngram_recent) > ngram_window:
                ngram_recent.pop(0)
            if len(ngram_recent) == ngram_window and ngram_cooldown > 0:
                recent_accepted = sum(item[0] for item in ngram_recent)
                recent_proposed = sum(item[1] for item in ngram_recent)
                recent_ratio = (
                    recent_accepted / recent_proposed
                    if recent_proposed > 0
                    else 0.0
                )
                if recent_ratio < ngram_threshold:
                    ngram_disabled_for_request = True
                    ngram_cooldown_remaining = ngram_cooldown
                    ngram_recent.clear()

            commit_started = time.perf_counter_ns()
            if ngram_draft_accepted == len(draft_tokens):
                target_hidden = extract_context_feature_from_dict(
                    verify_hidden_raw,
                    list(draft_model.config.target_layer_ids),
                )
            else:
                accepted_ids = mx.array([accepted_token_ids], dtype=mx.uint32)
                _, committed_hidden_raw = slow_path_commit(
                    target_model,
                    target_cache,
                    cache_snapshot,
                    accepted_ids,
                    capture_layer_ids=capture_layer_ids,
                )
                target_hidden = extract_context_feature_from_dict(
                    committed_hidden_raw,
                    list(draft_model.config.target_layer_ids),
                )
            phase_timings_us["commit"] += (
                time.perf_counter_ns() - commit_started
            ) / 1_000.0
        else:
            ngram_fallback_cycles += 1 if ngram_decoder is not None else 0
            proposed_count = block_len
            block_token_ids = mx.full(
                (block_len,),
                int(draft_model.config.mask_token_id),
                dtype=mx.uint32,
            )
            block_token_ids[0] = mx.array(root_token, dtype=mx.uint32)

            draft_started = time.perf_counter_ns()
            draft_logits = None
            if block_len > 1:
                draft_logits = draft_model(block_token_ids[None], target_hidden, draft_cache)
            phase_timings_us["draft"] += (time.perf_counter_ns() - draft_started) / 1_000.0

            build_started = time.perf_counter_ns()
            if draft_logits is None:
                tree = build_ddtree_tree_from_topk(
                    np.empty((0, 0), dtype=np.int64),
                    np.empty((0, 0), dtype=np.float32),
                    0,
                )
            else:
                tree = _build_tree_from_mlx_logits(
                    draft_logits[0, 1 - block_len:],
                    budget=tree_budget,
                    build_ddtree_tree_from_topk=build_ddtree_tree_from_topk,
                    suppress_mask=suppress_mask,
                )
            tree_node_count = _tree_node_count(tree)
            tree_node_count_history.append(tree_node_count)
            compiled_tree = compile_tree(tree, root_token, prefix_len=prompt_len + len(generated_token_ids))
            phase_timings_us["tree_build"] += (time.perf_counter_ns() - build_started) / 1_000.0

            verify_started = time.perf_counter_ns()
            tree_cache_state: dict[str, Any] = {}
            cache_snapshot = None if tree_aware_commit else snapshot_caches(target_cache)
            verify_logits, verify_hidden_raw = tree_verify_forward(
                target_model,
                compiled_tree=compiled_tree,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
                tree_aware_linear=True,
                tree_cache_state=tree_cache_state,
            )
            posterior_mx = greedy_tokens_with_mask(verify_logits[0], suppress_mask)
            mx.eval(posterior_mx)
            phase_timings_us["tree_verify"] += (time.perf_counter_ns() - verify_started) / 1_000.0

            posterior_tokens = posterior_mx.tolist()
            accepted_indices, bonus_token = follow_verified_tree(tree.child_maps, posterior_tokens)
            accepted_token_ids = _tree_token_ids(tree, root_token, accepted_indices)
            acceptance_len = len(accepted_token_ids)
            ddtree_cycles_completed += 1

            commit_started = time.perf_counter_ns()
            if tree_aware_commit:
                tree_aware_path_commit(
                    target_cache,
                    prefix_len=prompt_len + len(generated_token_ids),
                    accepted_indices=accepted_indices,
                    tree_cache_state=tree_cache_state,
                )
                accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
                target_hidden = _extract_context_feature_for_indices(
                    verify_hidden_raw,
                    list(draft_model.config.target_layer_ids),
                    accepted_idx_array,
                )
            else:
                if cache_snapshot is None:
                    raise RuntimeError("DDTree slow-path commit missing cache snapshot")
                accepted_ids = mx.array([accepted_token_ids], dtype=mx.uint32)
                _, committed_hidden_raw = slow_path_commit(
                    target_model,
                    target_cache,
                    cache_snapshot,
                    accepted_ids,
                    capture_layer_ids=capture_layer_ids,
                )
                target_hidden = extract_context_feature_from_dict(
                    committed_hidden_raw,
                    list(draft_model.config.target_layer_ids),
                )
            phase_timings_us["commit"] += (time.perf_counter_ns() - commit_started) / 1_000.0
            if tree_aware_commit:
                fast_path_count += 1

        block_size_history.append(proposed_count)
        acceptance_lengths.append(acceptance_len)
        acceptance_ratios.append(acceptance_len / max(proposed_count, 1))
        cycles_completed += 1
        if ngram_decoder is not None and len(accepted_token_ids) > 1:
            for token_id in accepted_token_ids[1:]:
                ngram_decoder.add_generated_token(token_id)

        emitted = accepted_token_ids
        for index, token_id in enumerate(accepted_token_ids):
            if token_id in eos_token_ids:
                emitted = accepted_token_ids[:index]
                stop_hit = True
                break
        generated_token_ids.extend(emitted)
        if emitted:
            generated_hidden_chunks.append(target_hidden[:, : len(emitted), :])
        staged_first = mx.array([bonus_token], dtype=mx.uint32)
        delta_text = tokenizer.decode(emitted) if emitted else ""

        if check_text_stops and delta_text:
            probe_text = stop_text_tail + delta_text
            _, text_stop_hit = _truncate_for_stop_text(
                probe_text,
                stop_strings=stop_strings,
                stop_after_strings=stop_after_strings,
            )
            if text_stop_hit:
                stop_hit = True
            stop_text_tail = probe_text[-max_stop_chars:]

        if on_step is not None and emitted:
            step_text = delta_text
            if step_text:
                try:
                    on_step(
                        {
                            "text": step_text,
                            "generated_token_ids": emitted,
                            "prompt_tokens": prompt_len,
                            "prefill_seconds": prefill_seconds,
                            "generated_tokens": len(generated_token_ids),
                            "proposed_tokens": sum(block_size_history),
                            "accepted_tokens": sum(acceptance_lengths),
                            "speculative_steps": cycles_completed,
                            "avg_acceptance_ratio": (
                                sum(acceptance_lengths) / max(sum(block_size_history), 1)
                            ),
                            "block_size_history": tuple(block_size_history),
                            "avg_tree_node_count": (
                                sum(tree_node_count_history)
                                / len(tree_node_count_history)
                                if tree_node_count_history
                                else 0.0
                            ),
                            "ddtree_fast_path_ratio": (
                                fast_path_count / ddtree_cycles_completed
                                if ddtree_cycles_completed > 0
                                else 0.0
                            ),
                            "tree_budget": tree_budget,
                            "ngram_acceptance_ratio": (
                                ngram_accepted_tokens / ngram_proposed_tokens
                                if ngram_proposed_tokens > 0
                                else 0.0
                            ),
                            "ngram_cycles_completed": ngram_cycles_completed,
                            "ngram_fallback_cycles": ngram_fallback_cycles,
                            "ngram_tool_guard_cycles": ngram_tool_guard_cycles,
                        }
                    )
                except Exception:
                    pass

        if stop_hit:
            break

        if not used_ngram:
            current_block_size = next_adaptive_block_size(
                current_block_size,
                acceptance_len,
                min(block_len, max(1, tree_node_count)),
                adaptive_block_size,
                hysteresis_state=_adaptive_hysteresis,
            )
        clear_interval = int(os.environ.get("DDTREE_CLEAR_CACHE_INTERVAL", "64"))
        if clear_interval > 0 and cycles_completed % clear_interval == 0:
            try:
                mx.clear_cache()
            except Exception:
                pass

    generated_token_ids = generated_token_ids[:max_new_tokens]
    text = tokenizer.decode(generated_token_ids) if generated_token_ids else ""
    if text:
        text, text_stop_hit = _truncate_for_stop_text(
            text,
            stop_strings=stop_strings,
            stop_after_strings=stop_after_strings,
        )
        stop_hit = stop_hit or text_stop_hit
    decode_seconds = max(time.perf_counter() - started, 1e-9)
    elapsed = prefill_seconds + decode_seconds
    proposed_tokens = sum(block_size_history)
    accepted_tokens = sum(acceptance_lengths)
    ngram_acceptance_ratio = (
        ngram_accepted_tokens / ngram_proposed_tokens
        if ngram_proposed_tokens > 0
        else 0.0
    )
    extended_prompt_cache_state = None
    if (
        capture_prefill_state
        and prefill.prefill_state is not None
        and generated_token_ids
        and not (stop_strings or stop_after_strings)
    ):
        try:
            extended_hidden = mx.concatenate(
                [prefill.hidden, *generated_hidden_chunks],
                axis=1,
            )
            extended_prompt_cache_state = PromptPrefillState(
                prompt_tokens=tuple([*prompt_token_ids, *generated_token_ids]),
                target_cache=copy.deepcopy(target_cache),
                hidden=extended_hidden,
                last_logits=None,
            )
        except Exception:
            extended_prompt_cache_state = None

    return {
        "text": text,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "finish_reason": "cancelled" if cancelled else ("stop" if stop_hit else "length"),
        "prompt_tokens": prompt_len,
        "prefill_seconds": prefill_seconds,
        "prompt_tps": prompt_tps,
        "reused_prefix_tokens": prefill.reused_prefix_tokens,
        "decode_seconds": decode_seconds,
        "generation_tps": (len(generated_token_ids) / decode_seconds if generated_token_ids else 0.0),
        "generated_tokens": len(generated_token_ids),
        "speculative_steps": cycles_completed,
        "proposed_tokens": proposed_tokens,
        "accepted_tokens": accepted_tokens,
        "avg_acceptance_length": (
            accepted_tokens / cycles_completed if cycles_completed > 0 else 0.0
        ),
        "avg_acceptance_ratio": (
            accepted_tokens / proposed_tokens if proposed_tokens > 0 else 0.0
        ),
        "acceptance_lengths": acceptance_lengths,
        "acceptance_ratios": acceptance_ratios,
        "block_size_history": block_size_history,
        "tree_node_count_history": tree_node_count_history,
        "avg_tree_node_count": (
            sum(tree_node_count_history) / len(tree_node_count_history)
            if tree_node_count_history
            else 0.0
        ),
        "max_tree_node_count": max(tree_node_count_history) if tree_node_count_history else 0,
        "adaptive_block_size": bool(adaptive_block_size and adaptive_block_size.enabled),
        "prefix_cache_source": "dflash_prefill_state" if prefill.reused_prefix_tokens else "none",
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "elapsed": elapsed,
        "prefill_hidden_bytes": prefill.hidden_bytes,
        "prefill_target_cache_bytes": prefill.target_cache_bytes,
        "prefill_logits_bytes": prefill.logits_bytes,
        "prefill_working_set_bytes": prefill.working_set_bytes,
        "prompt_cache_state_bytes": prefill.prefill_state_bytes,
        "prompt_cache_state": prefill.prefill_state if capture_prefill_state else None,
        "extended_prompt_cache_state": extended_prompt_cache_state,
        "prefix_boundary_state": prefix_boundary_state,
        "engine": "ddtree-ngram" if ngram_decoder is not None else "ddtree",
        "target_turboquant_bits": target_turboquant_bits,
        "ddtree_commit": "tree_aware" if tree_aware_commit else "slow_path",
        "tree_budget": tree_budget,
        "ddtree_cycles_completed": ddtree_cycles_completed,
        "ddtree_fast_path_ratio": (
            fast_path_count / ddtree_cycles_completed
            if ddtree_cycles_completed > 0
            else 0.0
        ),
        "ddtree_phase_timings_us": phase_timings_us,
        "ngram_enabled": ngram_decoder is not None,
        "ngram_num_draft_tokens": (
            ngram_decoder.num_draft_tokens if ngram_decoder is not None else 0
        ),
        "ngram_size": ngram_decoder.ngram_size if ngram_decoder is not None else 0,
        "ngram_min_matches": (
            ngram_decoder.min_matches if ngram_decoder is not None else 0
        ),
        "ngram_cycles_completed": ngram_cycles_completed,
        "ngram_fallback_cycles": ngram_fallback_cycles,
        "ngram_disabled_cycles": ngram_disabled_cycles,
        "ngram_tool_guard_cycles": ngram_tool_guard_cycles,
        "ngram_proposed_tokens": ngram_proposed_tokens,
        "ngram_accepted_tokens": ngram_accepted_tokens,
        "ngram_acceptance_ratio": ngram_acceptance_ratio,
        "ngram_cooldown_remaining": ngram_cooldown_remaining,
        "ngram_disabled_for_request": ngram_disabled_for_request,
    }
