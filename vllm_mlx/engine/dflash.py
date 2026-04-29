# SPDX-License-Identifier: Apache-2.0
"""
DFlash speculative-decoding engine.

Wraps the dflash MLX runtime (https://github.com/dflash/dflash) so that
`rapid-mlx serve <target> --drafter <drafter>` runs block-based draft+verify
with a separate drafter model that conditions on hidden states from selected
target layers.

This engine is single-request: only one prompt is generated at a time; further
requests wait on an asyncio.Lock. Continuous batching is not supported in this
mode.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..api.utils import clean_output_text
from .base import GenerationOutput
from .batched import BatchedEngine

logger = logging.getLogger(__name__)


_TOOL_STOP_AFTER_STRINGS = (
    "</tool_call>",
    "</minimax:tool_call>",
    "</invoke>",
    "<tool_call|>",
    "<|tool_call_end|>",
    "<｜tool▁call▁end｜>",
)


def _init_dflash_step_thread() -> None:
    """Mirror engine_core._init_mlx_step_thread for our private executor."""
    import mlx.core as mx

    stream = mx.new_stream(mx.default_device())
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    logger.info("DFlash step thread initialized: stream=%s", stream)


@dataclass
class _ActiveRequest:
    started_at: float
    mode: str = "dflash"
    first_token_at: float | None = None
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    generated_tokens: int = 0
    generation_tps: float = 0.0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    speculative_steps: int = 0
    acceptance_ratio: float = 0.0
    block_size: int = 0
    block_history: list[int] = field(default_factory=list)
    tree_budget: int = 0
    avg_tree_node_count: float = 0.0
    ddtree_fast_path_ratio: float = 0.0
    ngram_acceptance_ratio: float = 0.0
    ngram_cycles: int = 0
    ngram_fallback_cycles: int = 0
    ngram_tool_guard_cycles: int = 0
    cache_hit_type: str | None = None
    cached_tokens: int = 0


@dataclass
class _DDTreeCacheStats:
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


@dataclass
class _DDTreePrefixCacheFetch:
    state: Any | None
    hit_type: str = "miss"
    cached_tokens: int = 0


class _DDTreePrefixStateCache:
    """Small LRU cache for DFlash PromptPrefillState objects."""

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max(0, int(max_entries))
        self._entries: dict[tuple[int, ...], Any] = {}
        self._lru: list[tuple[int, ...]] = []
        self.stats = _DDTreeCacheStats()

    def fetch(self, prompt_tokens: list[int]) -> _DDTreePrefixCacheFetch:
        self.stats.total_queries += 1
        if self.max_entries <= 0 or not prompt_tokens:
            self.stats.misses += 1
            return _DDTreePrefixCacheFetch(None)

        prompt_tuple = tuple(int(token) for token in prompt_tokens)
        best_key: tuple[int, ...] | None = None
        for key in self._entries:
            if len(key) > len(prompt_tuple):
                continue
            if prompt_tuple[: len(key)] == key and (
                best_key is None or len(key) > len(best_key)
            ):
                best_key = key

        if best_key is None:
            self.stats.misses += 1
            return _DDTreePrefixCacheFetch(None)

        self.stats.hits += 1
        self.stats.tokens_saved += len(best_key)
        self._touch(best_key)
        hit_type = "exact" if len(best_key) == len(prompt_tuple) else "prefix"
        return _DDTreePrefixCacheFetch(
            copy.deepcopy(self._entries[best_key]),
            hit_type=hit_type,
            cached_tokens=len(best_key),
        )

    def store(self, prompt_tokens: list[int], state: Any | None) -> None:
        if self.max_entries <= 0 or state is None or not prompt_tokens:
            return
        key = tuple(int(token) for token in prompt_tokens)
        self._entries[key] = copy.deepcopy(state)
        self._touch(key)
        while len(self._entries) > self.max_entries:
            evict_key = self._lru.pop(0)
            if evict_key in self._entries:
                del self._entries[evict_key]
                self.stats.evictions += 1

    def _touch(self, key: tuple[int, ...]) -> None:
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def get_stats(self) -> dict[str, Any]:
        data = self.stats.to_dict()
        data.update(
            {
                "entry_count": len(self._entries),
                "max_entries": self.max_entries,
                "longest_prefix_tokens": max(
                    (len(key) for key in self._entries),
                    default=0,
                ),
            }
        )
        return data


class DFlashEngine(BatchedEngine):
    """Speculative-decoding engine using a separate drafter model."""

    def __init__(
        self,
        model_name: str,
        drafter_path: str,
        block_size: int | None = None,
        adaptive: bool = True,
        adaptive_min: int = 8,
        adaptive_max: int = 22,
        turboquant_bits: float | None = None,
        ddtree_budget: int = 0,
        ddtree_block_size: int | None = None,
        fallback_mode: str | None = None,
        ngram_num_draft_tokens: int | None = None,
        ngram_size: int | None = None,
        ngram_min_matches: int | None = None,
        ngram_disable_threshold: float | None = None,
        ngram_disable_window: int | None = None,
        ngram_disable_cooldown: int | None = None,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if self._is_mllm:
            raise ValueError(
                "DFlash mode is text-only; do not pass --mllm or use a multimodal model."
            )

        self._drafter_path = drafter_path
        self._drafter: Any = None
        self._block_size_override = block_size
        self._adaptive_enabled = bool(adaptive)
        self._adaptive_min = int(adaptive_min)
        self._adaptive_max = int(adaptive_max)
        self._turboquant_bits = turboquant_bits
        self._ddtree_budget = max(0, int(ddtree_budget or 0))
        self._ddtree_block_size = ddtree_block_size
        self._ddtree_last: dict[str, Any] = {}
        self._dflash_fallback_mode = fallback_mode
        self._ngram_first_enabled = (
            self._ddtree_budget > 0 and fallback_mode == "ngram"
        )
        self._ngram_num_draft_tokens = max(1, int(ngram_num_draft_tokens or 4))
        self._ngram_size = max(1, int(ngram_size or 3))
        self._ngram_min_matches = max(1, int(ngram_min_matches or 1))
        self._ngram_disable_threshold = float(
            0.55 if ngram_disable_threshold is None else ngram_disable_threshold
        )
        self._ngram_disable_window = max(1, int(ngram_disable_window or 4))
        self._ngram_disable_cooldown = max(0, int(ngram_disable_cooldown or 8))

        self._lock = asyncio.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._active: _ActiveRequest | None = None
        self._inflight = 0

        self._adaptive_cfg: Any = None
        self._current_block_size = 0
        self._observed_block_min = 0
        self._observed_block_max = 0

        self._lifetime_proposed = 0
        self._lifetime_accepted = 0
        self._lifetime_responses = 0
        self._ddtree_responses = 0
        cache_entries = getattr(scheduler_config, "prefix_cache_size", 2) or 2
        cache_entries = int(os.environ.get("DFLASH_DDTREE_PREFIX_CACHE_ENTRIES", cache_entries))
        self._ddtree_prefix_cache = _DDTreePrefixStateCache(max_entries=cache_entries)
        self._last_memory_stats = (0.0, 0.0, 0.0)
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _start_llm(self) -> None:  # noqa: D401 — overrides BatchedEngine
        # Create the dedicated MLX worker thread BEFORE loading the model.
        # All model + drafter operations (including the initial weight load)
        # must run on this thread so that mlx-lm's per-thread generation
        # stream owns the arrays used during inference (PR 161 invariant).
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dflash-step",
            initializer=_init_dflash_step_thread,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_models_in_thread)
        # mark as not-engine-started (we don't run AsyncEngineCore)
        self._engine_started = False

    def _load_models_in_thread(self) -> None:
        """Runs on the dflash-step worker thread."""
        try:
            from dflash.model_mlx import (
                AdaptiveBlockSizeConfig,
                load_draft,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DFlash mode requires the `dflash[mlx]` package. Install with:\n"
                "  pip install -e /Users/samuelfajreldines/dev/dflash[mlx]\n"
                f"Original error: {exc}"
            ) from exc

        from ..utils.tokenizer import load_model_with_fallback

        logger.info("[DFlash] Loading target model: %s", self._model_name)
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}
        if "qwen3" in self._model_name.lower():
            tokenizer_config["eos_token"] = "<|im_end|>"
        self._model, self._tokenizer = load_model_with_fallback(
            self._model_name,
            tokenizer_config=tokenizer_config,
        )

        logger.info("[DFlash] Loading drafter: %s", self._drafter_path)
        self._drafter = load_draft(
            self._drafter_path,
            turboquant_bits=self._turboquant_bits,
        )
        self._drafter.bind(self._model)

        self._current_block_size = (
            int(self._block_size_override)
            if self._block_size_override is not None
            else int(self._drafter.config.block_size)
        )

        if self._adaptive_enabled:
            self._adaptive_cfg = AdaptiveBlockSizeConfig(
                enabled=True,
                min_block_size=self._adaptive_min,
                max_block_size=self._adaptive_max,
                grow_threshold=0.88,
                shrink_threshold=0.55,
                grow_streak=2,
                shrink_streak=2,
            )
        else:
            self._adaptive_cfg = None

        try:
            import mlx.core as mx

            if mx.metal.is_available():
                info = mx.device_info()
                max_rec = info.get(
                    "max_recommended_working_set_size",
                    info.get("memory_size", 0),
                )
                if max_rec > 0:
                    soft = int(max_rec * self._gpu_memory_utilization)
                    mx.set_memory_limit(soft)
                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)
        except Exception as exc:
            logger.warning("[DFlash] Failed to set Metal memory limits: %s", exc)

    async def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._drafter = None

    # ------------------------------------------------------------------
    # Generation core
    # ------------------------------------------------------------------

    async def _stream_dflash(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Run dflash.stream_generate on the dflash-step thread, yielding GenerationResponse objects."""
        from dflash.model_mlx import stream_generate as _ds

        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"

        def _make_gen():
            return _ds(
                self._model,
                self._drafter,
                self._tokenizer,
                prompt,
                block_size=self._block_size_override,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                adaptive_block_size=self._adaptive_cfg,
            )

        gen = await loop.run_in_executor(executor, _make_gen)

        sentinel = object()

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        while True:
            resp = await loop.run_in_executor(executor, _next)
            if resp is sentinel:
                return
            yield resp

    async def _stream_ddtree(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        tools_requested: bool,
        prefix_boundary: int = 0,
    ):
        """Run the Rapid-MLX DDTree loop on the DFlash MLX worker thread."""
        from ..speculative.ddtree.engine import generate_ddtree

        if temperature not in (0, 0.0) or top_p not in (0, 0.0, 1, 1.0):
            logger.debug(
                "[DDTree] greedy DDTree path ignores sampler settings: temperature=%s top_p=%s",
                temperature,
                top_p,
            )

        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"
        stop_event = threading.Event()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        ddtree_block_size = (
            self._ddtree_block_size
            if self._ddtree_block_size is not None
            else self._block_size_override
        )
        if (
            ddtree_block_size is not None
            and self._ddtree_budget > 0
            and int(ddtree_block_size) <= self._ddtree_budget
        ):
            logger.warning(
                "[DDTree] Ignoring block_size=%s because it is <= tree_budget=%s; using drafter default.",
                ddtree_block_size,
                self._ddtree_budget,
            )
            ddtree_block_size = None

        def _run():
            from ..speculative.ddtree.engine import tokenize_prompt

            prompt_array = tokenize_prompt(self._tokenizer, prompt)
            prompt_token_ids = prompt_array.tolist()
            cache_fetch = self._ddtree_prefix_cache.fetch(prompt_token_ids)

            def _on_step(step: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, step)

            try:
                result = generate_ddtree(
                target_model=self._model,
                draft_model=self._drafter,
                tokenizer=self._tokenizer,
                prompt_tokens=prompt_array,
                max_new_tokens=max_tokens,
                tree_budget=self._ddtree_budget,
                block_size=ddtree_block_size,
                adaptive_block_size=self._adaptive_cfg,
                prefix_state=cache_fetch.state,
                capture_prefill_state=True,
                target_turboquant_bits=self._turboquant_bits,
                stop_strings=stop,
                stop_after_strings=(
                    list(_TOOL_STOP_AFTER_STRINGS) if tools_requested else None
                ),
                prefix_boundary=prefix_boundary,
                ngram_num_draft_tokens=(
                    self._ngram_num_draft_tokens
                    if self._ngram_first_enabled
                    else None
                ),
                ngram_size=(
                    self._ngram_size
                    if self._ngram_first_enabled
                    else None
                ),
                ngram_min_matches=(
                    self._ngram_min_matches
                    if self._ngram_first_enabled
                    else None
                ),
                ngram_disable_threshold=(
                    self._ngram_disable_threshold
                    if self._ngram_first_enabled
                    else None
                ),
                ngram_disable_window=(
                    self._ngram_disable_window
                    if self._ngram_first_enabled
                    else None
                ),
                ngram_disable_cooldown=(
                    self._ngram_disable_cooldown
                    if self._ngram_first_enabled
                    else None
                ),
                should_stop=stop_event.is_set,
                on_step=_on_step,
            )
                if result.get("prompt_cache_state") is not None:
                    self._ddtree_prefix_cache.store(
                        list(result.get("prompt_token_ids") or ()),
                        result.get("prompt_cache_state"),
                    )
                if result.get("extended_prompt_cache_state") is not None:
                    self._ddtree_prefix_cache.store(
                        [
                            *list(result.get("prompt_token_ids") or ()),
                            *list(result.get("generated_token_ids") or ()),
                        ],
                        result.get("extended_prompt_cache_state"),
                    )
                if (
                    prefix_boundary > 0
                    and prefix_boundary < len(prompt_token_ids)
                    and result.get("prefix_boundary_state") is not None
                ):
                    self._ddtree_prefix_cache.store(
                        prompt_token_ids[:prefix_boundary],
                        result.get("prefix_boundary_state"),
                    )
                self._ddtree_last = {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "prompt_cache_state",
                        "prompt_token_ids",
                        "prefix_boundary_state",
                        "extended_prompt_cache_state",
                    }
                }
                result["cache_hit_type"] = cache_fetch.hit_type
                result["cached_tokens"] = cache_fetch.cached_tokens
                loop.call_soon_threadsafe(queue.put_nowait, result)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        future = loop.run_in_executor(executor, _run)
        saw_step = False
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                is_final = "finish_reason" in item
                item_text = "" if is_final and saw_step else item.get("text", "")
                saw_step = saw_step or not is_final
                yield SimpleNamespace(
                    text=item_text,
                    tokens=list(item.get("generated_token_ids", []) or []),
                    prompt_tokens=int(item.get("prompt_tokens") or 0),
                    prefill_seconds=float(item.get("prefill_seconds") or 0.0),
                    generation_tokens=int(item.get("generated_tokens") or 0),
                    generation_tps=float(item.get("generation_tps") or 0.0),
                    finish_reason=item.get("finish_reason") if is_final else None,
                    proposed_tokens=int(item.get("proposed_tokens") or 0),
                    accepted_tokens=int(item.get("accepted_tokens") or 0),
                    speculative_steps=int(item.get("speculative_steps") or 0),
                    avg_acceptance_ratio=float(
                        item.get("avg_acceptance_ratio") or 0.0
                    ),
                    block_size_history=tuple(item.get("block_size_history") or ()),
                    avg_tree_node_count=float(
                        item.get("avg_tree_node_count") or 0.0
                    ),
                    ddtree_fast_path_ratio=float(
                        item.get("ddtree_fast_path_ratio") or 0.0
                    ),
                    tree_budget=int(item.get("tree_budget") or self._ddtree_budget),
                    ngram_acceptance_ratio=float(
                        item.get("ngram_acceptance_ratio") or 0.0
                    ),
                    ngram_cycles=int(item.get("ngram_cycles_completed") or 0),
                    ngram_fallback_cycles=int(
                        item.get("ngram_fallback_cycles") or 0
                    ),
                    ngram_tool_guard_cycles=int(
                        item.get("ngram_tool_guard_cycles") or 0
                    ),
                    cache_hit_type=item.get("cache_hit_type"),
                    cached_tokens=int(item.get("cached_tokens") or 0),
                )
            await future
        except BaseException:
            stop_event.set()
            raise

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        last = None
        text_parts: list[str] = []
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            images=images,
            videos=videos,
            **kwargs,
        ):
            last = output
            if output.new_text:
                text_parts.append(output.new_text)
        return GenerationOutput(
            text=clean_output_text("".join(text_parts)),
            prompt_tokens=last.prompt_tokens if last else 0,
            completion_tokens=last.completion_tokens if last else 0,
            finished=True,
            finish_reason=(last.finish_reason if last else None) or "stop",
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        if images or videos:
            raise ValueError("DFlash mode does not support images or videos.")

        self._inflight += 1
        try:
            async with self._lock:
                tools_requested = bool(kwargs.pop("tools_requested", False))
                prefix_boundary = int(kwargs.pop("prefix_boundary", 0) or 0)
                greedy_request = temperature in (0, 0.0)
                if self._ddtree_budget > 0 and not greedy_request:
                    logger.warning(
                        "[DDTree] falling back to DFlash for non-greedy request: temperature=%s top_p=%s",
                        temperature,
                        top_p,
                    )
                mode = (
                    "ddtree-ngram"
                    if self._ngram_first_enabled and greedy_request
                    else (
                        "ddtree"
                        if self._ddtree_budget > 0 and greedy_request
                        else "dflash"
                    )
                )
                self._track_request_start(mode)
                cumulative = ""
                last_resp = None
                try:
                    stream = (
                        self._stream_ddtree
                        if mode in ("ddtree", "ddtree-ngram")
                        else self._stream_dflash
                    )
                    if mode in ("ddtree", "ddtree-ngram"):
                        response_stream = stream(
                            prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            stop,
                            tools_requested,
                            prefix_boundary,
                        )
                    else:
                        response_stream = stream(prompt, max_tokens, temperature, top_p)
                    async for resp in response_stream:
                        last_resp = resp
                        new_text = resp.text or ""
                        cumulative += new_text
                        self._update_active(resp, new_text=new_text)

                        yield GenerationOutput(
                            text=clean_output_text(cumulative),
                            new_text=new_text,
                            tokens=list(resp.tokens) if resp.tokens else [],
                            prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.generation_tokens,
                            finished=bool(resp.finish_reason),
                            finish_reason=resp.finish_reason,
                        )
                finally:
                    self._track_request_end()

                # Some clients expect a final yield with finished=True
                if last_resp is not None and not last_resp.finish_reason:
                    yield GenerationOutput(
                        text=clean_output_text(cumulative),
                        new_text="",
                        tokens=[],
                        prompt_tokens=last_resp.prompt_tokens,
                        completion_tokens=last_resp.generation_tokens,
                        finished=True,
                        finish_reason="stop",
                    )
        finally:
            self._inflight -= 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _track_request_start(self, mode: str = "dflash") -> None:
        self._active = _ActiveRequest(
            started_at=time.time(),
            mode=mode,
            block_size=self._current_block_size,
            tree_budget=(
                self._ddtree_budget if mode in ("ddtree", "ddtree-ngram") else 0
            ),
        )

    def _track_request_end(self) -> None:
        if self._active is not None:
            self._lifetime_responses += 1
            self._lifetime_proposed += self._active.proposed_tokens
            self._lifetime_accepted += self._active.accepted_tokens
            if self._active.mode in ("ddtree", "ddtree-ngram"):
                self._ddtree_responses += 1
        self._active = None

    def _update_active(self, resp: Any, new_text: str = "") -> None:
        a = self._active
        if a is None:
            return
        if a.first_token_at is None and (new_text or resp.tokens):
            a.first_token_at = time.time()
        a.prompt_tokens = int(resp.prompt_tokens or 0)
        a.prefill_seconds = float(getattr(resp, "prefill_seconds", 0.0) or 0.0)
        a.generated_tokens = int(resp.generation_tokens or 0)
        a.generation_tps = float(getattr(resp, "generation_tps", 0.0) or 0.0)
        a.proposed_tokens = int(resp.proposed_tokens or 0)
        a.accepted_tokens = int(resp.accepted_tokens or 0)
        a.speculative_steps = int(resp.speculative_steps or 0)
        a.acceptance_ratio = float(resp.avg_acceptance_ratio or 0.0)
        a.avg_tree_node_count = float(getattr(resp, "avg_tree_node_count", 0.0) or 0.0)
        a.ddtree_fast_path_ratio = float(
            getattr(resp, "ddtree_fast_path_ratio", 0.0) or 0.0
        )
        a.ngram_acceptance_ratio = float(
            getattr(resp, "ngram_acceptance_ratio", 0.0) or 0.0
        )
        a.ngram_cycles = int(getattr(resp, "ngram_cycles", 0) or 0)
        a.ngram_fallback_cycles = int(
            getattr(resp, "ngram_fallback_cycles", 0) or 0
        )
        a.ngram_tool_guard_cycles = int(
            getattr(resp, "ngram_tool_guard_cycles", 0) or 0
        )
        a.cache_hit_type = getattr(resp, "cache_hit_type", None)
        a.cached_tokens = int(getattr(resp, "cached_tokens", 0) or 0)
        history = list(resp.block_size_history or ())
        if history:
            a.block_history = history
            a.block_size = int(history[-1])
            self._current_block_size = a.block_size
            mn = min(history)
            mx_ = max(history)
            self._observed_block_min = (
                mn if self._observed_block_min == 0 else min(self._observed_block_min, mn)
            )
            self._observed_block_max = max(self._observed_block_max, mx_)

    def get_stats(self) -> dict[str, Any]:
        active_mem_gb, peak_mem_gb, cache_mem_gb = self._last_memory_stats
        if self._active is None or _env_bool("DFLASH_STATS_MEMORY_DURING_ACTIVE", False):
            try:
                import mlx.core as mx

                active_mem_gb = mx.get_active_memory() / 1e9
                peak_mem_gb = mx.get_peak_memory() / 1e9
                cache_mem_gb = mx.get_cache_memory() / 1e9
                self._last_memory_stats = (active_mem_gb, peak_mem_gb, cache_mem_gb)
            except Exception:
                active_mem_gb = peak_mem_gb = cache_mem_gb = 0.0

        running_requests: list[dict[str, Any]] = []
        if self._active is not None:
            now = time.time()
            elapsed = now - self._active.started_at
            ttft = self._active.prefill_seconds if self._active.prefill_seconds > 0 else None
            if ttft is None and self._active.first_token_at:
                ttft = self._active.first_token_at - self._active.started_at
            tps = self._active.generation_tps if self._active.generation_tps > 0 else None
            if (
                tps is None
                and self._active.first_token_at
                and self._active.generated_tokens > 0
            ):
                window = now - self._active.first_token_at
                if window > 0.01:
                    tps = self._active.generated_tokens / window
            running_requests.append(
                {
                    "request_id": f"{self._active.mode}-active",
                    "mode": self._active.mode,
                    "status": "running",
                    "phase": (
                        "generation" if self._active.first_token_at else "prefill"
                    ),
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": self._active.prompt_tokens,
                    "completion_tokens": self._active.generated_tokens,
                    "max_tokens": 0,
                    "tokens_per_second": tps,
                    "ttft_s": ttft,
                    "acceptance_ratio": self._active.acceptance_ratio,
                    "block_size": self._active.block_size,
                    "speculative_steps": self._active.speculative_steps,
                    "accepted_tokens": self._active.accepted_tokens,
                    "proposed_tokens": self._active.proposed_tokens,
                    "tree_budget": self._active.tree_budget,
                    "avg_tree_node_count": self._active.avg_tree_node_count,
                    "ddtree_fast_path_ratio": self._active.ddtree_fast_path_ratio,
                    "ngram_acceptance_ratio": self._active.ngram_acceptance_ratio,
                    "ngram_cycles": self._active.ngram_cycles,
                    "ngram_fallback_cycles": self._active.ngram_fallback_cycles,
                    "ngram_tool_guard_cycles": self._active.ngram_tool_guard_cycles,
                    "cache_hit_type": self._active.cache_hit_type,
                    "cached_tokens": self._active.cached_tokens,
                    "progress": 0.0,
                }
            )

        lifetime_ratio = (
            (self._lifetime_accepted / self._lifetime_proposed)
            if self._lifetime_proposed > 0
            else 0.0
        )

        num_running = 1 if self._active is not None else 0
        num_waiting = max(0, self._inflight - num_running)

        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "is_mllm": False,
            "loaded": self._loaded,
            "running": self._active is not None,
            "uptime_seconds": time.time() - self._start_time,
            "num_running": num_running,
            "num_waiting": num_waiting,
            "total_requests_processed": self._lifetime_responses,
            "metal_active_memory_gb": active_mem_gb,
            "metal_peak_memory_gb": peak_mem_gb,
            "metal_cache_memory_gb": cache_mem_gb,
            "requests": running_requests,
            "dflash": {
                "mode": (
                    "ddtree-ngram"
                    if self._ngram_first_enabled
                    else ("ddtree" if self._ddtree_budget > 0 else "dflash")
                ),
                "lifetime_acceptance_ratio": lifetime_ratio,
                "current_block_size": self._current_block_size,
                "adaptive_enabled": self._adaptive_enabled,
                "adaptive_min": self._adaptive_min,
                "adaptive_max": self._adaptive_max,
                "observed_block_min": self._observed_block_min,
                "observed_block_max": self._observed_block_max,
                "ddtree_budget": self._ddtree_budget,
                "ddtree_block_size": self._ddtree_block_size,
                "ddtree_requests": self._ddtree_responses,
                "ddtree_last_fast_path_ratio": self._ddtree_last.get(
                    "ddtree_fast_path_ratio", 0.0
                ),
                "ddtree_last_avg_tree_node_count": self._ddtree_last.get(
                    "avg_tree_node_count", 0.0
                ),
                "ddtree_last_generation_tps": self._ddtree_last.get(
                    "generation_tps", 0.0
                ),
                "ngram_first_enabled": self._ngram_first_enabled,
                "ngram_num_draft_tokens": self._ngram_num_draft_tokens,
                "ngram_size": self._ngram_size,
                "ngram_min_matches": self._ngram_min_matches,
                "ngram_disable_threshold": self._ngram_disable_threshold,
                "ngram_disable_window": self._ngram_disable_window,
                "ngram_disable_cooldown": self._ngram_disable_cooldown,
                "ngram_last_acceptance_ratio": self._ddtree_last.get(
                    "ngram_acceptance_ratio", 0.0
                ),
                "ngram_last_cycles": self._ddtree_last.get(
                    "ngram_cycles_completed", 0
                ),
                "ngram_last_fallback_cycles": self._ddtree_last.get(
                    "ngram_fallback_cycles", 0
                ),
                "ngram_last_tool_guard_cycles": self._ddtree_last.get(
                    "ngram_tool_guard_cycles", 0
                ),
            },
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return self._ddtree_prefix_cache.get_stats()
