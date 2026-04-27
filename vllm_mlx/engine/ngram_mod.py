# SPDX-License-Identifier: Apache-2.0
"""
ngram-mod speculative-decoding engine.

Wraps `ngram_mod_generate_step` so that
`rapid-mlx serve <target> --spec-type ngram-mod` runs draft+verify with a
persistent hash-pool drafter (no separate draft model). Mirrors
DFlashEngine: single-request, owns a private MLX worker thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..api.utils import clean_output_text
from ..speculative.ngram_mod import (
    MultiLevelNGramDecoder,
    NGramModDecoder,
    ngram_mod_generate_step,
)
from .base import GenerationOutput
from .batched import BatchedEngine

logger = logging.getLogger(__name__)


def _looks_like_tool_prompt(prompt: str) -> bool:
    return any(
        marker in prompt
        for marker in (
            "<tool_call>",
            "<function=",
            "<minimax:tool_call>",
            '<invoke name="',
            "<|tool_call>",
            "[TOOL_CALLS]",
        )
    )


def _init_ngram_mod_step_thread() -> None:
    stream = mx.new_stream(mx.default_device())
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    logger.info("ngram-mod step thread initialized: stream=%s", stream)


@dataclass
class _ActiveRequest:
    started_at: float
    first_token_at: float | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    accepted_tokens: int = 0
    proposed_tokens: int = 0


class NGramModEngine(BatchedEngine):
    """Speculative-decoding engine using a persistent n-gram hash pool."""

    def __init__(
        self,
        model_name: str,
        n: int | list[int] = [16, 12, 8, 4, 2],
        pool_size: int = 1 << 20,
        n_min: int = 1,
        n_max: int = 16,
        reset_threshold: float = 0.05,
        reset_streak: int = 20,
        prefill_step_size: int = 512,
        force_greedy: bool = False,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=None,
            stream_interval=1,
            force_mllm=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if self._is_mllm:
            raise ValueError(
                "ngram-mod is text-only; do not pass --mllm or use a multimodal model."
            )

        if isinstance(n, list):
            self._decoder: NGramModDecoder | MultiLevelNGramDecoder = MultiLevelNGramDecoder(
                ns=n,
                pool_sizes=pool_size,
                n_min=n_min,
                n_max=n_max,
                reset_threshold=reset_threshold,
                reset_streak=reset_streak,
            )
        else:
            self._decoder = NGramModDecoder(
                n=n,
                pool_size=pool_size,
                n_min=n_min,
                n_max=n_max,
                reset_threshold=reset_threshold,
                reset_streak=reset_streak,
            )
        self._decoder.preseed_qwen3()
        self._prefill_step_size = int(prefill_step_size)
        self._force_greedy = bool(force_greedy)

        self._lock = asyncio.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._active: _ActiveRequest | None = None
        self._inflight = 0

        self._lifetime_responses = 0
        self._lifetime_proposed = 0
        self._lifetime_accepted = 0
        self._start_time = time.time()

    async def _start_llm(self) -> None:
        from mlx_lm import load as mlx_load

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ngram-mod-step",
            initializer=_init_ngram_mod_step_thread,
        )

        loop = asyncio.get_running_loop()

        def _load() -> None:
            logger.info("[ngram-mod] Loading target model: %s", self._model_name)
            self._model, self._tokenizer = mlx_load(self._model_name)

            try:
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
                logger.warning("[ngram-mod] Failed to set Metal memory limits: %s", exc)

        await loop.run_in_executor(self._executor, _load)
        self._engine_started = False

    async def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def _stream_ngram_mod(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.0,
    ):
        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "NGramModEngine not started"

        tokenizer = self._tokenizer
        token_ids = tokenizer.encode(prompt)
        prompt_tokens_len = len(token_ids)
        prompt_arr = mx.array(token_ids, mx.uint32)

        eos_ids: set[int] = set()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_ids.add(int(eos_id))
        for tid in getattr(tokenizer, "all_special_ids", []) or []:
            tok_str = (
                tokenizer.convert_ids_to_tokens(tid)
                if hasattr(tokenizer, "convert_ids_to_tokens")
                else None
            )
            if tok_str and ("im_end|>" in tok_str or "endoftext" in tok_str):
                eos_ids.add(int(tid))

        tool_prompt = _looks_like_tool_prompt(prompt)
        effective_temperature = (
            0.0
            if (self._force_greedy or tool_prompt)
            else float(temperature or 0.0)
        )
        logger.info(
            "[ngram-mod] request: temperature=%s (effective=%s) top_p=%s max_tokens=%s prompt_tokens=%d tool_prompt=%s",
            temperature,
            effective_temperature,
            top_p,
            max_tokens,
            prompt_tokens_len,
            tool_prompt,
        )

        _rep_penalty = float(repetition_penalty)
        # Mutable list shared with sampler closure; updated after each yielded token.
        # ngram_mod_generate_step maintains its own internal `seq` that we can't
        # reference from outside, so we maintain a parallel tracking list here.
        _recent: list[int] = list(token_ids)

        def _apply_repetition_penalty(logprobs: mx.array) -> mx.array:
            recent = _recent[-20:]
            if not recent:
                return logprobs
            vocab = logprobs.shape[-1]
            result = logprobs

            # Soft multiplicative penalty for recently generated tokens.
            # log-probs are ≤ 0; multiplying by >1 makes them more negative.
            unique = list(set(recent))
            idx = mx.array(unique, mx.int32)
            mask = mx.one_hot(idx, vocab, dtype=mx.float32).sum(axis=0) > 0
            for _ in range(result.ndim - 1):
                mask = mask[None]
            result = mx.where(mask, result * _rep_penalty, result)

            # Hard ban: tokens repeating ≥ 3x in last 10 positions get -1e9.
            # Soft penalty alone cannot break temperature=0 attractor states
            # (logprob near 0 × 1.3 is still near 0 vs competitors at -5+).
            tail = recent[-10:]
            banned = [t for t in set(tail) if tail.count(t) >= 3]
            if banned:
                bidx = mx.array(banned, mx.int32)
                ban_mask = mx.one_hot(bidx, vocab, dtype=mx.float32).sum(axis=0) > 0
                for _ in range(result.ndim - 1):
                    ban_mask = ban_mask[None]
                result = mx.where(ban_mask, mx.array(-1e9, dtype=result.dtype), result)

            return result

        if effective_temperature > 0:
            def sampler(logprobs):
                lp = _apply_repetition_penalty(logprobs) if _rep_penalty != 1.0 else logprobs
                scaled = lp / effective_temperature
                if 0.0 < top_p < 1.0:
                    sorted_logits = mx.sort(scaled, axis=-1)[..., ::-1]
                    sorted_probs = mx.softmax(sorted_logits, axis=-1)
                    cum = mx.cumsum(sorted_probs, axis=-1)
                    cutoff = mx.sum(cum < top_p, axis=-1, keepdims=True)
                    threshold = mx.take_along_axis(sorted_logits, cutoff, axis=-1)
                    scaled = mx.where(scaled < threshold, -mx.inf, scaled)
                return mx.random.categorical(scaled)
        else:
            def sampler(logprobs):
                lp = _apply_repetition_penalty(logprobs) if _rep_penalty != 1.0 else logprobs
                return mx.argmax(lp, axis=-1)

        def _make_gen():
            return ngram_mod_generate_step(
                prompt_arr,
                self._model,
                decoder=self._decoder,
                max_tokens=max_tokens,
                sampler=sampler,
                prefill_step_size=self._prefill_step_size,
                eos_ids=eos_ids,
            )

        gen = await loop.run_in_executor(executor, _make_gen)
        sentinel = object()

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        generated = 0
        proposed_before = self._decoder.lifetime_proposed
        accepted_before = self._decoder.lifetime_accepted
        first_token_at: float | None = None

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()

        finish_reason: str | None = None

        stop_seqs = [s for s in (stop or []) if s]
        max_stop_len = max((len(s) for s in stop_seqs), default=0)
        accumulated = ""
        emitted_chars = 0

        def _snapshot_counts() -> tuple[int, int]:
            return (
                self._decoder.lifetime_proposed - proposed_before,
                self._decoder.lifetime_accepted - accepted_before,
            )

        while True:
            item = await loop.run_in_executor(executor, _next)
            if item is sentinel:
                break

            tok, _logprobs, _from_draft = item
            if first_token_at is None:
                first_token_at = time.time()

            generated += 1

            # Update shared recent-token list so the sampler closure sees
            # the current generation history on subsequent steps.
            _recent.append(int(tok))
            if len(_recent) > 200:
                del _recent[:100]

            if tok in eos_ids:
                finish_reason = "stop"
                break

            detokenizer.add_token(int(tok))
            seg = detokenizer.last_segment
            if not seg:
                continue
            accumulated += seg

            if stop_seqs:
                hit_idx = -1
                for s in stop_seqs:
                    idx = accumulated.find(s, max(0, emitted_chars - (max_stop_len - 1)))
                    if idx != -1 and (hit_idx == -1 or idx < hit_idx):
                        hit_idx = idx
                if hit_idx != -1:
                    if hit_idx > emitted_chars:
                        to_emit = accumulated[emitted_chars:hit_idx]
                        proposed, accepted = _snapshot_counts()
                        yield {
                            "text": to_emit,
                            "prompt_tokens": prompt_tokens_len,
                            "generation_tokens": generated,
                            "accepted_tokens": accepted,
                            "proposed_tokens": proposed,
                            "first_token_at": first_token_at,
                            "finish_reason": None,
                        }
                    accumulated = accumulated[:hit_idx]
                    emitted_chars = len(accumulated)
                    finish_reason = "stop"
                    break

            safe_end = (
                max(emitted_chars, len(accumulated) - (max_stop_len - 1))
                if stop_seqs
                else len(accumulated)
            )
            if safe_end > emitted_chars:
                to_emit = accumulated[emitted_chars:safe_end]
                proposed, accepted = _snapshot_counts()
                yield {
                    "text": to_emit,
                    "prompt_tokens": prompt_tokens_len,
                    "generation_tokens": generated,
                    "accepted_tokens": accepted,
                    "proposed_tokens": proposed,
                    "first_token_at": first_token_at,
                    "finish_reason": None,
                }
                emitted_chars = safe_end

        detokenizer.finalize()
        tail = detokenizer.last_segment
        if tail:
            accumulated += tail
        proposed, accepted = _snapshot_counts()
        if finish_reason != "stop" and len(accumulated) > emitted_chars:
            yield {
                "text": accumulated[emitted_chars:],
                "prompt_tokens": prompt_tokens_len,
                "generation_tokens": generated,
                "accepted_tokens": accepted,
                "proposed_tokens": proposed,
                "first_token_at": first_token_at,
                "finish_reason": None,
            }
            emitted_chars = len(accumulated)

        if finish_reason is None:
            finish_reason = "length" if generated >= max_tokens else "stop"

        yield {
            "text": "",
            "prompt_tokens": prompt_tokens_len,
            "generation_tokens": generated,
            "accepted_tokens": accepted,
            "proposed_tokens": proposed,
            "first_token_at": first_token_at,
            "finish_reason": finish_reason,
            "_final": True,
        }

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
        if not self._loaded:
            await self.start()
        if images or videos:
            raise ValueError("ngram-mod does not support images or videos.")

        self._inflight += 1
        try:
            async with self._lock:
                self._track_request_start()
                parts: list[str] = []
                last: dict | None = None
                try:
                    async for chunk in self._stream_ngram_mod(
                        prompt, max_tokens, temperature, top_p, stop=stop
                    ):
                        last = chunk
                        if chunk.get("text"):
                            parts.append(chunk["text"])
                        self._update_active(chunk)
                finally:
                    self._track_request_end()

                return GenerationOutput(
                    text=clean_output_text("".join(parts)),
                    prompt_tokens=last["prompt_tokens"] if last else 0,
                    completion_tokens=last["generation_tokens"] if last else 0,
                    finished=True,
                    finish_reason=(last.get("finish_reason") if last else None) or "stop",
                )
        finally:
            self._inflight -= 1

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
            raise ValueError("ngram-mod does not support images or videos.")

        self._inflight += 1
        try:
            async with self._lock:
                self._track_request_start()
                cumulative = ""
                last: dict | None = None
                _rep_penalty = float(kwargs.pop("repetition_penalty", 1.0))
                try:
                    async for chunk in self._stream_ngram_mod(
                        prompt, max_tokens, temperature, top_p, stop=stop,
                        repetition_penalty=_rep_penalty,
                    ):
                        last = chunk
                        new_text = chunk.get("text") or ""
                        cumulative += new_text
                        self._update_active(chunk)

                        is_final = bool(chunk.get("_final"))
                        yield GenerationOutput(
                            text=clean_output_text(cumulative),
                            new_text=new_text,
                            tokens=[],
                            prompt_tokens=chunk["prompt_tokens"],
                            completion_tokens=chunk["generation_tokens"],
                            finished=is_final,
                            finish_reason=chunk.get("finish_reason"),
                        )
                finally:
                    self._track_request_end()
        finally:
            self._inflight -= 1

    def _track_request_start(self) -> None:
        self._active = _ActiveRequest(started_at=time.time())

    def _track_request_end(self) -> None:
        if self._active is not None:
            self._lifetime_responses += 1
            self._lifetime_proposed += self._active.proposed_tokens
            self._lifetime_accepted += self._active.accepted_tokens
        self._active = None
        # Reset per-request streak so cross-turn low-acceptance doesn't
        # trigger a pool reset at the wrong time in agentic tool-call loops.
        if self._decoder is not None:
            self._decoder._low_streak = 0

    def _update_active(self, chunk: dict) -> None:
        a = self._active
        if a is None:
            return
        if a.first_token_at is None and chunk.get("first_token_at"):
            a.first_token_at = chunk["first_token_at"]
        a.prompt_tokens = int(chunk.get("prompt_tokens") or 0)
        a.generated_tokens = int(chunk.get("generation_tokens") or 0)
        a.accepted_tokens = int(chunk.get("accepted_tokens") or 0)
        a.proposed_tokens = int(chunk.get("proposed_tokens") or 0)

    def get_stats(self) -> dict[str, Any]:
        try:
            active_mem_gb = mx.get_active_memory() / 1e9
            peak_mem_gb = mx.get_peak_memory() / 1e9
            cache_mem_gb = mx.get_cache_memory() / 1e9
        except Exception:
            active_mem_gb = peak_mem_gb = cache_mem_gb = 0.0

        running_requests: list[dict[str, Any]] = []
        if self._active is not None:
            now = time.time()
            elapsed = now - self._active.started_at
            ttft = (
                (self._active.first_token_at - self._active.started_at)
                if self._active.first_token_at
                else None
            )
            tps = None
            if (
                self._active.first_token_at
                and self._active.generated_tokens > 0
            ):
                window = now - self._active.first_token_at
                if window > 0.01:
                    tps = self._active.generated_tokens / window
            ratio = (
                self._active.accepted_tokens / self._active.proposed_tokens
                if self._active.proposed_tokens > 0
                else 0.0
            )
            running_requests.append(
                {
                    "request_id": "ngram-mod-active",
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
                    "acceptance_ratio": ratio,
                    "block_size": (
                        round(self._active.accepted_tokens / self._active.proposed_tokens, 2)
                        if self._active.proposed_tokens > 0
                        else None
                    ),
                    "accepted_tokens": self._active.accepted_tokens,
                    "proposed_tokens": self._active.proposed_tokens,
                    "cache_hit_type": None,
                    "cached_tokens": 0,
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

        decoder_stats = self._decoder.get_stats()
        is_multi = isinstance(self._decoder, MultiLevelNGramDecoder)

        ngram_section: dict[str, Any] = {
            "lifetime_acceptance_ratio": lifetime_ratio,
            "pool_size": decoder_stats["pool_size"],
            "pool_used": decoder_stats["used"],
            "pool_load": decoder_stats["load"],
            "pool_resets": decoder_stats["resets"],
            "n_min": self._decoder.n_min,
            "n_max": self._decoder.n_max,
        }
        if is_multi:
            ngram_section["ns"] = decoder_stats["ns"]
            ngram_section["levels"] = decoder_stats["levels"]
        else:
            ngram_section["n"] = decoder_stats["n"]

        return {
            "engine_type": "ngram-mod",
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
            "ngram_mod": ngram_section,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None
